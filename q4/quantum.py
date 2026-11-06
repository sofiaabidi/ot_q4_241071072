"""Pure NumPy six-qubit gate-based QAOA statevector simulation."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import numpy as np
from scipy.optimize import minimize

from .assignment import Assignment, evaluate_mask


@dataclass(frozen=True)
class QAOAResult:
    assignment: Assignment
    parameters: np.ndarray
    expectation: float
    state_norm: float
    probabilities: np.ndarray
    sampled_counts: dict[int, int]
    feasible_shots: int
    most_likely_feasible_mask: int
    chosen_probability: float
    optimizer_runs: tuple[dict, ...]
    runtime_seconds: float


def imbalance(times: Iterable[float], mask: int) -> float:
    values = np.asarray(tuple(times), dtype=float)
    bits = np.array([(mask >> i) & 1 for i in range(6)], dtype=float)
    return float(2.0 * values @ bits - values.sum())


def qubo_energy(times: Iterable[float], mask: int) -> float:
    values = np.asarray(tuple(times), dtype=float)
    total = float(values.sum())
    bits = np.array([(mask >> i) & 1 for i in range(6)], dtype=float)
    linear = np.sum((4.0 * values**2 - 4.0 * total * values) * bits)
    quadratic = sum(
        8.0 * values[i] * values[j] * bits[i] * bits[j]
        for i in range(6)
        for j in range(i + 1, 6)
    )
    return float(total**2 + linear + quadratic)


def ising_energy(times: Iterable[float], mask: int) -> float:
    values = np.asarray(tuple(times), dtype=float)
    z = np.array([1.0 - 2.0 * ((mask >> i) & 1) for i in range(6)])
    identity = float(values @ values)
    interactions = sum(
        2.0 * values[i] * values[j] * z[i] * z[j]
        for i in range(6)
        for j in range(i + 1, 6)
    )
    return float(identity + interactions)


def scaled_energies(times: Iterable[float]) -> np.ndarray:
    values = np.asarray(tuple(times), dtype=float)
    scale = float(values.sum() ** 2)
    return np.array([ising_energy(values, mask) / scale for mask in range(64)])


def _apply_zz_cost(state: np.ndarray, times: np.ndarray, gamma: float) -> np.ndarray:
    """Apply product of ZZ phases; the omitted identity is only a global phase."""
    total_squared = float(times.sum() ** 2)
    indices = np.arange(64)
    phase_argument = np.zeros(64, dtype=float)
    for i in range(6):
        zi = 1.0 - 2.0 * ((indices >> i) & 1)
        for j in range(i + 1, 6):
            zj = 1.0 - 2.0 * ((indices >> j) & 1)
            phase_argument += (2.0 * times[i] * times[j] / total_squared) * zi * zj
    return state * np.exp(-1j * gamma * phase_argument)


def _apply_rx_mixers(state: np.ndarray, beta: float) -> np.ndarray:
    result = state.copy()
    cosine, sine = np.cos(beta), np.sin(beta)
    for qubit in range(6):
        step = 1 << qubit
        block = step << 1
        for start in range(0, 64, block):
            low = slice(start, start + step)
            high = slice(start + step, start + block)
            a = result[low].copy()
            b = result[high].copy()
            result[low] = cosine * a - 1j * sine * b
            result[high] = -1j * sine * a + cosine * b
    return result


def qaoa_state(times: Iterable[float], parameters: Iterable[float], p: int = 2) -> np.ndarray:
    values = np.asarray(tuple(times), dtype=float)
    angles = np.asarray(tuple(parameters), dtype=float)
    if values.shape != (6,) or angles.shape != (2 * p,):
        raise ValueError("Expected six times and p gamma plus p beta angles")
    state = np.full(64, 1.0 / 8.0, dtype=complex)  # H^6 |000000>
    for layer in range(p):
        state = _apply_zz_cost(state, values, angles[layer])
        state = _apply_rx_mixers(state, angles[p + layer])
    return state


def run_qaoa(
    times: Iterable[float], *, p: int = 2, starts: int = 6, shots: int = 4096,
    seed: int = 72, maxiter: int = 180
) -> QAOAResult:
    """Optimize QAOA expectation and choose the best observed feasible sample."""
    values = np.asarray(tuple(times), dtype=float)
    energies = scaled_energies(values)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    def objective(parameters: np.ndarray) -> float:
        probabilities = np.abs(qaoa_state(values, parameters, p)) ** 2
        return float(probabilities @ energies)

    runs: list[dict] = []
    best = None
    bounds = [(0.0, 2.0 * np.pi)] * p + [(0.0, np.pi)] * p
    for start in range(starts):
        initial = np.concatenate((rng.uniform(0, 2*np.pi, p), rng.uniform(0, np.pi, p)))
        result = minimize(
            objective, initial, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": maxiter, "ftol": 1e-12}
        )
        runs.append({
            "start": start,
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "expectation": float(result.fun),
            "parameters": result.x.tolist(),
        })
        if best is None or (float(result.fun), start) < (float(best.fun), best.start_index):
            result.start_index = start
            best = result
    assert best is not None
    state = qaoa_state(values, best.x, p)
    probabilities = np.abs(state) ** 2
    norm = float(np.linalg.norm(state))
    if not np.isclose(norm, 1.0, atol=1e-10) or not np.isclose(probabilities.sum(), 1.0, atol=1e-10):
        raise AssertionError("QAOA statevector is not normalized")
    samples = rng.choice(64, size=shots, p=probabilities)
    unique, counts = np.unique(samples, return_counts=True)
    sampled_counts = {int(mask): int(count) for mask, count in zip(unique, counts)}
    feasible_masks = [mask for mask in sampled_counts if 2 <= mask.bit_count() <= 4]
    if not feasible_masks:
        raise RuntimeError("No feasible QAOA measurements were observed")
    chosen_mask = min(feasible_masks, key=lambda mask: (evaluate_mask(mask, values).difference, mask))
    most_likely = min(feasible_masks, key=lambda mask: (-sampled_counts[mask], mask))
    runtime = time.perf_counter() - started
    return QAOAResult(
        assignment=evaluate_mask(chosen_mask, values),
        parameters=best.x.copy(),
        expectation=float(best.fun),
        state_norm=norm,
        probabilities=probabilities,
        sampled_counts=sampled_counts,
        feasible_shots=sum(sampled_counts[mask] for mask in feasible_masks),
        most_likely_feasible_mask=most_likely,
        chosen_probability=float(probabilities[chosen_mask]),
        optimizer_runs=tuple(runs),
        runtime_seconds=runtime,
    )


def verify_hamiltonian(times: Iterable[float], tolerance: float = 1e-8) -> None:
    values = tuple(float(value) for value in times)
    for mask in range(64):
        expected = imbalance(values, mask) ** 2
        if not np.isclose(qubo_energy(values, mask), expected, rtol=tolerance, atol=tolerance):
            raise AssertionError(f"QUBO mismatch for mask {mask}")
        if not np.isclose(ising_energy(values, mask), expected, rtol=tolerance, atol=tolerance):
            raise AssertionError(f"Ising mismatch for mask {mask}")

