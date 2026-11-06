"""Assignment evaluation, greedy baseline, and exhaustive search."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import multiprocessing
import os
import queue
import time
from typing import Iterable

import numpy as np

from .data import JOB_IDS


@dataclass(frozen=True)
class Assignment:
    mask: int
    vehicle1: tuple[str, ...]
    vehicle2: tuple[str, ...]
    count1: int
    count2: int
    t1: float
    t2: float
    difference: float
    squared_cost: float
    feasible: bool

    def to_dict(self) -> dict:
        value = asdict(self)
        value["vehicle1"] = list(self.vehicle1)
        value["vehicle2"] = list(self.vehicle2)
        return value


def evaluate_mask(mask: int, times: Iterable[float]) -> Assignment:
    """Decode bit i=1 as job i on vehicle 1; 0 means vehicle 2."""
    values = np.asarray(tuple(times), dtype=float)
    if values.shape != (6,):
        raise ValueError("Exactly six delivery times are required")
    if not 0 <= mask < 64:
        raise ValueError("Mask must be in [0, 63]")
    on_v1 = np.array([(mask >> i) & 1 for i in range(6)], dtype=bool)
    vehicle1 = tuple(JOB_IDS[i] for i in range(6) if on_v1[i])
    vehicle2 = tuple(JOB_IDS[i] for i in range(6) if not on_v1[i])
    t1 = float(values[on_v1].sum())
    t2 = float(values[~on_v1].sum())
    difference = abs(t1 - t2)
    return Assignment(
        mask=mask,
        vehicle1=vehicle1,
        vehicle2=vehicle2,
        count1=len(vehicle1),
        count2=len(vehicle2),
        t1=t1,
        t2=t2,
        difference=difference,
        squared_cost=difference * difference,
        feasible=2 <= len(vehicle1) <= 4,
    )


def greedy_longest_first(times: Iterable[float]) -> Assignment:
    """Assign descending jobs to the less-loaded feasible vehicle."""
    values = np.asarray(tuple(times), dtype=float)
    order = sorted(range(6), key=lambda i: (-values[i], JOB_IDS[i]))
    groups: list[list[int]] = [[], []]
    loads = [0.0, 0.0]
    for position, job in enumerate(order):
        remaining_after = len(order) - position - 1
        choices: list[int] = []
        for vehicle in (0, 1):
            new_counts = [len(groups[0]), len(groups[1])]
            new_counts[vehicle] += 1
            if new_counts[vehicle] > 4:
                continue
            minimum_needed = sum(max(0, 2 - count) for count in new_counts)
            if minimum_needed <= remaining_after:
                choices.append(vehicle)
        if not choices:
            raise RuntimeError("Greedy assignment reached an infeasible state")
        chosen = min(choices, key=lambda vehicle: (loads[vehicle], vehicle))
        groups[chosen].append(job)
        loads[chosen] += float(values[job])
    mask = sum(1 << job for job in groups[0])
    result = evaluate_mask(mask, values)
    if not result.feasible:
        raise AssertionError("Greedy result must be feasible")
    return result


def serial_enumeration(times: Iterable[float]) -> list[Assignment]:
    values = tuple(float(value) for value in times)
    return [evaluate_mask(mask, values) for mask in range(64)]


def _evaluate_chunk(
    masks: tuple[int, ...], times: tuple[float, ...], ready_queue, start_event
) -> tuple[int, list[Assignment]]:
    """Wait at a start barrier so both process workers participate."""
    ready_queue.put(os.getpid())
    if not start_event.wait(timeout=20):
        raise RuntimeError("Parallel worker start barrier timed out")
    return os.getpid(), [evaluate_mask(mask, times) for mask in masks]


def parallel_exhaustive_search(
    times: Iterable[float], workers: int = 2
) -> tuple[list[Assignment], Assignment, list[Assignment], list[int], float]:
    """Evaluate every mask and return the representative and all tied optima."""
    if workers < 2:
        raise ValueError("Parallel search requires at least two workers")
    values = tuple(float(value) for value in times)
    chunks = tuple(tuple(range(offset, 64, workers)) for offset in range(workers))
    started = time.perf_counter()
    with multiprocessing.Manager() as manager:
        ready_queue = manager.Queue()
        start_event = manager.Event()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_evaluate_chunk, chunk, values, ready_queue, start_event)
                for chunk in chunks
            ]
            ready_pids: set[int] = set()
            try:
                for _ in range(workers):
                    ready_pids.add(int(ready_queue.get(timeout=20)))
            except queue.Empty as exc:
                start_event.set()
                raise RuntimeError("Not all parallel workers reached the start barrier") from exc
            start_event.set()
            results = [future.result() for future in futures]
    runtime = time.perf_counter() - started
    pids = sorted({pid for pid, _ in results} | ready_pids)
    if len(pids) < 2:
        raise RuntimeError("Parallel execution did not use at least two worker processes")
    assignments = sorted(
        (item for _, chunk in results for item in chunk), key=lambda item: item.mask
    )
    feasible = [item for item in assignments if item.feasible]
    if len(assignments) != 64 or len(feasible) != 50:
        raise AssertionError("Expected 64 total and 50 feasible assignments")
    best = min(feasible, key=lambda item: (item.difference, item.mask))
    tied = [
        item for item in feasible
        if np.isclose(item.difference, best.difference, rtol=1e-12, atol=1e-12)
    ]
    return assignments, best, tied, pids, runtime

