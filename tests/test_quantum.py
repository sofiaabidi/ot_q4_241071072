import numpy as np

from q4.assignment import serial_enumeration
from q4.data import train_and_predict
from q4.quantum import imbalance, ising_energy, qaoa_state, qubo_energy, run_qaoa, verify_hamiltonian


def test_qubo_and_ising_match_squared_imbalance_for_every_mask():
    times = train_and_predict().predictions
    verify_hamiltonian(times)
    for mask in range(64):
        expected = imbalance(times, mask) ** 2
        assert np.isclose(qubo_energy(times, mask), expected, rtol=1e-8, atol=1e-8)
        assert np.isclose(ising_energy(times, mask), expected, rtol=1e-8, atol=1e-8)


def test_qaoa_statevector_is_normalized():
    times = train_and_predict().predictions
    state = qaoa_state(times, [0.2, 0.7, 0.4, 1.1])
    probabilities = np.abs(state) ** 2
    assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-10)
    assert np.all(probabilities >= 0)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-10)


def test_qaoa_candidate_is_observed_feasible_and_not_better_than_exact():
    times = train_and_predict().predictions
    result = run_qaoa(times, starts=2, shots=1024, maxiter=60)
    exact = min((item for item in serial_enumeration(times) if item.feasible), key=lambda item: (item.difference, item.mask))
    assert result.assignment.mask in result.sampled_counts
    assert result.sampled_counts[result.assignment.mask] > 0
    assert result.assignment.feasible
    assert result.feasible_shots <= 1024
    assert result.assignment.difference + 1e-10 >= exact.difference
    assert np.isclose(result.probabilities.sum(), 1.0, atol=1e-10)

