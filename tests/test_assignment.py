import numpy as np

from q4.assignment import evaluate_mask, greedy_longest_first, parallel_exhaustive_search, serial_enumeration
from q4.data import train_and_predict


def test_assignment_space_and_totals():
    times = train_and_predict().predictions
    assignments = serial_enumeration(times)
    assert len(assignments) == 64
    feasible = [item for item in assignments if item.feasible]
    assert len(feasible) == 50
    for item in assignments:
        assert set(item.vehicle1).isdisjoint(item.vehicle2)
        assert len(item.vehicle1) + len(item.vehicle2) == 6
        assert np.isclose(item.t1 + item.t2, times.sum())
        assert np.isclose(item.difference, abs(item.t1 - item.t2))
    assert all(2 <= item.count1 <= 4 and 2 <= item.count2 <= 4 for item in feasible)


def test_greedy_is_deterministic_and_feasible():
    times = train_and_predict().predictions
    first = greedy_longest_first(times)
    second = greedy_longest_first(times)
    assert first == second
    assert first.feasible


def test_parallel_matches_serial_and_uses_processes():
    times = train_and_predict().predictions
    parallel, best, ties, pids, _ = parallel_exhaustive_search(times)
    serial = serial_enumeration(times)
    serial_best = min((item for item in serial if item.feasible), key=lambda item: (item.difference, item.mask))
    assert parallel == serial
    assert best == serial_best
    serial_ties = [
        item for item in serial
        if item.feasible and np.isclose(item.difference, serial_best.difference, rtol=1e-12, atol=1e-12)
    ]
    assert ties == serial_ties
    assert [item.mask for item in ties] == [22, 41]
    assert len(pids) >= 2

