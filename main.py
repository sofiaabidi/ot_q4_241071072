"""Run the complete Q4 experiment with one command."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
import sklearn

from q4.assignment import greedy_longest_first, parallel_exhaustive_search, serial_enumeration
from q4.data import JOB_IDS, train_and_predict
from q4.evidence import new_run_directory, write_evidence
from q4.quantum import run_qaoa, verify_hamiltonian


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _show(name, assignment, runtime):
    print(
        f"{name:20s} V1={list(assignment.vehicle1)} V2={list(assignment.vehicle2)} "
        f"T1={assignment.t1:.6f} T2={assignment.t2:.6f} "
        f"|diff|={assignment.difference:.6f} valid={assignment.feasible} runtime={runtime:.6f}s"
    )


def run() -> tuple[str, Path, dict]:
    run_id, directory = new_run_directory()
    log_path = directory / "console.log"
    with log_path.open("w", encoding="utf-8") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        print(f"Q4 cold-chain experiment run {run_id}")
        ml_started = time.perf_counter()
        regression = train_and_predict()
        ml_runtime = time.perf_counter() - ml_started
        print("\nPredicted job times (hours)")
        for job, prediction in zip(JOB_IDS, regression.predictions):
            print(f"  {job}: {prediction:.6f}")
        print(f"Holdout MAE={regression.mae:.6f} hours, R^2={regression.r2:.6f}")

        verify_hamiltonian(regression.predictions)
        greedy_started = time.perf_counter()
        greedy = greedy_longest_first(regression.predictions)
        greedy_runtime = time.perf_counter() - greedy_started
        qaoa = run_qaoa(regression.predictions)
        assignments, exact, exact_ties, pids, parallel_runtime = parallel_exhaustive_search(regression.predictions)
        serial_started = time.perf_counter()
        serial = serial_enumeration(regression.predictions)
        serial_best = min((item for item in serial if item.feasible), key=lambda item: (item.difference, item.mask))
        serial_runtime = time.perf_counter() - serial_started
        if exact.mask != serial_best.mask:
            raise AssertionError("Parallel result disagrees with independent serial enumeration")

        print("\nMethod comparison")
        _show("Greedy", greedy, greedy_runtime)
        _show("QAOA", qaoa.assignment, qaoa.runtime_seconds)
        _show("Parallel exhaustive", exact, parallel_runtime)
        print(f"Parallel worker PIDs: {pids}; serial runtime={serial_runtime:.6f}s")
        print(f"All tied exact optimum masks: {[item.mask for item in exact_ties]}")
        print(f"QAOA chosen mask {qaoa.assignment.mask} was observed {qaoa.sampled_counts[qaoa.assignment.mask]} times")
        print(f"QAOA matches exact optimum: {np.isclose(qaoa.assignment.difference, exact.difference, atol=1e-10)}")

        config = {
            "training_seed": 72,
            "training_records": 100,
            "split_random_state": 42,
            "qaoa": {"p": 2, "starts": 6, "shots": 4096, "seed": 72, "maxiter": 180},
            "parallel_workers": 2,
            "python": platform.python_version(),
            "dependencies": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        }
        runtimes = {
            "ml": ml_runtime,
            "greedy": greedy_runtime,
            "qaoa": qaoa.runtime_seconds,
            "parallel_exhaustive": parallel_runtime,
            "serial_exhaustive": serial_runtime,
        }
        metrics = write_evidence(
            run_id, directory, regression, assignments, greedy, exact, exact_ties, qaoa,
            config, runtimes, pids
        )
        print(f"\nEvidence saved to {directory.resolve()}")
    return run_id, directory, metrics


if __name__ == "__main__":
    run()

