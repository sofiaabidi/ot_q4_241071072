"""Persist immutable experiment evidence and update the latest-run pointer."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
from typing import Any

import numpy as np

from .assignment import Assignment
from .data import JOB_FEATURES, JOB_IDS, RegressionResult
from .quantum import QAOAResult


def new_run_directory(root: Path = Path("results/runs")) -> tuple[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(2)
    directory = root / run_id
    directory.mkdir(exist_ok=False)
    return run_id, directory


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_evidence(
    run_id: str,
    directory: Path,
    regression: RegressionResult,
    assignments: list[Assignment],
    greedy: Assignment,
    exact: Assignment,
    qaoa: QAOAResult,
    config: dict,
    runtimes: dict,
    worker_pids: list[int],
) -> dict:
    metrics = {
        "run_id": run_id,
        "ml": {
            "holdout_mae": regression.mae,
            "holdout_r2": regression.r2,
            "coefficients": regression.coefficients.tolist(),
            "intercept": regression.intercept,
        },
        "counts": {"all_assignments": len(assignments), "feasible_assignments": sum(a.feasible for a in assignments)},
        "methods": {
            "greedy": greedy.to_dict(),
            "qaoa": qaoa.assignment.to_dict(),
            "parallel_exhaustive": exact.to_dict(),
        },
        "qaoa_matches_exact": bool(np.isclose(qaoa.assignment.difference, exact.difference, atol=1e-10)),
        "qaoa_gap_hours": float(qaoa.assignment.difference - exact.difference),
        "runtimes_seconds": runtimes,
        "parallel_worker_pids": worker_pids,
    }
    _json_dump(directory / "config.json", config)
    _json_dump(directory / "metrics.json", metrics)

    with (directory / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["record_type", "id", "dist_km", "load_100kg", "traffic", "target_hours", "prediction_hours"])
        for index, (row, target) in enumerate(zip(regression.features, regression.target), start=1):
            writer.writerow(["training", index, *row, target, ""])
        for job, row, prediction in zip(JOB_IDS, JOB_FEATURES, regression.predictions):
            writer.writerow(["job", job, *row, "", prediction])

    with (directory / "assignments.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["mask", "bitstring", "vehicle1", "vehicle2", "count1", "count2", "t1", "t2", "difference", "squared_cost", "feasible", "qaoa_samples"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for assignment in assignments:
            writer.writerow({
                "mask": assignment.mask,
                "bitstring": format(assignment.mask, "06b")[::-1],
                "vehicle1": " ".join(assignment.vehicle1),
                "vehicle2": " ".join(assignment.vehicle2),
                "count1": assignment.count1,
                "count2": assignment.count2,
                "t1": assignment.t1,
                "t2": assignment.t2,
                "difference": assignment.difference,
                "squared_cost": assignment.squared_cost,
                "feasible": assignment.feasible,
                "qaoa_samples": qaoa.sampled_counts.get(assignment.mask, 0),
            })

    _json_dump(directory / "qaoa_details.json", {
        "parameters": qaoa.parameters.tolist(),
        "expectation_scaled": qaoa.expectation,
        "state_norm": qaoa.state_norm,
        "shots": sum(qaoa.sampled_counts.values()),
        "feasible_shots": qaoa.feasible_shots,
        "sampled_counts": {str(mask): count for mask, count in sorted(qaoa.sampled_counts.items())},
        "most_likely_feasible_mask": qaoa.most_likely_feasible_mask,
        "chosen_mask": qaoa.assignment.mask,
        "chosen_probability": qaoa.chosen_probability,
        "optimizer_runs": qaoa.optimizer_runs,
        "candidate_selection": "lowest-imbalance observed feasible QAOA measurement; no exact-search input",
    })
    latest = Path("results/LATEST.md")
    latest.parent.mkdir(exist_ok=True)
    latest.write_text(
        f"# Latest successful run\n\nRun: `{run_id}`\n\n"
        f"- [Configuration](runs/{run_id}/config.json)\n"
        f"- [Metrics](runs/{run_id}/metrics.json)\n"
        f"- [Console log](runs/{run_id}/console.log)\n"
        f"- [Predictions](runs/{run_id}/predictions.csv)\n"
        f"- [Assignments](runs/{run_id}/assignments.csv)\n"
        f"- [QAOA details](runs/{run_id}/qaoa_details.json)\n",
        encoding="utf-8",
    )
    return metrics

