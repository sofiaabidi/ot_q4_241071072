import json
from pathlib import Path

from main import run


def test_main_workflow_produces_consistent_evidence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_id, directory, metrics = run()
    expected = {"config.json", "metrics.json", "console.log", "predictions.csv", "assignments.csv", "qaoa_details.json"}
    assert expected == {path.name for path in directory.iterdir()}
    persisted = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert persisted == metrics
    assert persisted["counts"] == {"all_assignments": 64, "feasible_assignments": 50}
    assert persisted["exact_tied_optimal_masks"] == [22, 41]
    assert len(persisted["parallel_worker_pids"]) >= 2
    assert Path("results/LATEST.md").exists()

