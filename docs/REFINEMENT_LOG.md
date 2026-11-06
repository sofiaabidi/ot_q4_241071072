# Refinement Log

## 2026-11-06 — Build from empty implementation state

- Trigger: the user requested a robust, token-efficient, validated solution with reproducible results and prompt/refinement tracking. Inspection found only planning files and no Python executable on PATH.
- Change: added the modular ML, assignment, parallel-search, QAOA, evidence, CLI, documentation, and test structure.
- Reason: satisfy the examination domains while keeping one obvious command and an auditable distinction between sampled quantum output and the independent exact result.
- Files: `main.py`, `q4/`, `tests/`, `README.md`, `requirements.txt`, `.gitignore`, `docs/`, and `results/`.
- Verification: `python main.py` completed as run `20261106T053542Z-94cf`; `python -m pytest -q` passed all 9 tests. The run recorded 64 assignments, 50 feasible assignments, and two distinct worker PIDs.

## 2026-11-06 — Defer formal report

- Trigger: the current user instruction overrides the older plan's immediate one-page PDF requirement.
- Change: no PDF/DOCX report generator or styled result sheet was added. Machine-readable and Markdown evidence remains available for a later report request.
- Verification: repository contains no generated formal report.

## 2026-11-06 — Make parallel participation deterministic

- Trigger: the first full workflow correctly rejected a parallel run in which one Windows worker completed both tiny chunks before the second process became ready.
- Change: replaced the timing delay with an explicit start barrier. Each of two chunk workers reports its PID and waits until both are ready before evaluating its half of the 64 masks.
- Files: `q4/assignment.py`.
- Verification: run `20261106T053542Z-94cf` recorded worker PIDs 10600 and 18860; parallel and independent serial optima agreed. All 9 tests passed. Included in implementation commit `bf8e3d0`.

