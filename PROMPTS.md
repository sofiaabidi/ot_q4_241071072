# Coding-agent prompts — Q4 Cold-Chain Optimization

**How to use:** Put `PLAN.md` and `PROMPTS.md` in the root of a new empty repo. Open it in your coding agent. **Paste Prompt 1 first.** Use Prompts 2 and 3 to verify/polish; Prompt 4 only if blocked. Keep all outputs in this repository. The plan is the single source of truth; do not repeat the entire spec in every prompt.

## Prompt 1 — Build the complete working repo (paste this)

```text
Read PLAN.md fully and implement this exam Q4 in the CURRENT EMPTY REPO. Deliver a simple, correct, reproducible Python solution, not a demo scaffold or long commentary. Follow the exact seeded historical-data code and J1–J6 table, train/test regression, longest-job-first greedy baseline, all-64-assignment ProcessPoolExecutor exhaustive verification, and true 6-qubit gate-based NumPy statevector QAOA with Ising/QUBO derivation and feasible-shot postselection. Use one canonical assignment evaluator, fixed seeds, correct 2–4 rule, and honest measured runtimes. Generate outputs/results.json, outputs/results.csv, and a ONE-PAGE outputs/results_sheet.pdf plus a concise README with QUBO/Ising formulas, 3-domain explanation, comparison, 2 limitations, references. Make `python main.py` and `python -m unittest discover -s tests -v` work. Install dependencies if needed, run the whole pipeline, fix failures. Preserve a clear distinction between quantum sampled result and independent exact search. Avoid unnecessary frameworks/UI and do not invent metrics. At the end show actual command outputs, test pass count, produced files, and any real limitation, briefly.
```

## Prompt 2 — Independent correctness audit (paste after Prompt 1)

```text
Audit implementation against every MUST in PLAN.md and the pictured Q4. Actually rerun main.py and unit tests. Check the exact RNG code/call order; 6 job features; ML holdout and nonleaking refit; 64 total/50 feasible; bitstring to vehicle mapping; squared-imbalance QUBO vs Ising for all 64 masks; true H→ZZ cost→RX mixer QAOA with normalized state; measured-feasible QAOA candidate not copied from exhaustive; actual multiprocessing PIDs; deterministic greedy; correct runtimes and results; 1-page PDF. If any item fails, FIX CODE and rerun (do not merely describe it). Print a compact PASS/FAIL checklist with real output values and file paths. No unrelated refactors.
```

## Prompt 3 — Final submission polish (paste if build passes)

```text
Tighten the repo for a 5-mark, 45-minute practical: README must have copy-paste commands, clear variables/objective/constraints, how ML→QAOA→Parallel connect, measured baseline vs QAOA vs exhaustive results, QAOA exact-match answer, two honest limitations and references. PDF must have exactly ONE readable page with actual job assignments, T1/T2/|T1−T2| and runtimes for all 3 methods. Keep code compact and human-readable; avoid excessive jargon, art, dashboards, giant dependencies and fabricated claims. Run the tests and produce the final PDFs/CSV/JSON again. Provide a concise final submission checklist only.
```

## Prompt 4 — Recovery only if the agent is blocked

```text
Do NOT restart the repo. Fix only the current blocker, then rerun the full CLI and tests. Priorities: (1) exact given data + predicted six job times, (2) correct feasible assignment evaluator and parallel exact oracle, (3) real QAOA statevector cost/mixer and seeded sampling, (4) one-page PDF. If a quantum SDK cannot install, use the planned pure-NumPy gate simulator; do NOT downgrade to random search while calling it quantum. If ProcessPoolExecutor fails, fix pickling and __main__ guard, and do NOT claim parallel results until real worker PIDs are observed. If quantum does not hit the exact optimum, report an honest “No” and the gap rather than modifying the sampled answer. Return actual tracebacks/fixes/results succinctly.
```

## What your final repo should contain

```text
PLAN.md  PROMPTS.md  README.md  requirements.txt  main.py
q4/{__init__.py,data.py,assignment.py,quantum.py,report.py}
tests/test_core.py
outputs/results.json  outputs/results.csv  outputs/results_sheet.pdf
```

**One-minute final check:** run `python main.py`, run `python -m unittest discover -s tests -v`, open `outputs/results_sheet.pdf` and confirm it is exactly one page, then save/share the repository and report as required.
