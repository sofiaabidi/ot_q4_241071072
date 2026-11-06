# User Requests

## 2026-11-06 — Initial implementation request

The user requested a complete Q4 repository implementation, not a plan-only response. Direct requirements include:

- inspect the repository, environment, remote, `PLAN.md`, and `PROMPTS.md`;
- implement exact 100-record seeded regression and preserve data/predictions;
- implement genuine gate-based simulated QAOA with constraint handling and sampled provenance;
- evaluate all 64 assignments using actual parallel processing and identify all 50 feasible cases;
- compare deterministic longest-job-first greedy, QAOA, and exact search;
- preserve each run under `results/runs/<run-id>/` and maintain `results/LATEST.md`;
- add focused pytest tests, execute the workflow, and fix failures;
- maintain factual prompt and refinement logs;
- keep the repository minimal and defer the formal PDF/DOCX report until separately requested;
- commit meaningful milestones and push verified changes to `origin/main` if authentication is available.

Implementation choices not directly requested are documented in code, README, and the refinement log rather than represented as user instructions.

