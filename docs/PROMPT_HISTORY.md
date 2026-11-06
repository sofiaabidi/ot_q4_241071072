# Prompt History

## 1. Repository planning prompts (pre-existing)

`PROMPTS.md` was present when implementation began. It contains four proposed coding-agent prompts: complete build, correctness audit, submission polish, and blocker recovery. There was no evidence that any had previously been executed; therefore no execution outcome or commit is claimed.

## 2. Complete implementation request — 2026-11-06

Purpose: inspect the repository and implement a robust, token-efficient, validated Q4 solution with reproducible run evidence and prompt/refinement tracking. The request requires exact seeded ML data, genuine sampled QAOA, real parallel exhaustive search, greedy comparison, tests, factual logs, commits, and a push if authentication permits. It explicitly defers the formal report/results sheet.

Outcome: implementation completed from a repository containing only `PLAN.md` and `PROMPTS.md`. Verified run `20261106T053542Z-94cf` and 9 passing pytest tests provide execution evidence. Core implementation commit: `bf8e3d0`.

## 3. Final examination-readiness audit — 2026-11-06

Purpose: audit every Q4 marking criterion in `PLAN.md`, rerun tests and the workflow, verify persisted results against actual calculations, prove clean-clone reproducibility, fix only genuine defects, and keep the formal report deferred.

Outcome: all implementation criteria passed after adding explicit recording of both tied exact optima. Audited run `20261106T054359Z-f35d` passed 9 pytest tests and the complete workflow. The formal report remains intentionally deferred by direct user instruction.

