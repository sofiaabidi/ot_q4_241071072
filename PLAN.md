# Q4 — Cold-Chain Vehicle Assignment | Implementation Plan

**Goal:** A compact, reproducible, exam-ready Python repository solving the exact Q4 specification in the supplied question paper. **Time budget: 45 minutes, 5 marks.** Prioritize working execution, honest quantum verification, and a **one-page results sheet** over optional features.

## 1. Exact assignment / acceptance criteria

- Six jobs **J1–J6**, **two refrigerated vehicles**. Assign every job to exactly one vehicle. Each vehicle must get **2–4 jobs**.
- Predict each job's delivery time in **hours** using regression fitted to **100 synthetic historical records from the exact fixed-seed code below**. No outside training data; never use the generation equation directly as the job predictor.
- Minimize the absolute difference `abs(T1 - T2)` between vehicle totals (equivalently its square for an optimization Hamiltonian).
- **Quantum:** explicitly derive `(T1-T2)^2` as a **QUBO / Ising energy**, implement actual gate-based **QAOA with a NumPy statevector simulator** (Hadamard initialization, diagonal ZZ cost phase, RX mixing, measured bitstrings). Enforce the 2–4 rule by **filtering measured assignments**. Do not label random search or direct exhaustive search “QAOA.”
- **Parallel:** independently check **all 64 assignments** using `concurrent.futures.ProcessPoolExecutor` with at least two workers; filter infeasible results and determine the exact optimum. Capture worker process IDs to demonstrate real multiprocessing. Never silently claim parallel execution if it fell back to serial.
- **Baseline:** deterministic **greedy longest-job-first**, assigning each descending-time job to the currently less-loaded feasible vehicle; respect the 2–4 constraint.
- **Submit:** runnable repo, output, and one-page `outputs/results_sheet.pdf` with formulation, domain connections, methods, comparison table, correctness statement, two limitations, and references. Include `outputs/results.json` and `outputs/results.csv` for checking.

### Historical data — copy this *exactly*, including RNG call order

```python
import numpy as np
rng = np.random.default_rng(72); N = 100
dist = rng.uniform(5, 60, N); load = rng.uniform(1, 10, N)
traffic = rng.uniform(1, 5, N)
time = 0.04*dist + 0.15*load + 0.30*traffic + 0.5 + rng.normal(0, 0.1, N)
```

Do not change `72`, sample count, uniform distributions, noise, generation order, or columns. Features = `dist, load, traffic`; target = `time` (hours).

### The six real jobs (use this table verbatim)

| Job | dist (km) | load (×100 kg) | traffic index |
|---|---:|---:|---:|
| J1 | 38 | 1 | 2 |
| J2 | 49 | 9 | 3 |
| J3 | 20 | 8 | 3 |
| J4 | 59 | 8 | 3 |
| J5 | 50 | 1 | 4 |
| J6 | 27 | 10 | 4 |

Use `sklearn.linear_model.LinearRegression` with a deterministic 80/20 train/test split (`random_state=42`). Report held-out MAE and R² (no leakage); refit on all 100 records for the six job predictions. Reject nonfinite/nonpositive predictions rather than hiding them. Round only for display, not when optimizing.

## 2. Correct mathematical formulation (show in README and PDF)

Let `x_i ∈ {0,1}` indicate whether job i is assigned to **vehicle 1** (`x_i=1`) or **vehicle 2** (`x_i=0`); let `t_i>0` be the *predicted* hours.

```
T1(x) = Σ_i t_i x_i
T2(x) = Σ_i t_i(1−x_i)
Δ(x)  = T1 − T2 = 2 Σ_i t_i x_i − S, where S=Σ_i t_i
minimize |Δ(x)|  subject to 2 ≤ Σ_i x_i ≤ 4
```

The second vehicle automatically receives `6−Σ x_i` jobs, also 2–4. There are `2^6 = 64` assignments; `C(6,2)+C(6,3)+C(6,4)=15+20+15=50` feasible assignments (25 unique up to swapping vehicle labels).

**QUBO:** minimizing `E(x)=Δ(x)^2` is equivalent to minimizing `|Δ(x)|` on the same feasible set:

```
E(x) = S² + Σ_i (4 t_i² − 4S t_i) x_i + 8 Σ_{i<j} t_i t_j x_i x_j.
```

**Ising:** with `Z_i = 1−2x_i` (eigenvalues ±1), `Δ=−Σ_i t_i Z_i`, hence

```
H_C = (Σ_i t_i Z_i)² = (Σ_i t_i²) I + 2 Σ_{i<j} t_i t_j Z_i Z_j.
```

The identity term is a global phase, irrelevant to circuit measurement. For numerical conditioning, implement `H_C / S²`; this scaling does **not** change the minimizer. **No penalty needed** because we filter measured assignments to the 2–4 feasible set. Explicitly disclose the postselection choice.

## 3. Implementation architecture (keep it small)

```text
README.md                 # setup, one-command run, methodology, limitations
PLAN.md                   # this specification
PROMPTS.md                # coding-agent instructions
requirements.txt          # numpy, scipy, scikit-learn, reportlab
main.py                   # orchestrates pipeline; guarded __main__ for multiprocessing
q4/
  __init__.py
  data.py                 # exact generation snippet; regression + six job features
  assignment.py           # feasibility, totals, greedy, parallel exhaustive search
  quantum.py              # true QAOA statevector, Ising/QUBO, optimization, sampling
  report.py               # JSON/CSV + single-page PDF
 tests/
  test_core.py            # deterministic sanity/correctness tests
 outputs/                 # generated at runtime; PDF/CSV/JSON (can commit if allowed)
```

Avoid Flask, web apps, databases, notebooks, giant frameworks, unnecessary dashboards, lengthy abstractions or unrelated content. Prefer type hints and docstrings for the key functions; easy-to-follow undergraduate-level code.

## 4. Module responsibilities and computational behavior

**Data / ML (`data.py`)**
- Run the given historical-data block unchanged. Fit/test `LinearRegression`. Return six unrounded positive time predictions, train/test metrics, and training metadata.
- Keep the exact job identifiers and feature order; print predicted hours for J1–J6.

**Assignment / parallel (`assignment.py`)**
- A *single* `evaluate_mask(mask, times)` function decodes the six-bit assignment consistently and returns vehicle job lists, counts, totals, `abs(T1−T2)`, squared cost, feasible flag; use for baseline, quantum outputs and exhaustive verification.
- Deterministic greedy: sort by decreasing predicted time, break ties by job ID; assign to less-loaded vehicle, but enforce a maximum of four jobs and ensure that both vehicles can still end with at least two. Tie-break toward vehicle 1. Assert feasibility.
- `ProcessPoolExecutor(max_workers=2 or more)` and **top-level picklable worker function** evaluate **all 64 masks** (`0..63`). Use `if __name__ == '__main__'` in entry point. Return worker PIDs, 50 feasible candidates, best exact objective and all tied optima. Tie-break by ascending mask for a stable representative.
- Count time with `time.perf_counter()`; process startup may make parallel slower on 64 assignments. Never invent speedup.

**Quantum (`quantum.py`)**
- Six qubits; build |+⟩⁶ using H on each (uniform amplitudes are algebraically identical).
- For p=2 layers, apply real QAOA cost unitary `exp(-i γ_l H_C/S²)` through the exact diagonal ZZ phase and mixer `exp(-i β_l Σ_i X_i)` via RX(2β_l) on each qubit; maintain normalized complex amplitudes. This is a **statevector gate simulation**, not a heuristic probability model.
- Minimize `⟨H_C/S²⟩` with seeded multi-start `scipy.optimize.minimize` (bounded, modest iteration budget, e.g. 4–8 starts), with reproducible seed. Because only 64 basis states exist, computing the exact expectation from the statevector is cheap.
- Measure using the **QAOA final state probabilities**, e.g. 4096 seeded bitstring samples. Filter sampled bitstrings by the 2–4 rule, choose the lowest-imbalance **observed feasible measurement** (deterministic mask tie-break); record shots, valid shots, most likely measured feasible mask, chosen mask, chosen-mask probability, variational angles, and expectation. Note that “lowest observed sample” is not the same as “most probable sample”—report honestly. Do **not** use the parallel exhaustive optimum to select or modify the quantum answer.
- Verify Hamiltonian consistency for all 64 masks: `E_QUBO(x)=E_Ising(x)=Δ(x)^2` within tolerance before scaling; check state norm ≈1, nonnegative normalized probabilities summing to 1. Derive `Z_i=1−2x_i` and document bit order clearly.

**Report (`report.py`)**
- Generate `outputs/results.json`, `outputs/results.csv`, and **one-page** `outputs/results_sheet.pdf` (ReportLab) automatically from **actual run data**.
- PDF includes: heading; formulation (decision variable, objective, constraints, 64 vs 50 assignments); small ML summary (MAE, R² + six predicted job times); three-domain method summary; comparison table with **baseline greedy / quantum QAOA / parallel exhaustive** rows and columns **vehicle 1 jobs, vehicle 2 jobs, T1, T2, |T1−T2|, runtime**; clear “QAOA matches exact optimum: Yes/No” based on calculated numerical gap (vehicle-label permutations count as equal); interpretation; **two specific limitations** (e.g. synthetic training data and ideal noiseless statevector / 64-case overhead); 2–4 credible citations/URLs.
- Measure timings consistently. Label quantum runtime as training/circuit optimization + sampling; exhaustive runtime includes parallel process startup. Report units (hours, seconds) and enough digits to distinguish near ties. Do not claim speedup or quantum superiority without measured evidence.
- Assert the PDF has **exactly one page** (if no PDF reader dependency, inspect with a small stdlib-compatible method or optional `pypdf`). Keep it simple and readable; no unnecessary graphs.

## 5. Execution and verification gates

**Setup**
```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate  |  Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
python -m unittest discover -s tests -v
```

**Tests, all mandatory**
1. Data generation exactly reproducible with seed 72, shapes `(100,3)` and `(100,)`; six job predictions positive.
2. Exactly **64** assignments generated; exactly **50** feasible, each job exactly once; every reported assignment feasible.
3. For every mask, `Δ=T1−T2`; QUBO and Ising energies agree with `Δ²` (tolerance `1e-8` relative), including bit order.
4. Greedy baseline is deterministic and feasible; parallel exhaustive agrees with a serial oracle/reference for optimal objective and tied masks; `len(distinct_worker_pids) >= 2` when parallel mode is claimed.
5. QAOA norm/probabilities valid; quantum output must be observed in the measured sample and satisfy the 2–4 rule; quantum objective cannot beat the exact feasible optimum.
6. Results table, JSON/CSV, and one-page PDF derive from the same run; don't hardcode results or claim an unverified exact quantum match.

**Final terminal checks**
- `python main.py` exits 0 and prints predictions, ML metrics, greedy/QAOA/exact assignments/totals/gaps, worker PIDs, measured runtimes, exact-match check, output paths.
- `python -m unittest discover -s tests -v` exits 0.
- All three generated outputs exist and PDF is one page.
- README clearly says how this connects ML, Quantum and Parallel Computing, and provides two limitations and citations.

## 6. Priorities if time is tight

**Must work first:** exact data, regression, assignments/constraints, greedy baseline, **real parallel exhaustive**, **real six-qubit QAOA**, a single CLI command and honest results. **Then:** one-page PDF and tests (both required by question). **Last:** cosmetic polish. Do not omit a required domain to add styling.

**Suggested references (use URLs, not invented results):**
- scikit-learn `LinearRegression`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
- Farhi, Goldstone & Gutmann, *A Quantum Approximate Optimization Algorithm* (2014): https://arxiv.org/abs/1411.4028
- NumPy random Generator: https://numpy.org/doc/stable/reference/random/generator.html
- Python `ProcessPoolExecutor`: https://docs.python.org/3/library/concurrent.futures.html

## 7. Submission / academic integrity

The generated dataset is explicitly **synthetic** as required by the exam; the six job predictions are from a fitted model. The “quantum” result must come from a genuine QAOA measurement distribution, not exhaustive search. Exhaustive search supplies **verification**, not the quantum candidate. If QAOA fails to equal the exact optimum, state **No**, quantify the gap and explain it. Do not fabricate performance, runtime, model accuracy, experiment screenshots or citations. Submit a concise repo that genuinely executes and can be defended orally.
