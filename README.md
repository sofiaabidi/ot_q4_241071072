# Q4 — Cold-Chain Vehicle Assignment

A compact, reproducible solution connecting machine learning, gate-based quantum simulation, and real multiprocessing. It predicts six delivery times, assigns every job to one of two vehicles, and minimizes the difference between vehicle totals subject to 2–4 jobs per vehicle.

## Run

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py
python -m pytest -q
```

Every `python main.py` execution creates a distinct immutable folder under `results/runs/`; `results/LATEST.md` points to the newest successful run.

## Methods

The ML stage generates exactly 100 synthetic records using NumPy seed 72 and the examination formula. A deterministic 80/20 split evaluates `LinearRegression` with MAE and R²; a model refitted on all records predicts J1–J6 without using the generating formula directly.

For predicted times `t_i`, bit `x_i=1` puts job `i` on vehicle 1 and `x_i=0` puts it on vehicle 2:

```text
T1 = Σ t_i x_i,  T2 = Σ t_i(1-x_i),  S = Σ t_i
Δ = T1-T2 = 2Σ t_i x_i-S
minimize |Δ| subject to 2 ≤ Σx_i ≤ 4
```

The equivalent QUBO energy is

```text
E(x)=S² + Σ_i(4t_i²-4St_i)x_i + 8Σ_{i<j}t_it_j x_ix_j.
```

With `Z_i=1-2x_i`, the Ising cost is `H_C=(Σt_i Z_i)² = (Σt_i²)I + 2Σ_{i<j}t_it_j Z_iZ_j`. The simulator uses `H_C/S²`, which preserves the minimizer. It initializes `|+⟩⁶`, applies exact diagonal ZZ cost phases and RX mixers for two QAOA layers, optimizes expected energy, and samples 4096 measurements. Constraint handling is explicit postselection: only observed masks with 2–4 jobs on vehicle 1 are eligible. The QAOA answer is the lowest-imbalance observed feasible measurement and is never copied from exhaustive search.

The baseline is deterministic longest-job-first greedy allocation. Independently, `ProcessPoolExecutor` workers evaluate all 64 masks; exactly 50 satisfy the job-count rule. A serial enumeration cross-checks the parallel optimum. Runtime evidence includes worker process IDs; multiprocessing overhead can exceed useful work for only 64 candidates.

## Verified comparison

The latest audited run predicts `[2.791562, 4.694059, 3.373247, 4.954552, 3.871576, 4.243740]` hours for J1–J6, with holdout MAE `0.048516` hours and R² `0.992067`. Greedy assigns V1=`J3,J4,J5` and V2=`J1,J2,J6`, producing totals `12.199375` and `11.729361` hours (difference `0.470014`). QAOA and exact search both select representative mask 22: V1=`J2,J3,J5`, V2=`J1,J4,J6`, totals `11.938882` and `11.989854` hours (difference `0.050972`). Exact masks 22 and 41 are the two vehicle-label-swapped tied optima. Measured runtimes and sampling counts belong in the run evidence because they vary by machine and execution.

## Evidence and limitations

Each run preserves its configuration, console output, all 100 generated records, six predictions, all assignments, sampled QAOA counts and parameters, runtimes, and comparison metrics. See [results/LATEST.md](results/LATEST.md).

Two important limitations are that the training data is synthetic rather than operational cold-chain history, and QAOA is simulated as an ideal noiseless statevector rather than executed on noisy quantum hardware. The six-job problem is also too small for parallel or quantum speed claims.

References: [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), [Farhi et al., QAOA](https://arxiv.org/abs/1411.4028), [NumPy Generator](https://numpy.org/doc/stable/reference/random/generator.html), and [ProcessPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html).

