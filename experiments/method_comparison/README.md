# Method comparison on held-out structures (review 4)

**SYNTHETIC DATA.** Nothing here says anything about a real structure.

## Protocol — written and committed BEFORE any method was run

### Question

On recharge structures the model has **never seen**, which method best tells how much each
structure recharges, and in particular which structures are working and which are failing?

- **A. Trained network (amortised).** ARSNet trained on the training structures, applied
  unchanged to the test structures. This is what failed in run 4 (review 3); it is
  re-measured here on the same structures as B and C.
- **B. Per-structure physics-informed fitting (test-time fitting).** Start from A's
  trained network. For **each test structure separately**, continue optimising the
  network on that structure's own loss (physics + InSAR data + regularisers, which need
  no ground truth), then read off its recharge.
- **C. Non-neural baseline: ordinary least squares.** For each test structure, fit the
  physical model's single free amplitude by ordinary least squares. The model is a linear
  reservoir driven by rainfall, observed through the InSAR coupling. One recession time
  scale is shared by all structures and estimated on the training structures. No neural
  network is involved.

If B is not clearly better than C (rule below), the neural network is not needed for this
part of the problem, and the report will say so.

### Data

The independent simulator (`experiments/synthetic_recovery/simulator.py`) with its
default physics and noise:

- S = 0.05, S_k = 5e-3, L = S/60 day⁻¹;
- T log-uniform 20–200 m²/day;
- 40 acquisitions with 6/12/24-day gaps;
- 0.3 cm InSAR noise, 15 % low-coherence epochs.

| item | value |
|---|---|
| structures per seed | **N = 160**, of which 60 failing (η ~ U(0, 0.05)) and 100 working (η ~ U(0.3, 0.6)) |
| split | `src/data/splits.py`, fractions **0.6 / 0.2 / 0.2**, by connected structure group, stratified by working/failing, split seed = the run seed |
| held-out test structures | about 32 per seed; **at least 30 required**. If any seed gives fewer than 30, N is raised mechanically in steps of 16 until all five give ≥ 30, before any method is run. |
| seeds | **0, 1, 2, 3, 4**. Each seed sets the simulated data (structures, rain, noise), the split, and all training randomness. |

**Two truths** (a protocol choice, made now):

1. **"linear"** (main): planted recharge R = η · P/Δt, exactly as in runs 1–4. This is
   *exactly* the form method C assumes, so C is favoured here.
2. **"threshold"** (robustness, C misspecified): R = η · max(P/Δt − 2 mm/day, 0). Only
   rain above an intensity threshold infiltrates, a simple runoff-threshold behaviour
   that C's formula does not contain.

### Methods in detail (fixed now)

**A — trained network.** As run 4 (`heldout.py`):
- training structures only, full batch, normalisation fitted on them;
- `train.build_optimizer` (network lr 1e-3, global scalars 0.02), 600 epochs;
- validation loss every 10 epochs, best epoch kept.

Score of a test structure = its mean predicted recharge over steps 1…K−1.

**B — per-structure test-time fitting.** For each test structure *on its own* (a batch
of one structure, no neighbours):
- copy A's selected network;
- **freeze the global constants** S, S_k, L at A's values. They are regional properties
  learned from the training structures; refitting them per structure would give every
  structure its own unknown scale and break the comparison between structures;
- optimise all other weights with Adam (lr 1e-3) for **300 steps** on the default
  `CompositeARSLoss` of that structure alone (no wells, no truth);
- keep the final weights; score = mean predicted recharge over steps 1…K−1.

**C — ordinary least squares, no network.**
- Unit response g_k(τ): the linear-reservoir head for unit storativity driven by
  u_k = P_k/(1000 Δt_k). Backward Euler, g_0 = 0:
  g_k = (g_{k−1} + Δt_k u_k) / (1 + Δt_k/τ).
- Model: d_k = 100 · cos θ · c · g_k(τ). The single parameter is c = η·S_k/S.
- Fit: ordinary least squares on the reliable epochs (coherence ≥ 0.7, not flagged):
  ĉ = Σ_k w_k d_k / Σ_k w_k², with w_k = 100 cos θ g_k(τ).
- τ is one value for all structures: chosen from 41 log-spaced values between 10 and
  1000 days to minimise the total squared residual over the **training** structures.
- Score = ĉ. It ranks structures like η; a common unknown factor S_k/S does not change
  the ranking.
- Derivation: `docs/math/ols_baseline.md`.

### Metrics (per seed, per method, on the test structures only)

- Spearman ρ between the score and the true mean recharge.
- AUROC for working vs failing.
- Reported as **mean, sample standard deviation and minimum over the 5 seeds**, plus every
  per-seed value.
- Reference: the oracle (perfect knowledge of every constant, `simulator.oracle_eta`) on
  the same test structures.

Absolute recharge is **not** compared. From InSAR alone it is not identifiable (only
R·S_k/S is; review 2), for any of the three methods.

### Pass marks (fixed now)

1. **A method passes a truth** if, over the 5 seeds, mean ρ ≥ **0.70** and mean AUROC ≥
   **0.85**. These are the same values as the pre-registered marks of runs 1–4.
2. **"B is clearly better than C"** for a truth if **both** hold:
   - mean ρ(B) − mean ρ(C) ≥ **0.05**, and
   - ρ(B) > ρ(C) in **at least 4 of the 5 seeds** (paired by seed).

   Otherwise: **"B is not clearly better than C — the neural network is not needed for
   this part."** The same rule is reported for A vs C, and for B vs A.
3. The headline uses the **"linear"** truth. The **"threshold"** truth is reported as a
   robustness check, especially for C, whose formula it violates.

### What will be committed after the run

- `results.json` and `results.csv` (per seed, per method, per test structure);
- the results tables in this README;
- tests marking any method that misses its pass mark as `xfail(strict=True)` with its
  numbers.

Pass marks and rules will not be changed after the run.
