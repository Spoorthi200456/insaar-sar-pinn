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

---

## Results (one run of the protocol; commits `6f84aa3` protocol, `5816a8f` code)

Files: `results.json` (per seed, per method, summary) and `results.csv` (per test
structure). 30–31 held-out test structures per seed (11–12 failing). Runtime about 70 min on
CPU, almost all of it A's training and B's per-structure fitting; C takes 0.2–0.3 s per
seed.

### "linear" truth (headline)

| method | Spearman ρ, mean ± sd (min) | AUROC, mean ± sd (min) | pass mark (ρ ≥ 0.70, AUROC ≥ 0.85) |
|---|---|---|---|
| A trained network | **−0.05 ± 0.33** (−0.49) | **0.50 ± 0.20** (0.25) | **FAIL** |
| B per-structure fitting | 0.936 ± 0.008 (0.924) | 1.000 ± 0 (1.000) | pass |
| C least squares, no network | **0.967 ± 0.005** (0.960) | 1.000 ± 0 (1.000) | pass |
| oracle (every constant known) | 0.967 ± 0.006 (0.959) | 1.000 ± 0 (1.000) | — |

Per seed, ρ for A / B / C:
- seed 0: −0.49 / 0.93 / 0.97
- seed 1: 0.34 / 0.95 / 0.97
- seed 2: 0.10 / 0.92 / 0.97
- seed 3: 0.09 / 0.94 / 0.96
- seed 4: −0.28 / 0.94 / 0.96

### "threshold" truth (robustness)

| method | Spearman ρ, mean ± sd (min) | AUROC, mean ± sd (min) | pass mark |
|---|---|---|---|
| A | −0.02 ± 0.25 (−0.28) | 0.49 ± 0.14 (0.33) | **FAIL** |
| B | 0.918 ± 0.010 (0.910) | 1.000 ± 0 | pass |
| C | **0.955 ± 0.008** (0.948) | 1.000 ± 0 | pass |
| oracle | 0.954 ± 0.010 (0.941) | 1.000 ± 0 | — |

### Comparison rules (as pre-registered)

| comparison | truth | mean ρ difference | seeds won | clearly better? |
|---|---|---|---|---|
| **B vs C** | linear | **−0.030** | **0 of 5** | **NO** |
| **B vs C** | threshold | **−0.038** | **0 of 5** | **NO** |
| A vs C | both | −1.02 / −0.97 | 0 of 5 | no |
| B vs A | both | +0.98 / +0.94 | 5 of 5 | yes |

**Verdict: B is not clearly better than C. On the contrary, C beat B in every one of the
10 seed-truth runs. The neural network is not needed for this part of the problem**
(ranking structures by recharge and separating working from failing ones, from InSAR,
rainfall and design data, on structures not used for fitting). On this synthetic data a
one-parameter least-squares fit with one regional recession time does as well as an oracle
that knows every constant.

### Other observations

- **A, the trained network applied unchanged, is at chance on unseen structures in every
  seed.** That confirms run 4 (review 3) with 30+ test structures and 5 seeds. Its spread
  across seeds is large (ρ from −0.49 to +0.34): it is not even consistently wrong.
- **B rescues the network** (B vs A: +0.98, 5 of 5 seeds). Fitting the physics-informed
  model to each structure's own data works. But it is never better than the 1-parameter
  fit and costs about 3000× more compute (8–18 min per seed against 0.2–0.3 s). B lowered
  each structure's own loss to about 36–56 % of its starting value (median per seed).
- C's recession time was 56–63 days in every run, against 60 days in the simulator.

### Limitations: read before drawing conclusions

1. **The simulator is C's model.** The "linear" truth is exactly C's assumption: a linear
   reservoir with recharge proportional to rain. C matching the oracle there is expected.
2. **The "threshold" truth turned out to be a weaker robustness check than intended.**
   This was noticed after the run and is reported, not changed. All structures share one
   regional rainfall series, so the threshold changes the *time shape* of recharge
   identically for every structure. It scales nothing differently between them, so it
   cannot change their order, which is all ρ and AUROC measure. A real test of C's
   robustness needs truths that differ *between* structures in ways C's single amplitude
   cannot express, for example:
   - progressive clogging (a working structure whose infiltration declines over the
     series);
   - structure-specific thresholds or recession times;
   - local rain differences;
   - pumping near some structures.

   Those are where a richer model such as B might earn its cost. They have not been
   tested.
3. All the earlier caveats still hold:
   - synthetic data;
   - a compliant (alluvial) aquifer with a signal well above noise;
   - absolute recharge not identifiable from InSAR alone, for any of the three methods.

### Tests

`tests/test_method_comparison.py`:
- **`test_c_meets_pass_marks_on_unseen_structures`** (slow, in the normal suite):
  re-runs C's full protocol (5 seeds × 2 truths) in about 1 minute. It passes.
- **`test_seed0_a_fails_and_b_passes_on_unseen_structures`** (opt-in, about 20 min:
  `pytest -m comparison`): re-runs A and B on seed 0, "linear". It asserts that B meets
  the marks and that A does not.
