# Known-answer experiment: can the model recover recharge we planted?

**SYNTHETIC DATA.** Nothing here says anything about a real structure.

## Why

The model is meant to say whether a recharge structure is *working* (putting water into
the aquifer) or *failing* (e.g. clogged). Before trusting it on real data, we check that
it can recover recharge we planted ourselves, on data from a simulator whose truth we
know.

## Set-up

`simulator.py` is pure NumPy and shares **no code** with the network. It has its own
haversine, its own stencil weights, its own implicit time stepping and its own
line-of-sight coupling. `tests/test_synthetic_recovery.py` checks each part on its own:
mass balance, water-conserving lateral flow, a coupling that matches the model's, rain
binning that matches the dataset's, and an oracle that recovers the truth exactly without
noise.

- **Structures:** 16 in total. Two clusters of 3 at 1.5 km spacing provide lateral flow;
  10 are scattered tens of km apart.
- **Acquisitions:** 40, with irregular revisit gaps of 6/12/24 days (about 1.3 years,
  covering one monsoon).
- **Rain:** daily, monsoon-like (June–September wet), binned causally into
  (t_{k−1}, t_k].
- **Physics:**
  - S ∂h/∂t = ∇·(T∇h) − L h + R (backward Euler, h₀ = 0).
  - S = 0.05, L = S/60 day⁻¹, T ~ log-uniform(20, 200) m²/day.
  - a_ij = 1/d² within 5 km.
- **Coupling:** d = 100 cos θ S_k h, with S_k = 5e-3 (compliant alluvium, so the signal
  is above the noise) and θ ~ U(29°, 46°).
- **Noise:** 0.3 cm at good coherence. 15 % of epochs have low coherence (< 0.7) and
  1 cm of noise; the dataset masks them out of the data term.
- **Planted truth:** recharge R = η · P/Δt, with infiltration fraction η.
  - 10 working structures, η ~ U(0.3, 0.6).
  - 6 failing structures, η ~ U(0, 0.05).
- **Design capacity:** set as if each structure was designed for η ∈ [0.3, 0.6] at 30–90 %
  utilisation, so the failing structures do not stand out in their design data.

**Model:** ARSNet trained exactly as on real data.
- Same dataset class, fixed graph, loss with default weights and scales (σ_d = 0.3 cm,
  σ_r = 1e-4 m/day, σ_h = 0.1 m).
- AdamW (lr 1e-3, weight decay 1e-5), cosine schedule, 600 epochs, gradient clip 5.
- Full batch, seed 0, CPU.

## Scenarios

| | what the model is given | what it should be able to do |
|---|---|---|
| **A** | InSAR + rain + design only; every parameter learned | From InSAR alone only R·S_k/S is identifiable (finding-2 plan §7): a global rescaling of S_k, S and L leaves every observation unchanged. So we test **ranking** and **working-vs-failing separation**, which a global scale cannot change, not absolute R. |
| **B** | as A, but S and S_k fixed to their true values | absolute R |
| **C** | hard-rock aquifer, S_k = 2e-5: the signal (≈ 0.01 cm) is far below the noise | nothing. Report only: does the model invent structure in noise? |
| **D** | as A, plus noisy piezometer heads (σ = 5 cm) at every structure | report only: how much do wells help? |

## Metrics

Per structure, on recharge averaged over steps 1…K−1:

- **Spearman ρ** between recovered and true mean R. ρ is scale-free.
- **AUROC** for working vs failing: the chance that a random working structure is
  scored above a random failing one. 0.5 is chance, 1.0 is perfect.
- **Median relative error** |R̂ − R|/R over the working structures. Failing structures
  have R ≈ 0, where a relative error is meaningless.
- **Oracle floor:** least-squares η̂ with *perfect* knowledge of S, S_k, L, T, θ and the
  graph (`simulator.oracle_eta`). Its median relative error is the noise floor no model
  can beat on average.

## Thresholds (pre-registered: fixed and committed BEFORE the first training run)

| test | threshold | justification |
|---|---|---|
| A: Spearman ρ | **≥ 0.70** | A conventional "strong" monotone association. A predictor that ignores the structures (e.g. one following rainfall alone) scores ≈ 0, because η is drawn independently of rainfall. |
| A: AUROC | **≥ 0.85** | The working/failing call is the project's goal. 0.5 is chance; 0.85 means about 6 in 7 working/failing pairs are ordered correctly. Below that the flag is not useful. |
| B: median relative error of mean R | **≤ max(0.30, 2 × oracle floor)** | With the constants known, R should be recovered to better than the factor-of-about-2 spread of the recharge-factor norms a hydrogeologist would otherwise use (CGWB GEC-2015 gives ranges such as 0.1–0.2). It must also be within twice the noise floor. |

The pytest (`tests/test_synthetic_recovery.py::test_recovery_meets_preregistered_thresholds`)
asserts exactly these on scenarios A and B, with the same configuration as this report. If
the model misses them, the test fails and the result is reported as a failure. The
thresholds are not moved.

## Results

*Not yet run at the time the thresholds were committed.* See the next commit.
