# Project status — InSAR–PINN monitoring of Artificial Recharge Structures

*Updated 2026-10-24 after review 2 and the author's decisions 1–7. Merged into `main`
locally (not pushed); branch `review-2` kept. All numbers are from **SYNTHETIC DATA**.
No real Sentinel-1, rainfall or well data has been run through the model.*

**Goal (fixed):** assess whether Artificial Recharge Structures (ARS) in India are working
(recharging the aquifer) or failing (clogging, leakage, structural damage), using freely
available InSAR, rainfall and design data, constrained by groundwater physics.

**Scope (author decision 3):** satellite-only monitoring covers **alluvial aquifers
only**. In hard rock, the InSAR signal of recharge (~0.04 mm for 2 m of head) is about
100× below Sentinel-1 noise. The reasoning is in `docs/open_questions.md` (OQ1/OQ8).

**Summary.**
- Review 2 fixed every open finding from review 1, plus 15 of the 22 new ones (11–32).
  Each fix has a test that failed before it and passes after.
- Full suite: **135 passed, 1 xfailed**. ruff and black are clean, and the dashboard
  compiles (`npm run build`).
- **What the current code can do, on synthetic data from a compliant aquifer:**
  - From InSAR alone, it **ranks structures by recharge and separates working from
    failing ones**: ρ 0.95, AUROC 1.00 in known-answer run 3, which meets the
    pre-registered ranking thresholds.
  - This holds only under a training protocol revised **after** the official
    pre-registered run 1 **failed** (ρ 0.26, AUROC 0.57).
- **What it cannot do:** give absolute recharge.
  - From InSAR alone this is impossible in principle: only R·S_k/S is identifiable.
  - Even with the aquifer constants known, the current code misses the pre-registered
    accuracy (median error 0.42 > 0.30; cause not established).
- So the API and dashboard now show a **relative recharge ranking** when there is no well
  data, and a percentage, labelled provisional, only with well data (decision 5).

**The model's outputs are still not an assessment of any real structure.**

---

## Author decisions of 2026-10-24 (all made after the failed pre-registered run 1)

| # | Decision | Implemented in | Test |
|---|---|---|---|
| 1 | Global constants S, S_k, L get their own learning rate (0.02) | `train.build_optimizer` (train, demo, experiment run 3) | `tests/test_training.py` |
| 2 | Design capacity = soft limit on the seasonal (122-day) average recharge, not a per-pass ceiling. PROVISIONAL (OQ19) | `losses.CompositeARSLoss` | `TestSeasonalCapacity` |
| 3 | Hard rock out of scope; alluvial aquifers only | `CLAUDE.md`, `docs/open_questions.md` | (docs) |
| 4 | Well water levels: optional training input; missing wells reported | dataset, loss (offset-free well term), `train.py` warning, API `has_well_data` / `well_head_rmse_m` | `tests/test_wells.py` |
| 5 | No wells → relative recharge ranking + one-line note; percentage only with wells | API `relative_recharge_rank` / `_percentile`, `assessment_basis` / `_note`; dashboard map, hover and header | `TestRelativeRankingWithoutWells` |
| 6 | Dashboard must compile | `npm install` + `npm run build`: no errors, no fixes needed | build log |
| 7 | Merge into `main` locally, keep `review-2` | fast-forward merge, not pushed | — |

---

## Component status

| Component | State | Tested | Known issues / open |
|---|---|---|---|
| **Data contracts** (`schemas.py`) | Required per-observation incidence angle; one track per record; LOS "+ toward satellite" (provisional); shared reliability rule; optional `recharge_area_m2`; wells optional | `test_schemas.py`, `test_coupling.py`, `test_wells.py` | ±10 cm bound on *cumulative* displacement (21, OQ17); silent rainfall clamp (26) |
| **Dataset** | Causal rain binning; Δt in days; `valid_step_mask`; wells matched within ±3 days (OQ20) | `test_dataset.py`, `test_wells.py` | Fabricated Δt₀ (22, low) |
| **Fixed structure graph** | One graph for all structures; physics a_ij = c/d² [1/m²] separate from the GAT prior; seed + 3-hop batches | `test_graph.py` | Connectivity provisional (OQ11) |
| **Network** | GAT self-loop + skip; decoders conditioned at every step; R and v scaled to physical magnitudes; global S, S_k, L | `test_models.py` | — |
| **Flow physics** | S ∂h/∂t = ∇·(T∇h) − L h + R (m/day), real Δt, leakage sink | `test_losses.py`, `test_graph.py` | Pumping not modelled (OQ14); h_ref = 0 (OQ13) |
| **Deformation coupling** | d = 100·cos θ·S_k·Δh | `test_coupling.py` | Elastic, vertical only, no lag (OQ4, OQ5) |
| **Loss** | Dimensionless total (σ_d, σ_r, σ_h); seasonal capacity limit; mass balance in m³; optional well term; physics at low-coherence steps | `test_losses.py`, `test_wells.py` | Scales and window provisional (OQ18, OQ19) |
| **Training** | Fixed-graph minibatches; global-scalar learning rate; reports structures without wells | `test_training.py` | No validation split (24) |
| **API** | Relative ranking always; percentage only with wells (provisional: depends on fitted S); /explain on the same graph; thread-safe | `test_api.py`, `test_inference.py`, `test_export_onnx.py` | Network uses wells only through training, not at prediction time; `latent_drift_score` is not a drift (23); WebSocket untested (27) |
| **Dashboard** | Banner; ranking view without wells; percentage view with wells | `npm run build` passes | 11 npm audit vulnerabilities in pinned dependencies (not changed) |
| **CV branch** | Unused | none | Decide scope (28) |

---

## Known-answer experiment (`experiments/synthetic_recovery/`)

| run | protocol | A: InSAR only ρ / AUROC | B: S, S_k known, median rel. err | D: + wells ρ / AUROC | outcome |
|---|---|---|---|---|---|
| 1a | pre-registered, before decoder fix | 0.22 / 0.59 | 0.27 (vacuous: AUROC 0.50) | 0.97 / 1.00 | fail |
| **1b** | **pre-registered (OFFICIAL)** | **0.26 / 0.57** | 0.26 | 0.95 / 1.00 | **FAIL** |
| 2 | diagnostic: lr 0.02 for global scalars | 0.95 / 1.00 | 0.44 | 0.93 / 1.00 | A met, B failed |
| 3 | current code (decisions 1, 2, 4) | **0.95 / 1.00** | **0.42** | 0.96 / 1.00 (absolute heads, datum removed) | A met, B failed |

Thresholds: A needs ρ ≥ 0.70 and AUROC ≥ 0.85; B needs median relative error ≤ 0.30
(oracle floor 0.061). Runs 2 and 3 used protocols revised after seeing run 1.

In the suite:
- `test_current_code_ranks_structures_from_insar_alone` passes.
- `test_current_code_recovers_absolute_recharge_with_known_constants` is
  `xfail(strict=True)`.

The run-2 explanation of B (capacity ceiling) was withdrawn after run 3. B's cause is not
established.

## Demo — SYNTHETIC DATA (`scripts/demo_synthetic.py`, current code)

24 structures × 52 steps, seed 0, 300 epochs, CPU, 65 s; two runs byte-identical.

| quantity | value | planted |
|---|---|---|
| S / S_k / L | 0.142 / 5.4e-3 / 2.1e-4 | 0.05 / 1e-3 / 8.3e-4 |
| PDE residual | 8.5e-11 (m/day)² | — |
| data misfit | 0.0094 cm², falling steadily (RMSE ≈ 0.10 cm vs 0.05 cm noise) | — |
| mean R | ≈ 9× below planted | — |

Before decision 1 the misfit was 0.0325 cm² and rose after epoch 60. The absolute R and
the constants are wrong, as identifiability predicts; only R·S_k/S, and rankings, are
determined.

---

## What is still NOT proven

1. **Nothing on real data.** No Sentinel-1, IMD rainfall, CGWB wells or India-WRIS designs
   have been processed.
2. **Absolute recharge.** From InSAR alone it is impossible in principle (only R·S_k/S;
   `TestIdentifiability`). With wells alone it is still fixed only up to S (only R/S).
   And even with S and S_k given, the current code misses the pre-registered accuracy
   (B, cause unknown). **The "with wells" percentage therefore depends on the fitted
   storativity and is labelled provisional.**
3. **Ranking** has been shown only on synthetic data from a compliant aquifer with a
   strong signal, and only under a protocol revised after the official run failed.
4. **Hard rock** is out of scope (decision 3).
5. **Wells at prediction time.** The deployed network learns from wells only during
   training. At prediction time, `has_well_data` and `well_head_rmse_m` report the wells
   but do not change the network's output.
6. Many constants are provisional (`docs/open_questions.md`, OQ1–OQ20). Seven review
   findings remain open (21–24, 26–28).
