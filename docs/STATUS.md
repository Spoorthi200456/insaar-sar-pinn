# Project status — InSAR–PINN monitoring of Artificial Recharge Structures

*Updated 2026-10-24 after review 2 (branch `review-2`, not merged). All numbers are from
**SYNTHETIC DATA**. No real Sentinel-1, rainfall or well data has been run through the
model.*

**Goal (fixed):** assess whether Artificial Recharge Structures (ARS) in India are working
(recharging the aquifer) or failing (clogging, leakage, structural damage), using freely
available InSAR, rainfall and design data, constrained by groundwater physics.

**Summary.**
- Review 2 fixed every open finding from review 1 (2, 4, 5, 6, 7, 8, the `is_reliable` rule
  and the Δt-blind head integrator). A fresh review found 22 more problems (11–32); 12 are
  fixed and 10 are documented as open.
- Each fix has a test that failed before it and passes after. Full suite: **118 passed,
  1 xfailed**.
- The xfailed test is the honest headline result. In a **pre-registered known-answer
  test**, the model **failed** to recover which structures recharge from InSAR alone
  (ranking ρ 0.26, working/failing AUROC 0.57).
- With well heads, the same model recovered them almost perfectly (ρ 0.95, AUROC 1.00).
- A diagnostic re-run (a protocol change declared before running, but chosen after
  seeing the failure) shows the InSAR-only failure was an **optimiser** problem
  (ρ 0.95, AUROC 1.00 once the global constants could learn fast enough).
- Even then, **absolute** recharge cannot be recovered from InSAR alone. That is a
  property of the physics, not the code (`experiments/synthetic_recovery/README.md`).

**The model's outputs are still not an assessment of any real structure.** See "What
is still NOT proven" below and `docs/REVIEW_2_SUMMARY.md`.

---

## Component status

| Component | State | Tested | Known issues / open |
|---|---|---|---|
| **Data contracts** (`schemas.py`) | Required per-observation incidence angle; one track per record; LOS sign "+ toward satellite" (provisional); `is_reliable` = same rule as the dataset; optional `recharge_area_m2` | `test_schemas.py`, `test_coupling.py` | ±10 cm bound on *cumulative* displacement (finding 21, OQ17); silent rainfall clamp (26) |
| **Preprocessing / dataset** | Causal rain binning; Δt in days; `valid_step_mask` separate from coherence | `test_dataset.py` | Fabricated Δt₀ (22, low) |
| **Fixed structure graph** (`graph.py`) | One graph built from all structures; physics a_ij = c/d² [1/m²] kept separate from the GAT attention prior; seed + 3-hop batches | `test_graph.py`: physics identical for any batch size or shuffle | Connectivity provisional (OQ11) |
| **Graph attention** | Self-loop + skip: isolated structures keep their own features | `test_models.py` | — |
| **Decoders** (R and head rate) | Structure embedding conditions every step (finding 30) | `test_models.py` | — |
| **Flow physics** (`losses.pde_residual`) | S ∂h/∂t = ∇·(T∇h) − L h + R, every term m/day, real Δt, Hantush leakage sink | `test_losses.py`, `test_graph.py` | Pumping not modelled (OQ14); h_ref = 0 (OQ13) |
| **Head integrator** | h_k = h_{k−1} + Δt_k·v_k in metres; h = 0 at the first acquisition | `test_models.py` | — |
| **Deformation coupling** (`coupling.py`) | d = 100·cos θ·S_k·Δh: correct sign, units and LOS projection | `test_coupling.py` | Elastic, vertical only, no lag (OQ4, OQ5) |
| **Global constants** S, S_k, L | Bounded, learnable | range tests | Learn too slowly at the network learning rate (finding 32, open). Not identifiable from InSAR: only R·S_k/S is |
| **Loss** | Dimensionless total with stated scales (σ_d, σ_r, σ_h); regularisers in consistent units and masked to real steps; physics applied at low-coherence steps; no clamp | `test_losses.py` | Per-step design-capacity ceiling (finding 31, OQ19); scales provisional (OQ18) |
| **Training** (`train.py`) | Fixed-graph minibatches, Windows workers OK | smoke run | No validation split (24) |
| **API / inference** | Verdict = mean(R·A)/Q_design (dimensionless) on real steps; transmissivity reported with its units; /explain uses the same graph; thread-safe MC dropout | `test_api.py`, `test_inference.py`, `test_export_onnx.py` | `latent_drift_score` is not a drift (23); WebSocket untested (27); **the efficiency is not meaningful in absolute terms from InSAR alone** (identifiability) |
| **Dashboard** | "SYNTHETIC DATA, PROVISIONAL PHYSICS" banner; types updated | **not compiled** (no `npm install`) | — |
| **CI** | ruff and black now pass, so the test job can run | — | — |
| **CV branch** | Unused | none | Decide scope (28) |

---

## Known-answer experiment (`experiments/synthetic_recovery/`)

The simulator is independent: pure NumPy, sharing no code with the network. It has 16
structures (10 working, 6 failing), 40 irregular acquisitions and a compliant aquifer
(S_k = 5e-3), so the signal is about 6× the noise. Thresholds were committed before the
first run.

| run | protocol | A: InSAR only ρ / AUROC | B: S, S_k known, median rel. err | D: + wells ρ / AUROC | outcome |
|---|---|---|---|---|---|
| 1a | pre-registered, before decoder fix | 0.22 / 0.59 | 0.27 (vacuous: AUROC 0.50) | 0.97 / 1.00 | fail |
| **1b** | **pre-registered (official)** | **0.26 / 0.57** | 0.26 | 0.95 / 1.00 | **FAIL** |
| 2 | revised after 1b (lr 0.02 for S, S_k, L), declared before running | 0.95 / 1.00 | **0.44** | 0.93 / 1.00 | fails (B) |

Thresholds: A needs ρ ≥ 0.70 and AUROC ≥ 0.85; B needs median relative error ≤ 0.30.
Oracle noise floor: 0.061. The pytest is marked `xfail(strict=True)`.

## Demo — SYNTHETIC DATA (`scripts/demo_synthetic.py`)

24 structures × 52 steps, seed 0, 300 epochs, CPU, 65 s; two runs byte-identical
(`docs/demo/`).
- Learned S / S_k / L: 0.021 / 1.2e-4 / 9.9e-4, against planted 0.05 / 1e-3 / 8.3e-4.
- PDE residual 1.2e-10 (m/day)². Data misfit 0.0325 cm² (RMSE 0.18 cm against 0.05 cm
  noise); it *rises* after epoch ~60.
- Mean R is about 34× below planted.

This is the same optimiser limitation as run 1b (finding 32), not a new problem. The
design-capacity and mass-balance penalties are now active: their units are consistent, and
they are zero at the end only because R is small.

---

## What is still NOT proven

1. **Nothing on real data.** No Sentinel-1 product, IMD rainfall, CGWB wells or India-WRIS
   designs have been processed.
2. **Absolute recharge from InSAR alone is impossible in principle.** Only R·S_k/S is
   determined. Proven exactly on the simulator (`TestIdentifiability`). An absolute
   efficiency needs independent S_k and S (wells + pumping tests).
3. **Ranking from InSAR alone has been shown only under a protocol revised after seeing a
   failure** (run 2), and only on synthetic data with a strong signal. The training code in
   `src/` was not changed to that protocol (author's decision, finding 32).
4. **Hard-rock aquifers** (S_k ~ 1e-5, most peninsular check dams): the InSAR signal of
   recharge is far below the noise. The model can say nothing there (scenario C), and
   aquifer recharge may be invisible to InSAR (OQ1, OQ8).
5. **Many physical constants are provisional** (`docs/open_questions.md`, OQ1–OQ19): LOS
   sign, incidence source, S_k range, recharge area, runoff coefficient, connectivity,
   leakance reference, pumping, loss scales, meaning of design capacity.
6. **Well heads are not used in training** (finding 25). The experiment shows they are the
   single most valuable addition.
