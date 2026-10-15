# Project status — InSAR–PINN monitoring of Artificial Recharge Structures

*For faculty review, 2026-10-15. Code at `main`. All numbers below are from
**SYNTHETIC DATA** unless stated otherwise. No real Sentinel-1 data has been
run through the model yet.*

**Goal (fixed):** assess whether Artificial Recharge Structures (ARS) in India
are working (recharging the aquifer) or failing (clogging, leakage, structural
damage), using freely available InSAR, rainfall and design data, constrained
by groundwater physics.

**Summary.** The full pipeline runs end to end (data validation → model →
physics-informed loss → training → API/dashboard), and 60 automated tests
pass. A code review found 10 problems (`docs/review.md`):

- **Fixed (4 of 10):** the groundwater-flow equation now has the correct
  recharge sign and a proper time derivative (findings 1 and 3), and two
  data-handling bugs are fixed (findings 9 and 10).
- **Open (6 of 10):** the coupling between head and InSAR displacement is
  still wrong in sign and units (finding 2), along with five other problems.

**Because of these open problems, the model's outputs are not yet
physically meaningful and must not be read as an assessment of any
structure.**

---

## Component status

| Component | Implemented | Tested | Known issues | Next step |
|---|---|---|---|---|
| **Data contracts** (`src/data/schemas.py`) | Yes — Pydantic schemas for design, SAR, rainfall and well data; ±10 cm displacement clamp; quality flags | Yes — `tests/test_schemas.py` (13 tests) | `SARObservation.is_reliable` ignores `OUT_OF_BOUNDS_CLAMPED` (same omission as finding 10, not on the training path). Field doc says "positive = uplift", which conflicts with the provisional LOS convention "positive = toward satellite". | Confirm the LOS convention against the actual Sentinel-1 product; fix `is_reliable`. |
| **Preprocessing** (`src/data/preprocessing.py`) | Yes — causal rainfall binning into (t_{k−1}, t_k], gap interpolation, clamping, Δt in days | Yes — `tests/test_dataset.py` (rainfall-binning test fails on the pre-fix code) | Graph weights a_ij = exp(−d/r) have no units (see open unit problem 1). | Redefine a_ij with finite-volume geometry (finding 4). |
| **Dataset / batching** (`src/data/dataset.py`) | Yes — fixed-length series, left padding, reliability mask using coherence **and** quality flag | Yes — `tests/test_dataset.py` (6 dataset/loader tests) | The structure graph is rebuilt from whichever structures share a minibatch (finding 4). The demo avoids this by training on one full batch. | Build one fixed network-wide graph (finding 4). |
| **Spatial encoder — graph attention** (`src/models/layers.py`) | Yes | Yes — shapes, no-NaN on isolated node | No self-loop: a structure with no neighbour gets an all-zero embedding, so its design features have no effect (finding 5). In the demo, 22 of 24 structures have no neighbour. | Add self-loops / skip connection (finding 5). |
| **Temporal encoder** (transformer over displacement, rainfall, Δt) | Yes | Yes — shapes, padding mask | None known beyond the masking issue in finding 8. | — |
| **Transmissivity head** (network output `hydraulic_conductivity`, interpreted as T) | Yes — positive via exp | Yes — positivity | Units (m²/day) declared, not enforced. Demo values 0.0003–0.0075 m²/day are orders of magnitude below typical hard-rock values (roughly 1–100 m²/day). T is barely constrained: only 1 graph edge. | Enforce units; add a prior range from CGWB aquifer data. |
| **Recharge decoder** (network output `recharge_rate`, R) | Yes — non-negative via softplus | Yes — non-negativity | Units (m/day) declared, not enforced. **Demo R ≈ 0.17 m/day ≈ 170 mm/day, about 29× the synthetic rainfall (5.8 mm/day). Physically impossible.** The constraints that should catch this are inactive because of unit mismatches (finding 6). | Fix units of R and of the regularisers (finding 6). |
| **Head integrator** (`HeadChangeIntegrator`) | Yes | Indirectly (gradient flow) | Ignores Δt and has no physical units; h is declared to be in m but not enforced. | Rewrite as a Δt-aware time step of the flow equation (needs its own plan). |
| **Deformation coupling** (head → InSAR displacement, `network.py:182`) | Yes, but wrong | Not tested for physics | **Wrong sign** (a rise in head should cause uplift); uses the dimensionless Biot α as a m→cm coefficient; no line-of-sight projection with the incidence angle (finding 2). The synthetic generator shares the same error, so tests cannot catch it. | Derive the coupling (skeletal storage × thickness), add LOS projection; fix the generator (finding 2). **Highest priority.** |
| **Physics residual** (`losses.py::pde_residual`) | Yes — r = S ∂h/∂t − ∇·(T∇h) − R, form (b), every term m/day; correct recharge sign; ÷ real Δt; padding excluded | Yes — 4 physics tests, all shown to fail or misrank on the pre-fix code | Lateral-flow term has wrong units because of a_ij (open unit problem 1). Masked by coherence; hard ±1e4 clamp (finding 8). In the demo the residual stays large (RMS ≈ 0.19 m/day, about the size of R itself), so the physics is not yet satisfied. | Fix a_ij (finding 4) and masking (finding 8). |
| **Storativity S** (`ARSNet.storativity_logit`) | Yes — one global learnable value, 0 < S < 1, initialised to 0.02 | Yes — range and gradient tests | Not identifiable from the physics term alone ((S, T, R) → (cS, cT, cR) gives the same residual). Demo: 0.020679 after 300 epochs, barely moved. | Treat as a fitted coefficient; constrain with field data (pumping tests, CGWB norms). |
| **Regularisers** (design capacity, non-negativity, mass balance) | Yes | Yes — each activates in a unit test | **All three were 0 at every epoch of the demo**, so they constrain nothing. Design capacity compares R (m/day) with m³/day; mass balance compares summed R with summed rainfall in mm (finding 6). Non-negativity can never fire because softplus already makes R ≥ 0. Padded steps are not masked (finding 7). | Fix units and masking (findings 6, 7); drop or repurpose the non-negativity term. |
| **Training loop** (`src/training/train.py`) | Yes | Smoke-tested (2 epochs, no NaN) | Uses shuffled minibatches, so physics depends on batch composition (finding 4). | Switch to the fixed graph (finding 4). |
| **Reproducible demo** (`scripts/demo_synthetic.py`) | Yes — seed 0, CPU, full batch, about 50 s | Run twice: byte-identical results | Synthetic only; not a recovery test (no ground-truth T, R, S in form-(b) units). | Add a known-answer synthetic test once the coupling is fixed. |
| **Inference engine / API** (`src/backend/`) | Yes — FastAPI, eager PyTorch fallback, explainability endpoint; "PDE residual norm" uses the same residual as training | Yes — `tests/test_api.py` (7 tests) | Inherits all model issues above. | — (deployment plumbing, not the research contribution) |
| **ONNX export** (`src/backend/export_onnx.py`) | Yes | **No tests** | Not checked since the storativity output was added. | Add an export round-trip test. |
| **Dashboard** (`frontend/`) | Yes — Next.js map, time series, explainability panel | **No tests** | Displays model outputs that are not yet physically meaningful. | Add a "SYNTHETIC / provisional" banner until finding 2 is fixed. |
| **Computer-vision branch** (`CVEncoder`) | Code exists | Not exercised by any test, training or demo path | Unused. | Decide whether it is in scope. |

---

## Demo results — SYNTHETIC DATA

`scripts/demo_synthetic.py`, 24 structures × 52 weekly steps, seed 0, 300
epochs, CPU, one full batch. Plot: `docs/demo/loss_curves.png`; numbers:
`docs/demo/loss_history.csv`.

| Quantity | Value | Interpretation |
|---|---|---|
| Training time | ≈ 50 s | Within the 5-minute budget. |
| Reproducibility | 2 runs, identical CSV | Seeded and deterministic on CPU. |
| Learned S | 0.020679 (init 0.02) | Barely moved; not identifiable. |
| Graph edges | 1 of 276 pairs (5 km radius) | Lateral flow and T-smoothness are almost inactive. |
| PDE residual (MSE) | 0.53 → 0.035 (m/day)² | Falls, then plateaus at RMS ≈ 0.19 m/day. Physics not satisfied. |
| Data misfit (MSE) | 0.065 → 0.071 cm² | **Gets worse** as the physics term falls. Physics and data pull in opposite directions, as expected with the wrong coupling sign (finding 2). |
| Boundary, design-capacity, non-negativity, mass-balance terms | 0 at every epoch | Inactive (see Regularisers row). |
| T, structures ARS-0000/0001/0002 | 0.0075 / 0.0014 / 0.0003 m²/day | Implausibly low; units not enforced. |
| Mean R, same structures | 0.167 / 0.178 / 0.186 m/day | Implausibly high (≈ 29× rainfall); units not enforced. |

---

## What this does NOT do yet

**It does not tell you whether any real recharge structure is working or
failing.** No real data has been run, and the open problems below mean the
outputs are not physically meaningful even on synthetic data.

### Open unit problems (flagged, not hidden with scale factors)
1. **Graph weights a_ij have no units.** The lateral-flow term
   Σ_j a_ij (T_i+T_j)/2 (h_j − h_i) is in m/day only if a_ij carries 1/m²
   (shared boundary length / (distance × cell area)). Today it is the
   dimensionless exp(−d/r), so this term is in m³/day.
2. **Units of h and R are declared, not enforced.** Head h is declared in m
   and recharge R in m/day, but nothing in the network forces its outputs
   onto those scales. The demo's R ≈ 170 mm/day shows the consequence.
3. **The head integrator ignores Δt.** It steps once per acquisition with a
   fixed decay, whatever the gap (6, 12 or 24 days), and has no physical
   units. The PDE residual now uses the real Δt; the integrator does not.

### Open review findings (`docs/review.md`)
- **Finding 2 — deformation coupling is wrong.** Wrong sign (a head rise
  should give uplift), wrong coefficient (dimensionless Biot α used as m→cm),
  no line-of-sight projection with the incidence angle. The synthetic
  generator has the same error. **This blocks any working/failing
  interpretation.**
- **Finding 4 — physics depends on minibatch composition.** The structure
  graph is rebuilt from whichever structures share a batch; it must come from
  one fixed network-wide graph. (Avoided in the demo by full-batch training,
  not fixed in the code.)
- **Finding 5 — graph attention has no self-loops.** Structures without
  neighbours get zero spatial embeddings, so their design features are
  ignored.
- **Finding 6 — recharge is compared against incompatible units.** The
  design-capacity penalty compares R with m³/day, and mass balance compares
  it with rainfall in mm. Neither ever activates.
- **Finding 7 — regularisers are not masked.** Padded time steps add
  phantom recharge to the mass-balance sum.
- **Finding 8 — physics term masked by coherence and hard-clamped.** The
  physics is switched off exactly where InSAR data is poor, which is where it
  should help most; the ±1e4 clamp kills the gradient on large errors.

### Also not done
- No real Sentinel-1, rainfall or well data has been processed.
- No known-answer test (plant T, R, S in synthetic data, check the model
  recovers them).
- No validation against field observations (well levels, CGWB records).
- Line-of-sight sign and incidence-angle handling are recorded as
  **provisional** and are not yet used in code.
