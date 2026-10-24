# Code review findings — core model and data (2026-10-15)

Scope: `src/models/`, `src/data/`, `src/training/` reviewed as a whole (no diff;
the folder was not a git repo at review time). Line numbers refer to commit
`1563ecb` plus this commit.

Numbering is by severity for the research goal (does the model correctly say
whether an ARS is recharging or failing?), not by file order.

| # | Where | Short title | Status |
|---|---|---|---|
| 1 | `src/models/losses.py:130` | PDE residual has wrong sign — recharge acts as a sink | **Fixed** |
| 2 | `src/models/network.py:182` | Deformation coupling: wrong sign, wrong coefficient, no LOS projection | Open |
| 3 | `src/models/losses.py:123-125` | Storage term has no S_s and no ÷Δt | **Fixed** (integrator Δt still open) |
| 4 | `src/data/dataset.py:214-228` | Physics graph built from minibatch composition | Open |
| 5 | `src/models/layers.py:126-135` | GAT has no self-loop — isolated nodes get zero embedding | Open |
| 6 | `src/models/losses.py:147-159` | Q compared against m³/day and against mm | Open |
| 7 | `src/models/losses.py:147-159` | Regularizers not masked — padding adds phantom recharge | Open |
| 8 | `src/models/losses.py:131-132` | PDE loss masked by coherence; residual hard-clamped | Open |
| 9 | `src/data/preprocessing.py:91-113` | Rainfall binned by midpoint (non-causal) | **Fixed** |
| 10 | `src/data/dataset.py:115-116` | Clamped observations used as trusted targets | **Fixed** |

Findings 1 and 3 were fixed in form (b) (depth-integrated 2D, provisional) after
a plan-mode derivation approved by the author; see the commit message and
`CLAUDE.md` → Governing physics. Open findings 2, 4–8 change a PDE term, loss term, sign, unit or the
architecture, so per `CLAUDE.md` each needs a written derivation and the
author's approval in plan mode before any code changes.

---

## 1. PDE residual has the wrong sign

- **Where:** `src/models/losses.py:130` — `pde_residual = divergence - storage_term - recharge_rate`
- **Problem:** `CLAUDE.md` fixes the residual as r = S_s ∂h/∂t − ∇·(K∇h) − Q.
  The code computes ∇·(K∇h) − storage − Q = −(storage − ∇·(K∇h) + Q). Compared
  with the correct residual, Q has the opposite sign, i.e. it enters as
  extraction, not recharge.
- **Failure scenario:** a working structure (Q > 0, head rising) gets a large
  residual. Zero residual requires storage = ∇·(K∇h) − Q, so the model is
  pushed either to predict falling head under recharge or to drive Q → 0.
  Working and failing structures can be swapped.
- **Fix:** new shared `pde_residual()` in `src/models/losses.py` computes
  r = S ∂h/∂t − ∇·(T∇h) − R (form (b), every term m/day). Used by
  `CompositeARSLoss` and by the demo's "PDE residual norm"
  (`src/backend/inference_engine.py`), which previously used a third,
  different formula.
- **Tests:** `tests/test_losses.py::TestPDEResidualPhysics::test_recharge_with_rising_head_has_zero_residual`
  and `::test_recharge_with_falling_head_is_penalised`. Verified against the
  pre-fix code (`ba8467d`) with the same scenario: rising head under recharge
  got PDE loss 1.22e-4 (new: 0) and falling head got the *lower* loss
  1.18e-4 — the old code preferred the physically wrong answer.
- **Status:** Fixed.

## 2. Deformation coupling: wrong sign, wrong coefficient, no LOS projection

- **Where:** `src/models/network.py:182` — `d_pred = -biot_alpha.unsqueeze(-1) * delta_h`;
  also `src/data/schemas.py:107` (field doc: "positive = uplift") and the
  synthetic generator `src/training/train.py:61`.
- **Problem:** (a) A head rise lowers effective stress and causes uplift, so
  with "positive = uplift" a head rise must give positive displacement; the
  code gives negative. (b) The dimensionless Biot α ∈ (0, 1] is used as the
  coefficient mapping head [m] to displacement [cm]; that coefficient must
  have units of cm/m (e.g. S_k·b-type skeletal-storage term), not be α.
  (c) No projection from vertical to line of sight using the incidence angle.
  (d) The synthetic generator uses the same wrong sign plus an extra `0.01`
  factor that the model does not have, so synthetic tests cannot catch it.
- **Failure scenario:** recharge-driven uplift can only be fitted with Δh < 0,
  which forces Q (softplus, ≥ 0) toward 0; working structures look like
  failures.
- **Proposed fix:** derive the coupling (vertical displacement from head
  change via skeletal specific storage × aquifer thickness, units m/m → convert
  to cm), project to LOS with the per-structure incidence angle, and fix the
  generator to use the same relation. Requires the LOS convention below to be
  confirmed first.
- **Status:** Open. Provisional LOS conventions recorded in `CLAUDE.md`;
  coupling code unchanged.

## 3. Storage term has no S_s and no division by Δt

- **Where:** `src/models/losses.py:123-125` — `storage_term[:, 1:] = h[:, 1:] - h[:, :-1]`;
  related: `src/models/network.py:69-75` (integrator applies a per-step decay
  that ignores Δt).
- **Problem:** S_s ∂h/∂t is approximated by a raw step difference: no S_s
  [1/m] and no ÷Δt [days]. Sentinel-1 intervals are 6, 12 or 24 days (or more
  when passes are missing), so the same physical head rate gives different
  "∂h/∂t" depending on revisit. Padded steps have Δt = 0 and would divide by
  zero once Δt is introduced.
- **Failure scenario:** K and Q estimates depend on the satellite revisit
  pattern instead of the aquifer.
- **Fix:** storage term = S·(h_k − h_{k−1})/Δt_k with Δt from
  `time_delta_days`; learnable global storativity S = sigmoid(logit) ∈ (0, 1),
  init 0.02 (`ARSNet.storativity_logit`, form (b): S is dimensionless, so
  sigmoid rather than softplus, since S_y < porosity < 1). Steps where k or
  k−1 is padding (Δt = 0), and step 0, are excluded from the PDE loss; no
  division by zero.
- **Tests:** `tests/test_losses.py::TestPDEResidualPhysics::test_storage_uses_real_dt`,
  `::test_padding_and_first_step_excluded`,
  `TestCompositeARSLoss::test_gradient_reaches_storativity`,
  `tests/test_models.py::TestARSNet::test_storativity_is_dimensionless_fraction`.
  Verified against pre-fix code: for one constant physical rate sampled at
  Δt = 6/12/24/12/6 days, the old storage term was
  0.006/0.012/0.024/0.012/0.006 (proportional to Δt); new S·dh/dt is 1e-4 m/day
  at every step.
- **Still open (not part of this fix):** `HeadChangeIntegrator`
  (`src/models/network.py:69-75`) still integrates per step with a fixed decay
  that ignores Δt, and produces h with no physical units. h is *declared* to
  be in m, not enforced. Architecture change, needs its own plan.
- **Caveat:** S is not identifiable from the PDE term alone
  ((S, T, R) → (cS, cT, cR) leaves r unchanged). Report it as a fitted
  coefficient, not as a measured specific yield.
- **Status:** Fixed (storage term); integrator Δt open.

## 4. Physics graph built from minibatch composition

- **Where:** `src/data/dataset.py:214-228` (`ARSCollateFn.__call__` rebuilds
  the (B, B) adjacency from whichever samples share a batch, `shuffle=True`);
  consumed by the GAT (`src/models/network.py:156-158`) and by ∇·(K∇h)
  (`src/models/losses.py:127-129`).
- **Problem:** violates `CLAUDE.md` hard rule 3.
- **Failure scenario:** a structure whose hydraulic neighbours land in another
  batch gets divergence = 0 and an isolated GAT node; its K, Q and residual
  change each epoch, with `batch_size`, and at inference with whichever
  structures are submitted together.
- **Proposed fix:** build one fixed N×N structure graph for the whole network
  once; each batch indexes rows/columns for its structures and gathers
  neighbour states from a full-graph forward (or uses neighbour sampling from
  the fixed graph).
- **Units (found while fixing 1 and 3):** in form (b),
  Σ_j a_ij (T_i+T_j)/2 (h_j − h_i) is in m/day only if a_ij carries 1/m²
  (e.g. shared boundary length / (distance × cell area), finite-volume
  stencil). The current a_ij = exp(−d/r) is dimensionless, so this term is in
  m³/day. Not rescaled (hard rule 2); redefine a_ij as part of this finding.
  With the 5 km radius and ~50 km spacing of the synthetic set, a_ij = 0
  everywhere, so the term is currently 0.
- **Status:** Open.

## 5. GAT has no self-loop

- **Where:** `src/models/layers.py:126-135`; diagonal zeroed at
  `src/data/preprocessing.py:143`.
- **Problem:** a node's own features never reach its output; a node with no
  neighbours gets softmax(all −inf) → NaN → 0, so its embedding is relu(0) = 0.
- **Failure scenario:** with a 5 km radius and structures ~50 km apart (as in
  the synthetic set), every node is isolated, so design features (height,
  storage, catchment) have no effect on K or Q.
- **Proposed fix:** add self-loops (or a residual/skip connection) in the GAT
  layer, keep the physics adjacency (diagonal 0) separate from the GAT
  adjacency.
- **Status:** Open.

## 6. Q compared against two different units

- **Where:** `src/models/losses.py:147-150` (Q vs design capacity, m³/day) and
  `src/models/losses.py:155-159` (Σ Q vs Σ rainfall, mm per interval).
- **Problem:** Q cannot be both m³/day and mm; violates hard rule 2. The units
  table in `CLAUDE.md` still has Q as TODO. The flow equation additionally
  needs Q as a volumetric source per unit volume [1/day].
- **Failure scenario:** one of the two penalties is meaningless whatever Q's
  units are; the mass-balance hinge compares a volume to a depth without a
  catchment area.
- **Proposed fix:** fix Q's units from the PDE (1/day), convert to m³/day via
  aquifer area × thickness for the capacity check, and convert rainfall to
  volume via catchment area × runoff/infiltration coefficient for the mass
  balance. Record all conversions in the units table.
- **Status:** Open.

## 7. Regularizers not masked

- **Where:** `src/models/losses.py:147-159` (design capacity, non-negativity,
  mass balance); module docstring claims "Every term is masked".
- **Problem:** left-padded steps contribute softplus(Q) > 0 to cumulative
  recharge while their rainfall is 0.
- **Failure scenario:** a structure with 10 real steps padded to 52
  accumulates ~42 phantom Q values; the mass-balance penalty grows with padding
  length and pushes Q down for short series.
- **Proposed fix:** pass a validity mask (real step, independent of coherence)
  and apply it to all three regularizers.
- **Status:** Open.

## 8. PDE loss masked by coherence; residual hard-clamped

- **Where:** `src/models/losses.py:131-132`.
- **Problem:** the physics term uses `observation_mask` (coherence-based), so
  physics is switched off exactly at low-coherence / interpolated steps,
  where a PINN constraint is most useful. `torch.clamp(±1e4)` gives zero
  gradient for any residual beyond the bound.
- **Failure scenario:** gaps in InSAR coverage are filled with no physics
  constraint; a badly wrong state gets no correction from the physics term.
- **Proposed fix:** mask the PDE term only by validity (not padding), not by
  coherence; replace the hard clamp with residual normalisation or a Huber
  loss.
- **Status:** Open.

## 9. Rainfall binned by midpoint (non-causal) — FIXED

- **Where:** `src/data/preprocessing.py:91-113`, `aggregate_rainfall_to_insar_cadence`.
- **Problem (old code):** rain was assigned to the nearest acquisition by
  midpoint, so rain falling *after* acquisition k could be credited to step k,
  and rain after the last acquisition was clipped into the last step.
- **Failure scenario:** a storm 2 days after acquisition k (12-day revisit)
  was used as forcing for the displacement already observed at k — a
  non-causal leak, inconsistent with "mm per InSAR interval".
- **Fix:** step k sums rain in (t_{k-1}, t_k]; step 0 takes everything up to
  t_0. Rain after the last acquisition is **dropped** (not flagged, not
  returned).
- **Test:** `tests/test_dataset.py::TestPreprocessing::test_rainfall_binned_causally_and_rain_after_last_acquisition_dropped`
  — rain at t = 0, 5, 10, 11, 25 (1–5 mm) with acquisitions at 0, 10, 20
  must give `[1, 5, 4]`, and the 5 mm at t = 25 must not be in the last bin.
  Verified: **fails on the pre-fix code** (got `[1, 9, 5]`), passes on the fix.
- **Status:** Fixed.

## 10. Clamped observations used as trusted targets — FIXED

- **Where:** `src/data/dataset.py:115-116`.
- **Problem (old code):** `reliable = coherence >= 0.7` ignored `quality_flag`,
  so values the schema had clamped to ±9.999 cm (`OUT_OF_BOUNDS_CLAMPED`)
  were fitted as truth; the 0.7 threshold was hardcoded.
- **Failure scenario:** a real −14 cm subsidence event at coherence 0.9 is
  stored as −9.999 and used as a data-misfit target, biasing K and Q.
- **Fix:** reliable requires coherence ≥ `COHERENCE_THRESHOLD` **and**
  `quality_flag` not in {`OUT_OF_BOUNDS_CLAMPED`, `MISSING`}.
- **Test:** `tests/test_dataset.py::TestARSDataset::test_clamped_observation_is_not_a_reliable_target`.
  Verified: **fails on the pre-fix code** (mask was `True`), passes on the fix.
- **Related, still open:** `src/data/schemas.py:130-133`,
  `SARObservation.is_reliable`, has the same omission (ignores
  `OUT_OF_BOUNDS_CLAMPED`, hardcodes 0.7). Not used by the dataset path, but
  any caller relying on it gets the old behaviour.
- **Status:** Fixed (dataset path).

---

## Verification log

### Review 2, step 1 (2026-10-24, branch `review-2` from `main` @ `cd55d40`)

The four fixed findings were re-checked on the current code before any change:

- Finding 1 (recharge sign): `src/models/losses.py:99` returns
  `storage_term - divergence - recharge_m_per_day`, i.e. r = S∂h/∂t − ∇·(T∇h) − R. ✓
- Finding 3 (S and ÷Δt): `losses.py:87-97` divides by the real Δt and
  multiplies by S; padding and step 0 are masked. ✓
- Finding 9 (causal rainfall): `src/data/preprocessing.py:109` uses
  `searchsorted(side="left")` → bins (t_{k−1}, t_k]; rain after the last
  acquisition is dropped. ✓
- Finding 10 (clamped targets): `src/data/dataset.py:115-116` requires
  coherence ≥ threshold **and** flag ∉ {CLAMPED, MISSING}. ✓
- Their tests (`TestPDEResidualPhysics` ×4, rainfall binning, clamped target):
  **6 passed**. Full suite: **60 passed**.

- Pre-fix code was not in git (fixes were applied before `git init`). It was
  reconstructed exactly from the review agent's edit script (which asserted
  the old text before replacing it) in a scratch copy, and the two new tests
  were run there: **2 failed, 52 passed**.
- Fixed code (this commit): **54 passed** (`pytest -q`, Python 3.11.9,
  torch 2.14.1+cpu).
- Line numbers in the review agent's original report for `network.py`,
  `preprocessing.py` and `train.py` were offsets into concatenated listings
  and were wrong; the numbers above are the corrected ones.
