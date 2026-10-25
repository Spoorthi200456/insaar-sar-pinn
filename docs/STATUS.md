# Project status — InSAR–PINN monitoring of Artificial Recharge Structures

*Updated 2026-10-24 after review 4 (branch `review-4`, pushed to origin; not merged into
`main`). All numbers are from **SYNTHETIC DATA**. No real Sentinel-1, rainfall or well data
has been run through the model.*

**Goal (fixed):** assess whether Artificial Recharge Structures (ARS) in India are working
(recharging the aquifer) or failing (clogging, leakage, structural damage), using freely
available InSAR, rainfall and design data, constrained by groundwater physics.

**Scope (author decision 3):** satellite-only monitoring of **alluvial aquifers only**.

## Review 4 (2026-10-24): method comparison on unseen structures (read first)

Three methods were compared on 30–31 held-out structures per seed, 5 seeds, 2 synthetic
truths. The protocol and pass marks were committed before any run
(`experiments/method_comparison/README.md`).

| method | mean ρ ± sd ("linear" truth) | mean AUROC | passes? |
|---|---|---|---|
| A: trained network applied unchanged | −0.05 ± 0.33 | 0.50 | **no**: chance level |
| B: network fitted to each structure's own data | 0.936 ± 0.008 | 1.00 | yes |
| C: least squares on a 1-parameter physical model, **no network** | **0.967 ± 0.005** | 1.00 | yes |
| oracle (all constants known) | 0.967 ± 0.006 | 1.00 | — |

- **B is not clearly better than C. C beat B in all 10 seed-truth runs.** So the neural
  network is not needed for ranking structures and flagging failing ones, on this
  synthetic data.
- Per-structure fitting (B) does fix the generalisation failure of the trained network,
  but at about 3000× the compute of C, and never better.
- **Caveat:** the simulator *is* C's model, and the "threshold" robustness truth could not
  change the structures' order (one shared rainfall series). That was noticed after the
  run and reported. Behaviours that differ between structures (progressive clogging,
  local thresholds, pumping) have not been tested; that is where a richer model could
  still matter.
- Absolute recharge remains unidentifiable from InSAR alone, for every method.

## Headline (review 3; still true for the trained network, method A above)

**On structures it has never seen, the model cannot yet tell working structures from
failing ones.**

Review 3 added a proper train / validation / test split and re-ran the known-answer test,
scoring only structures held out from training. The recharge ranking was at chance level:
Spearman ρ −0.03, working/failing AUROC 0.54. That held in every scenario: InSAR only,
aquifer constants given, and wells at the training structures.

The earlier "success" (ρ 0.95, AUROC 1.00, runs 2–3) was measured on the same 16
structures the model trained on. It does not generalise.

The model *does* reproduce unseen structures' InSAR displacement reasonably well, and the
physics is satisfied. So the gap is specifically **recharge**, the quantity the project
needs.

- All code review findings are now fixed (1–33), except that the CV branch's scope is the
  author's call.
- Full suite: **164 passed, 3 xfailed**. The three expected failures are the known-answer
  tests that miss their pre-registered thresholds, each with its numbers in the reason.
- ruff and black are clean, and the dashboard builds.

**The model's outputs are not an assessment of any real structure.**

---

## Review 3 (2026-10-24): what changed

| finding | problem | fix | test |
|---|---|---|---|
| 24 | No validation split; "best" checkpoint chosen on training loss | Split by connected structure group (no structure and no graph edge in two splits), fixed seed, 70/15/15; train on train, select on validation, report test once; normalisation fitted on train only | `test_splits.py` |
| 21 | ±10 cm bound on *cumulative* displacement discarded long subsiding records | Bound the change since the last accepted acquisition; flag `STEP_OUT_OF_BOUNDS`, keep the value | `test_displacement_bounds.py` |
| 22 | First acquisition's Δt invented; all pre-series rain counted as inflow | Δt₀ = 0; "Δt > 0" means an interval exists; inflow only from monitored intervals | `test_time_intervals.py` |
| 23 | `latent_drift_score` measured spread, not drift | RMS z-score against the training-split embedding reference; `null` without a reference | `test_latent_drift.py` |
| 26 | Rainfall > 1000 mm clamped silently | Clamped **and** flagged | `TestRainfallClampIsFlagged` |
| 27 | WebSocket untested; predictions sent as binary frames the dashboard could not parse | Text frames; tests | `TestWebSocketTelemetry` |
| 28 | CV branch untested; MC dropout overwrote BatchNorm statistics | Tests; `mc_dropout_mode` keeps BatchNorm frozen | `test_cv_branch.py` |
| 33 (new) | API normalised design features on the request batch | Uses the training statistics saved in the checkpoint | `test_inference_normalisation.py` |

Three existing tests asserted the old cumulative clamp (the bug of finding 21). They were
rewritten to the per-interval rule with the same guarantee, as listed in commit `c07b1bd`.

---

## Component status

| Component | State | Tested | Open |
|---|---|---|---|
| Data contracts | Incidence angle required; one track; per-interval LOS bound; flagged rainfall clamp; optional wells and recharge area | yes | LOS sign and bound value provisional (OQ2, OQ17) |
| Dataset | Causal rain binning; Δt₀ = 0; real-step mask; wells matched ±3 days | yes | — |
| Splits | By connected group, fixed seed, stratifiable | yes | Fractions provisional |
| Graph | Fixed; physics a_ij = c/d² [1/m²] separate from the GAT prior | yes | Connectivity provisional (OQ11) |
| Network | Self-loop GAT; per-step conditioned decoders; physical output scales; global S, S_k, L; training reference for drift | yes | **Does not generalise to unseen structures (run 4)** |
| Physics and loss | Leaky form-(b) flow, elastic LOS coupling, dimensionless total, seasonal capacity, mass balance, optional wells | yes | Pumping not modelled (OQ14); scales provisional (OQ18) |
| Training | Train/val/test split, validation selection, test once, saved split/stats/reference | yes | — |
| API | Ranking / percentage by well availability; training normalisation; drift vs training; thread-safe; text WebSocket | yes | Percentage depends on fitted S |
| Dashboard | Banner; ranking view; drift "n/a" | `npm run build` passes | 11 npm audit warnings (not changed) |
| CV branch | Tested; BatchNorm frozen in MC dropout | yes | In scope? (author) |

---

## Known-answer experiment (`experiments/synthetic_recovery/`)

| run | protocol | scored on | A: InSAR only ρ / AUROC | B: S, S_k known, rel. err | D: wells ρ / AUROC | outcome |
|---|---|---|---|---|---|---|
| 1a | pre-registered, before decoder fix | training structures | 0.22 / 0.59 | 0.27 (vacuous) | 0.97 / 1.00 | fail |
| **1b** | **pre-registered (OFFICIAL)** | training structures | **0.26 / 0.57** | 0.26 | 0.95 / 1.00 | **FAIL** |
| 2 | diagnostic, lr for global scalars | training structures | 0.95 / 1.00 | 0.44 | 0.93 / 1.00 | A met, B failed |
| 3 | current code after decisions 1, 2, 4 | training structures | 0.95 / 1.00 | 0.42 | 0.96 / 1.00 | A met, B failed |
| **4** | **current code, held-out (review 3)** | **10 unseen test structures** | **−0.03 / 0.54** | **0.49** | −0.32 / 0.42 (wells at train only) | **FAIL** |
| review 4 | A / B / C comparison, 5 seeds | 30–31 unseen per seed | A −0.05 / 0.50; B 0.94 / 1.00; C 0.97 / 1.00 | — | — | A fails; B, C pass; B not better than C |

Thresholds (pre-registered, never moved): A needs ρ ≥ 0.70 and AUROC ≥ 0.85; B needs median
relative error ≤ max(0.30, 2 × oracle floor). Runs 2–4 used protocols revised after run 1.

## Demo — SYNTHETIC DATA (`scripts/demo_synthetic.py`, unseen structures)

24 structures split 17 / 4 / 3 (seed 0); trained on 17; 300 epochs, 67 s CPU; two runs
byte-identical.
- On the 3 test structures: InSAR RMSE 0.085 cm (noise 0.05 cm) and PDE residual 1.1e-10.
- Recharge: 2.8–2.9e-4 m/day for all three, against planted 7.4e-4 to 2.1e-3 m/day. The
  structures are not distinguished, and the values are 3–7× low.
- The validation loss tracks the training loss, so a low loss does **not** mean correct
  recharge.

---

## What is still NOT proven

1. **Anything about a structure the model was not trained on.** Run 4 and the demo show
   chance-level recharge on unseen structures. This is now the main open problem.
2. **Anything on real data.**
3. **Absolute recharge**, from InSAR alone (impossible: only R·S_k/S) or with wells
   (only R/S).
4. **Hard rock**: out of scope.

## Next step: review 3 proposal, tested in review 4 (it works, but a 1-parameter least-squares fit works as well or better)

The physics and data terms of the loss need **no ground truth**: they use only a
structure's own InSAR, rainfall and design data. So instead of applying one trained
network to a new structure, the model could be **fitted to each new structure's own data**
at assessment time (per-structure physics-informed inversion). The global constants could
be fixed from training structures or wells. Run 4 suggests that a single amortised network
trained on a few dozen structures does not carry over; per-structure fitting avoids needing
it to.
