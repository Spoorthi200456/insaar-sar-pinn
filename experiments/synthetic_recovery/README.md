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

### Official pre-registered outcome: **FAILED**

Under the pre-registered protocol and thresholds, the model **fails** scenario A. It
cannot recover the ranking of recharge, or separate working from failing structures,
from InSAR alone.

- InSAR only: ρ = 0.26, AUROC = 0.57.
- With well heads: ρ = 0.95, AUROC = 1.00.

Run 1b below is the official result (the author's decision). Run 1a is the very first run.
Both used the same pre-registered protocol and thresholds. They differ only by the
finding-30 decoder bug fix, which was itself made *after* seeing run 1a. It has its own
failing-then-passing unit test, but it is still a change made after a failure, and it is
reported as such.

### Run 1a (pre-registered protocol; code at `9742130`, before the finding-30 fix): FAILED

Files: `results_run1a.csv` (per structure), `results_run1a.json`. The run is
deterministic: a second run gave identical numbers.

| scenario | Spearman ρ | AUROC | median rel. err. (working) | fitted S / S_k / L | verdict |
|---|---|---|---|---|---|
| A, InSAR only | 0.22 | 0.59 | 0.82 | 0.021 / 2.6e-4 / 9.4e-4 | **fails** (needs ρ ≥ 0.70 and AUROC ≥ 0.85) |
| B, S and S_k known | 0.02 | 0.50 | 0.27 | 0.050 / 5.0e-3 / 8.8e-4 | meets ≤ 0.30, **but meaningless**: AUROC 0.50 |
| C, hard rock | −0.14 | 0.37 | 0.94 | — | (report only) no information, as expected |
| D, with wells | **0.97** | **1.00** | 0.26 | 0.018 / 2.1e-4 / 6.5e-4 | (report only) recovers ranking and the working/failing split |

Oracle noise floor (perfect constants): median relative error **0.061**.
B threshold = max(0.30, 2 × 0.061) = 0.30.

**Why it failed (diagnostics in `results_run1.json`).** The recovered mean recharge was
**the same for every structure**: its spread across structures (coefficient of variation)
was 7.8e-7 in A and 6.4e-6 in B, against 0.75 in the truth. B's relative error of 0.27
is just what a constant guess scores when the true η is spread over 0.3–0.6. That is why
it "passes" the number while separating nothing.

The cause is in the architecture. `RechargeDecoder` (and the head-rate decoder built from
it) gives an LSTM the structure's embedding only as its *initial state*. Its per-step
input is rainfall, which is the same for every structure in the region. An LSTM forgets
its initial state roughly geometrically, so after ~40 steps the structure's identity is
gone, and R(t) becomes a function of rainfall alone. In A the head fit was also poor
(data RMSE 0.55 cm against 0.3 cm noise).

With wells (D), the direct head constraint was strong enough to force some memory
through, and recovery worked.

This is recorded as review finding 30 and fixed in the next commit, with its own failing
test. The pre-registered thresholds are **not changed**.

### Run 1b — OFFICIAL (pre-registered protocol; after the finding-30 decoder fix, `a249eae`): **FAILED**

Files: `results_run1b.csv`, `results_run1b.json`.

| scenario | Spearman ρ | AUROC | median rel. err. (working) | R spread across structures (CV; truth 0.75) | verdict |
|---|---|---|---|---|---|
| A, InSAR only | 0.26 | 0.57 | 0.96 | 0.002 | **fails** |
| B, S and S_k known | 0.70 | 0.73 | 0.26 | 0.010 | ρ just reaches 0.70; R almost constant |
| C, hard rock | 0.03 | 0.47 | 0.98 | 0.00002 | (report only) no information, as expected |
| D, with wells | **0.95** | **1.00** | **0.19** | 0.69 | (report only) recovers |

**Why A fails.**

1. **The fitted S_k is 2.3e-4 against a true 5e-3**, and the fitted head is 2 % of the
   true head. The model never fits the displacement signal: RMSE 0.55 cm against
   0.30 cm noise, and 0.46 cm against the noise-free signal, which is about the signal
   itself.
2. **S_k is limited by the optimiser, not the data.** S_k = 10^(−6 + 4·sigmoid(raw))
   starts at raw = 0 (1e-4). Adam with lr 1e-3 and a cosine schedule can move a parameter
   by about Σ lr_t ≈ 0.3 in 600 epochs. raw moved **0.36**, right at that limit, still
   heading up. The true value needs raw ≈ 2.5. **With this protocol the model cannot
   reach the true S_k**, so it cannot reproduce the amplitude of the displacement.
3. Separately, **R is not identifiable in absolute terms from InSAR alone**: only
   R·S_k/S is (finding-2 plan §7). S stayed near its initial value (0.021 vs 0.05) because
   nothing in the data pins it. This does not affect ranking, but it does mean scenario A
   could never test absolute R.

**What it shows.** When the head is observed directly (D: piezometers with 5 cm noise),
the same model, loss and protocol rank the structures almost perfectly (ρ 0.95) and
separate every failing structure from every working one (AUROC 1.00). The physics and the
architecture can do the job. What fails is getting the head *from InSAR alone*, through a
skeletal-storage coefficient that starts 50× too small and is optimised too slowly.

**Decision needed (asked of the author, not taken here).** Fixing point 2, e.g. by
initialising S_k from the data's amplitude, giving the global scalars their own learning
rate, or training longer, is a change to the method chosen *after* seeing a failed test.
That is the garden-of-forking-paths risk the pre-registration exists to prevent, so it
needs the author's decision.

### Run 2 protocol — DIAGNOSTIC, declared before running (committed before run 2 was made)

Author's decision after run 1b: one stated protocol change, one run, the same thresholds.
Whatever happens, the official pre-registered outcome stays **run 1b (failed)**. If run 2
meets the thresholds, it will be reported as "passes under a protocol revised after seeing
run 1", not as "passes".

**The single change:** the three global physical scalars (`storativity_logit`,
`skeletal_storage_raw`, `leakance_raw`) get their own Adam learning rate,
**0.02** (weight decay 0 on them). The network weights keep lr 1e-3.

**Why this value (a rule, not a fit to the result):** each scalar is a bounded
sigmoid of a raw parameter. raw ∈ [−4, 4] covers 98 % of its range: S_k from about 1.1e-6
to 0.9e-2, starting at raw = 0. With a cosine schedule, the furthest Adam can move a
parameter is about Σ lr_t ≈ lr × epochs / 2. Reaching either end of the range from 0 needs
Σ ≥ 4, so lr ≥ 4 / 300 ≈ 0.013. 0.02 is the next round value above that. Run 1b's lr of
1e-3 allowed Σ ≈ 0.3.

**Not changed:**
- S_k still starts at **1e-4**, the geometric middle of the elastic S_k ≈ 1e-5–1e-3
  reported from InSAR + well studies (Hoffmann et al. 2001, *WRR* 37). Starting it at the
  simulator's true 5e-3 would put the answer in.
- Network learning rate, epochs, data, seeds, loss weights and scales, thresholds.

Run with `python experiments/synthetic_recovery/run_recovery.py --protocol revised-run2`
(writes `results_run2.*`).

### Run 2 — DIAGNOSTIC (revised protocol above, commit `0d4a6a2`): still FAILS overall

Files: `results_run2.csv`, `results_run2.json`. One run, same thresholds.

| scenario | Spearman ρ | AUROC | median rel. err. (working) | fitted S / S_k / L (truth 0.05 / 5e-3 / 8.3e-4) | verdict |
|---|---|---|---|---|---|
| A, InSAR only | **0.95** | **1.00** | 0.95 | 0.0033 / 8.6e-3 / 7.1e-5 | **meets A's thresholds under a protocol revised after seeing run 1** |
| B, S and S_k known | 0.85 | 0.92 | **0.44** | (fixed) / (fixed) / 5.3e-4 | **fails** B (needs ≤ 0.30) |
| C, hard rock | −0.13 | 0.43 | 0.98 | — | (report only) no information, as expected |
| D, with wells | 0.93 | 1.00 | 0.90 | 0.0050 / 5.2e-3 / 8.0e-5 | (report only) ranks perfectly; absolute R 10× off because S is 10× off |

So **even under the revised protocol the pre-registered test as a whole fails**, because
of B. The official outcome remains run 1b (failed).

In A the model now fits the InSAR data to the noise level: RMSE 0.293 cm against
0.30 cm noise, and 0.099 cm against the noise-free signal. Its ranking of the structures is
almost perfect, and it separates every failing structure from every working one. Its
*absolute* recharge is about 26× too small. That is exactly what the identifiability
argument below predicts: R·S_k/S is right, but S came out 15× too small and S_k 1.7× too
large.

**Why B fails (review finding 31).** The simulator sized each structure's design
capacity on its *mean* recharge. So during monsoon peaks the planted recharge exceeds
the design capacity: at **21 %** of working-structure steps, by up to **7.2×** (checked on
the simulator alone). The model's loss penalises *every step* above capacity. In A that
penalty can be dodged by moving the free global scale. In B, S and S_k are fixed to the
truth, so the scale is pinned and the penalty pushes the peaks down. Recovered recharge is
compressed (spread CV 0.33 against 0.75) and biased low. Two things disagree: is the design
capacity a per-step ceiling (the loss) or a seasonal average (the simulator)? That question
is for the author (open question OQ19). It is not fixed here and there is no further run.

## Identifiability: can InSAR alone tell S_k apart from the head change?

**No, and this is exact.** InSAR measures d = 100 · cos θ · S_k · Δh. Multiply S_k by any
factor c and divide the head change by c, and every InSAR number is unchanged. In the flow
equation the head change is proportional to recharge, so dividing recharge by c does it.
The model cannot tell a structure recharging 10 mm/day in a stiff aquifer from one
recharging 1 mm/day in an aquifer ten times more compressible. A second, independent
ambiguity: multiplying S, L, T and R all by c leaves the head unchanged, so even perfect
wells do not give absolute recharge without knowing S.

Both are verified on the simulator: the differences are 7e-16 cm and 3e-15 m
(`tests/test_synthetic_recovery.py::TestIdentifiability`). What InSAR alone *can* give is
R·S_k/S.

**But S_k and S are one number for the whole region, so the unknown factor is the same for
every structure.** Ranking the structures, and separating working from failing ones, does
*not* depend on that factor. With S_k deliberately set 10× wrong, the best possible
(oracle) estimate is 10× off in every structure but still ranks them at ρ = 0.97 (same
test file).

**So the poor ranking in run 1 was an optimiser problem, and the wrong absolute values
are an identifiability problem:**

| symptom | cause | can a better optimiser fix it? | what fixes it |
|---|---|---|---|
| Run 1b: ranking ρ 0.26, AUROC 0.57 (InSAR only) | **Optimiser:** S_k started 50× too small and could not move far enough, so the model never fitted the InSAR signal (finding 32) | Yes. Run 2 (faster learning rate for S_k, S, L only): ρ 0.95, AUROC 1.00 | Done in run 2 (diagnostic); `src/` unchanged |
| Run 2: absolute recharge 26× off (InSAR only), 10× off (with wells) | **Identifiability:** only R·S_k/S (InSAR) or R/S (wells) is determined | **No.** Many different (S_k, S, R) fit the data equally well | Independent knowledge of S_k and S: pumping tests, CGWB aquifer parameters, InSAR + well co-location (Hoffmann et al. 2001) |
| Run 2 B: relative error 0.44 with S, S_k known | **Model–experiment mismatch:** per-step capacity ceiling vs peak recharge above capacity (finding 31) | No | Decide what Q_design means (OQ19) |

**What this means for the project goal.** From InSAR alone, and under the conditions
simulated here (compliant aquifer, S_k ≈ 5e-3, signal well above noise), the model can tell
*which structures recharge more* and *which have failed*, provided it is trained so that
S_k can reach a realistic value. It cannot say *how much* any structure recharges, in
mm/day or m³/day, without independent S_k and S. So the API's `recharge_efficiency`
(R·A/Q_design) is not meaningful in absolute terms from InSAR alone. In hard rock (C) it
cannot do either.

### Run 3 protocol — current training code after the author's decisions (declared before running)

The author's decisions of 2026-10-24, all made **after** seeing run 1. Run 1b stays the
official pre-registered result.

1. `train.build_optimizer`: global scalars at lr 0.02, as in run 2, now in `src/`.
2. Design capacity is a soft limit on the 122-day seasonal-average recharge, not a
   per-pass ceiling (finding 31).
4. Wells go through the new optional well-head input. In scenario D they are given as
   **absolute** heads with a +87.3 m datum offset, which the model must remove.

Same simulator, seed, epochs (600), network lr (1e-3) and thresholds. One run:
`python experiments/synthetic_recovery/run_recovery.py --protocol current-code-run3`
(writes `results_run3.*`). Whatever it shows is reported, labelled "under a protocol
revised after seeing run 1".

### Run 3 — current code after the author's decisions (protocol `50bdf20`, declared before running)

Files: `results_run3.csv`, `results_run3.json`. One run, same thresholds. **This is a
protocol revised after seeing run 1. The official pre-registered result is still run 1b
(failed).**

| scenario | Spearman ρ | AUROC | median rel. err. (working) | fitted S / S_k / L (truth 0.05 / 5e-3 / 8.3e-4) | verdict |
|---|---|---|---|---|---|
| A, InSAR only | **0.95** | **1.00** | 0.95 | 0.0032 / 8.6e-3 / 7.2e-5 | **meets A** (ρ ≥ 0.70, AUROC ≥ 0.85) |
| B, S and S_k known | 0.81 | 0.90 | **0.42** | (fixed) / (fixed) / 5.2e-4 | **fails B** (needs ≤ 0.30) |
| C, hard rock | −0.10 | 0.45 | 0.98 | — | (report only) no information |
| D, wells (absolute heads, +87.3 m datum) | 0.96 | 1.00 | 0.89 | 0.0052 / 5.1e-3 / 9.3e-5 | (report only) ranks perfectly; datum offset removed (head fit 0.996× true); absolute R 10× off because S is 10× off |

**Correction to the run-2 explanation of B.** Run 2 attributed B's failure to the
per-pass design-capacity ceiling (finding 31). Decision 2 replaced that with a seasonal
soft limit, and B still fails (0.42 against 0.44). Looking back, the capacity penalty was
already small at the end of run 2 (0.008; 0.004 now). So **that explanation was at most a
minor part, and is withdrawn as the cause.**

What the diagnostics do show for B:
- The model **underfits** the InSAR data: RMSE 0.375 cm against 0.30 cm noise.
- Its head reaches only **69 %** of the true amplitude.
- The fitted leakance is 5.2e-4 against 8.3e-4.
- Recovered recharge is compressed: spread CV 0.30 against 0.75.

**The cause is not established.** Candidates, not tested:
- B fixes S and S_k, so the optimiser cannot use the global scale that A uses to fit the
  data.
- The softplus recharge output cannot reach exactly zero in dry intervals.
- 600 epochs may not be enough.

Finding out needs further runs, which the author has not asked for.

**What the current code has shown (synthetic, compliant aquifer):**
- From InSAR alone it **ranks structures and separates working from failing** (scenario
  A, meets the pre-registered A thresholds under the revised protocol).
- It **does not recover absolute recharge**, even with S and S_k given (B fails).

The test suite reflects exactly this:
- `test_current_code_ranks_structures_from_insar_alone` asserts the A thresholds and
  passes.
- `test_current_code_recovers_absolute_recharge_with_known_constants` asserts the B
  threshold and is `xfail(strict=True)`.
