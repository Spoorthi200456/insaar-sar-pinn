# Harder method comparison: a simulator that is NOT method C's model (review 5)

**SYNTHETIC DATA.** Nothing here says anything about a real structure.

## Protocol — written and committed BEFORE the simulator code exists and before any method runs

### Why this test

In review 4 a one-number least-squares fit (method C) beat the per-structure neural
fit (method B) and matched an oracle. But the simulator there **was** C's model:
- recharge proportional to rain;
- one regional rain series and one recession time;
- no pumping;
- no change over time.

That test could not show whether the network earns its cost when structures behave in
ways C cannot represent. This test is built to give B that chance.

### Methods (unchanged from review 4 unless stated)

- **A. Trained network**, applied unchanged to unseen structures:
  - trained on the training structures only, full batch;
  - `train.build_optimizer` (network lr 1e-3, global scalars 0.02), 600 epochs;
  - validation loss every 10 epochs, best epoch kept.
- **B. Per-structure test-time fitting**: a copy of A's network, global S, S_k, L frozen,
  Adam lr 1e-3, 300 steps, eval mode, each test structure alone, no truth
  (`experiments/method_comparison/methods.test_time_fit`).
- **C. Ordinary least squares, no network**: one amplitude ĉ per structure. One regional
  recession time τ is grid-searched (41 log values, 10–1000 days) on the **training**
  structures (`src/models/ols_baseline.py`).
  - **The only change from review 4:** each structure's unit response g is built from
    **its own** rain series, because rain now differs between structures. The formula is
    unchanged (`docs/math/ols_baseline.md`, step 7).

No method is tuned for this simulator. All settings above are review 4's.

### The new simulator (`experiments/harder_comparison/simulator.py`)

The simulator is pure NumPy. It imports **nothing** from
`experiments/synthetic_recovery/simulator.py`, `src/models/ols_baseline.py` or the
network.
- It runs on a **daily** time step, not the InSAR intervals C uses.
- It spins up for 365 days before the first acquisition, so the aquifer is not at rest
  at t₀.

**Geometry and acquisitions**
- N structures uniformly scattered over 12.4–13.6° N, 76.4–78.0° E.
- 60 Sentinel-1 acquisitions with 6/12/24-day gaps (probabilities 0.25/0.5/0.25), about
  2.2 years, starting on 1 January. That covers two monsoons (June–September).

**1. Rain differs per structure** (C assumed one regional series):
- **Regional wet day:** each day is wet with probability 0.6 (June–Sept) or 0.05
  (otherwise). The regional intensity is exponential with mean 15 mm (monsoon) or 5 mm.
- **Per structure on a regional wet day:**
  - dry with probability 0.3 (patchy convective rain);
  - otherwise intensity = regional intensity × m_i × a daily lognormal factor
    (σ_log = 0.5, mean 1).
- **m_i, a smooth spatial multiplier:** exp(0.35 z_i − 0.35²/2), with z a Gaussian random
  field with exponential correlation over 20 km.
- The record's rain gauge is that structure's own daily rain.

**2. Rain → recharge is a daily pond/soil bucket**, not "recharge ∝ rain". For each
structure i and day d:
- Runoff from the catchment exists only above a **structure-specific threshold**
  θ_i ~ U(2, 15) mm/day: inflow = c_i · max(P_d − θ_i, 0)/1000 · A_catch,i
  + P_d/1000 · A_i [m³]. Here c_i ~ U(0.1, 0.3) and A_catch,i ~ U(0.2, 3) km².
- Pond volume V is capped at V_max,i = A_i · H_i [m³], with A_i ~ U(2000, 8000) m² and
  H_i ~ U(2, 8) m. Overflow is lost.
- Evaporation is 5 mm/day × A_i whenever water is present.
- Infiltration (the **recharge volume**) is I_d = min(water available, k_i(d) · A_i) [m³],
  with infiltration capacity k_i(d) [m/day].
- **Recharge flux into the aquifer node:** R_i,d = I_d / A_cell, with A_cell = 2×10⁵ m²
  for every structure (the area its mound spreads over, about 250 m radius).

**3. Three kinds of structure, with gradual clogging:**
- **working:** k_i constant, k0_i ~ U(0.1, 0.3) m/day;
- **failed** (from the start): k_i = k0_i · U(0, 0.05);
- **progressively clogging:** k_i(t) = k0_i · exp(−t/t_c,i) over the acquisition period
  (constant during spin-up). t_c,i is chosen so that k at the last acquisition is
  f_i · k0_i, with f_i ~ U(0.03, 0.15).

Each seed has N = 160 structures: **70 working, 45 failed, 45 clogging**.

**4. Nearby pumping:**
- 30 % of the structures of each kind (21 / 14 / 14) have an irrigation well nearby.
- It pumps q_i ~ U(50, 300) m³/day during the rabi season (1 Nov – 31 Mar).
- This is a sink of −q_i/A_cell [m/day] at that node. It is **not** recharge: it is a
  confounder of the InSAR signal.

**5. Aquifer (form (b) on the structure graph, own implementation, daily backward Euler):**
S dh/dt = Σ_j a_ij (T_i+T_j)/2 (h_j − h_i) − L_i h + R_i − pumping_i
- a_ij = 1/d_ij² within 5 km;
- T ~ log-U(20, 200) m²/day;
- S = 0.05;
- **structure-specific recession time** τ_i = S/L_i ~ log-U(30, 150) days (C assumes one
  τ for the region).

**6. InSAR:**
- d_i,k = 100 cos θ_i S_k (h_i(t_k) − h_i(t₀)) + noise, with S_k = 5×10⁻³ and
  θ_i ~ U(29°, 46°).
- Noise is 0.3 cm, or 1.0 cm on 15 % low-coherence epochs (coherence 0.3–0.65; otherwise
  0.75–0.98). The first acquisition is always coherent.

**7. Design data:**
- area A_i;
- height H_i;
- storage volume A_i · H_i;
- catchment A_catch,i;
- design capacity Q_design,i = A_i · k0_i · U(0.7, 1.3) m³/day (what the structure was
  built for, before any failure).

**Expected signal size** (reasoning, not a tuned value), for a working structure that holds
water about 60 % of a monsoon:
- recharge flux 0.6 · 0.2 m/day · 5000 m² / 2×10⁵ m² ≈ 3×10⁻³ m/day;
- head rise over one monsoon ≈ R τ/S · (1 − e^(−122/τ)) ≈ 3 m;
- LOS ≈ 100 · 0.8 · 5×10⁻³ · 3 ≈ 1.2 cm, against 0.3 cm noise.

This is the same order as review 4 (about 1.7 cm), so a failure here is not simply a
weaker signal. Pumping drawdown is up to about −0.7 cm.

**Known mismatch (recorded, not fixed):** the network's regularisers convert R to a volume
with the structure area A_i (finding 6, OQ9). The simulator spreads recharge over A_cell.
This affects only the design-capacity and mass-balance penalties, and is a real open
question for field data too.

### Split and seeds

| item | value |
|---|---|
| seeds | **0, 1, 2, 3, 4**. Each sets the simulated data, the split and all training randomness |
| split | `src/data/splits.py`, fractions **0.6 / 0.2 / 0.2**, by connected structure group, stratified by the label below, split seed = run seed |
| held-out test structures | **at least 30 per seed**. If any seed gives fewer, N is raised in steps of 16 (keeping the 70:45:45 proportions, rounded) for all seeds before any method runs |

### What is scored (fixed now)

Steps k = 1…K−1 are the InSAR intervals. The **late window** is the intervals whose end
time is t_k ≥ t₀ + (2/3)(t_{K−1} − t₀), the last third of the record.

**Truth**
- **Amount:** the mean over k = 1…K−1 of the true interval-mean recharge flux R̄_i,k
  [m/day] (mean over the days in (t_{k−1}, t_k] of I/A_cell).
- **Label "failing now":** failed or clogging = failing; working = working. By the
  last acquisition every clogging structure has lost 85–97 % of its capacity.

**Scores** (the same definition for every method, "mean predicted recharge"; any common
unknown factor such as S/S_k cancels in ρ and AUROC)

| method | amount score | now score |
|---|---|---|
| A, B | mean predicted R over k = 1…K−1 | mean predicted R over the late window |
| C | ĉ · mean_k u_k over k = 1…K−1 (`mean_relative_recharge`) | ĉ · mean of u_k over the late window |

Here u_k = P_k/(1000 Δt_k) is the structure's own rain rate.

**Headline metrics per seed** (test structures only):
1. **ρ_amount** = Spearman(amount score, true amount).
2. **AUROC_now** = AUROC of the now score for working (positive) vs failing now.

**Secondary (reported, no pass mark):**
- AUROC_now on **clogging vs working** only (the subgroup C's constant amplitude
  cannot follow);
- AUROC_now on failed vs working;
- Spearman of the now score against the true late-window mean flux;
- ρ_amount within pumped and within unpumped structures;
- C's fitted τ;
- B's loss reduction;
- run times.

There is **no oracle** this time. Recharge is not linear in one parameter here, so the
review-4 oracle does not exist.

### Pass marks and comparison rule (fixed now; will not be changed after the run)

1. A method **passes** if, over the 5 seeds, mean ρ_amount ≥ **0.70** and mean
   AUROC_now ≥ **0.85** (the marks of runs 1–4 and review 4).
2. For a metric m, **X is clearly better than Y on m** if both:
   - mean m(X) − mean m(Y) ≥ **0.05**, and
   - m(X) > m(Y) in **at least 4 of the 5 seeds** (paired by seed).
3. **"B is clearly better than C"** if B is clearly better than C on **at least one**
   headline metric, **and** C is not clearly better than B on the other. Otherwise:
   **"B is not clearly better than C — the neural network is not needed for this part."**
   The same rule is reported for C vs B, A vs C and B vs A.
   - Note: "at least one of two metrics" is deliberately lenient to B, since this test
     exists to give B its best fair chance.

### What will be committed after the run

- `results.json` (per seed, per method, summary) and `results.csv` (per test structure);
- the results tables in this README, exactly as they come out;
- `tests/test_harder_comparison.py`:
  - simulator unit tests;
  - a slow test that re-runs C's full protocol and asserts its pre-registered outcome
    (a mark C misses becomes `xfail(strict=True)` with the numbers);
  - an opt-in (`-m comparison`) test re-running A and B on seed 0 and asserting their
    outcome the same way.

---

## Results (one run of the protocol; commits `4c0384e` protocol, `2403497` code)

Files:
- `results.json`: per seed, per method, summary;
- `results.csv`: per test structure.

Run details:
- N = 160 per seed gave 30–32 test structures in every seed, so no N raise was needed.
  Test structures per seed were 12–14 working, 6–11 failed, 9–12 clogging and 7–14
  pumped.
- The late window held 18–23 intervals.
- Runtime about 107 min on CPU:
  - A's training: 3–8 min per seed;
  - B: 9–33 min per seed;
  - C: 0.3–1.1 s per seed.

### Headline metrics (mean ± sd (min) over seeds 0–4)

| method | ρ_amount | AUROC_now | pass mark (ρ ≥ 0.70 and AUROC ≥ 0.85) |
|---|---|---|---|
| A trained network | 0.209 ± 0.175 (−0.076) | 0.567 ± 0.129 (0.397) | **FAIL** (both) |
| B per-structure fitting | **0.663 ± 0.057** (0.569) | **0.954 ± 0.032** (0.907) | **FAIL**: ρ below 0.70; AUROC passes |
| C least squares, no network | 0.528 ± 0.048 (0.446) | 0.903 ± 0.049 (0.866) | **FAIL**: ρ below 0.70; AUROC passes |

Per seed:

| seed | ρ_amount A / B / C | AUROC_now A / B / C |
|---|---|---|
| 0 | 0.20 / 0.71 / 0.57 | 0.47 / 0.91 / 0.87 |
| 1 | 0.40 / 0.65 / 0.53 | 0.66 / 0.96 / 0.88 |
| 2 | 0.26 / 0.70 / 0.56 | 0.71 / 0.995 / 0.986 |
| 3 | −0.08 / 0.57 / 0.45 | 0.40 / 0.96 / 0.91 |
| 4 | 0.26 / 0.68 / 0.54 | 0.60 / 0.95 / 0.88 |

### Comparison rule (as pre-registered)

| comparison | ρ_amount: mean diff, seeds won | AUROC_now: mean diff, seeds won | clearly better? |
|---|---|---|---|
| **B vs C** | **+0.135, 5 of 5** | **+0.051, 5 of 5** | **YES** (on both metrics) |
| C vs B | −0.135, 0 of 5 | −0.051, 0 of 5 | no |
| A vs C | −0.319, 0 of 5 | −0.335, 0 of 5 | no |
| B vs A | +0.454, 5 of 5 | +0.387, 5 of 5 | yes |

**Verdict: B is clearly better than C on this simulator.** B fits the physics-informed
network to each structure's own data. It beat C in every seed, on both headline
metrics. So where structures differ in the ways listed in the protocol, the neural network
**is** useful for this part of the problem. In review 4, where the world was C's own
model, it was not.

**But no method passes the pre-registered marks.**
- B detects failing structures well (AUROC 0.95).
- B misses the ranking mark: ρ 0.66 against 0.70.
- C misses the ranking mark by more: ρ 0.53.
- A, the trained network applied unchanged, fails both again. That is the third
  independent test in which it is close to chance on unseen structures.

### Secondary results (no pass mark; mean over seeds)

| | A | B | C |
|---|---|---|---|
| AUROC_now, clogging vs working | 0.61 | **0.95** | 0.88 |
| AUROC_now, failed vs working | 0.51 | 0.95 | 0.93 |
| ρ of now score vs true late-window recharge | 0.21 | 0.66 | 0.60 |
| ρ_amount within pumped structures (7–14 per seed) | 0.15 | 0.67 | 0.58 |
| ρ_amount within unpumped structures | 0.22 | 0.74 | 0.58 |

Other measurements:
- C's fitted regional τ: 45–63 days. The structures' true τ ranges over 30–150 days.
- B lowered each structure's own loss to 36–41 % of its starting value (median per
  seed).

What the secondary numbers say:
- C separates structures that failed **from the start** almost as well as B (0.93 vs
  0.95).
- B's detection advantage comes mainly from **gradually clogging** structures
  (0.95 vs 0.88). C's single constant amplitude averages the good early record with the
  clogged late record. B's time-varying recharge can follow the decline.
- On ranking, B is ahead of C in both pumped and unpumped subsets.

### Limitations: read before drawing conclusions

1. **I designed this simulator knowing what C cannot represent**: that was its stated
   purpose. It favours a flexible method by construction, just as review 4's simulator
   favoured C. The truth for real structures is somewhere in between and unknown until
   real data is used.
2. **B's network does not contain the simulator's processes either.** It has no pond
   bucket, no pumping term and no per-structure recession time. Its advantage comes from
   flexibility, not from matching physics. Its ranking is still short of the mark.
3. **Cost.** B took 9–33 min per seed, roughly 20–60 s per structure, against about 1 s
   for C per seed.
4. **The API's "neural_experimental" option is method A**, the trained network applied
   unchanged. It is **not** method B. A failed here too. Method B is not available in the
   API.
5. The model's area convention for its regularisers differs from the simulator's A_cell
   (protocol, "Known mismatch"; OQ9).
6. Synthetic data, an alluvial aquifer with signal above noise, and no absolute recharge
   from InSAR alone. All of this still holds for every method.

### Tests

`tests/test_harder_comparison.py`:
- simulator unit tests (12);
- `test_recorded_verdict_follows_from_the_recorded_per_seed_numbers`: re-applies the
  rules to the recorded per-seed numbers;
- slow `test_c_full_protocol_reproduces_the_recorded_numbers`: re-runs C on all 5 seeds
  in seconds; passes;
- slow `test_c_meets_pass_marks_on_the_harder_simulator`: **xfail(strict=True)** with
  C's numbers;
- opt-in `test_seed0_a_and_b_reproduce_the_recorded_outcome` (`pytest -m comparison`,
  about 18 min): re-runs A, B and C on seed 0. It checks the recorded values (± 0.02),
  that A is below both marks, and that B's ρ ≥ C's + 0.05.
