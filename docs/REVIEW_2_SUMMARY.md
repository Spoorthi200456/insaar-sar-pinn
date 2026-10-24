# Review 2 — summary for a non-expert reader

*2026-10-24. Branch `review-2`, merged into `main` locally after the author's decisions
(not pushed; the branch is kept). Everything here is about the code and **synthetic**
data. No real satellite, rainfall or well data has been used.*

## The question the project asks

Is a given recharge structure in India (check dam, percolation tank and so on) actually
putting water into the ground, or has it failed? The idea is to watch the ground surface
from space. When a structure recharges, the water table under it rises, the ground rises
very slightly, and the Sentinel-1 satellite can measure that movement (InSAR). A
physics-informed neural network combines this with rainfall, design data and the
equations of groundwater flow.

## What was wrong (in plain words)

Review 1 had found 10 problems and fixed 4. Review 2 fixed the other 6 and looked at the
whole project again. The most important problems were:

1. **The ground moved the wrong way.** In the model, recharge made the ground *sink*. It
   should rise. The size was also about 1000× too large, and the satellite's slanted
   viewing angle was ignored. A working structure could only be explained as a failing
   one. *(finding 2)*
2. **The water never drained away.** The groundwater equation had no way for water to
   leave a structure's location (most structures are far apart). So the water level could
   only ever rise, and normal dry-season falls were "impossible". *(finding 11)*
3. **Units were mixed throughout.**
   - Recharge (metres per day) was compared with design capacity (cubic metres per day)
     and with rainfall (millimetres). *(6)*
   - The working/failing verdict that the website shows divided a depth-rate by a
     volume-rate. It labelled almost every structure "failing" whatever the data said.
     *(12)*
   - The training objective added physics errors and satellite errors measured in
     different units, which made the physics about a million times too weak to matter.
     *(29)*
4. **The answer depended on how the data was shuffled.** Which structures counted as
   neighbours changed from one training batch to the next. *(4)*
5. **The model could not tell structures apart.**
   - A structure with no neighbours ignored its own design data. *(5)*
   - More seriously, the part of the network that produces recharge forgot which structure
     it was working on after a few dozen time steps. Every structure got exactly the same
     recharge. *(30, found by the new known-answer test)*
6. **Physics was switched off where it was needed most.**
   - The physics was skipped wherever the satellite data was poor. *(8)*
   - Fake padding steps counted as real recharge. *(7)*
   - The water level was computed without regard to the real number of days between
     satellite passes. *(review-1 note)*
7. **Software and process problems:**
   - Concurrent web requests could get random answers. *(14)*
   - The "explain" page used a different network of structures from the prediction.
     *(13)*
   - The README promised things the code does not do. *(15)*
   - The automatic checks on GitHub had never actually run the tests. *(16)*

## Your decisions (2026-10-24) and what they changed

All of these were made **after** the official pre-registered known-answer run failed.
That run stays the official result.

1. **The aquifer constants learn faster.** S, S_k and L now have their own learning rate
   in the training code. This is what made ranking from InSAR alone work (run 3 below).
2. **Design capacity is a soft limit on the season's average recharge,** not a ceiling at
   every satellite pass. The season is a 122-day monsoon window (provisional).
3. **Hard rock is out of scope.** The project now covers alluvial aquifers only. The
   0.04 mm reasoning for your supervisor is in `docs/open_questions.md`.
4. **Well water levels are an optional training input.** Wells give absolute water levels
   and the model works with changes, so only the *shape* of the well curve is compared.
   Training lists the structures without wells, and the API says `has_well_data: false`
   for them.
5. **Without wells, the website shows a ranking, not a percentage.** Example: "rank 3 of
   16, relative recharge", with a one-line explanation. The efficiency percentage appears
   only when well data is present, and is labelled provisional (see "What is still NOT
   proven").
6. **The dashboard compiles.** `npm install` and `npm run build` ran with no errors, so no
   fixes were needed. npm reports 11 known vulnerabilities in the pinned packages; these
   were not changed.
7. **Merged into `main` locally,** after the full suite passed: 135 passed, 1 expected
   failure. Nothing was pushed.

## What was changed

- Every fix is a separate commit. Each says what was wrong, gives the derivation with
  textbook sources (Freeze & Cherry, Bear, Hantush & Jacob, Hoffmann et al., MODFLOW
  documentation), and shows the new test **failing before** the fix and **passing after**.
- Plain-English derivations are in `docs/math/` (7 notes).
- Every guess the code has to make without real information (sign of the satellite data,
  aquifer stiffness, area of each structure, runoff, pumping and more) is listed in
  `docs/open_questions.md` (OQ1–OQ20). Each is marked PROVISIONAL.
- Tests went from 60 to **135 passing**, plus 1 deliberately marked "expected to fail"
  (below). No existing test was deleted or weakened. A few were updated with their checks
  unchanged:
  - to supply the new required satellite-angle field;
  - to cover new parameters;
  - to give well data to records whose percentage they check (decision 5).
  The single recovery test was split into two tests of the current code (below).

## What the tests now prove

Each item below is checked by an automatic test against a hand calculation or an exact
physical property:

- Recharge raises the ground, by exactly 100 · cos(angle) · S_k · (water-level change) cm.
- Water drains back with the right time scale, and a pure dry-season recession satisfies
  the physics exactly.
- Every term of the groundwater equation is in metres per day. Lateral flow between two
  structures matches a hand calculation, and water is conserved.
- The result for a structure is the same whatever the batch size or shuffle order.
- Recharge, design capacity and rainfall are compared as volumes. The website's
  efficiency is a true ratio: 25 % comes out as 0.25, not 0.0066.
- Padding never counts as data. Physics still applies when the satellite data is poor.
- **InSAR alone can never determine absolute recharge.** It can only determine recharge
  × S_k / S. This is exact: two different combinations give identical satellite data to
  15 decimal places. But because S_k and S are shared by the whole region, InSAR *can* in
  principle rank structures and separate working from failing ones.

## The known-answer test: the honest headline

We built a separate, simple simulator that knows the true answer: 16 structures, 6 of them
deliberately "clogged". We wrote down the pass marks **before** running the model:

- ranking ρ ≥ 0.70 and working/failing AUROC ≥ 0.85 from InSAR alone;
- absolute recharge within 30 % when the aquifer constants are known.

**The model failed** (official result): ranking ρ 0.26, AUROC 0.57. That is barely better
than a coin.

- **With well water levels added, it succeeded:** ρ 0.95, AUROC 1.00. Every clogged
  structure was ranked below every working one.
- **Why InSAR-only failed:** one constant, the aquifer's stiffness S_k, started 50× too
  small and was learned too slowly to ever reach a realistic value. So the model never
  matched the size of the ground movement.
- **A diagnostic second run, with that constant allowed to learn faster** (the change was
  written down before running, but decided after seeing the failure): InSAR-only ranking
  rose to ρ 0.95 and AUROC 1.00. The "known constants" check still failed (44 % error,
  against a 30 % limit).
- **Run 3, the current code after your decisions 1, 2 and 4:**
  - Ranking from InSAR alone: ρ 0.95, AUROC 1.00. This **meets the pre-registered
    ranking marks**, but under a protocol revised after seeing run 1.
  - "Known constants" check: 42 % error, so it **still fails**.
  - With wells given as absolute levels with an 87 m datum offset: ρ 0.96, AUROC 1.00. The
    model removed the offset correctly.
- **A correction.** After run 2 I said the "known constants" failure came from the
  per-pass capacity ceiling. Your decision 2 removed that ceiling, and the failure stayed.
  The capacity penalty had in fact been tiny. So that explanation is withdrawn. What is
  visible is that the model fits the satellite data less well in that scenario and
  reaches only 69 % of the true water-level rise. The cause is **not established**.
- **So:**
  - The ranking failure was an optimisation problem, and it is now fixed.
  - Absolute numbers from InSAR alone are impossible: that is physics.
  - Absolute numbers even with known constants are not yet achieved, for a reason not
    found.

In the test suite:
- "ranks structures from InSAR alone" passes.
- "recovers absolute recharge with known constants" is marked "expected to fail
  (strict)". If it ever starts passing, the suite flags it so this write-up is revisited.

## What is still NOT proven

- That any of this works on **real** data.
- That InSAR-only ranking works on real aquifers. It is shown only on synthetic alluvial
  data, and only with training settings changed after the official test failed.
- **Hard rock** is now out of scope (decision 3): the 0.04 mm signal is invisible to
  Sentinel-1. InSAR might still detect *structural* failure there (dam settlement), which
  is a different signal this model does not look for.
- **Absolute** recharge or efficiency percentages.
  - From InSAR alone they cannot be determined at all.
  - **Even with well data** the percentage the website shows depends on the aquifer's
    storativity S. Wells alone do not pin that down: in run 3 with wells, the ranking was
    perfect but the absolute recharge was 10× off. The percentage is therefore labelled
    "provisional", and should not be read as a real percentage until S (and S_k) are
    measured, e.g. by pumping tests.
- **Wells at prediction time.** The network learns from wells during training. When you
  submit new data, the API reports whether wells are present and how well the predicted
  water level matches them, but the wells do not change the prediction itself.
- Seven review findings remain open (21–24, 26–28), for example a ±10 cm limit that would
  throw away later data in fast-subsiding areas, and the lack of a validation split. See
  `docs/review.md`.

## What real data is needed next

1. **Sentinel-1 time series** for the study area, with the processor's documented sign
   convention and the incidence angle at each structure (e.g. MintPy or LiCSBAS products).
   (OQ2, OQ3)
2. **Well water levels** (CGWB / state groundwater department) near the structures. This is
   the single most useful addition: it turned failure into success in the simulation.
   (finding 25)
3. **Aquifer constants** S_k and S (pumping tests, CGWB aquifer maps, or InSAR + well
   co-location studies) to turn rankings into absolute recharge. (OQ7)
4. **IMD gridded rainfall.**
5. **India-WRIS design records**, including each structure's water-spread area and how its
   design capacity is defined. (OQ9, OQ19)
6. A choice of **alluvial study basin** (decision 3), e.g. in the Indo-Gangetic plain or a
   coastal delta. (OQ1)

## Where to look

- `docs/review.md`: every finding with file/line, failure scenario, fix, and
  before/after test results.
- `docs/math/`: derivations.
- `docs/open_questions.md`: every provisional choice.
- `experiments/synthetic_recovery/README.md`: the known-answer test in full.
- `docs/STATUS.md`: component-by-component status.
