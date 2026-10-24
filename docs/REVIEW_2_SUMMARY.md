# Review 2 — summary for a non-expert reader

*2026-10-24. Branch `review-2` (not merged, not pushed). Everything here is about the code
and **synthetic** data. No real satellite, rainfall or well data has been used.*

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

## What was changed

- Every fix is a separate commit. Each says what was wrong, gives the derivation with
  textbook sources (Freeze & Cherry, Bear, Hantush & Jacob, Hoffmann et al., MODFLOW
  documentation), and shows the new test **failing before** the fix and **passing after**.
- Plain-English derivations are in `docs/math/` (7 notes).
- Every guess the code has to make without real information (sign of the satellite data,
  aquifer stiffness, area of each structure, runoff, pumping and more) is listed in
  `docs/open_questions.md` (OQ1–OQ19). Each is marked PROVISIONAL.
- Tests went from 60 to **118 passing**, plus 1 deliberately marked "expected to fail"
  (below). No existing test was deleted or weakened. A few were updated to supply the new
  required satellite-angle field, or extended to cover new parameters, with their checks
  unchanged.

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
  against a 30 % limit). The reason is that the simulated peak recharge breaks the
  model's rule that a structure never exceeds its design capacity at any moment. Is
  design capacity a hard ceiling or an average? That is a definition question for you
  (OQ19).
- **So:** the first failure was an optimisation problem, which can be fixed. Not being able
  to get absolute numbers is a physics problem, which cannot be fixed without extra data.

The test stays in the suite, marked "expected to fail (strict)". If it ever starts passing,
the suite will flag it so the write-up is revisited.

## What is still NOT proven

- That any of this works on **real** data.
- That InSAR-only ranking works under the *original* protocol (it did not), or on real
  aquifers. The faster-learning fix was not put into the training code (finding 32, your
  decision).
- Anything about **hard-rock** aquifers, where most Indian check dams are. There the ground
  movement from recharge is about 0.04 mm, far below what Sentinel-1 can see. In the
  simulation the model could say nothing. Please raise this with your supervisor (OQ8).
  InSAR might instead detect *structural* failure (dam settlement), which is a different
  signal this model does not yet look for.
- **Absolute** recharge or "efficiency" numbers from InSAR alone. They are not
  determinable. The website's efficiency number should not be read as a real percentage
  until S_k and S are measured.
- Ten review findings remain open (21–28, 31, 32), for example a ±10 cm limit that would
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
6. A decision on the **study region**, preferably alluvial for a first test. (OQ1)

## Where to look

- `docs/review.md`: every finding with file/line, failure scenario, fix, and
  before/after test results.
- `docs/math/`: derivations.
- `docs/open_questions.md`: every provisional choice.
- `experiments/synthetic_recovery/README.md`: the known-answer test in full.
- `docs/STATUS.md`: component-by-component status.
