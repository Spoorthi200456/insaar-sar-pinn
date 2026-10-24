# Open questions — decisions taken without the information they need

Each entry: the question, why it matters, the **PROVISIONAL** choice used in
the code right now, and what would settle it. Nothing here is a confirmed
fact. Search the code for `PROVISIONAL` to find where each choice lives.

| # | Question | Why it matters | PROVISIONAL choice in code | What settles it |
|---|---|---|---|---|
| OQ1 | Study region (which state / basin / aquifer type)? | Aquifer type decides whether InSAR can see recharge at all (hard rock: S_k ~1e-5, ≈ 0.04 mm uplift for 2 m of head; alluvium: S_k ~1e-4–1e-3). See `docs/plans/finding2.md` §5. | None in code; synthetic data uses Bengaluru-area coordinates (12.9° N, 77.5° E) only as placeholders. | Author + supervisor choose the region. |
| OQ2 | Which InSAR product, and its LOS sign convention? | A wrong sign makes every working structure look like a failing one. | `los_displacement_cm` positive = motion **toward the satellite** (uplift gives positive values). Any flip happens once, at ingestion, never in the model. | Read the processor's documentation (e.g. MintPy documents positive = toward satellite; LiCSBAS and commercial products differ). |
| OQ3 | Source of the incidence angle θ? | d_LOS = u_z·cos θ; cos 29° / cos 46° = 1.26, so a constant θ gives up to ±13 % error. | Required per-observation field `incidence_angle_deg` ∈ [20°, 50°]; per structure, the mean over its acquisitions is used. | Take θ from the Sentinel-1 product metadata (or the processor's geometry file) at the structure's pixel. |
| OQ4 | Elastic deformation only (finding-2 plan A1)? | Heads below the historic minimum cause inelastic compaction that does not rebound; uplift would then be over-predicted. | Elastic only. | Well records showing whether heads are above the pre-consolidation head. |
| OQ5 | Horizontal motion neglected (A3)? | Near a recharge mound horizontal motion can be a noticeable fraction of vertical. | Neglected (d_LOS = u_z cos θ). | Ascending + descending decomposition. |
| OQ6 | Is the InSAR series relative to its first acquisition? | The model compares head *increments* h(t) − h(t₀); an unknown per-structure offset in d would bias everything. | Yes: observed series used as-is; predicted head referenced to the first real acquisition. | Processor documentation (reference date and reference point). |
| OQ7 | Range of the skeletal storage coefficient S_k? | Sets how much displacement one metre of head produces. | One global S_k, `log10 S_k ∈ (−6, −2)`, init 1e-4. | Field estimates for the chosen region (InSAR + wells, e.g. Hoffmann et al. 2001 method). |
| OQ8 | Should the hard-rock feasibility risk (finding-2 plan §5) be raised with the supervisor? | It may decide which structures stay in scope. | Raised in `docs/REVIEW_2_SUMMARY.md`; code unaffected. | Supervisor discussion. |

More entries are appended below as later fixes need them.
