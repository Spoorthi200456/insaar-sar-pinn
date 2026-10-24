# Open questions — decisions taken without the information they need

Each entry: the question, why it matters, the **PROVISIONAL** choice used in
the code right now, and what would settle it. Nothing here is a confirmed
fact. Search the code for `PROVISIONAL` to find where each choice lives.

| # | Question | Why it matters | PROVISIONAL choice in code | What settles it |
|---|---|---|---|---|
| OQ1 | Study region (which state / basin / aquifer type)? | Aquifer type decides whether InSAR can see recharge at all (see OQ8). | **Aquifer type decided (author decision 3): alluvial only.** The specific basin is still open; synthetic data uses Bengaluru-area coordinates (12.9° N, 77.5° E) only as placeholders, and Bengaluru is itself hard rock. | Author + supervisor choose an alluvial basin, e.g. in the Indo-Gangetic plain or a coastal delta. |
| OQ2 | Which InSAR product, and its LOS sign convention? | A wrong sign makes every working structure look like a failing one. | `los_displacement_cm` positive = motion **toward the satellite** (uplift gives positive values). Any flip happens once, at ingestion, never in the model. | Read the processor's documentation (e.g. MintPy documents positive = toward satellite; LiCSBAS and commercial products differ). |
| OQ3 | Source of the incidence angle θ? | d_LOS = u_z·cos θ; cos 29° / cos 46° = 1.26, so a constant θ gives up to ±13 % error. | Required per-observation field `incidence_angle_deg` ∈ [20°, 50°]; per structure, the mean over its acquisitions is used. | Take θ from the Sentinel-1 product metadata (or the processor's geometry file) at the structure's pixel. |
| OQ4 | Elastic deformation only (finding-2 plan A1)? | Heads below the historic minimum cause inelastic compaction that does not rebound; uplift would then be over-predicted. | Elastic only. | Well records showing whether heads are above the pre-consolidation head. |
| OQ5 | Horizontal motion neglected (A3)? | Near a recharge mound horizontal motion can be a noticeable fraction of vertical. | Neglected (d_LOS = u_z cos θ). | Ascending + descending decomposition. |
| OQ6 | Is the InSAR series relative to its first acquisition? | The model compares head *increments* h(t) − h(t₀); an unknown per-structure offset in d would bias everything. | Yes: observed series used as-is; predicted head referenced to the first real acquisition. | Processor documentation (reference date and reference point). |
| OQ7 | Range of the skeletal storage coefficient S_k? | Sets how much displacement one metre of head produces. | One global S_k, `log10 S_k ∈ (−6, −2)`, init 1e-4. | Field estimates for the chosen region (InSAR + wells, e.g. Hoffmann et al. 2001 method). |
| OQ8 | Should the hard-rock feasibility risk (finding-2 plan §5) be raised with the supervisor? | It decides which structures stay in scope. | **Decided (author decision 3): hard rock is out of scope for satellite-only monitoring; alluvial aquifers only.** Reasoning for the supervisor below the table ("Why hard rock is out of scope"). Code unaffected. | Supervisor confirmation. |
| OQ9 | Recharge area A of each structure? | Converts R [m/day] to a volume rate R·A [m³/day] for the design-capacity and mass-balance checks (finding 6). | `recharge_area_m2` if supplied, else storage_volume_m3 / height_m (prismatic basin; real sloping basins are up to ~3× larger). | Water-spread area from India-WRIS design records, or a Sentinel-2 water mask at full storage. |
| OQ10 | Runoff coefficient c_r of the catchment? | Sets the most water that can reach the structure (mass-balance upper bound). | 0.3 (`losses.RUNOFF_COEFFICIENT`), within the 0.1–0.4 typical of semi-arid Indian catchments. | Catchment land use / soil data, or gauged inflows. |
| OQ11 | Which structures are hydraulically connected, and how strongly? | Sets the lateral-flow stencil a_ij = c_ij / d_ij² (finding 4). | Surveyed `GraphEdge`s if given (c = weight); else every pair within 5 km, c = 1. | Hydrogeological maps (aquifer units, faults, lineaments) from CGWB / state groundwater departments. |
| OQ12 | Characteristic head rate for scaling the head-rate output? | Only sets the size of v at initialisation (v = 0.01 m/day × network output); a poor choice slows training, it cannot change units. | 1 cm/day (`HEAD_RATE_SCALE_M_PER_DAY`). | Observed seasonal head rates from CGWB wells in the study region. |
| OQ13 | Reference level h_ref of the leakage sink, and is one global leakance L enough? | The sink −L(h − h_ref) drains recharge mounds (finding 11). | h_ref = 0 (series starts at the regional level); one global L ∈ (1e-6, 1e-1) /day, init 1e-3. | Well hydrographs (recession rates) in the study region. |
| OQ14 | Pumping / abstraction? | Unmodelled pumping looks like a large L or a small R, so a working structure in a heavily pumped area can look as if it is failing. | Not modelled. | CGWB well census, abstraction estimates, power-use proxies. |
| OQ15 | Aquifer thickness b at each structure? | Needed to turn the fitted transmissivity T [m²/day] into a conductivity K = T/b [m/day]; the API field `hydraulic_conductivity_m_per_day` actually holds T. | Not converted; T reported as `transmissivity_m2_per_day`, the old field kept and documented as T. | CGWB aquifer maps / borehole logs. |
| OQ16 | Characteristic recharge flux through a structure bed? | Sets the size of R at initialisation (R = 0.01 m/day × softplus(·)); it cannot change units, but a value 100× off slows training and biases early epochs. | 1 cm/day (`RECHARGE_SCALE_M_PER_DAY`). Percolation-tank bed infiltration rates are typically mm to a few cm per day. | Measured infiltration (pond drawdown) at the structures, or CGWB GEC-2015 norms for the region. |
| OQ17 | Should the ±10 cm bound apply to cumulative displacement or to per-interval change? | Applied to cumulative LOS (as now), long series in subsiding aquifers lose their later epochs (review finding 21). | Unchanged: cumulative, as in the original data contract. | Decide from the InSAR product's expected range; most likely bound the per-interval change instead. |
| OQ18 | Characteristic error scales of the loss terms (σ_d, σ_r, σ_h)? | They set how strongly physics, InSAR and wells pull against each other (finding 29). | σ_d = 0.3 cm, σ_r = 1e-4 m/day, σ_h = 0.1 m; fixed before the known-answer experiment, not tuned to it. | σ_d from the InSAR product's per-epoch uncertainty; σ_h from the well network's metadata; σ_r is a modelling choice to revisit with real data. |
| OQ19 | Is the design recharge capacity a per-step ceiling or a seasonal / annual average? | A per-pass ceiling penalises normal monsoon peaks and biases recovered recharge low (finding 31). | **Author decision 2 (2026-10-24): soft limit on the seasonal average.** Δt-weighted mean of R·A over the preceding 122 days (June–September), quadratic hinge. Window length PROVISIONAL. | Confirm the definition in India-WRIS / state design documents for each structure type, and whether the season should follow the local monsoon calendar (e.g. NE monsoon in Tamil Nadu). |
| OQ20 | How close in time must a well reading be to an InSAR acquisition to be used, and should wells be interpolated? | Sets which steps the well-head term constrains (author decision 4). | Nearest reading within ±3 days (`WELL_MATCH_TOLERANCE_DAYS`); no interpolation; absolute head compared after removing a per-structure constant offset. | Sampling frequency of the CGWB / state well network in the study basin (manual readings are often quarterly; loggers daily). |

## Why hard rock is out of scope (OQ1, OQ8): the 0.04 mm estimate

*For the supervisor. Order-of-magnitude reasoning, all values PROVISIONAL.*

1. **Uplift per metre of head.** Elastic aquifer deformation is Δb = S_k·Δh, with skeletal
   storage coefficient S_k = ρ_w·g·α_c·b (`docs/math/deformation_coupling.md`;
   Freeze & Cherry 1979 §2.10; Hoffmann et al. 2001).
2. **Hard rock.** Granite or basalt with a weathered zone has skeleton compressibility
   α_c ~ 1e-10 1/Pa (Freeze & Cherry Table 2.5, sound rock; check your edition). The
   deforming thickness is b ~ 20 m. So
   S_k ≈ 1000 kg/m³ × 9.81 m/s² × 1e-10 1/Pa × 20 m ≈ **2e-5**.
3. **A typical seasonal head rise under a check dam or percolation tank** is Δh ~ 2 m, so
   Δb ≈ 2e-5 × 2 m = 4e-5 m = **0.04 mm**. Projected on the line of sight
   (cos 29°–46° = 0.87–0.69), that is ≈ 0.03 mm.
4. **What Sentinel-1 can see.** Per-epoch noise of a time series is a few mm (the
   experiment uses 3 mm). Long-term velocity precision is of order 1 mm/yr. The signal is
   about **100× below** the per-epoch noise.
5. **Alluvium for comparison.** S_k ~ 1e-4 to 1e-3 gives **0.2–2 mm per metre of head**,
   i.e. 0.4–4 mm for 2 m. That is measurable over a season with many acquisitions.
6. **Evidence in this repo.** In the known-answer experiment's hard-rock scenario C
   (S_k = 2e-5), the model's ranking of structures was no better than chance in every run
   (ρ between −0.14 and +0.03). The best possible (oracle) estimate had a median relative
   error of about 15, i.e. 1500 %. The data carry no information about recharge there.
7. **What InSAR may still see in hard rock:** deformation *of the structure itself* (bund
   settlement, cracking, seepage-driven subsidence at the dam). That is a different signal,
   at the structure footprint, and not what this model measures. It would need its own
   method.
