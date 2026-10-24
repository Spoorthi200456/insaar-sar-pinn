# Head change → InSAR line-of-sight displacement (review finding 2)

**Result used in code** (`src/models/coupling.py::head_to_los_cm`):

> **d_LOS [cm] = 100 [cm/m] · cos θ · S_k · (h − h_ref) [m]**, positive = toward the satellite.

**In plain English:** when recharge raises the water level, the water pressure
inside the aquifer rises and takes some weight off the rock grains. The grain
skeleton springs back a little, so the ground rises. How much it rises per
metre of water level is a material property, the *skeletal storage coefficient*
S_k: about 1/10 000 for typical aquifers, so a 2 m rise gives about 0.2 mm.
The satellite looks at the ground at a slant (angle θ from vertical), so it
sees only the part of the vertical motion that points along its line of sight,
which is cos θ of it.

The old code used `d = −α·Δh`, with α the dimensionless Biot coefficient (≈ 0.6). It had
the **wrong sign** (a head rise gave subsidence), and its size was about **1000× too
large**: α·Δh is in metres, and the result was read as centimetres. It also had **no
cos θ**.

## Symbols

| symbol | meaning | units |
|---|---|---|
| h | hydraulic head | m |
| h_ref | head at the first real acquisition (InSAR is relative to it) | m |
| p | pore-water pressure | Pa = N/m² |
| σ | total vertical stress | Pa |
| σ′ | effective stress (carried by the grain skeleton) | Pa |
| ρ_w | density of water ≈ 1000 | kg/m³ |
| g | gravitational acceleration ≈ 9.81 | m/s² |
| α_c | vertical compressibility of the aquifer skeleton | 1/Pa |
| b | thickness of the deforming aquifer system | m |
| S_sk = ρ_w g α_c | skeletal specific storage | 1/m |
| S_k = S_sk · b | skeletal storage coefficient | dimensionless |
| Δb | change in aquifer-system thickness | m |
| u_z | vertical ground displacement, + up | m |
| θ | radar incidence angle (from vertical) | degrees |
| d_LOS | line-of-sight displacement, + toward satellite (PROVISIONAL, OQ2) | cm |

## Derivation

**Step 1. Effective stress (Terzaghi).** σ = σ′ + p.
*Plain English:* the weight above is carried partly by the grains (σ′) and partly by the
water (p).
*Source:* Terzaghi (1925); Freeze & Cherry (1979) *Groundwater* §2.10.
For hard rock, Biot's form σ = σ′ + α p applies, and α is then absorbed into the measured
S_k (Wang 2000, *Theory of Linear Poroelasticity*, ch. 2).

**Step 2. Total stress does not change.** Δσ = 0, so Δσ′ = −Δp.
*Plain English:* the overlying rock does not get heavier when the water level changes, so
any extra water pressure is taken off the grains.
*Source:* Freeze & Cherry (1979) §2.10, the same assumption used to derive S_s.

**Step 3. Pressure from head.** h = z + p/(ρ_w g). At a fixed point Δz = 0, so
Δp = ρ_w g Δh, and therefore **Δσ′ = −ρ_w g Δh**.
Units: kg/m³ · m/s² · m = Pa ✓.
*Source:* Freeze & Cherry (1979) §2.2 (hydraulic head).

**Step 4. Elastic skeleton strain.** Δb/b = −α_c Δσ′. Units: 1/Pa · Pa = dimensionless ✓.
*Plain English:* more load on the grains squeezes the layer thinner; less load lets it
expand.
*Source:* Freeze & Cherry (1979) §2.10 (aquifer compressibility); Jacob (1940).

**Step 5. Combine.** Δb = −α_c b (−ρ_w g Δh) = ρ_w g α_c b Δh = **S_k Δh**.
Units: 1/m · m · m = m ✓.
Sign: Δh > 0 gives Δb > 0, i.e. uplift.
*Source:* Hoffmann, Zebker, Galloway & Amelung (2001), *Water Resources Research* 37,
which uses exactly Δb = S_k Δh to estimate S_k from InSAR; Galloway & Burbey (2011),
*Hydrogeology Journal* 19.

**Step 6. Ground motion.** If nothing below the aquifer deforms, u_z = Δb = S_k Δh.
*Source:* Galloway & Burbey (2011).

**Step 7. Line-of-sight projection.** d_LOS = **u**·ê, where ê is the unit vector from the
ground to the satellite. With horizontal motion neglected (OQ5), d_LOS = u_z cos θ.
Uplift moves the ground toward the satellite, so d_LOS > 0 ✓.
*Source:* Hanssen (2001) *Radar Interferometry* ch. 2.

**Step 8. Metres to centimetres.** d_LOS [cm] = 100 · cos θ · S_k · Δh.
The 100 is the named constant `CM_PER_M`, a unit conversion, not a fitted factor.

**Reference (Q6).** InSAR time series are relative to the first acquisition, so the model
uses Δh = h(t) − h(t_first real acquisition) (`network.py`, `first_real`).

## Worked numbers (tested in `tests/test_coupling.py`)

- S_k = 1e-4, Δh = 1 m, θ = 0°: d = 100 · 1 · 1e-4 · 1 = **0.01 cm**.
- At θ = 60°, cos 60° = 0.5, so d = **0.005 cm**.
- d(29°)/d(46°) = cos 29°/cos 46° = 0.8746/0.6947 = **1.259**.
- Old code, same Δh = 1 m with α = 0.6: d = **−0.6 cm**. That is the wrong sign and 60×
  the correct θ = 0° value (about 70× the value at θ = 35°).

## How S_k is represented

There is one global learnable S_k, with log10 S_k = −6 + 4·sigmoid(raw), so 1e-6 < S_k < 1e-2.
It is initialised at 1e-4. The range spans hard rock (~1e-5) to compliant alluvium
(~1e-3 and above) (Freeze & Cherry 1979, Table 2.5 compressibilities; check against your
edition). It is global because, from InSAR alone, only S_k·Δh is identifiable (finding-2
plan §7).

## Assumptions (all PROVISIONAL, see `docs/open_questions.md`)

- Elastic deformation only (OQ4).
- No time lag.
- Vertical motion only (OQ5).
- One track per series. Enforced by the validator in `schemas.py`.
- InSAR is relative to the first acquisition (OQ6).
- LOS sign: positive toward the satellite (OQ2).

## Consequence for the project goal

In hard rock (S_k ~ 1e-5), 2 m of head gives about 0.04 mm of uplift. That is far below
Sentinel-1 noise (a few mm), so aquifer recharge may be **invisible** there (OQ1, OQ8).
The model now reports this honestly: with a small S_k it predicts near-zero displacement.
