# Units of recharge vs design capacity vs rainfall (review finding 6)

**In plain English:** the model's recharge R says how many metres of water
enter the aquifer *per day, per square metre* of ground under the structure.
The design capacity says how many *cubic metres per day* the whole structure
was built to recharge. Rainfall is a *depth* (mm) that fell in one satellite
interval. You cannot compare these three numbers directly, just as you cannot
compare "5 km/h" with "5 km". The old code did compare them directly, and the
two safety checks built on them could never fire: in the demo, R was 29× the
rainfall and neither check noticed. The fix converts everything to volumes of
water (m³) or volume rates (m³/day) first.

## Symbols

| symbol | meaning | units |
|---|---|---|
| R_k | recharge flux at step k (model output `recharge_rate`) | m/day |
| A | recharge area of the structure (water-spread / bed area) | m² |
| A_catch | catchment area draining into the structure | m² (schema stores km²; ×1e6) |
| Q_design | design recharge capacity | m³/day |
| P_k | rain in InSAR interval k (t_{k−1}, t_k] | mm |
| Δt_k | length of interval k | day |
| c_r | runoff coefficient (fraction of catchment rain that reaches the structure) | dimensionless |

## Why R is in m/day

In the depth-integrated flow equation (form (b), CLAUDE.md), S ∂h/∂t = ∇·(T∇h) + R. The
storage term is S [–] × ∂h/∂t [m/day] = m/day. So R must also be in m/day, i.e. m³ of
water per m² of aquifer per day.
*Source:* Bear (1979) *Hydraulics of Groundwater* ch. 5 (the Dupuit–Forchheimer
depth-integrated equation with a distributed source); Freeze & Cherry (1979) §5.

## Step 1. Flux to volume rate

A flux [m³/(m²·day)] times the area it acts on [m²] is a volume rate [m³/day]:
**V̇_k = R_k · A**. Units: m/day · m² = m³/day ✓. Only now is it comparable with Q_design.

## Step 2. Design-capacity hinge (dimensionless)

penalty_k = max(0, V̇_k / Q_design − 1)². It is zero while the structure recharges at or
below its design rate. The ratio is dimensionless, so the penalty does not depend on the
unit of area or volume (tested: rescaling A, A_catch and Q_design together by 37 leaves it
unchanged).

## Step 3. Mass balance (dimensionless)

Water recharged over the series: V_rech = Σ_k R_k · A · Δt_k. Units: m/day · m² · day = m³ ✓.

The most water that can reach the structure is direct rain on its footprint plus runoff
from its catchment:
V_in = Σ_k (P_k / 1000) · (A + c_r · A_catch). Units: m · m² = m³ ✓.
The 1000 is mm → m (named constant `MM_PER_M`).
*Source:* the rational-method idea that runoff = c × rain × area (Chow, Maidment & Mays
1988, *Applied Hydrology* §15.1; check against your edition).

The hinge is max(0, (V_rech − V_in) / (V_in + 0.001 m · A))². The 0.001 m · A term (1 mm of
water over the footprint) keeps the ratio finite in a completely dry series.

This is an **upper bound** only. It ignores water stored in the structure before the
series starts, and evaporation, which would make the real bound tighter.

Padded steps have Δt = 0 and P = 0, so they add nothing to V_rech or V_in. The
design-capacity hinge is still averaged over padded steps; that is finding 7, fixed
separately.

## PROVISIONAL choices (docs/open_questions.md)

- **OQ9, area A.** If `recharge_area_m2` is not supplied, A ≈ storage_volume / height.
  A basin with vertical sides (a prism) of volume V and depth h has area V/h. Real basins
  have sloping sides, which makes A larger, up to about 3V/h for a cone. This estimate is
  rough. India-WRIS design records or a satellite water-spread map would replace it.
- **OQ10, runoff coefficient c_r = 0.3.** Typical semi-arid Indian catchments are about
  0.1–0.4.

## Worked numbers (tested in `tests/test_losses.py::TestRegularizerUnits`)

- **Capacity:** R = 0.17 m/day (the old demo value) over A = 1000 m² gives V̇ = 170 m³/day.
  With Q_design = 100 m³/day the penalty is (1.7 − 1)² = **0.49**. The old code compared
  0.17 with 100 directly: penalty **0**.
- **Mass balance:** 1 mm per 7-day step on 1000 + 0.3 · 500 000 = 151 000 m² gives
  151 m³ in per step. Out per step: 0.17 · 1000 · 7 = 1190 m³ (impossible).
  New penalty ((7140 − 906)/(906 + 1))² ≈ 47. Old code: Σ R = 1.02 vs Σ P = 6 "mm",
  penalty **0**.
