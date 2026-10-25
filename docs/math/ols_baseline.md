# Method C: ordinary least squares on a linear-reservoir model (review 4)

**In plain English:** instead of a neural network, use the simplest physical description
and fit one number per structure.
1. Assume each structure turns a fixed fraction of the rain into recharge.
2. Assume the raised water table drains back with a fixed time scale.
3. Assume the ground rises in proportion to the water level.

The size of the movement the satellite sees is then one number times a curve that depends
only on the rainfall. Fitting that one number is a straight-line fit through the origin:
ordinary least squares, solved with a single formula.

## Symbols

| symbol | meaning | units |
|---|---|---|
| P_k | rain in interval k | mm |
| Δt_k | interval length (0 for the first acquisition) | day |
| u_k = P_k/(1000 Δt_k) | rain rate | m/day |
| η | infiltrating fraction, R = η·u | dimensionless |
| S, S_k, L | storativity, skeletal storage, leakance | –, –, 1/day |
| τ = S/L | recession time scale | day |
| g_k(τ) | unit response: head for η/S = 1 | m |
| c = η·S_k/S | the one fitted amplitude | dimensionless |
| d_k | LOS displacement | cm |
| θ | incidence angle | deg |

## Derivation

**Step 1. Water balance of one isolated structure** (form (b) without lateral flow,
`docs/math/flow_equation_leakage.md`): S dh/dt = −L h + R, with R = η u.

**Step 2. Backward Euler with Δt_k** (as in the simulator and the model):
S (h_k − h_{k−1})/Δt_k = −L h_k + η u_k.
Divide by S and write τ = S/L:
h_k = (h_{k−1} + Δt_k (η/S) u_k) / (1 + Δt_k/τ).
The equation is linear in η, so h_k = (η/S) · g_k(τ), where g_k solves the same recursion
with η/S = 1:
**g_k = (g_{k−1} + Δt_k u_k)/(1 + Δt_k/τ)**, with g_0 = 0.
Units: g has units of u × time = m (per unit η/S) ✓.
*Source:* linear-reservoir / exponential recession model of groundwater storage
(e.g. Brutsaert 2005, *Hydrology: An Introduction*, ch. 10 on baseflow recession; check
your edition). Backward Euler, as in `docs/math/head_integrator.md`.

**Step 3. InSAR coupling** (`docs/math/deformation_coupling.md`):
d_k = 100 · cos θ · S_k · h_k = 100 · cos θ · (η S_k/S) · g_k = **100 cos θ · c · g_k**.

**Step 4. Ordinary least squares for c.** With w_k = 100 cos θ g_k, the model is
d_k = c·w_k: a line through the origin. Minimising Σ_k (d_k − c w_k)² over the reliable
epochs gives d/dc = −2 Σ w_k (d_k − c w_k) = 0, so
**ĉ = Σ_k w_k d_k / Σ_k w_k²**.
*Source:* least squares through the origin (any regression text; e.g. Draper & Smith,
*Applied Regression Analysis*, §3.1).

**Step 5. The recession time τ.** τ is one regional value. For each candidate τ on a
41-point log grid from 10 to 1000 days, fit ĉ for every **training** structure and add up
the squared residuals. Keep the τ with the smallest total. Test structures are not used.

**Step 6. Score.** ĉ = η·S_k/S. S_k/S is unknown but common to all structures (it is
regional), so ĉ ranks structures exactly as η does. Absolute recharge is not available
(identifiability, review 2).

## What C assumes that may be wrong

- Recharge proportional to rain. The "threshold" truth of the comparison breaks this on
  purpose.
- No lateral flow between structures. Lateral flow is small for structures kilometres
  apart.
- Elastic, instantaneous coupling.
- One recession time for the region.

## Tests (`tests/test_method_comparison.py::TestOLSBaseline`)

- g equals the simulator's linear-reservoir head with S = 1 and L = 1/τ.
- ĉ = η·S_k/S exactly on noise-free data.
- Unreliable epochs have no influence.
- On noise-free training structures, τ is the grid value nearest the true 60 days.
