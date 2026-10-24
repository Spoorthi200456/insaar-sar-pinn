# A sink in the flow equation: Hantush leakage (review finding 11)

**In plain English:** water put into the ground under a recharge structure does not stay
there for ever. The small mound of raised water table slowly drains away into the
surrounding regional aquifer, and the level falls back between rains. The old equation had
only one way for water to leave a structure's location: flow to a *neighbouring structure*.
Real structures are usually kilometres apart with no modelled neighbour, so the equation
said the water level could only ever go up (recharge is never negative). Every dry-season
fall in water level was then a contradiction the model could not explain.

The fix adds the standard "leaky aquifer" term. The water level relaxes back towards its
starting level at a rate set by one number, the leakance L. The larger L is, the faster
the mound drains.

## Symbols

| symbol | meaning | units |
|---|---|---|
| h | head change since the first acquisition | m |
| h_ref | head the aquifer relaxes to with no recharge (the regional level) | m |
| S | storativity | dimensionless |
| T | transmissivity | m²/day |
| R | recharge flux | m/day |
| K′ | vertical hydraulic conductivity of the leaky layer / connection to the regional system | m/day |
| b′ | thickness of that layer | m |
| L = K′/b′ | leakance | 1/day |
| τ = S/L | recession time scale | day |

## Derivation

**Step 1. Leakage flux.** Water crosses a leaky layer of thickness b′ and vertical
conductivity K′ under the head difference (h − h_ref). By Darcy's law the flux per unit
area is q_L = K′ (h − h_ref) / b′ = L (h − h_ref). Units: m/day / m · m = m/day ✓.
*Source:* Hantush & Jacob (1955), *Trans. AGU* 36, "Non-steady radial flow in an infinite
leaky aquifer"; Bear (1979) *Hydraulics of Groundwater* §5 (leaky aquifers); Freeze &
Cherry (1979) §8.3.

**Step 2. Add it to the depth-integrated balance.** The leakage leaves the aquifer, so it
enters with a minus sign:
S ∂h/∂t = ∇·(T∇h) − L (h − h_ref) + R.
Every term is in m/day: S·m/day, (1/m²)(m²/day)(m), (1/day)(m), m/day ✓.

**Step 3. Reference level (PROVISIONAL, OQ13).** h is measured from the first
acquisition, and we take h_ref = 0. This assumes the series starts at the regional level,
for example at the end of a dry season. It is not exact. A per-structure h_ref would trade
one-for-one with the mean recharge (R − L·h_ref is all the data can see), so it would make
R unidentifiable. One global L is learned; per-structure L would need wells.

**Step 4. Residual used in code** (`losses.pde_residual`):
r = S (h_k − h_{k−1})/Δt_k − ∇·(T∇h)_k + L h_k − R_k.
The leakage uses h_k (implicit), which matches the backward difference of the storage
term. The discrete recession h_k = h_{k−1}/(1 + Δt L/S) then gives r = 0 exactly.

**Step 5. What the term predicts.** With no recharge and no neighbours, S dh/dt = −L h, so
h(t) = h₀ e^{−t/τ} with τ = S/L. For S = 0.05 and L = 8.3e-4 /day, τ = 60 days: a
recharge mound halves in about 42 days. That is plausible for a structure-scale mound;
regional data would settle it.

## Range used in the model

There is one global L, with log10 L = −6 + 5·sigmoid(raw), so 1e-6 < L < 1e-1 1/day. It is
initialised at 1e-3 /day. For S ≈ 0.02 this spans recession times from about 0.2 days to
about 50 years.

## Not modelled (PROVISIONAL, OQ14)

**Pumping.** In much of India, abstraction is the largest sink. An unmodelled pumping
signal looks to the model like a large L or a small R: a working structure in a heavily
pumped area could look as if it is failing. Fixing this needs abstraction estimates
(e.g. CGWB well census, power-use proxies).

## Tests (`tests/test_losses.py::TestLeakageSink`)

- A pure recession (R = 0) has residual 0 with L, and −1.4e-3 … −7.5e-4 m/day without it.
  Since R ≥ 0, no allowed recharge can explain it.
- One-step balance: R = S Δh/Δt + L h₁ gives r = 0.
- `CompositeARSLoss` passes L through: the PDE loss on a recession is > 1e-7 without L and
  < 1e-12 with it.
- `ARSNet.leakance` is a bounded global scalar, initialised at 1e-3 /day.
