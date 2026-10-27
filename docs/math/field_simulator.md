# The field-like simulator of the harder comparison (review 5)

**In plain English:** review 4's test world followed exactly the simple rule that method C
assumes: recharge proportional to rain, one rainfall series, one drainage time. This
simulator is written so that **C's rule is wrong**, in the ways real structures differ from
each other.

Code: `experiments/harder_comparison/simulator.py`. Protocol and parameter values:
`experiments/harder_comparison/README.md`.

## Symbols

| symbol | meaning | units |
|---|---|---|
| d | day index (365 spin-up days, then the acquisition period) | day |
| P_i,d | rain at structure i on day d | mm/day |
| θ_i | runoff threshold (initial abstraction) | mm/day |
| c_i | runoff coefficient | – |
| A_i, A_catch,i | pond area, catchment area | m² |
| V_i, V_max,i | pond volume, capacity A_i H_i | m³ |
| E | pond evaporation, 5 | mm/day |
| k_i(d) | infiltration capacity of the pond bed | m/day |
| I_i,d | infiltrated volume (the recharge volume) | m³/day |
| A_cell | area over which a structure's recharge spreads, 2×10⁵ | m² |
| R_i,d = I_i,d / A_cell | recharge flux into the aquifer node | m/day |
| Q_i,d | pumping sink, q_i/A_cell in rabi, else 0 | m/day |
| S, S_k | storativity 0.05, skeletal storage 5×10⁻³ | – |
| τ_i, L_i = S/τ_i | recession time, leakance | day, 1/day |
| T_i, a_ij | transmissivity, graph weight 1/d_ij² | m²/day, 1/m² |

## Step 1. Rain → pond inflow (threshold runoff)

The catchment yields runoff only on days when rain exceeds what the soil absorbs first (the
"initial abstraction"). The pond also catches the rain that falls directly on it:

inflow_i,d = c_i · max(P_i,d − θ_i, 0)/1000 · A_catch,i + P_i,d/1000 · A_i   [m³]

Units: mm/1000 = m, and m × m² = m³ ✓.

*Source:* threshold (initial-abstraction) runoff, as in the SCS curve-number method (USDA
NRCS, *National Engineering Handbook* Part 630, ch. 10), reduced here to a linear excess.

## Step 2. Pond water balance (a bucket)

Each day, in this order:
1. V ← min(V + inflow, V_max). Water above capacity spills and is lost.
2. I = min(V, k_i(d) · A_i). The bed can pass at most k·A m³ per day.
3. V ← V − I.
4. V ← V − min(V, E/1000 · A_i). Evaporation acts only while water is present.

So I ≤ k·A always, I ≤ the water available, and the pond can keep recharging for days
after the rain stops. That memory is not proportional to that day's rain.

*Source:* single-store "bucket" water balance (Thornthwaite & Mather 1955; Manabe 1969).

## Step 3. Clogging

- working: k = k0;
- failed: k = k0 · U(0, 0.05);
- clogging: k(t) = k0 · exp(−t/t_c), with t_c = t_end / ln(1/f), so that k(t_end) = f · k0.

Exponential decline of infiltration rate through progressive clogging of the bed is the
standard simple description (Bouwer 2002, *Hydrogeology Journal* 10, 121–142, "Artificial
recharge of groundwater").

## Step 4. Aquifer (form (b), daily backward Euler)

Starting from the depth-integrated flow equation with leakage
(`docs/math/flow_equation_leakage.md`), with a pumping sink:

S ∂h/∂t = ∇·(T∇h) − L_i h + R − Q.

The graph stencil (`docs/math/graph_stencil.md`) gives
(∇·(T∇h))_i ≈ Σ_j a_ij (T_i + T_j)/2 (h_j − h_i) = (G h)_i.

Backward Euler with Δt = 1 day:

S (h_d − h_{d−1}) = G h_d − L h_d + R_d − Q_d
⇔ (S I + diag(L) − G) h_d = S h_{d−1} + R_d − Q_d.

The matrix does not change from day to day, so it is inverted once. Units: every term is
m/day × 1 day = m ✓.

What differs from C:
- each node has its own τ_i = S/L_i;
- nodes exchange water laterally;
- pumping lowers the head without being "negative recharge" in the truth.

## Step 5. InSAR and the truth that is scored

- InSAR at acquisition k: d_k = 100 cos θ S_k (h(t_k) − h(t₀)) + noise
  (`docs/math/deformation_coupling.md`).
- The true recharge of interval k is the mean of R_i,d over the days in (t_{k−1}, t_k],
  which are the same days the dataset bins the rain into.

## Why C cannot represent this

C models d = ĉ · g(rain; τ), with g linear in rain and one τ. Here:
- the response to rain is non-linear (threshold, spill, capacity limit);
- the response is delayed (pond storage);
- the response changes over time (clogging);
- part of the signal is not caused by rain at all (pumping);
- τ differs between structures.

`tests/test_harder_comparison.py::test_simulator_is_not_method_c_model` measures this on
noise-free data:
- on review 4's simulator, C's best fit leaves under 2 % of the signal unexplained;
- here, more than 10 %.
