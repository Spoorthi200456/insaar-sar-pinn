# Lateral flow between structures on a fixed graph (review finding 4)

**In plain English:** groundwater flows from high head to low head. Between two
neighbouring structures, the flow depends on how easily the aquifer passes
water (transmissivity T), the head difference, and the distance. The model
estimates this on a *graph* (structures = nodes, hydraulic connections =
edges), which is a coarse version of the grid a groundwater model such as
MODFLOW uses. Two things were wrong:

1. The graph was rebuilt for every training batch from whichever structures happened to be
   in it. A structure's neighbours, and so its physics, changed from batch to batch. Now one
   graph of **all** structures is built once.
2. The edge weights had no units, so the "flow" term was not in m/day like the other terms
   of the equation. Now they carry 1/m².

## Symbols

| symbol | meaning | units |
|---|---|---|
| h_i | head at node i | m |
| T_i | transmissivity at node i | m²/day |
| d_ij | distance between structures i and j (haversine) | m |
| w_ij | width of the face shared by cells i and j | m |
| A_i | plan area of cell i | m² |
| c_ij | connectivity (1 = full contact, 0 = no connection) | dimensionless |
| a_ij | physics stencil weight | 1/m² |

## Derivation

**Step 1. Darcy flow through a face.** The volume flow from cell j into cell i through a
face of width w_ij is Q_ij = T_ij · w_ij · (h_j − h_i) / d_ij.
Units: m²/day · m · m / m = m³/day ✓.
*Plain English:* more transmissive aquifer, wider contact or a steeper head drop all mean
more flow.
*Source:* Darcy's law integrated over the saturated thickness (Freeze & Cherry 1979 §2.3,
§5.1). This is the "conductance" form CR = T·w/d in MODFLOW (Harbaugh 2005, USGS TM 6-A16,
ch. 2; Wang & Anderson 1982, *Introduction to Groundwater Modeling* ch. 3).

**Step 2. Interface transmissivity.** T_ij = (T_i + T_j)/2, the arithmetic mean. This is
what the code uses. MODFLOW's default is the harmonic mean; with T varying smoothly the
difference is small. This is a provisional simplification.

**Step 3. Water balance of cell i (finite volume).** The net inflow per unit area of cell i
is (1/A_i) · Σ_j Q_ij. This approximates ∇·(T∇h) at node i [m/day], so
∇·(T∇h)_i ≈ Σ_j [w_ij / (d_ij · A_i)] · (T_i + T_j)/2 · (h_j − h_i).
*Source:* finite-volume discretisation of the divergence theorem (Wang & Anderson 1982
ch. 3; Versteeg & Malalasekera 2007 ch. 4).

**Step 4. The weight.** a_ij = w_ij / (d_ij · A_i). Units: m / (m · m²) = **1/m²** ✓.
On a regular square grid of spacing Δ, w = Δ, d = Δ and A = Δ², so a = 1/Δ². The structure
graph is not a grid, so we use this regular-grid limit with Δ = d_ij, scaled by a
dimensionless connectivity: **a_ij = c_ij / d_ij²**.
This is symmetric (a_ij = a_ji), so the flow out of i equals the flow into j. Water is
conserved.

**Step 5. Units of the lateral term.** Σ_j a_ij (T_i+T_j)/2 (h_j − h_i):
1/m² · m²/day · m = m/day ✓. This now matches S ∂h/∂t and R in the residual.

## Fixed graph and minibatches (hard rule 3)

- `src/data/graph.py::StructureGraph` is built once from all records. Its matrices are
  `physics_adjacency` = a_ij [1/m²] and `attention_prior` (dimensionless, for the
  graph-attention layers only).
- A training batch has `batch_size` *seed* structures plus every structure within
  `hops` = (GAT layers + 1) = 3 edges of them.
  - Each seed's network outputs need its 2-hop neighbourhood (2 GAT layers).
  - Its ∇·(T∇h) needs its neighbours' heads, which need *their* 2-hop neighbourhoods.
  - Hence 3 hops.
- Only seeds contribute to the loss (`loss_node_mask`). The result is exactly the same as a
  forward pass over the whole network. Tested: K and the PDE residual of every structure
  are identical for batch sizes 1, 3, 5 and 7, in any shuffle order.

## Worked number (tested in `tests/test_graph.py`)

Two cells 2 km apart, T = 100 m²/day, heads 1 m and 0 m:
a = 1/(2000 m)² = 2.5e-7 /m², so ∇·(T∇h) at the high cell = 2.5e-7 · 100 · (0 − 1) =
**−2.5e-5 m/day** (water leaves it). The low cell gets +2.5e-5 m/day.
The old weight was exp(−2/5) = **0.67** (no units), which would give −67 "m/day"
(really m³/day).

## PROVISIONAL (docs/open_questions.md OQ11)

- Which structures are hydraulically connected. Fallback: all pairs within 5 km, with
  c = 1. Surveyed edges, if supplied, give c = edge weight.
- d_ij as the cell spacing.
- In the synthetic data, structures are about 50 km apart, so most have no edge and the
  lateral term is 0. That is physically plausible, since recharge mounds rarely spread
  kilometres, but the term cannot then constrain T.
