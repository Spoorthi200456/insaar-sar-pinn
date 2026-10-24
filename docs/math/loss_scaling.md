# Making the total loss dimensionless (review finding 29)

**In plain English:** the training objective adds several penalties together, but
they were measured in different units. The physics error was in (metres per day)², the
satellite misfit in cm², the well-head misfit in m². Adding them is like adding 3 kg
to 5 km: the answer depends on the units you happen to pick. Here, a physically large
flow-equation error of 1 mm/day cost 0.000001, while a tiny 0.3 cm displacement misfit
cost 0.09. So training effectively ignored the physics.

The fix divides each term by a "how big an error do we consider normal" scale for that
quantity before adding. Every term then counts in units of "typical errors" and has no
physical unit.

## Equation

total = w_pde · L_pde/σ_r² + w_data · L_data/σ_d² + w_BC · L_BC/σ_h² + (dimensionless
regularisers).

Each ratio is dimensionless: (m/day)²/(m/day)², cm²/cm², m²/m² ✓.
*Source:* this is the Gaussian negative log-likelihood up to constants. A misfit with
standard deviation σ contributes (error/σ)². See Tarantola (2005) *Inverse Problem Theory*
ch. 1 and 3 (least squares as a likelihood), and Raissi, Perdikaris & Karniadakis (2019),
*J. Comput. Phys.* 378 (PINN loss terms), for the practice of balancing PINN terms.

## Scales (PROVISIONAL, docs/open_questions.md OQ18)

They were chosen **before** the known-answer experiment was run, and not tuned to it.

| scale | value | why |
|---|---|---|
| σ_d (displacement) | 0.3 cm | typical per-epoch Sentinel-1 LOS time-series noise (a few mm) |
| σ_r (PDE residual) | 1e-4 m/day = 0.1 mm/day | a few % of a structure's recharge flux (≈ mm/day), i.e. "the physics holds to a few %" |
| σ_h (well head) | 0.1 m | typical manual / logger head uncertainty |

## Tests (`tests/test_losses.py::TestTotalLossIsDimensionless`)

- The same data expressed in mm, with σ_d also in mm, gives the same total. The old code
  gave 1.75 in cm and 175 in mm.
- A residual of exactly σ_r at every valid step adds exactly w_pde = 1 to the total. The
  breakdown term stays in (m/day)².

## Well-head term (author decision 4)

Wells report **absolute** head (m above a datum); the model's head h is the change since
the first InSAR acquisition. So the two can differ by a constant per structure. With
e_k = h_k − h_well,k on the steps with a matched reading:

L_well = mean over readings of (e_k − ē_i)², where ē_i is the mean of e over structure i's
readings. Units: m².
Total contribution: w_well · L_well / σ_h² (dimensionless, σ_h = 0.1 m).

*Plain English:* only the *shape* of the water-level curve is compared, not its height
above sea level. A structure with no readings adds nothing. The training log lists such
structures, and the API reports `has_well_data = false` for them.

*Source:* least squares with an unknown constant offset is ordinary least squares on
de-meaned data (Tarantola 2005 ch. 3; any regression text).

Tests: `tests/test_wells.py`.
- A 312.5 m datum offset gives a misfit of 0.
- A de-meaned ±0.2 m mismatch gives 0.02 m² and total 2.0.
- Structures without readings are ignored.
- Readings more than 3 days from an acquisition are not used.
