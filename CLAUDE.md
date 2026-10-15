# InSAR–PINN ARS Monitoring — project rules for Claude Code

## What this project is
Monitoring the integrity of Artificial Recharge Structures (ARS) by fusing
Sentinel-1 InSAR line-of-sight (LOS) displacement with a physics-informed
model of groundwater flow and poroelastic deformation.
Core code: `src/models/` (network, layers, losses), `src/data/` (schemas, preprocessing, dataset).
`src/backend/` and `frontend/` are deployment plumbing — NOT the research contribution.

## What must NOT change (the goal) vs. what MAY change (the method)
- Fixed goal: assess whether Artificial Recharge Structures in India are
  working (recharging the aquifer) or failing (clogging, leakage, structural
  damage), using freely available satellite InSAR + rainfall + design data,
  constrained by groundwater physics.
- The method (equations, loss terms, architecture) MAY change if it is
  physically or mathematically wrong. Every such change must be proposed in
  plan mode first, with the reason, and approved by the author.

## About the author
MSc Data Science student: strong in ML/DL and PyTorch, NEW to groundwater
hydrology and poroelasticity. When explaining physics or math, show every
step of the derivation, define every symbol with units, and say which
textbook result each step relies on. Never skip steps.

## Governing physics (source of truth — fill in and keep updated)
- Flow: S_s ∂h/∂t = ∇·(K ∇h) + Q        (Q > 0 = recharge INTO the aquifer)
- Residual must be:  r = S_s ∂h/∂t − ∇·(K ∇h) − Q
- Deformation coupling: TODO — write the exact relation, the coefficient's
  name and units (it is NOT the dimensionless Biot α if it maps head [m] to displacement [cm]).
- LOS sign convention: positive = motion toward the satellite.
  (provisional, confirm against data source)
  CONFLICT, not yet resolved: `src/data/schemas.py:107` documents
  `los_displacement_cm` as "positive = uplift", which is a vertical
  convention, not a LOS one. Code not changed; see docs/review.md finding 2.
  Incidence angle handling: per-structure input θ taken from the Sentinel-1
  product metadata (roughly 29–46° in IW mode), not a constant.
  (provisional, confirm against data source) Not yet used in code.

## Units table (every physical tensor must appear here)
| quantity | symbol | units | where |
|---|---|---|---|
| LOS displacement | d | cm | schemas.py |
| rainfall | R | mm per InSAR interval | dataset.py |
| design recharge capacity | Q_design | m³/day | schemas.py |
| recharge (model output) | Q | TODO | network.py |
| hydraulic conductivity | K | TODO | network.py |
| time step | Δt | days | dataset.py |

## Hard rules
1. Never change a PDE term, loss term, sign, or unit without first writing the
   derivation in the PR/commit message and updating the tables above.
2. Any term that adds/compares two quantities must have matching units — if
   not, stop and flag it instead of "fixing" it with a scale factor.
3. Physics must not depend on minibatch composition (graph neighbours must come
   from a fixed structure network, not from whichever samples share a batch).
4. Don't touch `docs/INVENTION_LOG.md` — it is written by the human author only.
5. Run `pytest -q` after any change in `src/`. Don't weaken or delete tests to make them pass.
6. Never read or print `.env` / `.env.local` contents.

## Review style
Be adversarial. Prefer "this is wrong because …" over "consider …".
Report findings with file:line, the failure scenario, and the fix.
