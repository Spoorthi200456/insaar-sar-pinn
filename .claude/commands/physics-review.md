---
description: Adversarial physics + numerics review of the PINN model and loss
---
Review `src/models/network.py`, `src/models/layers.py`, `src/models/losses.py`
and `src/data/dataset.py` against the physics and units tables in CLAUDE.md.
Do NOT edit any file. For each check, answer PASS / FAIL with file:line evidence:

1. Sign of every term in the PDE residual vs. S_s ∂h/∂t = ∇·(K∇h) + Q.
2. Is ∂h/∂t divided by the actual (irregular) Δt? Is S_s present?
3. Dimensional consistency of every sum, difference and comparison
   (PDE residual, mass balance, design-capacity penalty, data misfit).
4. Displacement ↔ head coupling: sign, coefficient units, LOS projection.
5. Does any physics term depend on minibatch composition or shuffling?
6. Identifiability: can K and Q be separated from LOS displacement alone?
   If not, what extra data (piezometer heads, pumping tests) is required?
7. Masking: do padded / low-coherence steps leak into any loss term?
8. Anything the README/docstrings claim that the code does not actually do.

Finish with a ranked list: the 3 problems that would get this rejected by a
hydrogeology reviewer or a patent examiner, and what experiment would settle each.
$ARGUMENTS
