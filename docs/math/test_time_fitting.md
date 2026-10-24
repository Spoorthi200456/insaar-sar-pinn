# Method B: fitting the physics-informed model to each new structure (review 4)

**In plain English:** the trained network (method A) is applied to a new structure as a
fixed formula, and on structures it had not seen it failed (review 3). But the model's
training objective needs **no right answers**. It only asks that:
- the predicted ground movement matches the structure's satellite data;
- the predicted water level obeys the groundwater equation;
- recharge stays within what rain and design allow.

All of these can be computed for a new structure from its own data. Method B therefore
starts from the trained network and keeps adjusting it **on that one structure's own data**
before reading off its recharge.

## Objective (for one test structure i)

min over θ_net of L_i(θ_net; S*, S_k*, L*)

L_i is the default `CompositeARSLoss` evaluated on structure i alone:
- the dimensionless sum of the PDE residual / σ_r²;
- the InSAR misfit / σ_d²;
- the seasonal design-capacity and mass-balance penalties;
- non-negativity and T-smoothness (`docs/math/loss_scaling.md`).

θ_net are all network weights. S*, S_k* and L* are the **global constants from training,
held fixed**.

**Why freeze the global constants.** From InSAR alone only R·S_k/S is identifiable
(review 2, `TestIdentifiability`). If each structure refitted its own S_k and S, each would
get its own unknown scale factor, and comparing recharge between structures, the purpose
of the ranking, would be meaningless. Holding them at the regional values learned from the
training structures gives every test structure the same factor.

**Settings (fixed by the protocol):**
- Adam, lr 1e-3, 300 steps, gradient-norm clip 5;
- eval mode (no dropout);
- the structure alone in its batch (no neighbours);
- start from method A's selected weights, on a copy, so the trained model is unchanged;
- final weights used, because there is no ground truth to early-stop on.

Score = mean predicted recharge over steps 1…K−1.

*Source:* this is "test-time training" / per-instance optimisation of a physics-informed
model. The data and physics residuals are self-supervised (Raissi, Perdikaris &
Karniadakis 2019, *J. Comput. Phys.* 378, for PINN inverse problems; Sun et al. 2020,
ICML, "Test-time training", for the general idea).

## Tests (`tests/test_method_comparison.py::TestTestTimeFitting`)

- The function takes no truth argument.
- It lowers the structure's own loss.
- The global constants are unchanged.
- The trained model passed in is not modified.
