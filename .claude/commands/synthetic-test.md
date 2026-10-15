---
description: Plan a synthetic "known answer" test — can the model recover K and Q we planted?
---
Use plan mode. Do not write code until I approve the plan.

Goal: prove (or disprove) that this model can recover physical parameters
when we KNOW the true answer.

Plan the following:
1. A small, independent forward simulator (NumPy, not the neural network) for
   a few ARS nodes: given true K_i, S_s, Q_i(t), rainfall, irregular Δt, it
   produces head h_i(t) and then LOS displacement d_i(t) using the coupling
   and sign convention in CLAUDE.md. Add realistic InSAR noise and gaps.
2. Train ARSNet on this synthetic data only.
3. Report: error in recovered K and Q vs. truth, per node; whether different
   (K, Q) pairs give the same displacement (identifiability); how results
   change with noise level and with/without head (piezometer) observations.
4. Put it in `experiments/synthetic_recovery/` with a fixed random seed and a
   README of the results. Add a pytest that fails if recovery error exceeds a
   threshold we agree on.
$ARGUMENTS
