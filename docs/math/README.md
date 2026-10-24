# Maths notes

One file per physics / maths change made in review 2. Each note explains the
equations in plain English, defines every symbol with units, and names the
textbook result each step relies on. Edition-dependent section numbers are
marked "check against your edition".

| File | Topic | Review finding |
|---|---|---|
| [deformation_coupling.md](deformation_coupling.md) | Head change → line-of-sight displacement | 2 |
| [recharge_units.md](recharge_units.md) | Recharge vs design capacity vs rainfall | 6 |
| [graph_stencil.md](graph_stencil.md) | Lateral flow on a fixed structure graph, a_ij in 1/m² | 4 |
| [masking.md](masking.md) | Which time steps each loss term uses; no hard clamp | 8, 7 |
| [head_integrator.md](head_integrator.md) | Head from head rate with the real Δt | 3 (note) |
| [flow_equation_leakage.md](flow_equation_leakage.md) | Hantush leakage sink in the flow equation | 11 |
| [loss_scaling.md](loss_scaling.md) | Dimensionless total loss (σ_d, σ_r, σ_h) | 29 |
| [validation_split.md](validation_split.md) | Train / validation / test split by structure group | 24 |

The governing equations used by the code are summarised in `CLAUDE.md` →
"Governing physics"; these notes are the full derivations behind it.
