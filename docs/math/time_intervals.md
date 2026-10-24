# The first acquisition has no interval (review finding 22)

**In plain English:** the model works in steps, one per satellite pass. For every pass
except the first there is a definite period since the previous pass: 6, 12 or 24 days. The
rain that fell, the recharge, and the change in water level all belong to that period.
The first pass has no previous pass, so there is no period to speak of. The old code
pretended there was, by copying the length of the second period. It also lumped *all*
rain recorded before the first pass into that imaginary period, which could be years of
rain. The mass-balance check therefore thought far more water had been available than
really was, so it could not catch impossible recharge.

## Convention (now used everywhere)

Δt_k = t_k − t_{k−1} for k ≥ 1, and **Δt_0 = 0**. So:

- **Δt_k > 0** ⟺ an interval (t_{k−1}, t_k] exists between two real acquisitions.
- The first real acquisition is still a real step (`valid_step_mask` is True). It is the
  reference date of the InSAR series, so it has Δt = 0 and h = 0 there.
- Padded steps have Δt = 0 and `valid_step_mask` False.

## What uses it

| quantity | rule |
|---|---|
| head integrator | h_k = h_{k−1} + Δt_k·v_k for k ≥ 1 with Δt_k > 0, so h = 0 at the reference step (no separate reference subtraction needed) |
| PDE residual | valid at k if Δt_k > 0 and steps k and k−1 are both real (`real_mask`) |
| recharge volume | Σ R_k·A·Δt_k: the reference step contributes 0 |
| mass-balance inflow | Σ over real steps **with Δt_k > 0** of P_k: rain before the first acquisition is excluded |
| seasonal capacity mean | Δt-weighted: the reference step has weight 0 |
| positional encoding | cumulative time starts at 0 at the reference step |

*Source:* this is just the definition of a backward difference / finite volume in time.
The quantity on step k is attached to the interval (t_{k−1}, t_k], and there is no interval
before the first sample (any numerical-methods text, e.g. Burden & Faires ch. 5).

## Tests (`tests/test_time_intervals.py`)

- `time_deltas_days([0, 7, 21] days)` gives [0, 7, 14]. The old code gave [7, 7, 14].
- In the dataset, the reference acquisition has Δt = 0 and is real; padding has Δt = 0 and
  is not real.
- 900 mm of rain 400 days before the first acquisition does not change the mass-balance
  penalty. The old code counted it as inflow.
- The PDE is valid from the step after the reference acquisition.
- The model's head and displacement are 0 up to and including the reference acquisition.
  This guard already held before the change.
