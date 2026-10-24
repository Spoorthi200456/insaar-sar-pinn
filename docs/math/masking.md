# Which time steps each loss term uses (review findings 7 and 8)

**In plain English:** each structure's time series is padded at the start
with fake zero steps so that all series have the same length, and some real
satellite measurements are untrustworthy (low coherence, clamped, missing).
These are two different things:

- a **padded** step is not real at all. Nothing should be computed there.
- a **low-coherence** step is a real moment in time where the *measurement* is bad. The
  physics still happened. Groundwater keeps flowing whether or not the satellite saw it
  clearly.

So the data term should skip bad measurements, but the physics term should not. Using
physics where the data is missing is the main point of a physics-informed model. The old
code skipped physics wherever the data was bad.

| term | uses step k when | reason |
|---|---|---|
| data misfit | k is real **and** the measurement is reliable (coherence ≥ 0.7, not clamped, not interpolated) **and** the structure is a seed | Never fit untrustworthy numbers. |
| PDE residual | k **and** k−1 are real (Δt_k > 0, Δt_{k−1} > 0) **and** the structure is a seed | The backward difference (h_k − h_{k−1})/Δt_k needs both steps. Coherence is irrelevant to physics. |
| design capacity, mass balance | k is real, structure is a seed (masking added by the finding-7 fix) | Padding has no rain and no recharge. |

## Why there is no hard clamp on the residual

The old code computed clamp(r, −1e4, 1e4)². For |r| > 1e4 m/day the clamp is flat, so the
gradient was **zero**: the worse the physics, the less the model was pushed to fix it.
The clamp is removed. Exploding gradients are already guarded by
`clip_grad_norm_(max_norm=5)` in training. That rescales the step instead of switching the
signal off.

Tests: `tests/test_losses.py::TestPDEMaskAndClamp`.
- A low-coherence step with r = −1e-3 m/day now contributes (1e-3)²/4 to the loss; the old
  code gave 0.
- At |r| = 5e4 m/day, d(loss)/dR is non-zero; the old code gave exactly 0.
