# Integrating head with the real time gap (review-2 fix; finding 3 note)

**In plain English:** the model predicts how fast the water level is changing at each
satellite pass (the head rate v, in metres per day), then adds up those changes to get the
water level itself. The old version added one fixed-size step per satellite pass. A 24-day
gap between passes therefore counted the same as a 6-day gap, and the "water level" it
produced had no real unit. The new version multiplies each rate by the actual number of
days since the previous pass, as you would compute distance from speed × time.

## Symbols

| symbol | meaning | units |
|---|---|---|
| v_k | predicted head rate at step k (`ARSNetOutput.head_rate`) | m/day |
| Δt_k | t_k − t_{k−1} (0 on padding) | day |
| h_k | head change since the first real acquisition (`predicted_head_change`) | m |

## Scheme

h_k = h_{k−1} + Δt_k · v_k, applied only when steps k and k−1 are both real acquisitions.
Otherwise the increment is 0. So h = 0 on padding and at the first real acquisition, which
is the reference date of the InSAR series (OQ6).
Units: day · m/day = m ✓.
*Source:* the explicit (forward) Euler / rectangle rule for integrating dh/dt = v
(any numerical-methods text, e.g. Burden & Faires, *Numerical Analysis*, ch. 5).

**Consistency with the PDE residual.** The residual (`losses.pde_residual`) uses the
backward difference (h_k − h_{k−1})/Δt_k. With this integrator that difference equals v_k
exactly, so the storage term is S·v_k with no discretisation mismatch. The physics loss
then ties the predicted head rate to recharge and lateral flow:
S·v_k − ∇·(T∇h)_k − R_k → 0.

**Why the head is a separate output.** Head is tied to recharge only through the soft PDE
residual, which is the standard PINN arrangement. If head were computed *from* R by the
same discretised equation, the residual would be zero by construction and the physics term
would only push lateral flow towards zero. The old K-modulated decay integrator also built
in a relation, "higher K discharges head faster", which is not part of form (b).

## Scale (PROVISIONAL, docs/open_questions.md OQ12)

v = 0.01 m/day × (decoder output). The decoder output is a plain number of order 1, so this
characteristic rate of 1 cm/day only sets the size at initialisation. The unit stays m/day,
and training can reach any value. This is non-dimensionalisation, standard practice for
PINNs, not a unit-fixing factor.

## Tests (`tests/test_models.py::TestHeadIntegratorUsesRealDt`)

- h_k − h_{k−1} = Δt_k · v_k for irregular gaps of 12/6/12/24 days.
- h = 0 on 8 padded steps and on the first real step. The old code had already
  accumulated 0.37 → 0.73 "m" of head on padding.
- Doubling every Δt doubles h for the same rates. The old integrator had no Δt input at
  all: constant q gave the same h for 6-day and 24-day revisits.
