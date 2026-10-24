from __future__ import annotations

import pytest
import torch

from src.models.losses import (
    CompositeARSLoss,
    LossWeights,
    _masked_mse,
    _weighted_laplacian_divergence,
    pde_residual,
)


class TestMaskedMSE:
    def test_matches_plain_mse_without_mask(self):
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.5, 2.5, 2.0])
        expected = torch.mean((pred - target) ** 2)
        assert torch.isclose(_masked_mse(pred, target, None), expected)

    def test_ignores_masked_elements(self):
        pred = torch.tensor([1.0, 100.0])
        target = torch.tensor([1.0, 0.0])
        mask = torch.tensor([True, False])
        result = _masked_mse(pred, target, mask)
        assert torch.isclose(result, torch.tensor(0.0))

    def test_empty_mask_does_not_produce_nan(self):
        pred = torch.tensor([1.0, 2.0])
        target = torch.tensor([9.0, 9.0])
        mask = torch.tensor([False, False])
        result = _masked_mse(pred, target, mask)
        assert torch.isfinite(result)


class TestWeightedLaplacianDivergence:
    def test_zero_for_uniform_head_and_conductivity(self):
        k = torch.ones(4)
        h = torch.ones(4, 3)  # uniform head across nodes -> zero divergence
        adj = torch.rand(4, 4) * (1 - torch.eye(4))
        div = _weighted_laplacian_divergence(k, h, adj)
        assert torch.allclose(div, torch.zeros_like(div), atol=1e-5)

    def test_zero_when_disconnected(self):
        k = torch.ones(3)
        h = torch.randn(3, 5)
        adj = torch.zeros(3, 3)
        div = _weighted_laplacian_divergence(k, h, adj)
        assert torch.allclose(div, torch.zeros_like(div))


class TestCompositeARSLoss:
    def _dummy_batch(self, b=3, t=6):
        torch.manual_seed(0)
        adj = torch.rand(b, b) * (1 - torch.eye(b))
        return dict(
            predicted_displacement_cm=torch.randn(b, t) * 0.1,
            observed_displacement_cm=torch.randn(b, t) * 0.1,
            predicted_head_change=torch.randn(b, t) * 0.1,
            hydraulic_conductivity=torch.rand(b) + 0.1,
            recharge_rate=torch.rand(b, t) * 10,
            design_recharge_capacity=torch.full((b,), 50.0),
            rainfall_mm=torch.rand(b, t) * 40,
            adjacency=adj,
            time_delta_days=torch.full((b, t), 12.0),
            storativity=torch.tensor(0.02),
            observation_mask=torch.ones(b, t, dtype=torch.bool),
            recharge_area_m2=torch.full((b,), 1000.0),
            catchment_area_m2=torch.full((b,), 2.0e6),
        )

    def test_total_loss_is_non_negative_and_finite(self):
        loss_fn = CompositeARSLoss(LossWeights())
        breakdown = loss_fn(**self._dummy_batch())
        assert breakdown.total.item() >= 0
        assert torch.isfinite(breakdown.total)

    def test_design_capacity_penalty_activates_when_exceeded(self):
        loss_fn = CompositeARSLoss(LossWeights())
        batch = self._dummy_batch()
        # way over capacity
        batch["recharge_rate"] = torch.full_like(batch["recharge_rate"], 1000.0)
        breakdown = loss_fn(**batch)
        assert breakdown.design_capacity.item() > 0

    def test_non_negativity_penalty_zero_for_non_negative_recharge(self):
        loss_fn = CompositeARSLoss(LossWeights())
        batch = self._dummy_batch()
        batch["recharge_rate"] = torch.abs(batch["recharge_rate"])
        breakdown = loss_fn(**batch)
        assert breakdown.non_negativity.item() == 0.0

    def test_mass_balance_penalty_activates_when_recharge_exceeds_rainfall(self):
        loss_fn = CompositeARSLoss(LossWeights())
        batch = self._dummy_batch()
        batch["rainfall_mm"] = torch.zeros_like(batch["rainfall_mm"])
        batch["recharge_rate"] = torch.ones_like(batch["recharge_rate"]) * 5.0
        breakdown = loss_fn(**batch)
        assert breakdown.mass_balance.item() > 0

    def test_breakdown_as_dict_has_expected_keys(self):
        loss_fn = CompositeARSLoss(LossWeights())
        breakdown = loss_fn(**self._dummy_batch())
        d = breakdown.as_dict()
        for key in ("total", "pde_residual", "data_misfit", "mass_balance_penalty"):
            assert key in d

    def test_gradients_flow_to_conductivity_and_recharge(self):
        loss_fn = CompositeARSLoss(LossWeights())
        batch = self._dummy_batch()
        batch["hydraulic_conductivity"] = batch["hydraulic_conductivity"].requires_grad_(True)
        batch["recharge_rate"] = batch["recharge_rate"].requires_grad_(True)
        breakdown = loss_fn(**batch)
        breakdown.total.backward()
        assert batch["hydraulic_conductivity"].grad is not None
        assert batch["recharge_rate"].grad is not None

    def test_gradient_reaches_storativity(self):
        loss_fn = CompositeARSLoss(LossWeights())
        batch = self._dummy_batch()
        batch["storativity"] = torch.tensor(0.02, requires_grad=True)
        loss_fn(**batch).total.backward()
        assert batch["storativity"].grad is not None
        assert batch["storativity"].grad.abs().item() > 0


class TestPDEResidualPhysics:
    """
    Form (b): S ∂h/∂t = ∇·(T∇h) + R, residual r = S ∂h/∂t − ∇·(T∇h) − R [m/day].
    No neighbours (adjacency = 0) so ∇·(T∇h) = 0 and S ∂h/∂t must equal R.
    """

    S = 0.1
    R = 1e-4                 # m/day, recharge INTO the aquifer
    RATE = R / S             # m/day head rise that exactly balances R

    def _batch(self, head, dt):
        b, t = head.shape
        return dict(
            predicted_displacement_cm=torch.zeros(b, t),
            observed_displacement_cm=torch.zeros(b, t),
            predicted_head_change=head,
            hydraulic_conductivity=torch.ones(b),
            recharge_rate=torch.full((b, t), self.R),
            design_recharge_capacity=torch.full((b,), 1e6),
            rainfall_mm=torch.full((b, t), 1e6),
            adjacency=torch.zeros(b, b),
            time_delta_days=dt,
            storativity=torch.tensor(self.S),
            observation_mask=torch.ones(b, t, dtype=torch.bool),
            recharge_area_m2=torch.ones(b),
            catchment_area_m2=torch.ones(b),
        )

    @staticmethod
    def _head_from_dt(dt_days, rate):
        # h_k = rate * (t_k - t_0), with t_k = sum of Δt_1..Δt_k
        dt = torch.tensor(dt_days, dtype=torch.float64)
        t = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(dt[1:], 0)])
        return (rate * t).float().unsqueeze(0), dt.float().unsqueeze(0)

    def test_recharge_with_rising_head_has_zero_residual(self):
        head, dt = self._head_from_dt([12.0] * 6, self.RATE)
        breakdown = CompositeARSLoss(LossWeights())(**self._batch(head, dt))
        assert breakdown.pde.item() < 1e-14

    def test_recharge_with_falling_head_is_penalised(self):
        rising, dt = self._head_from_dt([12.0] * 6, self.RATE)
        falling, _ = self._head_from_dt([12.0] * 6, -self.RATE)
        loss_fn = CompositeARSLoss(LossWeights())
        loss_rising = loss_fn(**self._batch(rising, dt)).pde.item()
        loss_falling = loss_fn(**self._batch(falling, dt)).pde.item()
        # Positive recharge must favour rising head, never falling head.
        assert loss_falling > loss_rising
        assert abs(loss_falling - (2 * self.R) ** 2) < 1e-12  # r = S·(−RATE) − R = −2R

    def test_storage_uses_real_dt(self):
        # Same physical head rate sampled at irregular Sentinel-1 gaps.
        head, dt = self._head_from_dt([12.0, 6.0, 12.0, 24.0, 12.0, 6.0], self.RATE)
        residual, valid = pde_residual(
            head, dt, torch.tensor(self.S), torch.ones(1),
            torch.full_like(head, self.R), torch.zeros(1, 1),
        )
        assert valid[0, 1:].all()
        assert torch.allclose(residual[0, 1:], torch.zeros(5), atol=1e-9)

    def test_padding_and_first_step_excluded(self):
        head, dt = self._head_from_dt([12.0] * 6, self.RATE)
        head = head + 0.05  # non-zero head, so a pad→real jump would show up if not excluded
        pad = 4
        head_p = torch.cat([torch.zeros(1, pad), head], dim=1)
        dt_p = torch.cat([torch.zeros(1, pad), dt], dim=1)

        _, valid_p = pde_residual(
            head_p, dt_p, torch.tensor(self.S), torch.ones(1),
            torch.full_like(head_p, self.R), torch.zeros(1, 1),
        )
        # padding, the first real step (predecessor is padding) and k=0 are invalid
        assert not valid_p[0, : pad + 1].any()
        assert valid_p[0, pad + 1 :].all()

        loss_fn = CompositeARSLoss(LossWeights())
        unpadded = loss_fn(**self._batch(head, dt))
        batch_p = self._batch(head_p, dt_p)
        padded = loss_fn(**batch_p)
        assert torch.isfinite(padded.pde)
        assert abs(padded.pde.item() - unpadded.pde.item()) < 1e-14


class TestRegularizerUnits:
    """
    Review finding 6. R [m/day] is a flux per unit area; it must be converted to
    a volume rate R*A [m3/day] (A = structure recharge area [m2]) before it is
    compared with the design capacity Q_design [m3/day], and to a volume
    sum_k R_k A dt_k [m3] before it is compared with the rain volume that can
    reach the structure, sum_k P_k/1000 (A + c_r A_catch) [m3].
    Derivation: docs/math/recharge_units.md.
    """

    A = 1000.0          # m2 water-spread area
    A_CATCH = 0.5e6     # m2 (0.5 km2)

    def _batch(self, r, p_mm, q_design, b=2, t=6, area=A, catch=A_CATCH):
        return dict(
            predicted_displacement_cm=torch.zeros(b, t),
            observed_displacement_cm=torch.zeros(b, t),
            predicted_head_change=torch.zeros(b, t),
            hydraulic_conductivity=torch.ones(b),
            recharge_rate=torch.full((b, t), r),
            design_recharge_capacity=torch.full((b,), q_design),
            rainfall_mm=torch.full((b, t), p_mm),
            adjacency=torch.zeros(b, b),
            time_delta_days=torch.full((b, t), 7.0),
            storativity=torch.tensor(0.02),
            observation_mask=torch.ones(b, t, dtype=torch.bool),
            recharge_area_m2=torch.full((b,), area),
            catchment_area_m2=torch.full((b,), catch),
        )

    def test_capacity_fires_when_volume_rate_exceeds_design(self):
        # Demo value R = 0.17 m/day over 1000 m2 = 170 m3/day > Q_design = 100 m3/day.
        # Old code compared 0.17 with 100 directly and never fired.
        out = CompositeARSLoss(LossWeights())(**self._batch(0.17, 40.0, 100.0))
        assert out.design_capacity.item() == pytest.approx((170.0 / 100.0 - 1.0) ** 2, rel=1e-5)

    def test_mass_balance_fires_when_recharge_volume_exceeds_rain_volume(self):
        # 1 mm per 7-day step on A + c_r*A_catch = 1000 + 0.3*5e5 = 151000 m2 -> 151 m3/step
        # in; R = 0.17 m/day * 1000 m2 * 7 day = 1190 m3/step out -> impossible.
        # Old code compared sum R = 1.02 with sum P = 6 (mm) and never fired.
        out = CompositeARSLoss(LossWeights())(**self._batch(0.17, 1.0, 1e9))
        assert out.mass_balance.item() > 1.0

    def test_feasible_recharge_gives_zero_penalties(self):
        # R*A = 10 m3/day <= 100; 70 m3/step out << 40 mm * 151000 m2 = 6040 m3/step in.
        out = CompositeARSLoss(LossWeights())(**self._batch(0.01, 40.0, 100.0))
        assert out.design_capacity.item() == 0.0
        assert out.mass_balance.item() == 0.0

    def test_penalties_are_dimensionally_consistent(self):
        # Rescaling every area and the design capacity by the same factor is a
        # change of units of area (e.g. m2 -> 10 m2 blocks); dimensionless
        # penalties must not change.
        loss_fn = CompositeARSLoss(LossWeights())
        base = loss_fn(**self._batch(0.17, 1.0, 100.0))
        c = 37.0
        scaled = loss_fn(
            **self._batch(0.17, 1.0, 100.0 * c, area=self.A * c, catch=self.A_CATCH * c)
        )
        assert scaled.design_capacity.item() == pytest.approx(base.design_capacity.item(), rel=1e-5)
        assert scaled.mass_balance.item() == pytest.approx(base.mass_balance.item(), rel=1e-5)
        assert base.design_capacity.item() > 0 and base.mass_balance.item() > 0


class TestPDEMaskAndClamp:
    """
    Review finding 8. The physics term must apply at every real step (Δt > 0
    for k and k-1), including low-coherence / interpolated InSAR steps where
    there is no trustworthy data and physics is the only constraint. It must
    also keep a gradient for large residuals (no hard clamp).
    """

    def _batch(self, head, recharge, mask):
        b, t = head.shape
        return dict(
            predicted_displacement_cm=torch.zeros(b, t),
            observed_displacement_cm=torch.zeros(b, t),
            predicted_head_change=head,
            hydraulic_conductivity=torch.ones(b),
            recharge_rate=recharge,
            design_recharge_capacity=torch.full((b,), 1e12),
            rainfall_mm=torch.full((b, t), 1e6),
            adjacency=torch.zeros(b, b),
            time_delta_days=torch.full((b, t), 12.0),
            storativity=torch.tensor(0.1),
            recharge_area_m2=torch.ones(b),
            catchment_area_m2=torch.ones(b),
            observation_mask=mask,
        )

    def test_low_coherence_step_still_constrained_by_physics(self):
        head = torch.zeros(1, 5)                  # flat head ...
        recharge = torch.zeros(1, 5)
        recharge[0, 3] = 1e-3                     # ... but recharge at step 3: r = -1e-3
        mask = torch.ones(1, 5, dtype=torch.bool)
        mask[0, 3] = False                        # step 3 has low coherence
        out = CompositeARSLoss(LossWeights())(**self._batch(head, recharge, mask))
        # valid PDE steps are k = 1..4 (4 steps); only k = 3 is non-zero
        assert out.pde.item() == pytest.approx((1e-3) ** 2 / 4, rel=1e-5)

    def test_large_residual_keeps_gradient(self):
        head = torch.zeros(1, 4)
        recharge = torch.full((1, 4), 5e4, requires_grad=True)  # |r| = 5e4 m/day > old 1e4 clamp
        mask = torch.ones(1, 4, dtype=torch.bool)
        out = CompositeARSLoss(LossWeights(pde=1.0, data=0, boundary=0, design_capacity=0,
                                           non_negativity=0, mass_balance=0,
                                           conductivity_smoothness=0))(
            **self._batch(head, recharge, mask))
        out.total.backward()
        assert recharge.grad is not None
        assert recharge.grad[0, 1:].abs().min().item() > 0


class TestRegularizersMasked:
    """
    Review finding 7. Left-padded steps (Δt = 0) are not real; the recharge
    decoder still emits softplus(.) > 0 there, which must not count as recharge.
    T-smoothness must be dimensionless: it compares log T on graph edges, so a
    change of units of T (a constant factor) cannot change it.
    """

    def _batch(self, r, dt, rain, area=1000.0, q=100.0):
        b, t = r.shape
        return dict(
            predicted_displacement_cm=torch.zeros(b, t),
            observed_displacement_cm=torch.zeros(b, t),
            predicted_head_change=torch.zeros(b, t),
            hydraulic_conductivity=torch.ones(b),
            recharge_rate=r,
            design_recharge_capacity=torch.full((b,), q),
            rainfall_mm=rain,
            adjacency=torch.zeros(b, b),
            time_delta_days=dt,
            storativity=torch.tensor(0.02),
            recharge_area_m2=torch.full((b,), area),
            catchment_area_m2=torch.full((b,), 1e5),
            observation_mask=dt > 0,
        )

    def test_padding_does_not_change_regularizers(self):
        r = torch.tensor([[0.05, 0.2, 0.3, 0.01]])        # 50, 200, 300, 10 m3/day vs Q = 100
        dt = torch.full((1, 4), 12.0)
        rain = torch.tensor([[1.0, 0.0, 2.0, 0.5]])
        pad = 6
        r_p = torch.cat([torch.full((1, pad), 0.5), r], dim=1)  # phantom softplus recharge
        dt_p = torch.cat([torch.zeros(1, pad), dt], dim=1)
        rain_p = torch.cat([torch.zeros(1, pad), rain], dim=1)
        loss_fn = CompositeARSLoss(LossWeights())
        plain = loss_fn(**self._batch(r, dt, rain))
        padded = loss_fn(**self._batch(r_p, dt_p, rain_p))
        assert plain.design_capacity.item() > 0 and plain.mass_balance.item() > 0
        assert padded.design_capacity.item() == pytest.approx(
            plain.design_capacity.item(), rel=1e-6
        )
        assert padded.mass_balance.item() == pytest.approx(plain.mass_balance.item(), rel=1e-6)
        assert padded.non_negativity.item() == pytest.approx(plain.non_negativity.item(), abs=1e-12)

    def test_explicit_valid_mask_overrides_dt(self):
        # A single-acquisition series has Δt = 0 but is real: the caller says so.
        r = torch.tensor([[0.3]])
        out = CompositeARSLoss(LossWeights())(
            **self._batch(r, torch.zeros(1, 1), torch.zeros(1, 1)),
            valid_mask=torch.ones(1, 1, dtype=torch.bool),
        )
        assert out.design_capacity.item() == pytest.approx((300.0 / 100.0 - 1.0) ** 2, rel=1e-5)

    def test_t_smoothness_is_dimensionless_log_difference(self):
        args = self._batch(torch.zeros(3, 2), torch.full((3, 2), 12.0), torch.zeros(3, 2))
        args["adjacency"] = torch.tensor([[0.0, 2.5e-7, 0.0],
                                          [2.5e-7, 0.0, 1e-8],
                                          [0.0, 1e-8, 0.0]])            # 1/m², unequal
        args["hydraulic_conductivity"] = torch.tensor([1.0, 10.0, 100.0])   # m²/day
        loss_fn = CompositeARSLoss(LossWeights())
        base = loss_fn(**args).conductivity_smoothness.item()
        # every edge is one decade apart: mean (ln 10)^2, independent of a_ij size
        assert base == pytest.approx(torch.log(torch.tensor(10.0)).item() ** 2, rel=1e-5)
        # any change of units of T
        args["hydraulic_conductivity"] = args["hydraulic_conductivity"] * 86400.0
        assert loss_fn(**args).conductivity_smoothness.item() == pytest.approx(base, rel=1e-5)


class TestLeakageSink:
    """
    Review finding 11. With R >= 0 and no sink, an isolated node's head can only
    rise, so a dry-season recession is unexplainable. Form (b) with Hantush
    leakage: S ∂h/∂t = ∇·(T∇h) − L (h − h_ref) + R, h_ref = 0 (h is relative to
    the first acquisition). Residual r = S ∂h/∂t − ∇·(T∇h) + L h − R  [m/day].
    Derivation: docs/math/flow_equation_leakage.md.
    """

    S, L = 0.05, 0.05 / 60.0  # 60-day recession time scale S/L

    def _recession(self, h0=2.0, dt_days=(12.0, 12.0, 6.0, 24.0, 12.0)):
        # exact backward-Euler recession with R = 0: h_k = h_{k-1} / (1 + Δt L / S)
        h = [h0]
        for dt in dt_days[1:]:
            h.append(h[-1] / (1.0 + dt * self.L / self.S))
        return (torch.tensor([h], dtype=torch.float64),
                torch.tensor([list(dt_days)], dtype=torch.float64))

    def test_pure_recession_without_recharge_has_zero_residual(self):
        h, dt = self._recession()
        r, valid = pde_residual(h, dt, torch.tensor(self.S, dtype=torch.float64),
                                torch.ones(1, dtype=torch.float64), torch.zeros_like(h),
                                torch.zeros(1, 1, dtype=torch.float64),
                                leakance_per_day=torch.tensor(self.L, dtype=torch.float64))
        assert valid[0, 1:].all()
        assert torch.allclose(r[0, 1:], torch.zeros(4, dtype=torch.float64), atol=1e-15)

    def test_leakance_units_balance_storage(self):
        # one step: S (h1 - h0)/Δt + L h1 - R = 0  <=>  R = S Δh/Δt + L h1 [m/day]
        h = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        dt = torch.tensor([[10.0, 10.0]], dtype=torch.float64)
        r, _ = pde_residual(h, dt, torch.tensor(0.05, dtype=torch.float64),
                            torch.ones(1, dtype=torch.float64),
                            torch.full_like(h, 0.05 * 0.1 + 1e-3),
                            torch.zeros(1, 1, dtype=torch.float64),
                            leakance_per_day=torch.tensor(1e-3, dtype=torch.float64))
        assert r[0, 1].item() == pytest.approx(0.0, abs=1e-15)

    def test_model_exposes_bounded_global_leakance(self, synthetic_records):
        from src.data.dataset import ARSDataset, make_collate_fn
        from src.models.network import ARSNet

        ds = ARSDataset(synthetic_records, sequence_length=16)
        batch = make_collate_fn()([ds[i] for i in range(len(ds))])
        out = ARSNet(design_feature_dim=3)(
            design_features=batch["design_features"], displacement_cm=batch["displacement_cm"],
            rainfall_mm=batch["rainfall_mm"], time_delta_days=batch["time_delta_days"],
            adjacency=batch["adjacency"], incidence_angle_deg=batch["incidence_angle_deg"],
            coherence_mask=batch["coherence_mask"])
        assert out.leakance.dim() == 0
        assert out.leakance.item() == pytest.approx(1e-3, rel=1e-4)  # init [1/day]
        assert 1e-6 < out.leakance.item() < 1e-1

    def test_composite_loss_uses_leakance(self):
        h, dt = self._recession()
        h, dt = h.float(), dt.float()
        args = dict(
            predicted_displacement_cm=torch.zeros_like(h),
            observed_displacement_cm=torch.zeros_like(h),
            predicted_head_change=h, hydraulic_conductivity=torch.ones(1),
            recharge_rate=torch.zeros_like(h), design_recharge_capacity=torch.full((1,), 1e6),
            rainfall_mm=torch.zeros_like(h), adjacency=torch.zeros(1, 1), time_delta_days=dt,
            storativity=torch.tensor(self.S), recharge_area_m2=torch.ones(1),
            catchment_area_m2=torch.ones(1))
        loss_fn = CompositeARSLoss(LossWeights())
        assert loss_fn(**args).pde.item() > 1e-7  # no sink: recession unexplained
        assert loss_fn(**args, leakance=torch.tensor(self.L)).pde.item() < 1e-12
