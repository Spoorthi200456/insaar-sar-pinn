from __future__ import annotations

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
        )

    def test_total_loss_is_non_negative_and_finite(self):
        loss_fn = CompositeARSLoss(LossWeights())
        breakdown = loss_fn(**self._dummy_batch())
        assert breakdown.total.item() >= 0
        assert torch.isfinite(breakdown.total)

    def test_design_capacity_penalty_activates_when_exceeded(self):
        loss_fn = CompositeARSLoss(LossWeights())
        batch = self._dummy_batch()
        batch["recharge_rate"] = torch.full_like(batch["recharge_rate"], 1000.0)  # way over capacity
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
