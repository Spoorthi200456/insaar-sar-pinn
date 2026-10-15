from __future__ import annotations

import torch

from src.models.losses import CompositeARSLoss, LossWeights, _masked_mse, _weighted_laplacian_divergence


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
