from __future__ import annotations

import torch

from src.data.dataset import ARSDataset, make_collate_fn
from src.models.layers import HydrologicGATLayer, HydrologicGATStack, TemporalAlignmentTransformer
from src.models.network import ARSNet


def _batch_from_records(records, sequence_length=16):
    dataset = ARSDataset(records, sequence_length=sequence_length)
    samples = [dataset[i] for i in range(len(dataset))]
    return make_collate_fn()(samples)


class TestHydrologicGATLayer:
    def test_output_shape_concat_heads(self):
        layer = HydrologicGATLayer(in_dim=8, out_dim=4, num_heads=3, concat_heads=True)
        v = torch.randn(2, 5, 8)
        adj = torch.rand(2, 5, 5)
        adj = adj * (1 - torch.eye(5))
        out = layer(v, adj)
        assert out.shape == (2, 5, 12)

    def test_output_shape_averaged_heads(self):
        layer = HydrologicGATLayer(in_dim=8, out_dim=4, num_heads=3, concat_heads=False)
        v = torch.randn(2, 5, 8)
        adj = torch.rand(2, 5, 5) * (1 - torch.eye(5))
        out = layer(v, adj)
        assert out.shape == (2, 5, 4)

    def test_isolated_node_does_not_produce_nan(self):
        layer = HydrologicGATLayer(in_dim=4, out_dim=4, num_heads=2)
        v = torch.randn(1, 3, 4)
        adj = torch.zeros(1, 3, 3)  # fully disconnected graph
        out = layer(v, adj)
        assert torch.isfinite(out).all()


class TestHydrologicGATStack:
    def test_multi_layer_output_dim(self):
        stack = HydrologicGATStack(in_dim=6, hidden_dim=8, num_layers=2, num_heads=2)
        v = torch.randn(1, 4, 6)
        adj = torch.rand(1, 4, 4) * (1 - torch.eye(4))
        out = stack(v, adj)
        assert out.shape == (1, 4, stack.output_dim)


class TestTemporalAlignmentTransformer:
    def test_pooled_output_shape(self):
        model = TemporalAlignmentTransformer(embed_dim=16, num_layers=2, num_heads=2)
        disp = torch.randn(3, 10)
        rain = torch.rand(3, 10)
        delta = torch.full((3, 10), 7.0)
        out = model(disp, rain, delta)
        assert out.shape == (3, 16)

    def test_handles_key_padding_mask(self):
        model = TemporalAlignmentTransformer(embed_dim=16, num_layers=1, num_heads=2)
        disp = torch.randn(2, 6)
        rain = torch.rand(2, 6)
        delta = torch.full((2, 6), 7.0)
        mask = torch.zeros(2, 6, dtype=torch.bool)
        mask[:, :2] = True  # first two steps padded/ignored
        out = model(disp, rain, delta, key_padding_mask=mask)
        assert torch.isfinite(out).all()


class TestARSNet:
    def test_forward_output_shapes(self, synthetic_records):
        batch = _batch_from_records(synthetic_records, sequence_length=16)
        model = ARSNet(design_feature_dim=3)
        out = model(
            design_features=batch["design_features"],
            displacement_cm=batch["displacement_cm"],
            rainfall_mm=batch["rainfall_mm"],
            time_delta_days=batch["time_delta_days"],
            adjacency=batch["adjacency"],
            biot_alpha=batch["biot_alpha"],
            coherence_mask=batch["coherence_mask"],
        )
        n = batch["displacement_cm"].shape[0]
        assert out.hydraulic_conductivity.shape == (n,)
        assert out.recharge_rate.shape == (n, 16)
        assert out.predicted_displacement_cm.shape == (n, 16)

    def test_conductivity_is_positive(self, synthetic_records):
        batch = _batch_from_records(synthetic_records, sequence_length=16)
        model = ARSNet(design_feature_dim=3)
        out = model(
            design_features=batch["design_features"],
            displacement_cm=batch["displacement_cm"],
            rainfall_mm=batch["rainfall_mm"],
            time_delta_days=batch["time_delta_days"],
            adjacency=batch["adjacency"],
            biot_alpha=batch["biot_alpha"],
            coherence_mask=batch["coherence_mask"],
        )
        assert (out.hydraulic_conductivity > 0).all()

    def test_recharge_rate_is_non_negative(self, synthetic_records):
        batch = _batch_from_records(synthetic_records, sequence_length=16)
        model = ARSNet(design_feature_dim=3)
        out = model(
            design_features=batch["design_features"],
            displacement_cm=batch["displacement_cm"],
            rainfall_mm=batch["rainfall_mm"],
            time_delta_days=batch["time_delta_days"],
            adjacency=batch["adjacency"],
            biot_alpha=batch["biot_alpha"],
            coherence_mask=batch["coherence_mask"],
        )
        assert (out.recharge_rate >= 0).all()

    def test_storativity_is_dimensionless_fraction(self, synthetic_records):
        batch = _batch_from_records(synthetic_records, sequence_length=16)
        model = ARSNet(design_feature_dim=3)
        out = model(
            design_features=batch["design_features"],
            displacement_cm=batch["displacement_cm"],
            rainfall_mm=batch["rainfall_mm"],
            time_delta_days=batch["time_delta_days"],
            adjacency=batch["adjacency"],
            biot_alpha=batch["biot_alpha"],
            coherence_mask=batch["coherence_mask"],
        )
        assert out.storativity.dim() == 0
        assert 0.0 < out.storativity.item() < 1.0
        assert abs(out.storativity.item() - 0.02) < 1e-6  # provisional init

    def test_gradients_flow_to_all_parameters(self, synthetic_records):
        batch = _batch_from_records(synthetic_records, sequence_length=16)
        model = ARSNet(design_feature_dim=3)
        out = model(
            design_features=batch["design_features"],
            displacement_cm=batch["displacement_cm"],
            rainfall_mm=batch["rainfall_mm"],
            time_delta_days=batch["time_delta_days"],
            adjacency=batch["adjacency"],
            biot_alpha=batch["biot_alpha"],
            coherence_mask=batch["coherence_mask"],
        )
        # Include predicted_displacement_cm so the loss also touches
        # `head_integrator` (and therefore `decay_logit`), which
        # `recharge_rate`/`hydraulic_conductivity` alone do not exercise.
        # `storativity` is the only path to `storativity_logit` (it enters the
        # PDE residual in the loss, not the forward outputs above).
        loss = (
            out.recharge_rate.sum()
            + out.hydraulic_conductivity.sum()
            + out.predicted_displacement_cm.sum()
            + out.storativity
        )
        loss.backward()
        missing_grad = [name for name, p in model.named_parameters() if p.grad is None]
        assert not missing_grad, f"Parameters with no gradient: {missing_grad}"
