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
            incidence_angle_deg=batch["incidence_angle_deg"],
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
            incidence_angle_deg=batch["incidence_angle_deg"],
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
            incidence_angle_deg=batch["incidence_angle_deg"],
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
            incidence_angle_deg=batch["incidence_angle_deg"],
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
            incidence_angle_deg=batch["incidence_angle_deg"],
            coherence_mask=batch["coherence_mask"],
        )
        # Include predicted_displacement_cm so the loss also touches
        # `head_rate_decoder` (head enters only via displacement), which
        # `recharge_rate`/`hydraulic_conductivity` alone do not exercise.
        # `storativity` is the only path to `storativity_logit` (it enters the
        # PDE residual in the loss, not the forward outputs above); the same
        # holds for `leakance` -> `leakance_raw` (finding 11).
        loss = (
            out.recharge_rate.sum()
            + out.hydraulic_conductivity.sum()
            + out.predicted_displacement_cm.sum()
            + out.storativity
            + out.leakance
        )
        loss.backward()
        missing_grad = [name for name, p in model.named_parameters() if p.grad is None]
        assert not missing_grad, f"Parameters with no gradient: {missing_grad}"


class TestIsolatedNodeKeepsOwnFeatures:
    """Review finding 5: a structure with no neighbours must still use its own features."""

    def test_gat_output_of_isolated_node_depends_on_its_features(self):
        torch.manual_seed(0)
        layer = HydrologicGATLayer(in_dim=4, out_dim=4, num_heads=2, dropout=0.0).eval()
        adj = torch.zeros(1, 3, 3)  # nobody has a neighbour
        v1 = torch.randn(1, 3, 4)
        v2 = v1.clone()
        v2[0, 1] += 1.0  # change only node 1's features
        out1, out2 = layer(v1, adj), layer(v2, adj)
        assert not torch.allclose(out1[0, 1], out2[0, 1])   # node 1 sees its own change
        assert torch.allclose(out1[0, 0], out2[0, 0])       # isolated others do not
        assert out1[0, 1].abs().sum() > 0                   # embedding is not all-zero

    def test_design_features_change_k_of_isolated_structure(self, synthetic_records):
        batch = _batch_from_records(synthetic_records, sequence_length=16)
        assert (batch["adjacency"] == 0).all()  # synthetic structures are ~50 km apart
        torch.manual_seed(0)
        model = ARSNet(design_feature_dim=3).eval()

        def k_of(design):
            with torch.no_grad():
                return model(
                    design_features=design,
                    displacement_cm=batch["displacement_cm"],
                    rainfall_mm=batch["rainfall_mm"],
                    time_delta_days=batch["time_delta_days"],
                    adjacency=batch["adjacency"],
                    incidence_angle_deg=batch["incidence_angle_deg"],
                    coherence_mask=batch["coherence_mask"],
                ).hydraulic_conductivity

        design = batch["design_features"].clone()
        changed = design.clone()
        changed[0] += 2.0  # e.g. a much taller dam with a larger catchment
        assert abs(k_of(design)[0].item() - k_of(changed)[0].item()) > 1e-6


class TestHeadIntegratorUsesRealDt:
    """
    Review finding 3 note / review-2 item: head must be integrated with the
    real time gap, h_k = h_{k-1} + Δt_k · v_k, with v_k the predicted head rate
    [m/day], h = 0 at the first real acquisition and on padding.
    """

    def _forward(self, model, batch, dt):
        return model(
            design_features=batch["design_features"],
            displacement_cm=batch["displacement_cm"],
            rainfall_mm=batch["rainfall_mm"],
            time_delta_days=dt,
            adjacency=batch["adjacency"],
            incidence_angle_deg=batch["incidence_angle_deg"],
            coherence_mask=batch["coherence_mask"],
        )

    def test_head_increment_is_rate_times_real_dt(self, synthetic_records):
        batch = _batch_from_records(synthetic_records, sequence_length=16)
        torch.manual_seed(0)
        model = ARSNet(design_feature_dim=3).eval()
        gaps = torch.tensor([12.0, 6.0, 12.0, 24.0] * 4)
        dt = gaps.expand_as(batch["time_delta_days"]).clone()
        with torch.no_grad():
            out = self._forward(model, batch, dt)
        h, v = out.predicted_head_change, out.head_rate
        assert torch.allclose(h[:, 0], torch.zeros_like(h[:, 0]))              # reference step
        assert torch.allclose(h[:, 1:] - h[:, :-1], dt[:, 1:] * v[:, 1:], atol=1e-6)

    def test_head_is_zero_on_padding_and_first_real_step(self, synthetic_records):
        batch = _batch_from_records(synthetic_records, sequence_length=24)  # 16 real + 8 pad
        torch.manual_seed(0)
        model = ARSNet(design_feature_dim=3).eval()
        with torch.no_grad():
            out = self._forward(model, batch, batch["time_delta_days"])
        h = out.predicted_head_change
        assert (batch["time_delta_days"][:, :8] == 0).all()
        assert torch.equal(h[:, :9], torch.zeros_like(h[:, :9]))  # 8 pads + first real step
        assert (h[:, 9:].abs() > 0).any()

    def test_doubling_every_gap_doubles_head_for_the_same_rate(self, synthetic_records):
        # The decoder sees Δt only through h; with the rate held fixed (same
        # inputs except Δt is not an input to it), h must scale with elapsed time.
        batch = _batch_from_records(synthetic_records, sequence_length=16)
        torch.manual_seed(0)
        model = ARSNet(design_feature_dim=3).eval()
        dt = batch["time_delta_days"]
        with torch.no_grad():
            v = self._forward(model, batch, dt).head_rate
            h1 = model.head_integrator(v, dt)
            h2 = model.head_integrator(v, 2 * dt)
        assert torch.allclose(h2, 2 * h1, atol=1e-6)


class TestRechargeOutputScale:
    """
    Review finding 20. Before training, R must start at a physically plausible
    magnitude for infiltration through a structure bed (mm to a few cm per
    day), not ~0.7 m/day (700 mm/day) from an unscaled softplus.
    """

    def test_initial_recharge_is_plausible(self, synthetic_records):
        batch = _batch_from_records(synthetic_records, sequence_length=16)
        torch.manual_seed(0)
        model = ARSNet(design_feature_dim=3).eval()
        with torch.no_grad():
            out = model(
                design_features=batch["design_features"],
                displacement_cm=batch["displacement_cm"],
                rainfall_mm=batch["rainfall_mm"],
                time_delta_days=batch["time_delta_days"],
                adjacency=batch["adjacency"],
                incidence_angle_deg=batch["incidence_angle_deg"],
                coherence_mask=batch["coherence_mask"],
            )
        mean_r = out.recharge_rate.mean().item()
        assert 1e-4 < mean_r < 5e-2  # 0.1 mm/day .. 50 mm/day
