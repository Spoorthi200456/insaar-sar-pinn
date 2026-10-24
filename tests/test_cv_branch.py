"""
Review finding 28: the computer-vision branch (CVEncoder: footprint mask +
displacement-map patches) was not exercised by any test. Its BatchNorm layers
also expose a defect in MC-dropout uncertainty: switching the whole model to
train() makes BatchNorm use batch statistics and overwrite its running
averages on every request. MC dropout must sample dropout masks only
(Gal & Ghahramani 2016), with BatchNorm frozen.
"""

from __future__ import annotations

import pytest
import torch

from src.models.network import ARSNet

B, T, C, H = 4, 8, 3, 16


def _inputs():
    g = torch.Generator().manual_seed(0)
    return dict(
        design_features=torch.randn(B, 3, generator=g),
        displacement_cm=torch.randn(B, T, generator=g) * 0.3,
        rainfall_mm=torch.rand(B, T, generator=g) * 20,
        time_delta_days=torch.cat([torch.zeros(B, 1), torch.full((B, T - 1), 12.0)], dim=1),
        adjacency=torch.zeros(B, B),
        incidence_angle_deg=torch.full((B,), 35.0),
        cv_patch=torch.randn(B, C, H, H, generator=g),
    )


class TestCVBranch:
    def test_forward_shapes_and_gradient_reach_the_cv_encoder(self):
        torch.manual_seed(0)
        model = ARSNet(design_feature_dim=3, cv_in_channels=C)
        out = model(**_inputs())
        assert out.recharge_rate.shape == (B, T)
        out.recharge_rate.sum().backward()
        grads = [p.grad for p in model.cv_encoder.parameters()]
        assert all(g is not None for g in grads)
        assert any(g.abs().sum() > 0 for g in grads)

    def test_missing_patch_is_an_error(self):
        model = ARSNet(design_feature_dim=3, cv_in_channels=C)
        args = _inputs()
        args.pop("cv_patch")
        with pytest.raises(ValueError, match="cv_patch"):
            model(**args)

    def test_cv_patch_changes_the_prediction(self):
        torch.manual_seed(0)
        model = ARSNet(design_feature_dim=3, cv_in_channels=C).eval()
        a = _inputs()
        b = dict(a, cv_patch=a["cv_patch"] + 1.0)
        with torch.no_grad():
            assert not torch.allclose(model(**a).recharge_rate, model(**b).recharge_rate)


class TestMCDropoutFreezesBatchNorm:
    def test_batchnorm_statistics_unchanged_and_dropout_active(self):
        from src.backend.inference_engine import mc_dropout_mode

        torch.manual_seed(0)
        model = ARSNet(design_feature_dim=3, cv_in_channels=C).eval()
        bn = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        assert bn
        before = [m.running_mean.clone() for m in bn]
        with torch.no_grad(), mc_dropout_mode(model):
            assert all(not m.training for m in bn)
            r1 = model(**_inputs()).recharge_rate
            r2 = model(**_inputs()).recharge_rate
        assert not torch.allclose(r1, r2)  # dropout really samples
        assert all(torch.equal(b0, m.running_mean) for b0, m in zip(before, bn, strict=True))
        assert not model.training  # restored to eval afterwards
