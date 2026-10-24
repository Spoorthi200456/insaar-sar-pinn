"""
Review finding 2: head change -> InSAR line-of-sight displacement.

    d_LOS [cm] = 100 [cm/m] * cos(theta) * S_k * (h - h_ref) [m]

positive d_LOS = motion toward the satellite (PROVISIONAL, docs/open_questions.md
OQ2), so a head RISE (recharge) must give a POSITIVE displacement (uplift).
Derivation: docs/math/deformation_coupling.md.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
import torch
from pydantic import ValidationError

from src.data.dataset import ARSDataset, make_collate_fn
from src.data.schemas import (
    ARSDesignParameters,
    ARSTelemetryRecord,
    RainfallObservation,
    SARObservation,
)
from src.models.coupling import CM_PER_M, head_to_los_cm
from src.models.network import ARSNet
from src.training.train import make_synthetic_records_with_truth


class TestHeadToLOS:
    def test_head_rise_gives_uplift_toward_satellite(self):
        d = head_to_los_cm(torch.tensor([[0.0, 1.0]]), torch.tensor(1e-4), torch.tensor([35.0]))
        assert d[0, 1].item() > 0  # +1 m of head -> uplift -> toward satellite -> positive

    def test_coupling_units_exact(self):
        dh = torch.tensor([[1.0]])  # 1 m of head change
        s_k = torch.tensor(1e-4)    # dimensionless
        # theta = 0: 1e-4 * 1 m = 1e-4 m = 0.01 cm
        assert head_to_los_cm(dh, s_k, torch.tensor([0.0])).item() == pytest.approx(0.01, rel=1e-6)
        # theta = 60 deg: cos 60 = 0.5 -> 0.005 cm
        assert head_to_los_cm(dh, s_k, torch.tensor([60.0])).item() == pytest.approx(0.005, rel=1e-5)
        assert CM_PER_M == 100.0

    def test_incidence_angle_scales_by_cosine(self):
        dh = torch.tensor([[2.0], [2.0]])
        d = head_to_los_cm(dh, torch.tensor(1e-3), torch.tensor([29.0, 46.0]))
        expected = math.cos(math.radians(29.0)) / math.cos(math.radians(46.0))
        assert (d[0, 0] / d[1, 0]).item() == pytest.approx(expected, rel=1e-5)

    def test_per_structure_skeletal_storage_broadcasts(self):
        dh = torch.ones(2, 3)
        d = head_to_los_cm(dh, torch.tensor([1e-4, 2e-4]), torch.tensor([0.0, 0.0]))
        assert torch.allclose(d[1], 2 * d[0])


def _batch(records, sequence_length=16):
    ds = ARSDataset(records, sequence_length=sequence_length)
    return make_collate_fn()([ds[i] for i in range(len(ds))])


def _forward(model, batch):
    return model(
        design_features=batch["design_features"],
        displacement_cm=batch["displacement_cm"],
        rainfall_mm=batch["rainfall_mm"],
        time_delta_days=batch["time_delta_days"],
        adjacency=batch["adjacency"],
        incidence_angle_deg=batch["incidence_angle_deg"],
        coherence_mask=batch["coherence_mask"],
    )


class TestARSNetCoupling:
    def test_model_displacement_is_the_shared_coupling_of_head_increment(self, synthetic_records):
        torch.manual_seed(0)
        batch = _batch(synthetic_records)
        model = ARSNet(design_feature_dim=3).eval()
        with torch.no_grad():
            out = _forward(model, batch)
        h = out.predicted_head_change
        h_inc = h - h[:, :1]  # no padding at T=16, so the first real step is index 0
        expected = head_to_los_cm(h_inc, out.skeletal_storage, batch["incidence_angle_deg"])
        assert torch.allclose(out.predicted_displacement_cm, expected, atol=1e-7)
        # sign: head risen since t0 -> uplift (d > 0); head fallen -> subsidence (d < 0)
        rising, falling = h_inc > 1e-6, h_inc < -1e-6
        assert rising.any() or falling.any()
        assert (out.predicted_displacement_cm[rising] > 0).all()
        assert (out.predicted_displacement_cm[falling] < 0).all()

    def test_skeletal_storage_bounded(self, synthetic_records):
        batch = _batch(synthetic_records)
        model = ARSNet(design_feature_dim=3)
        out = _forward(model, batch)
        assert out.skeletal_storage.item() == pytest.approx(1e-4, rel=1e-5)  # init
        for direction in (+1.0, -1.0):  # huge pushes both ways: S_k must stay in its range
            model = ARSNet(design_feature_dim=3)
            opt = torch.optim.SGD(model.parameters(), lr=1e3)
            loss = direction * 1e6 * model.skeletal_storage()
            loss.backward()
            opt.step()
            s_k = _forward(model, batch).skeletal_storage.item()
            # closed bounds: float32 sigmoid saturates to exactly 0 or 1
            assert 1e-6 * (1 - 1e-6) <= s_k <= 1e-2 * (1 + 1e-6)
            assert s_k != pytest.approx(1e-4, rel=0.5)  # it really moved


class TestSyntheticGeneratorUsesModelCoupling:
    def test_noise_free_displacement_equals_coupling_of_true_head(self):
        records, truth = make_synthetic_records_with_truth(
            n_structures=3, n_timesteps=20, seed=3, noise_cm=0.0
        )
        for rec, t in zip(records, truth):
            sar = sorted(rec.sar_series, key=lambda o: o.timestamp)
            d_obs = torch.tensor([o.los_displacement_cm for o in sar], dtype=torch.float64)
            h = torch.tensor(t["head_m"], dtype=torch.float64).unsqueeze(0)
            d_exp = head_to_los_cm(
                h - h[:, :1],
                torch.tensor(t["skeletal_storage"], dtype=torch.float64),
                torch.tensor([t["incidence_angle_deg"]], dtype=torch.float64),
            )[0]
            assert torch.allclose(d_obs, d_exp, atol=1e-9)
            # planted recharge raises head, and raised head must show as uplift
            assert max(t["head_m"]) > 0
            assert d_obs.max().item() > 0


def _sar(ts, **kw):
    base = dict(structure_id="ARS-0001", timestamp=ts, los_displacement_cm=0.1, coherence=0.9,
                incidence_angle_deg=35.0)
    base.update(kw)
    return SARObservation(**base)


class TestSchemaIncidenceAndTrack:
    def test_missing_incidence_angle_rejected(self):
        with pytest.raises(ValidationError):
            SARObservation(structure_id="A", timestamp=datetime.now(timezone.utc),
                           los_displacement_cm=0.1, coherence=0.9)

    @pytest.mark.parametrize("theta", [5.0, 70.0])
    def test_out_of_range_incidence_angle_rejected(self, theta):
        with pytest.raises(ValidationError):
            _sar(datetime.now(timezone.utc), incidence_angle_deg=theta)

    def test_mixed_tracks_rejected(self):
        t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        design = ARSDesignParameters(structure_id="ARS-0001", latitude=12.9, longitude=77.5,
                                     height_m=5.0, storage_volume_m3=1000.0, catchment_area_km2=2.0,
                                     design_recharge_capacity_m3_per_day=100.0)
        sar = [_sar(t0, track_direction="ascending"),
               _sar(t0 + timedelta(days=12), track_direction="descending")]
        rain = [RainfallObservation(structure_id="ARS-0001", timestamp=t0, rainfall_mm=1.0)]
        with pytest.raises(ValidationError):
            ARSTelemetryRecord(design=design, sar_series=sar, rainfall_series=rain)
