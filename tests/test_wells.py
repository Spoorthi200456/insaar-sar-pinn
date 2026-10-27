"""
Author decision 4: well water levels are an OPTIONAL training input
(`ARSTelemetryRecord.well_series`). Wells report absolute head; the model's head
is relative to the first InSAR acquisition, so the well term compares them
after removing a per-structure constant offset. When a structure has no well
data, the system must say so (training log, dataset flag, API field).
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
import torch

from src.data.dataset import ARSDataset, make_collate_fn
from src.data.schemas import WellHeadObservation
from src.models.losses import CompositeARSLoss, LossWeights
from src.training.train import make_synthetic_records


def _with_wells(record, heads_m, every=2, jitter_days=1.0):
    sar = sorted(record.sar_series, key=lambda o: o.timestamp)
    wells = [
        WellHeadObservation(
            structure_id=record.design.structure_id,
            timestamp=sar[k].timestamp + timedelta(days=jitter_days),
            hydraulic_head_m=float(heads_m[k]),
        )
        for k in range(0, len(sar), every)
    ]
    return record.model_copy(update={"well_series": wells})


class TestDatasetAlignsWells:
    def test_wells_aligned_to_acquisitions_and_missing_flagged(self):
        recs = make_synthetic_records(n_structures=2, n_timesteps=8, seed=3)
        heads = 100.0 + np.arange(8) * 0.5  # absolute head [m]
        recs[0] = _with_wells(recs[0], heads)  # every 2nd acquisition, 1 day late
        ds = ARSDataset(recs, sequence_length=10)  # 2 padded steps
        s0, s1 = ds[0], ds[1]
        assert s0.has_well_data.item() is True and s1.has_well_data.item() is False
        assert s0.well_mask.tolist() == [False, False] + [True, False] * 4
        assert torch.allclose(s0.well_head_m[2::2], torch.tensor(heads[0::2], dtype=torch.float32))
        assert not s1.well_mask.any()
        batch = make_collate_fn()([s0, s1])
        assert batch["has_well_data"].tolist() == [True, False]
        assert batch["well_head_m"].shape == (2, 10)

    def test_reading_too_far_from_an_acquisition_is_not_used(self):
        recs = make_synthetic_records(n_structures=1, n_timesteps=6, seed=3)
        recs[0] = _with_wells(recs[0], np.zeros(6), every=1, jitter_days=3.5)  # > 3-day tolerance
        s = ARSDataset(recs, sequence_length=6)[0]
        assert not s.well_mask.any() and s.has_well_data.item() is False


class TestWellLoss:
    def _args(self, h_pred, h_well, mask):
        b, t = h_pred.shape
        return dict(
            predicted_displacement_cm=torch.zeros(b, t),
            observed_displacement_cm=torch.zeros(b, t),
            predicted_head_change=h_pred,
            hydraulic_conductivity=torch.ones(b),
            recharge_rate=torch.zeros(b, t),
            design_recharge_capacity=torch.full((b,), 1e6),
            rainfall_mm=torch.full((b, t), 1e3),
            adjacency=torch.zeros(b, b),
            time_delta_days=torch.full((b, t), 12.0),
            storativity=torch.tensor(0.05),
            recharge_area_m2=torch.ones(b),
            catchment_area_m2=torch.ones(b),
            well_head_m=h_well,
            well_mask=mask,
        )

    def test_constant_datum_offset_is_ignored(self):
        h = torch.tensor([[0.0, 0.4, 1.0, 0.7]])
        out = CompositeARSLoss(LossWeights())(
            **self._args(h, h + 312.5, torch.ones(1, 4, dtype=torch.bool))
        )
        assert out.well.item() == pytest.approx(0.0, abs=1e-10)

    def test_shape_mismatch_is_penalised_in_metres_squared(self):
        h = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        well = torch.tensor([[10.0, 10.2, 9.8, 10.0]])  # de-meaned: 0, .2, -.2, 0
        w = LossWeights(
            pde=0,
            data=0,
            boundary=0,
            design_capacity=0,
            non_negativity=0,
            mass_balance=0,
            conductivity_smoothness=0,
            well=1.0,
        )
        out = CompositeARSLoss(w, head_scale_m=0.1)(
            **self._args(h, well, torch.ones(1, 4, dtype=torch.bool))
        )
        assert out.well.item() == pytest.approx(0.02, rel=1e-5)  # mean of (0, .04, .04, 0) [m²]
        assert out.total.item() == pytest.approx(0.02 / 0.1**2, rel=1e-5)  # dimensionless
        assert "well_head_misfit" in out.as_dict()

    def test_structures_without_wells_contribute_nothing(self):
        h = torch.zeros(2, 4)
        well = torch.tensor([[0.0, 0.1, 0.0, 0.1], [50.0, -50.0, 50.0, -50.0]])
        mask = torch.tensor([[True] * 4, [False] * 4])
        out = CompositeARSLoss(LossWeights())(**self._args(h, well, mask))
        assert out.well.item() == pytest.approx(0.0025, rel=1e-5)  # only structure 0


def test_training_reports_structures_without_well_data():
    from src.training.train import structures_without_well_data

    recs = make_synthetic_records(n_structures=3, n_timesteps=6, seed=3)
    recs[1] = _with_wells(recs[1], np.zeros(6))
    assert structures_without_well_data(recs, sequence_length=6) == ["ARS-0000", "ARS-0002"]


def test_api_prediction_says_whether_well_data_was_present():
    from src.backend.inference_engine import InferenceEngine

    recs = make_synthetic_records(n_structures=2, n_timesteps=8, seed=3)
    recs[0] = _with_wells(recs[0], 100.0 + np.arange(8) * 0.1)
    # Well data are used by the network path (the default least-squares method ignores
    # them), so select it explicitly (review 5).
    preds = InferenceEngine(onnx_path=None, torch_checkpoint_path=None, sequence_length=8).predict(
        recs, method="neural_experimental"
    )
    assert preds[0].has_well_data is True and preds[0].well_head_rmse_m is not None
    assert preds[1].has_well_data is False and preds[1].well_head_rmse_m is None
