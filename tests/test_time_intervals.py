"""
Review finding 22. The first acquisition has no preceding interval, so its
Δt is 0, not a copy of the second gap. Everything that needs an interval
(storage term, head increment, recharge volume, rain inflow) uses only steps
with Δt_k > 0; the reference acquisition still counts as a real step.
See docs/math/time_intervals.md.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
import torch

from src.data.dataset import ARSDataset
from src.data.preprocessing import time_deltas_days
from src.data.schemas import RainfallObservation
from src.models.losses import CompositeARSLoss, LossWeights, pde_residual
from src.training.train import make_synthetic_records


def test_first_step_has_no_interval():
    ts = np.array([0.0, 7.0, 21.0]) * 86400.0
    assert time_deltas_days(ts).tolist() == [0.0, 7.0, 14.0]


def test_dataset_marks_first_acquisition_real_but_without_interval():
    rec = make_synthetic_records(n_structures=1, n_timesteps=6, seed=0)[0]
    s = ARSDataset([rec], sequence_length=9)[0]  # 3 padded steps
    assert s.time_delta_days[:4].tolist() == [0.0, 0.0, 0.0, 0.0]  # 3 pads + first real
    assert s.valid_step_mask.tolist() == [False] * 3 + [True] * 6
    assert (s.time_delta_days[4:] > 0).all()


def test_rain_before_the_first_acquisition_is_not_inflow():
    # Step 0 aggregates ALL rain up to t0 (possibly years); it fell before the
    # monitored period and must not count as water available for recharge.
    rec = make_synthetic_records(n_structures=1, n_timesteps=6, seed=0)[0]
    t0 = min(o.timestamp for o in rec.sar_series)
    wet = rec.model_copy(
        update={
            "rainfall_series": rec.rainfall_series
            + [
                RainfallObservation(
                    structure_id=rec.design.structure_id,
                    timestamp=t0 - timedelta(days=400),
                    rainfall_mm=900.0,
                )
            ]
        }
    )
    loss_fn = CompositeARSLoss(LossWeights())

    def mass_balance(r):
        s = ARSDataset([r], sequence_length=6)[0]
        t = s.time_delta_days.shape[0]
        return loss_fn(
            predicted_displacement_cm=torch.zeros(1, t),
            observed_displacement_cm=torch.zeros(1, t),
            predicted_head_change=torch.zeros(1, t),
            hydraulic_conductivity=torch.ones(1),
            recharge_rate=torch.full((1, t), 0.5),
            design_recharge_capacity=torch.tensor([1e9]),
            rainfall_mm=s.rainfall_mm.unsqueeze(0),
            adjacency=torch.zeros(1, 1),
            time_delta_days=s.time_delta_days.unsqueeze(0),
            storativity=torch.tensor(0.05),
            recharge_area_m2=torch.tensor([1000.0]),
            catchment_area_m2=torch.tensor([1e5]),
            valid_mask=s.valid_step_mask.unsqueeze(0),
        ).mass_balance.item()

    dry, wet_ = mass_balance(rec), mass_balance(wet)
    assert dry > 0
    assert wet_ == pytest.approx(dry, rel=1e-6)


def test_pde_uses_the_step_after_the_reference_acquisition():
    # padding (2), reference acquisition (Δt=0), then two real intervals
    dt = torch.tensor([[0.0, 0.0, 0.0, 12.0, 12.0]])
    real = torch.tensor([[False, False, True, True, True]])
    _, valid = pde_residual(
        torch.zeros(1, 5),
        dt,
        torch.tensor(0.05),
        torch.ones(1),
        torch.zeros(1, 5),
        torch.zeros(1, 1),
        real_mask=real,
    )
    assert valid.tolist() == [[False, False, False, True, True]]


def test_model_head_and_displacement_are_zero_at_the_reference_acquisition():
    from src.data.dataset import make_collate_fn
    from src.models.network import ARSNet

    recs = make_synthetic_records(n_structures=2, n_timesteps=6, seed=0)
    ds = ARSDataset(recs, sequence_length=9)
    b = make_collate_fn()([ds[i] for i in range(2)])
    torch.manual_seed(0)
    with torch.no_grad():
        out = ARSNet(design_feature_dim=3).eval()(
            design_features=b["design_features"],
            displacement_cm=b["displacement_cm"],
            rainfall_mm=b["rainfall_mm"],
            time_delta_days=b["time_delta_days"],
            adjacency=b["adjacency"],
            incidence_angle_deg=b["incidence_angle_deg"],
            coherence_mask=b["coherence_mask"],
        )
    assert torch.equal(out.predicted_head_change[:, :4], torch.zeros(2, 4))
    assert torch.equal(out.predicted_displacement_cm[:, :4], torch.zeros(2, 4))
    assert (out.predicted_head_change[:, 4:] != 0).any()
