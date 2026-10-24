"""
Review finding 12: the API verdict must compare like with like.

The model's R is a flux [m/day]. The structure's recharge volume rate is
R·A [m³/day] (A = recharge area, finding 6), so
  predicted_recharge_series_m3_per_day = R·A
  recharge_efficiency = mean over REAL steps of R·A / Q_design   [dimensionless]
The network's `hydraulic_conductivity` output is transmissivity T [m²/day].
"""

from __future__ import annotations

import numpy as np
import pytest

from src.backend.inference_engine import InferenceEngine
from src.training.train import make_synthetic_records

R_TRUE = 0.004    # m/day
AREA = 2500.0     # m²  -> R·A = 10 m³/day
Q_DESIGN = 40.0   # m³/day -> efficiency 0.25
N_REAL, SEQ = 10, 16  # 6 padded steps


def _records():
    recs = make_synthetic_records(n_structures=2, n_timesteps=N_REAL, seed=11)
    out = []
    for r in recs:
        design = r.design.model_copy(update={"recharge_area_m2": AREA,
                                             "design_recharge_capacity_m3_per_day": Q_DESIGN})
        out.append(r.model_copy(update={"design": design}))
    return out


@pytest.fixture()
def engine(monkeypatch):
    eng = InferenceEngine(onnx_path=None, torch_checkpoint_path=None, sequence_length=SEQ)

    def fake_run(batch):
        b, t = batch["displacement_cm"].shape
        r = np.zeros((b, t), dtype=np.float32)
        r[:, t - N_REAL:] = R_TRUE
        r[:, : t - N_REAL] = 0.7  # what an untrained softplus emits on padding
        return {"hydraulic_conductivity": np.full(b, 12.0, dtype=np.float32),
                "recharge_rate": r,
                "predicted_head_change": np.zeros((b, t), dtype=np.float32),
                "predicted_displacement_cm": np.zeros((b, t), dtype=np.float32)}

    monkeypatch.setattr(eng, "_run_torch", fake_run)
    monkeypatch.setattr(eng, "_mc_dropout_uncertainty", lambda batch, n_samples=12: None)
    return eng


def test_recharge_series_is_volume_rate_on_real_steps(engine):
    pred = engine.predict(_records())[0]
    series = np.array(pred.predicted_recharge_series_m3_per_day)
    assert series.shape == (N_REAL,)                          # padding not reported
    assert np.allclose(series, R_TRUE * AREA, rtol=1e-5)     # 10 m³/day


def test_efficiency_is_dimensionless_ratio_of_volume_rates(engine):
    preds = engine.predict(_records(), recharge_efficiency_threshold=0.3)
    for p in preds:
        assert p.recharge_efficiency == pytest.approx(R_TRUE * AREA / Q_DESIGN, rel=1e-5)  # 0.25
        assert p.is_underperforming is True                   # 0.25 < 0.3
        lo, hi = p.confidence_interval_m3_per_day
        assert lo <= R_TRUE * AREA <= hi


def test_transmissivity_reported_with_its_units(engine):
    p = engine.predict(_records())[0]
    assert p.transmissivity_m2_per_day == pytest.approx(12.0)


def test_displacement_series_aligned_with_observations(engine):
    # The dashboard plots predicted_displacement_cm against the record's own
    # sar_series timestamps, so it must have one value per real acquisition.
    recs = _records()
    p = engine.predict(recs)[0]
    assert len(p.predicted_displacement_cm) == len(recs[0].sar_series) == N_REAL


def test_concurrent_requests_never_predict_with_dropout_on():
    """
    Review finding 14. /predict runs in worker threads (asyncio.to_thread). The
    MC-dropout pass switches the SHARED model to train mode; a concurrent
    request must never run its point prediction with dropout active.
    """
    import threading

    eng = InferenceEngine(onnx_path=None, torch_checkpoint_path=None, sequence_length=SEQ)
    model = eng._torch_model
    real_forward = model.forward
    in_mc = threading.Event()
    seen = []  # (thread name, model.training) for every forward call

    def spy(*args, **kwargs):
        name = threading.current_thread().name
        seen.append((name, model.training))
        if name == "A" and model.training and not in_mc.is_set():
            in_mc.set()          # A is now inside MC dropout, model in train mode
            threading.Event().wait(1.0)  # hold it there long enough for B to collide
        return real_forward(*args, **kwargs)

    model.forward = spy
    records = _records()
    a = threading.Thread(target=eng.predict, args=(records,), name="A")
    a.start()
    assert in_mc.wait(10.0)
    b = threading.Thread(target=eng.predict, args=(records,), name="B")
    b.start()
    a.join(30.0)
    b.join(30.0)
    b_point = [training for name, training in seen if name == "B"][0]  # B's first call = point prediction
    assert b_point is False
