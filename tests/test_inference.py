"""
Review finding 12: the API verdict must compare like with like.

The model's R is a flux [m/day]. The structure's recharge volume rate is
R·A [m³/day] (A = recharge area, finding 6), so
  predicted_recharge_series_m3_per_day = R·A
  recharge_efficiency = mean over REAL steps of R·A / Q_design   [dimensionless]
The network's `hydraulic_conductivity` output is transmissivity T [m²/day].

Since review 5 the network is the optional "neural_experimental" method (the
default is least squares, tests/test_assessment_method.py), so these tests of the
network path select it explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.backend.inference_engine import InferenceEngine
from src.data.schemas import WellHeadObservation
from src.training.train import make_synthetic_records

R_TRUE = 0.004  # m/day
AREA = 2500.0  # m²  -> R·A = 10 m³/day
Q_DESIGN = 40.0  # m³/day -> efficiency 0.25
N_REAL, SEQ = 10, 16  # 6 padded steps


def _records(with_wells: bool = True):
    # Author decision 5: an efficiency percentage is only reported when the
    # structure has well data, so the efficiency tests below use records with
    # wells (one reading per acquisition, absolute head).
    recs = make_synthetic_records(n_structures=2, n_timesteps=N_REAL, seed=11)
    out = []
    for r in recs:
        design = r.design.model_copy(
            update={"recharge_area_m2": AREA, "design_recharge_capacity_m3_per_day": Q_DESIGN}
        )
        wells = []
        if with_wells:
            wells = [
                WellHeadObservation(
                    structure_id=r.design.structure_id,
                    timestamp=o.timestamp,
                    hydraulic_head_m=100.0,
                )
                for o in r.sar_series
            ]
        out.append(r.model_copy(update={"design": design, "well_series": wells}))
    return out


@pytest.fixture()
def engine(monkeypatch):
    eng = InferenceEngine(onnx_path=None, torch_checkpoint_path=None, sequence_length=SEQ)

    def fake_run(batch):
        b, t = batch["displacement_cm"].shape
        r = np.zeros((b, t), dtype=np.float32)
        r[:, t - N_REAL :] = R_TRUE
        r[:, : t - N_REAL] = 0.7  # what an untrained softplus emits on padding
        return {
            "hydraulic_conductivity": np.full(b, 12.0, dtype=np.float32),
            "recharge_rate": r,
            "predicted_head_change": np.zeros((b, t), dtype=np.float32),
            "predicted_displacement_cm": np.zeros((b, t), dtype=np.float32),
        }

    monkeypatch.setattr(eng, "_run_torch", fake_run)
    monkeypatch.setattr(eng, "_mc_dropout_uncertainty", lambda batch, n_samples=12: None)
    return eng


def test_recharge_series_is_volume_rate_on_real_steps(engine):
    pred = engine.predict(_records(), method="neural_experimental")[0]
    series = np.array(pred.predicted_recharge_series_m3_per_day)
    assert series.shape == (N_REAL,)  # padding not reported
    assert np.allclose(series, R_TRUE * AREA, rtol=1e-5)  # 10 m³/day


def test_efficiency_is_dimensionless_ratio_of_volume_rates(engine):
    preds = engine.predict(
        _records(), recharge_efficiency_threshold=0.3, method="neural_experimental"
    )
    for p in preds:
        assert p.recharge_efficiency == pytest.approx(R_TRUE * AREA / Q_DESIGN, rel=1e-5)  # 0.25
        assert p.is_underperforming is True  # 0.25 < 0.3
        lo, hi = p.confidence_interval_m3_per_day
        assert lo <= R_TRUE * AREA <= hi


def test_transmissivity_reported_with_its_units(engine):
    p = engine.predict(_records(), method="neural_experimental")[0]
    assert p.transmissivity_m2_per_day == pytest.approx(12.0)


def test_displacement_series_aligned_with_observations(engine):
    # The dashboard plots predicted_displacement_cm against the record's own
    # sar_series timestamps, so it must have one value per real acquisition.
    recs = _records()
    p = engine.predict(recs, method="neural_experimental")[0]
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
            in_mc.set()  # A is now inside MC dropout, model in train mode
            threading.Event().wait(1.0)  # hold it there long enough for B to collide
        return real_forward(*args, **kwargs)

    model.forward = spy
    records = _records()
    a = threading.Thread(
        target=eng.predict, args=(records,), kwargs={"method": "neural_experimental"}, name="A"
    )
    a.start()
    assert in_mc.wait(10.0)
    b = threading.Thread(
        target=eng.predict, args=(records,), kwargs={"method": "neural_experimental"}, name="B"
    )
    b.start()
    a.join(30.0)
    b.join(30.0)
    # B's first forward call is its point prediction
    b_point = [training for name, training in seen if name == "B"][0]
    assert b_point is False


class TestRelativeRankingWithoutWells:
    """
    Author decision 5. From InSAR alone only R*S_k/S is identifiable (one unknown
    factor shared by every structure), so an absolute efficiency percentage is
    not meaningful; a ranking within the submitted network is. Without well
    data the API reports the ranking and NO percentage, with a one-line note.
    """

    def _scored_engine(self, monkeypatch, r_values):
        eng = InferenceEngine(onnx_path=None, torch_checkpoint_path=None, sequence_length=SEQ)

        def fake_run(batch):
            b, t = batch["displacement_cm"].shape
            r = np.zeros((b, t), dtype=np.float32)
            for i, v in enumerate(r_values):
                r[i, t - N_REAL :] = v
            return {
                "hydraulic_conductivity": np.ones(b, dtype=np.float32),
                "recharge_rate": r,
                "predicted_head_change": np.zeros((b, t), dtype=np.float32),
                "predicted_displacement_cm": np.zeros((b, t), dtype=np.float32),
            }

        monkeypatch.setattr(eng, "_run_torch", fake_run)
        monkeypatch.setattr(eng, "_mc_dropout_uncertainty", lambda batch, n_samples=12: None)
        return eng

    def _three_records(self, with_wells):
        recs = make_synthetic_records(n_structures=3, n_timesteps=N_REAL, seed=11)
        out = []
        for r in recs:
            design = r.design.model_copy(
                update={"recharge_area_m2": AREA, "design_recharge_capacity_m3_per_day": Q_DESIGN}
            )
            wells = (
                [
                    WellHeadObservation(
                        structure_id=r.design.structure_id,
                        timestamp=o.timestamp,
                        hydraulic_head_m=50.0,
                    )
                    for o in r.sar_series
                ]
                if with_wells
                else []
            )
            out.append(r.model_copy(update={"design": design, "well_series": wells}))
        return out

    def test_no_wells_gives_ranking_and_no_percentage(self, monkeypatch):
        eng = self._scored_engine(monkeypatch, [0.002, 0.006, 0.004])
        preds = eng.predict(self._three_records(with_wells=False), method="neural_experimental")
        assert [p.relative_recharge_rank for p in preds] == [3, 1, 2]
        assert [p.relative_recharge_percentile for p in preds] == pytest.approx([0.0, 1.0, 0.5])
        for p in preds:
            assert p.recharge_efficiency is None
            assert p.is_underperforming is None
            assert p.assessment_basis == "relative_ranking_insar_only"
            assert "relative" in p.assessment_note.lower() and "\n" not in p.assessment_note

    def test_ranking_is_unchanged_by_a_common_scale_factor(self, monkeypatch):
        # the unidentifiable global factor (S_k/S) cannot change the ranking
        a = self._scored_engine(monkeypatch, [0.002, 0.006, 0.004]).predict(
            self._three_records(False), method="neural_experimental"
        )
        b = self._scored_engine(monkeypatch, [0.02, 0.06, 0.04]).predict(
            self._three_records(False), method="neural_experimental"
        )
        assert [p.relative_recharge_rank for p in a] == [p.relative_recharge_rank for p in b]

    def test_wells_give_percentage_marked_provisional(self, monkeypatch):
        eng = self._scored_engine(monkeypatch, [0.002, 0.006, 0.004])
        preds = eng.predict(
            self._three_records(with_wells=True),
            recharge_efficiency_threshold=0.3,
            method="neural_experimental",
        )
        assert preds[0].recharge_efficiency == pytest.approx(0.002 * AREA / Q_DESIGN, rel=1e-5)
        assert preds[0].is_underperforming is True  # 0.002*2500/40 = 0.125 < 0.3
        assert preds[1].is_underperforming is False  # 0.006*2500/40 = 0.375 >= 0.3
        for p in preds:
            assert p.assessment_basis == "well_calibrated_provisional"
            assert "storativity" in p.assessment_note.lower()
            assert p.relative_recharge_rank in (1, 2, 3)
