"""
Author decision (review 5): method C, least squares with no neural network, is
the DEFAULT assessment method of the API and dashboard. The neural network stays
available as an option, clearly labelled EXPERIMENTAL (it ranked unseen
synthetic structures no better than chance in reviews 3-4).
"""

from __future__ import annotations

import json
from dataclasses import replace
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.stats import spearmanr

from experiments.synthetic_recovery import run_recovery as rr
from experiments.synthetic_recovery.simulator import TruthConfig, simulate
from src.backend.inference_engine import InferenceEngine
from src.data.schemas import PredictionRequest, TelemetryBatch
from src.training.train import make_synthetic_records

K = 40


@pytest.fixture(scope="module")
def sim_records():
    sim = simulate(replace(TruthConfig(), seed=3, n_structures=20, n_failing=8))
    return sim, rr.to_records(sim)


def _engine():
    return InferenceEngine(onnx_path=None, torch_checkpoint_path=None, sequence_length=K)


def test_request_defaults_to_the_least_squares_method():
    recs = make_synthetic_records(n_structures=1, n_timesteps=6, seed=1)
    req = PredictionRequest(batch=TelemetryBatch(batch_id=uuid4(), records=recs))
    assert req.method == "ols_baseline"


def test_engine_default_ranks_by_least_squares_without_the_network(sim_records, monkeypatch):
    sim, records = sim_records
    eng = _engine()

    def no_network(*args, **kwargs):
        raise AssertionError("the default method must not run the neural network")

    monkeypatch.setattr(eng, "_run_torch", no_network)
    monkeypatch.setattr(eng, "_run_onnx", no_network)
    preds = eng.predict(records)
    assert all(p.method == "ols_baseline" for p in preds)
    assert all("no neural network" in p.method_note for p in preds)
    # Ranking score = mean recharge volume rate / Q_design (up to S_k/S), as for the
    # network. Truth of that quantity in the simulator:
    truth = sim.recharge[:, 1:].mean(axis=1) * sim.recharge_area_m2 / sim.design_capacity_m3_per_day
    scores = np.array([p.relative_recharge_score for p in preds])
    assert spearmanr(scores, truth).statistic >= 0.9
    ranks = [p.relative_recharge_rank for p in preds]
    assert sorted(ranks) == list(range(1, len(preds) + 1))
    assert ranks[int(np.argmax(scores))] == 1


def test_least_squares_reports_a_ranking_and_no_absolute_numbers(sim_records):
    _, records = sim_records
    for p in _engine().predict(records[:3]):
        assert p.assessment_basis == "relative_ranking_insar_only"
        assert p.recharge_efficiency is None and p.is_underperforming is None
        assert p.transmissivity_m2_per_day is None
        assert p.predicted_recharge_series_m3_per_day is None
        assert p.confidence_interval_m3_per_day is None
        assert len(p.predicted_displacement_cm) == K


def test_fixed_recession_time_is_used_when_configured(sim_records):
    _, records = sim_records
    eng = InferenceEngine(
        onnx_path=None, torch_checkpoint_path=None, sequence_length=K, ols_tau_days=60.0
    )
    assert "60 days" in eng.predict(records[:2])[0].method_note


def test_network_is_still_available_and_labelled_experimental(sim_records):
    _, records = sim_records
    preds = _engine().predict(records[:3], method="neural_experimental")
    for p in preds:
        assert p.method == "neural_experimental"
        assert p.method_note.startswith("EXPERIMENTAL")
        assert p.transmissivity_m2_per_day is not None
        assert p.predicted_recharge_series_m3_per_day is not None


def test_unknown_method_is_rejected():
    recs = make_synthetic_records(n_structures=1, n_timesteps=6, seed=1)
    with pytest.raises(ValueError):
        InferenceEngine(onnx_path=None, torch_checkpoint_path=None, sequence_length=6).predict(
            recs, method="random_forest"
        )


class TestApi:
    @pytest.fixture()
    def client(self):
        from src.backend.api import app

        with TestClient(app) as c:
            yield c

    def _payload(self, **extra):
        records = make_synthetic_records(n_structures=3, n_timesteps=12, seed=7)
        req = PredictionRequest(batch=TelemetryBatch(batch_id=uuid4(), records=records), **extra)
        return json.loads(req.model_dump_json())

    def test_predict_without_a_method_uses_least_squares(self, client):
        payload = self._payload()
        del payload["method"]
        body = client.post("/api/v1/predict", json=payload).json()
        assert {p["method"] for p in body["predictions"]} == {"ols_baseline"}

    def test_predict_can_select_the_experimental_network(self, client):
        body = client.post(
            "/api/v1/predict", json=self._payload(method="neural_experimental")
        ).json()
        assert {p["method"] for p in body["predictions"]} == {"neural_experimental"}
        assert all(p["method_note"].startswith("EXPERIMENTAL") for p in body["predictions"])

    def test_health_names_the_default_and_the_experimental_method(self, client):
        body = client.get("/api/v1/health").json()
        assert body["default_method"] == "ols_baseline"
        assert body["experimental_methods"] == ["neural_experimental"]

    def test_websocket_default_is_least_squares(self, client):
        payload = self._payload()
        del payload["method"]
        with client.websocket_connect("/ws/telemetry") as ws:
            ws.send_text(json.dumps(payload))
            msg = json.loads(ws.receive_text())
        assert {p["method"] for p in msg["predictions"]} == {"ols_baseline"}
