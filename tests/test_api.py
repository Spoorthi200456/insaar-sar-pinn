from __future__ import annotations

import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.backend.api import app
from src.data.schemas import PredictionRequest, TelemetryBatch
from src.training.train import make_synthetic_records


def _prediction_request_payload(
    n_structures: int = 3, n_timesteps: int = 12, method: str = "ols_baseline"
) -> dict:
    records = make_synthetic_records(n_structures=n_structures, n_timesteps=n_timesteps, seed=7)
    batch = TelemetryBatch(batch_id=uuid4(), records=records, edges=[])
    request = PredictionRequest(batch=batch, recharge_efficiency_threshold=0.3, method=method)
    return json.loads(request.model_dump_json())


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert "backend" in body


class TestPredictEndpoint:
    def test_predict_returns_one_prediction_per_structure(self, client):
        # T and the m3/day series come from the network path (review 5: optional,
        # experimental), so select it explicitly.
        payload = _prediction_request_payload(n_structures=4, method="neural_experimental")
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert len(body["predictions"]) == 4
        for pred in body["predictions"]:
            assert pred["hydraulic_conductivity_m_per_day"] > 0
            assert all(v >= 0 for v in pred["predicted_recharge_series_m3_per_day"])

    def test_predict_rejects_malformed_payload(self, client):
        response = client.post("/api/v1/predict", json={"batch": {"records": []}})
        assert response.status_code == 422

    def test_predict_rejects_unknown_extra_field(self, client):
        payload = _prediction_request_payload(n_structures=1)
        payload["not_a_real_field"] = True
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422

    def test_predict_response_includes_request_id_header(self, client):
        payload = _prediction_request_payload(n_structures=1)
        response = client.post("/api/v1/predict", json=payload)
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time-Ms" in response.headers


class TestExplainEndpoint:
    def test_explain_returns_attributions_for_known_structure(self, client):
        payload = _prediction_request_payload(n_structures=2)
        batch = payload["batch"]
        structure_id = batch["records"][0]["design"]["structure_id"]
        response = client.request(
            "GET", f"/api/v1/explain/{structure_id}", content=json.dumps(batch)
        )
        assert response.status_code == 200
        body = response.json()
        assert body["structure_id"] == structure_id
        assert "design_features_grad_l1" in body["feature_attributions"]

    def test_explain_unknown_structure_returns_404(self, client):
        payload = _prediction_request_payload(n_structures=2)
        batch = payload["batch"]
        response = client.request("GET", "/api/v1/explain/NOT-A-REAL-ID", content=json.dumps(batch))
        assert response.status_code == 404


class TestExplainUsesBatchGraph:
    """Review finding 13: /explain must use the same graph (incl. surveyed edges) as /predict."""

    def test_explain_residual_uses_submitted_edges(self, client, monkeypatch):
        import src.backend.inference_engine as ie

        captured = {}
        real = ie.pde_residual

        def spy(*args, **kwargs):
            captured["adjacency"] = args[5]
            return real(*args, **kwargs)

        monkeypatch.setattr(ie, "pde_residual", spy)
        payload = _prediction_request_payload(n_structures=2)
        batch = payload["batch"]
        a, b = (r["design"]["structure_id"] for r in batch["records"])
        batch["edges"] = [{"source_id": a, "target_id": b, "weight": 1.0}]  # ~tens of km apart
        response = client.request("GET", f"/api/v1/explain/{a}", content=json.dumps(batch))
        assert response.status_code == 200
        adj = captured["adjacency"]
        assert adj.shape == (2, 2)
        assert adj[0, 1].item() > 0 and adj[1, 0].item() > 0  # surveyed edge is present


class TestWebSocketTelemetry:
    """
    Review finding 27: /ws/telemetry had no test. Every message must be a TEXT
    frame of JSON: the dashboard does JSON.parse(event.data), and a binary frame
    arrives in the browser as a Blob, which JSON.parse cannot read.
    """

    def test_prediction_is_a_text_json_frame(self, client):
        payload = _prediction_request_payload(n_structures=3)
        with client.websocket_connect("/ws/telemetry") as ws:
            ws.send_text(json.dumps(payload))
            message = ws.receive()
            assert message.get("text") is not None, "prediction sent as a binary frame"
            body = json.loads(message["text"])
        assert body["type"] == "prediction"
        assert len(body["predictions"]) == 3
        assert body["batch_id"] == payload["batch"]["batch_id"]

    def test_invalid_payload_returns_error_and_keeps_connection(self, client):
        with client.websocket_connect("/ws/telemetry") as ws:
            ws.send_text("{not json")
            err = ws.receive_json()
            assert err["type"] == "error" and err["detail"] == "invalid_payload"
            ws.send_text(json.dumps(_prediction_request_payload(n_structures=1)))
            assert json.loads(ws.receive()["text"])["type"] == "prediction"
