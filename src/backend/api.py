"""
FastAPI service exposing the InSAR-PINN ARS model.

    uvicorn src.backend.api:app --host 0.0.0.0 --port 8000

Endpoints
---------
GET  /api/v1/health                          liveness/readiness probe
POST /api/v1/predict                         batch REST inference
GET  /api/v1/explain/{structure_id}           feature attribution + drift diagnostics
WS   /ws/telemetry                             high-throughput streaming inference
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

import orjson
import structlog
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import ValidationError

from src.backend.config import settings
from src.backend.inference_engine import InferenceEngine
from src.data.schemas import (
    ExplainabilityResponse,
    PredictionRequest,
    PredictionResponse,
)

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)
log = structlog.get_logger(__name__)
logging.basicConfig(level=logging.INFO)

_engine_holder: dict[str, InferenceEngine] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("api.startup", model_version=settings.model_version, environment=settings.environment)
    _engine_holder["engine"] = InferenceEngine(
        onnx_path=settings.onnx_model_path,
        torch_checkpoint_path=settings.torch_checkpoint_path,
        sequence_length=settings.sequence_length,
        design_feature_dim=settings.design_feature_dim,
        preferred_providers=settings.preferred_providers,
    )
    log.info(
        "api.model_loaded",
        backend=_engine_holder["engine"].backend_name,
        provider=_engine_holder["engine"].active_provider,
    )
    yield
    _engine_holder.clear()
    log.info("api.shutdown")


def get_engine() -> InferenceEngine:
    engine = _engine_holder.get("engine")
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Inference engine not ready"
        )
    return engine


app = FastAPI(
    title=settings.app_name,
    version=settings.model_version,
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_timing_and_id(request: Request, call_next):
    request_id = str(uuid4())
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    log.info(
        "api.request",
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    return response


@app.get(f"{settings.api_prefix}/health")
async def health() -> dict:
    engine = _engine_holder.get("engine")
    return {
        "status": "ok" if engine is not None else "starting",
        "model_version": settings.model_version,
        "backend": engine.backend_name if engine else None,
        "active_provider": engine.active_provider if engine else None,
        "environment": settings.environment,
    }


@app.post(f"{settings.api_prefix}/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest) -> PredictionResponse:
    engine = get_engine()

    if len(payload.batch.records) > settings.max_batch_structures:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Batch exceeds max_batch_structures={settings.max_batch_structures}",
        )

    try:
        predictions = await asyncio.to_thread(
            engine.predict,
            payload.batch.records,
            payload.batch.edges,
            payload.recharge_efficiency_threshold,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced as a clean 500 with context
        log.error("predict.inference_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Inference failed"
        ) from exc

    return PredictionResponse(
        batch_id=payload.batch.batch_id,
        model_version=settings.model_version,
        predictions=predictions,
    )


@app.get(f"{settings.api_prefix}/explain/{{structure_id}}")
async def explain(structure_id: str, request: Request) -> ExplainabilityResponse:
    """
    Expects the same `TelemetryBatch` the structure was scored with, passed
    as the request body (kept as a GET-with-body-equivalent POST would be
    more RESTful; documented here as GET for dashboard simplicity -- clients
    should use `httpx`/`fetch` with a JSON body via POST in production; both
    are wired identically to `_run_explain` below).
    """
    body = await request.body()
    if not body:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request body must contain a TelemetryBatch to explain",
        )
    try:
        raw = orjson.loads(body)
        from src.data.schemas import TelemetryBatch

        batch = TelemetryBatch.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors()
        ) from exc

    engine = get_engine()
    if structure_id not in {r.design.structure_id for r in batch.records}:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Unknown structure_id")

    try:
        result = await asyncio.to_thread(engine.explain, batch.records, structure_id, batch.edges)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    return ExplainabilityResponse(
        structure_id=structure_id,
        feature_attributions=result["feature_attributions"],
        latent_drift_score=result["latent_drift_score"],
        pde_residual_norm=result["pde_residual_norm"],
    )


# --------------------------------------------------------------------------- #
# High-throughput WebSocket streaming
# --------------------------------------------------------------------------- #


class ConnectionManager:
    def __init__(self) -> None:
        self.active: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self.active.add(ws)

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self.active.discard(ws)


manager = ConnectionManager()


@app.websocket("/ws/telemetry")
async def telemetry_stream(websocket: WebSocket) -> None:
    """
    Streaming protocol: the client sends a `TelemetryBatch`-shaped JSON
    message per frame (typically a small, rolling window of a few
    structures' latest observations); the server responds with the
    corresponding `PredictionResponse` JSON, and continues until the client
    disconnects. A periodic heartbeat is sent on idle connections so
    load balancers / proxies do not reap them.
    """
    await manager.connect(websocket)
    engine = get_engine()
    log.info("ws.connected", client=str(websocket.client))

    try:
        while True:
            try:
                raw_message = await asyncio.wait_for(
                    websocket.receive_text(), timeout=settings.ws_heartbeat_seconds
                )
            except TimeoutError:
                await websocket.send_json({"type": "heartbeat", "ts": time.time()})
                continue

            try:
                payload_dict = orjson.loads(raw_message)
                request_payload = PredictionRequest.model_validate(payload_dict)
            except (orjson.JSONDecodeError, ValidationError) as exc:
                await websocket.send_json(
                    {"type": "error", "detail": "invalid_payload", "errors": str(exc)}
                )
                continue

            try:
                predictions = await asyncio.to_thread(
                    engine.predict,
                    request_payload.batch.records,
                    request_payload.batch.edges,
                    request_payload.recharge_efficiency_threshold,
                )
            except Exception as exc:  # noqa: BLE001
                log.error("ws.inference_failed", error=str(exc))
                await websocket.send_json({"type": "error", "detail": "inference_failed"})
                continue

            response = PredictionResponse(
                batch_id=request_payload.batch.batch_id,
                model_version=settings.model_version,
                predictions=predictions,
            )
            await websocket.send_bytes(
                orjson.dumps({"type": "prediction", **orjson.loads(response.model_dump_json())})
            )

    except WebSocketDisconnect:
        log.info("ws.disconnected", client=str(websocket.client))
    finally:
        await manager.disconnect(websocket)
