# InSAR–PINN Artificial Recharge Structure (ARS) Monitoring Platform

Research prototype of an InSAR–PINN framework for monitoring Artificial
Recharge Structures (ARS): it fuses Sentinel-1 InSAR line-of-sight
displacement with a physics-informed neural network constrained by a
depth-integrated groundwater-flow equation (with leakage) and an elastic
head-to-displacement coupling.

> **Status: research prototype, synthetic data only.** No real Sentinel-1,
> rainfall or well data has been run through it, several physical constants
> are provisional (`docs/open_questions.md`), and its outputs are **not** an
> assessment of any real structure. See `docs/STATUS.md` and
> `docs/REVIEW_2_SUMMARY.md` for what is and is not proven.

## Repository layout

```
insar-pinn-ars/
├── src/
│   ├── data/                  # Pydantic v2 data contracts + PyTorch Dataset/DataLoader
│   │   ├── schemas.py
│   │   ├── preprocessing.py
│   │   └── dataset.py
│   ├── models/                # Custom architecture + composite physics loss
│   │   ├── layers.py
│   │   ├── network.py
│   │   └── losses.py
│   ├── training/
│   │   └── train.py
│   └── backend/                # Export + optimized inference + FastAPI service
│       ├── config.py
│       ├── export_onnx.py
│       ├── inference_engine.py
│       └── api.py
├── frontend/                   # Next.js 14 / React / TailwindCSS analytics dashboard
│   ├── app/
│   ├── components/
│   └── lib/
├── tests/                      # pytest suite (schemas, dataset, model, loss, API contract)
├── docker/Dockerfile           # Multi-stage, security-hardened production image
├── docker-compose.yml
└── .github/workflows/deploy.yml
```

## Quickstart (backend)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.backend.api:app --host 0.0.0.0 --port 8000 --reload
```

REST inference: `POST http://localhost:8000/api/v1/predict`
Streaming telemetry: `ws://localhost:8000/ws/telemetry`
Health: `GET http://localhost:8000/api/v1/health`
Model explainability: `GET http://localhost:8000/api/v1/explain/{structure_id}`

## Quickstart (frontend)

```bash
cd frontend
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_BASE_URL` and `NEXT_PUBLIC_MAPBOX_TOKEN` in
`frontend/.env.local` before running.

## Training

```bash
# synthetic smoke run (no real data needed)
python -m src.training.train --synthetic 24 --epochs 50
# your own records: a JSON list of ARSTelemetryRecord objects (src/data/schemas.py)
python -m src.training.train --data path/to/records.json --epochs 50
```

## Reproducible demo and known-answer test (synthetic)

```bash
python scripts/demo_synthetic.py                       # loss curves -> docs/demo/
python experiments/synthetic_recovery/run_recovery.py  # recovery report -> experiments/synthetic_recovery/
```

## Physics and maths

Governing equations and units: `CLAUDE.md` → "Governing physics" and
"Units table". Step-by-step derivations: `docs/math/`. Review findings:
`docs/review.md`.

## Exporting for optimized inference

```bash
python -m src.backend.export_onnx --checkpoint checkpoints/ars_net.pt \
    --output artifacts/ars_net.onnx --quantize int8
```

`inference_engine.py` prefers a TensorRT execution provider when
`onnxruntime-gpu` + TensorRT libraries are present, falls back to the
CUDA execution provider, then to CPU. FP16 and INT8 paths exist but only the
FP32 CPU round trip is tested (`tests/test_export_onnx.py`).

## Testing

```bash
pytest -q
```

## Docker

```bash
docker build -f docker/Dockerfile -t ars-monitor:latest .
docker compose up
```
