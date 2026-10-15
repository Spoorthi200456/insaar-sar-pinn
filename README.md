# InSAR–PINN Artificial Recharge Structure (ARS) Monitoring Platform

Production implementation of the InSAR–PINN framework for monitoring the
integrity of Artificial Recharge Structures (ARS) by fusing Sentinel-1 InSAR
deformation measurements with a physics-informed neural network (PINN) that
inverts a coupled poroelastic groundwater-flow PDE.

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
python -m src.training.train --config configs/train.yaml
```

## Exporting for optimized inference

```bash
python -m src.backend.export_onnx --checkpoint checkpoints/ars_net.pt \
    --output artifacts/ars_net.onnx --quantize int8
```

`inference_engine.py` prefers a TensorRT execution provider when
`onnxruntime-gpu` + TensorRT libraries are present, falls back to the
CUDA execution provider, then to CPU — with FP16 acceleration on either
GPU path and dynamic INT8 quantization available on CPU.

## Testing

```bash
pytest -q
```

## Docker

```bash
docker build -f docker/Dockerfile -t ars-monitor:latest .
docker compose up
```
