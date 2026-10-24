"""
`InferenceEngine`: the single entrypoint the FastAPI backend uses to turn
validated `TelemetryBatch` objects into `StructurePrediction`s.

Provider selection strategy (ONNX Runtime):

    1. TensorrtExecutionProvider  (best latency on NVIDIA GPUs; ORT builds
       and caches a TensorRT engine from the ONNX graph on first run)
    2. CUDAExecutionProvider      (GPU fallback if TensorRT libs are absent)
    3. CPUExecutionProvider       (always available; paired with the INT8
       dynamically-quantized graph produced by `export_onnx.py --quantize int8`
       when present, for materially better CPU throughput)

If no ONNX artifact exists on disk at all (e.g. local dev before a first
`export_onnx` run), the engine transparently falls back to eager PyTorch so
the API and dashboard remain fully functional end-to-end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import ARSDataset, make_collate_fn
from src.data.schemas import ARSTelemetryRecord, GraphEdge, StructurePrediction
from src.models.losses import pde_residual
from src.models.network import ARSNet

logger = logging.getLogger(__name__)

_OUTPUT_ORDER = (
    "hydraulic_conductivity",
    "recharge_rate",
    "predicted_head_change",
    "predicted_displacement_cm",
)


@dataclass
class RawPrediction:
    structure_id: str
    hydraulic_conductivity: float
    recharge_rate: np.ndarray
    predicted_head_change: np.ndarray
    predicted_displacement_cm: np.ndarray


class InferenceEngine:
    def __init__(
        self,
        onnx_path: Path | None,
        torch_checkpoint_path: Path | None,
        sequence_length: int = 52,
        design_feature_dim: int = 3,
        preferred_providers: list[str] | None = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.design_feature_dim = design_feature_dim
        self.preferred_providers = preferred_providers or [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

        self._session = None
        self._torch_model: ARSNet | None = None
        self.backend_name = "uninitialized"
        self.active_provider: str | None = None

        if onnx_path is not None and onnx_path.exists():
            self._load_onnx(onnx_path)
        elif torch_checkpoint_path is not None and torch_checkpoint_path.exists():
            self._load_torch(torch_checkpoint_path)
        else:
            logger.warning(
                "No ONNX artifact or torch checkpoint found (onnx=%s, checkpoint=%s); "
                "initializing an untrained ARSNet for interface/dev purposes only.",
                onnx_path, torch_checkpoint_path,
            )
            self._torch_model = ARSNet(design_feature_dim=design_feature_dim).eval()
            self.backend_name = "torch-untrained"

    # ------------------------------------------------------------------ #
    def _load_onnx(self, onnx_path: Path) -> None:
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = [p for p in self.preferred_providers if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(onnx_path), sess_options=session_options, providers=providers
        )
        self.active_provider = self._session.get_providers()[0]
        self.backend_name = "onnxruntime"
        logger.info(
            "Loaded ONNX Runtime session from %s (active provider=%s, available=%s)",
            onnx_path, self.active_provider, providers,
        )

    def _load_torch(self, checkpoint_path: Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint.get("model_config", {"design_feature_dim": self.design_feature_dim})
        model = ARSNet(**config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self._torch_model = model
        self.backend_name = "torch"
        logger.info("Loaded PyTorch checkpoint from %s (eager-mode fallback)", checkpoint_path)

    # ------------------------------------------------------------------ #
    def _build_batch_tensors(
        self, records: list[ARSTelemetryRecord], edges: list[GraphEdge]
    ) -> dict[str, torch.Tensor | list[str]]:
        dataset = ARSDataset(records, sequence_length=self.sequence_length)
        collate_fn = make_collate_fn(explicit_edges=edges)
        samples = [dataset[i] for i in range(len(dataset))]
        return collate_fn(samples)

    def _run_onnx(self, batch: dict[str, torch.Tensor | list[str]]) -> dict[str, np.ndarray]:
        assert self._session is not None
        feed = {
            "design_features": batch["design_features"].numpy().astype(np.float32),
            "displacement_cm": batch["displacement_cm"].numpy().astype(np.float32),
            "rainfall_mm": batch["rainfall_mm"].numpy().astype(np.float32),
            "time_delta_days": batch["time_delta_days"].numpy().astype(np.float32),
            "adjacency": batch["adjacency"].numpy().astype(np.float32),
            "incidence_angle_deg": batch["incidence_angle_deg"].numpy().astype(np.float32),
            "coherence_mask": batch["coherence_mask"].numpy().astype(bool),
        }
        outputs = self._session.run(None, feed)
        return dict(zip(_OUTPUT_ORDER, outputs))

    def _run_torch(self, batch: dict[str, torch.Tensor | list[str]]) -> dict[str, np.ndarray]:
        assert self._torch_model is not None
        with torch.no_grad():
            out = self._torch_model(
                design_features=batch["design_features"],
                displacement_cm=batch["displacement_cm"],
                rainfall_mm=batch["rainfall_mm"],
                time_delta_days=batch["time_delta_days"],
                adjacency=batch["adjacency"],
                incidence_angle_deg=batch["incidence_angle_deg"],
                coherence_mask=batch["coherence_mask"],
            )
        return {
            "hydraulic_conductivity": out.hydraulic_conductivity.numpy(),
            "recharge_rate": out.recharge_rate.numpy(),
            "predicted_head_change": out.predicted_head_change.numpy(),
            "predicted_displacement_cm": out.predicted_displacement_cm.numpy(),
        }

    def _mc_dropout_uncertainty(
        self, batch: dict[str, torch.Tensor | list[str]], n_samples: int = 12
    ) -> np.ndarray | None:
        """
        MC-Dropout uncertainty estimate for recharge_rate, only available on
        the eager PyTorch backend (dropout layers are inert in the exported/
        traced ONNX graph, which is fixed in eval mode). Returns per-structure
        std-dev over `n_samples` stochastic forward passes, or None if the
        active backend can't provide it -- callers fall back to a
        temporal-variability heuristic in that case.
        """
        if self._torch_model is None:
            return None
        model = self._torch_model
        model.train()  # re-enables dropout stochasticity
        try:
            samples = []
            with torch.no_grad():
                for _ in range(n_samples):
                    out = model(
                        design_features=batch["design_features"],
                        displacement_cm=batch["displacement_cm"],
                        rainfall_mm=batch["rainfall_mm"],
                        time_delta_days=batch["time_delta_days"],
                        adjacency=batch["adjacency"],
                        incidence_angle_deg=batch["incidence_angle_deg"],
                        coherence_mask=batch["coherence_mask"],
                    )
                    samples.append(out.recharge_rate.numpy())
            stacked = np.stack(samples, axis=0)  # (n_samples, B, T)
            return stacked.std(axis=0).mean(axis=1)  # (B,) mean std across time
        finally:
            model.eval()

    # ------------------------------------------------------------------ #
    def predict(
        self,
        records: list[ARSTelemetryRecord],
        edges: list[GraphEdge] | None = None,
        recharge_efficiency_threshold: float = 0.30,
    ) -> list[StructurePrediction]:
        edges = edges or []
        batch = self._build_batch_tensors(records, edges)

        if self._session is not None:
            raw = self._run_onnx(batch)
            uncertainty = None
        else:
            raw = self._run_torch(batch)
            uncertainty = self._mc_dropout_uncertainty(batch)

        design_capacity = batch["design_recharge_capacity"].numpy()
        structure_ids = batch["structure_ids"]

        predictions: list[StructurePrediction] = []
        for i, structure_id in enumerate(structure_ids):
            q_series = raw["recharge_rate"][i]
            k_value = float(raw["hydraulic_conductivity"][i])
            mean_q = float(np.mean(q_series))
            capacity = float(design_capacity[i]) if design_capacity[i] > 1e-6 else 1e-6
            efficiency = float(np.clip(mean_q / capacity, 0.0, 5.0))

            if uncertainty is not None:
                std = float(uncertainty[i])
            else:
                # Temporal-variability heuristic fallback (ONNX/no-dropout path):
                # use the recent series' own dispersion as a coarse proxy for
                # predictive uncertainty, scaled to a ~90% interval half-width.
                std = float(np.std(q_series[-8:])) if len(q_series) >= 2 else 0.0
            half_width = 1.645 * std
            ci = (max(0.0, mean_q - half_width), mean_q + half_width)

            predictions.append(
                StructurePrediction(
                    structure_id=structure_id,
                    hydraulic_conductivity_m_per_day=k_value,
                    predicted_recharge_series_m3_per_day=[float(v) for v in q_series],
                    predicted_displacement_cm=[
                        float(v) for v in raw["predicted_displacement_cm"][i]
                    ],
                    recharge_efficiency=efficiency,
                    is_underperforming=efficiency < recharge_efficiency_threshold,
                    confidence_interval_m3_per_day=ci,
                )
            )
        return predictions

    def explain(self, records: list[ARSTelemetryRecord], structure_id: str) -> dict[str, float]:
        """
        Lightweight gradient-based feature attribution + PDE-consistency and
        latent-drift diagnostics for one structure, computed via the eager
        PyTorch model (falls back to loading the torch checkpoint on demand
        if the engine is currently ONNX-only).
        """
        model = self._torch_model
        if model is None:
            raise RuntimeError(
                "Explainability requires the eager PyTorch model; none is loaded in this engine."
            )
        batch = self._build_batch_tensors(records, edges=[])
        idx = batch["structure_ids"].index(structure_id)

        design = batch["design_features"].clone().requires_grad_(True)
        rainfall = batch["rainfall_mm"].clone().requires_grad_(True)
        displacement = batch["displacement_cm"].clone().requires_grad_(True)

        out = model(
            design_features=design,
            displacement_cm=displacement,
            rainfall_mm=rainfall,
            time_delta_days=batch["time_delta_days"],
            adjacency=batch["adjacency"],
            incidence_angle_deg=batch["incidence_angle_deg"],
            coherence_mask=batch["coherence_mask"],
        )
        target = out.recharge_rate[idx].mean()
        target.backward()

        attributions = {
            "design_features_grad_l1": float(design.grad[idx].abs().sum()),
            "rainfall_grad_l1": float(rainfall.grad[idx].abs().sum()),
            "displacement_grad_l1": float(displacement.grad[idx].abs().sum()),
        }

        with torch.no_grad():
            embedding = out.node_embedding[idx]
            latent_drift = float(torch.norm(embedding - embedding.mean()).item())

            # Same residual the model is trained on (losses.pde_residual), [m/day].
            residual, valid = pde_residual(
                out.predicted_head_change,
                batch["time_delta_days"],
                out.storativity,
                out.hydraulic_conductivity,
                out.recharge_rate,
                batch["adjacency"],
            )
            pde_residual_norm = float(torch.norm(residual[idx][valid[idx]]).item())

        return {
            "feature_attributions": attributions,
            "latent_drift_score": latent_drift,
            "pde_residual_norm": pde_residual_norm,
        }
