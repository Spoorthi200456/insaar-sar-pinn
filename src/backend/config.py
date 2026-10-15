"""Centralized, environment-driven backend configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARS_", env_file=".env", extra="ignore")

    app_name: str = "InSAR-PINN ARS Monitor"
    model_version: str = "1.0.0"
    environment: str = "development"

    # Inference artifacts
    onnx_model_path: Path = Path("artifacts/ars_net.onnx")
    torch_checkpoint_path: Path = Path("checkpoints/ars_net.pt")
    sequence_length: int = 52
    design_feature_dim: int = 3

    # Execution provider preference order for ONNX Runtime
    preferred_providers: list[str] = Field(
        default_factory=lambda: [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    )
    enable_fp16: bool = True

    # API
    api_prefix: str = "/api/v1"
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    max_batch_structures: int = 256
    default_recharge_threshold: float = 0.30

    # WebSocket streaming
    ws_max_queue_size: int = 1000
    ws_heartbeat_seconds: int = 30


settings = Settings()
