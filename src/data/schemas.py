"""
Strict Pydantic v2 data contracts for the InSAR-PINN ARS monitoring pipeline.

These models are the single source of truth for what "valid telemetry" means
before it is allowed anywhere near a tensor. They encode the bounds and
invariants laid out in Module 3 of the framework:

  * SAR/InSAR line-of-sight displacement is bounded to +/- 10 cm.
  * InSAR coherence must lie in [0, 1]; points with coherence < 0.7 are
    considered unreliable for training (flagged, not silently dropped).
  * Rainfall is bounded to [0, 1000] mm over an aggregation interval.
  * ARS design parameters (height, storage volume, design recharge
    capacity) must be finite and non-negative.
  * All timestamps are timezone-aware UTC.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Optional
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from src.data.preprocessing import COHERENCE_THRESHOLD

# Stored values that are not the true measurement (clamped to the +/-10 cm
# bound, or missing): never reliable, whatever the coherence.
UNRELIABLE_FLAGS = frozenset({"out_of_bounds_clamped", "missing"})


# --------------------------------------------------------------------------- #
# Shared primitives
# --------------------------------------------------------------------------- #

class DataQualityFlag(str, Enum):
    OK = "ok"
    LOW_COHERENCE = "low_coherence"
    MISSING = "missing"
    INTERPOLATED = "interpolated"
    OUT_OF_BOUNDS_CLAMPED = "out_of_bounds_clamped"
    ANOMALY_NO_RAINFALL = "anomaly_no_rainfall"


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class StrictModel(BaseModel):
    """Base class: forbids unknown fields and enforces validation on assignment."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=False,
        str_strip_whitespace=True,
    )


# --------------------------------------------------------------------------- #
# ARS static design parameters (India-WRIS ARS viewer)
# --------------------------------------------------------------------------- #

class ARSDesignParameters(StrictModel):
    structure_id: str = Field(..., min_length=1, max_length=128)
    name: Optional[str] = Field(default=None, max_length=256)
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    height_m: float = Field(..., gt=0.0, le=200.0)
    storage_volume_m3: float = Field(..., gt=0.0)
    catchment_area_km2: float = Field(..., gt=0.0)
    construction_date: Optional[datetime] = None
    design_recharge_capacity_m3_per_day: float = Field(..., gt=0.0)
    recharge_area_m2: Optional[float] = Field(
        default=None, gt=0.0,
        description=(
            "Area through which the structure recharges the aquifer (water-spread / "
            "bed area) [m2]. If absent, approximated as storage_volume_m3 / height_m "
            "(PROVISIONAL, docs/open_questions.md OQ9)."
        ),
    )
    biot_coefficient_alpha: float = Field(
        default=0.6, gt=0.0, le=1.0,
        description=(
            "DEPRECATED, unused by the model (review finding 2). The dimensionless Biot "
            "coefficient cannot map head [m] to displacement [cm]; the model uses a "
            "learned skeletal storage coefficient S_k instead. Kept only so existing "
            "API payloads still validate."
        ),
    )

    @field_validator("construction_date")
    @classmethod
    def _tz_aware(cls, v: Optional[datetime]) -> Optional[datetime]:
        return _utc(v) if v is not None else None

    @property
    def effective_recharge_area_m2(self) -> float:
        """A [m2]: `recharge_area_m2`, else the prismatic-basin estimate V / h (PROVISIONAL)."""
        if self.recharge_area_m2 is not None:
            return self.recharge_area_m2
        return self.storage_volume_m3 / self.height_m

    @field_validator(
        "height_m", "storage_volume_m3", "catchment_area_km2",
        "design_recharge_capacity_m3_per_day",
    )
    @classmethod
    def _finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("value must be finite")
        return v


# --------------------------------------------------------------------------- #
# SAR / InSAR observation (per structure, per timestep)
# --------------------------------------------------------------------------- #

class SARObservation(StrictModel):
    structure_id: str = Field(..., min_length=1, max_length=128)
    timestamp: datetime
    los_displacement_cm: float = Field(
        ...,
        description=(
            "Line-of-sight displacement relative to the first acquisition, "
            "positive = motion toward the satellite (PROVISIONAL, docs/open_questions.md "
            "OQ2; uplift gives positive values). If a product uses the opposite sign, "
            "flip it once here at ingestion, never in the model."
        ),
    )
    incidence_angle_deg: float = Field(
        ..., ge=20.0, le=50.0,
        description=(
            "Radar incidence angle at the structure [deg], from the Sentinel-1 product "
            "metadata (IW mode ~29-46 deg). Required: a silent default would hide "
            "missing metadata (finding-2 plan Q5)."
        ),
    )
    coherence: float = Field(..., ge=0.0, le=1.0)
    track_direction: str = Field(default="descending", pattern="^(ascending|descending)$")
    quality_flag: DataQualityFlag = DataQualityFlag.OK

    @field_validator("timestamp")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        return _utc(v)

    @model_validator(mode="after")
    def _bounds_and_flags(self) -> "SARObservation":
        # Hard physical bound from the data validation contract: |d_i(t)| < 10 cm.
        if abs(self.los_displacement_cm) >= 10.0:
            clamped = math.copysign(9.999, self.los_displacement_cm)
            object.__setattr__(self, "los_displacement_cm", clamped)
            object.__setattr__(self, "quality_flag", DataQualityFlag.OUT_OF_BOUNDS_CLAMPED)
        elif self.coherence < COHERENCE_THRESHOLD and self.quality_flag == DataQualityFlag.OK:
            object.__setattr__(self, "quality_flag", DataQualityFlag.LOW_COHERENCE)
        return self

    @property
    def is_reliable(self) -> bool:
        """Same rule as `ARSDataset`: coherence >= threshold and not clamped/missing."""
        return (
            self.coherence >= COHERENCE_THRESHOLD
            and self.quality_flag.value not in UNRELIABLE_FLAGS
        )


# --------------------------------------------------------------------------- #
# Rainfall / meteorology observation
# --------------------------------------------------------------------------- #

class RainfallObservation(StrictModel):
    structure_id: str = Field(..., min_length=1, max_length=128)
    timestamp: datetime
    rainfall_mm: float = Field(..., ge=0.0)
    quality_flag: DataQualityFlag = DataQualityFlag.OK

    @field_validator("timestamp")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        return _utc(v)

    @field_validator("rainfall_mm")
    @classmethod
    def _clamp_bounds(cls, v: float) -> float:
        # Data validation contract: R_i(t) in [0, 1000] mm per aggregation interval.
        if v > 1000.0:
            return 1000.0
        return v


# --------------------------------------------------------------------------- #
# Optional in-situ groundwater well measurement (boundary/initial conditions)
# --------------------------------------------------------------------------- #

class WellHeadObservation(StrictModel):
    structure_id: str = Field(..., min_length=1, max_length=128)
    timestamp: datetime
    hydraulic_head_m: float
    is_boundary_node: bool = False

    @field_validator("timestamp")
    @classmethod
    def _tz_aware(cls, v: datetime) -> datetime:
        return _utc(v)


# --------------------------------------------------------------------------- #
# Composite telemetry batch as received from ingestion (one structure)
# --------------------------------------------------------------------------- #

class ARSTelemetryRecord(StrictModel):
    design: ARSDesignParameters
    sar_series: list[SARObservation] = Field(..., min_length=1)
    rainfall_series: list[RainfallObservation] = Field(..., min_length=1)
    well_series: list[WellHeadObservation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent_ids(self) -> "ARSTelemetryRecord":
        sid = self.design.structure_id
        for obs in self.sar_series:
            if obs.structure_id != sid:
                raise ValueError(
                    f"SAR observation structure_id={obs.structure_id!r} does not match "
                    f"design.structure_id={sid!r}"
                )
        for obs in self.rainfall_series:
            if obs.structure_id != sid:
                raise ValueError(
                    f"Rainfall observation structure_id={obs.structure_id!r} does not match "
                    f"design.structure_id={sid!r}"
                )
        # One track per series (finding-2 assumption A4): ascending and descending
        # tracks see different LOS directions, so one incidence angle per
        # structure is meaningless if they are mixed.
        tracks = {o.track_direction for o in self.sar_series}
        if len(tracks) > 1:
            raise ValueError(
                f"sar_series for {sid!r} mixes tracks {sorted(tracks)}; submit one track "
                "direction per record"
            )
        return self

    @property
    def coherent_fraction(self) -> float:
        if not self.sar_series:
            return 0.0
        reliable = sum(1 for o in self.sar_series if o.is_reliable)
        return reliable / len(self.sar_series)


class GraphEdge(StrictModel):
    source_id: str
    target_id: str
    weight: float = Field(..., ge=0.0, le=1.0, description="Hydraulic connectivity weight a_ij.")


class TelemetryBatch(StrictModel):
    """Top-level ingestion contract: a batch of ARS structures plus their graph topology."""

    batch_id: UUID
    records: list[ARSTelemetryRecord] = Field(..., min_length=1)
    edges: list[GraphEdge] = Field(default_factory=list)

    @model_validator(mode="after")
    def _edges_reference_known_structures(self) -> "TelemetryBatch":
        known = {r.design.structure_id for r in self.records}
        for e in self.edges:
            if e.source_id not in known or e.target_id not in known:
                raise ValueError(
                    f"Edge ({e.source_id} -> {e.target_id}) references an unknown structure_id"
                )
        return self


# --------------------------------------------------------------------------- #
# Inference API contracts
# --------------------------------------------------------------------------- #

class PredictionRequest(StrictModel):
    batch: TelemetryBatch
    recharge_efficiency_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = 0.30


class StructurePrediction(StrictModel):
    structure_id: str
    hydraulic_conductivity_m_per_day: float
    predicted_recharge_series_m3_per_day: list[float]
    predicted_displacement_cm: list[float]
    recharge_efficiency: float = Field(
        ..., description="mean(Q_recharge) / Q_design, clamped to [0, 5] for reporting."
    )
    is_underperforming: bool
    confidence_interval_m3_per_day: tuple[float, float]

    @model_validator(mode="after")
    def _ci_ordered(self) -> "StructurePrediction":
        lo, hi = self.confidence_interval_m3_per_day
        if lo > hi:
            raise ValueError("confidence interval lower bound exceeds upper bound")
        return self


class PredictionResponse(StrictModel):
    batch_id: UUID
    model_version: str
    predictions: list[StructurePrediction]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExplainabilityResponse(StrictModel):
    structure_id: str
    feature_attributions: dict[str, float]
    latent_drift_score: float = Field(..., ge=0.0)
    pde_residual_norm: float = Field(..., ge=0.0)
