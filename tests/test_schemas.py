from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from src.data.schemas import (
    ARSDesignParameters,
    ARSTelemetryRecord,
    DataQualityFlag,
    GraphEdge,
    RainfallObservation,
    SARObservation,
    TelemetryBatch,
)


def _design(**overrides) -> ARSDesignParameters:
    base = dict(
        structure_id="ARS-0001",
        latitude=12.97,
        longitude=77.59,
        height_m=5.0,
        storage_volume_m3=1000.0,
        catchment_area_km2=2.0,
        design_recharge_capacity_m3_per_day=100.0,
    )
    base.update(overrides)
    return ARSDesignParameters(**base)


class TestARSDesignParameters:
    def test_valid_construction(self):
        d = _design()
        assert d.structure_id == "ARS-0001"
        assert d.biot_coefficient_alpha == pytest.approx(0.6)

    def test_rejects_negative_height(self):
        with pytest.raises(ValidationError):
            _design(height_m=-1.0)

    def test_rejects_out_of_range_latitude(self):
        with pytest.raises(ValidationError):
            _design(latitude=95.0)

    def test_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            ARSDesignParameters(**{**_design().model_dump(), "extra_field": 1})


class TestSARObservation:
    def test_flags_out_of_bounds_interval_change(self):
        # Review finding 21: was test_clamps_out_of_bounds_displacement, which
        # asserted the cumulative +/-10 cm clamp (the bug). The bound now applies
        # to the change between acquisitions: a 15 cm jump is flagged (and kept,
        # not clamped); a 15 cm cumulative value on its own is legitimate.
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        single = SARObservation(
            structure_id="ARS-0001",
            timestamp=t0,
            los_displacement_cm=15.0,
            coherence=0.9,
            incidence_angle_deg=35.0,
        )
        assert single.los_displacement_cm == 15.0
        assert single.quality_flag == DataQualityFlag.OK
        sar = [
            SARObservation(
                structure_id="ARS-0001",
                timestamp=t0,
                los_displacement_cm=0.1,
                coherence=0.9,
                incidence_angle_deg=35.0,
            ),
            SARObservation(
                structure_id="ARS-0001",
                timestamp=t0 + timedelta(days=12),
                los_displacement_cm=15.1,
                coherence=0.9,
                incidence_angle_deg=35.0,
            ),
        ]
        rain = [RainfallObservation(structure_id="ARS-0001", timestamp=t0, rainfall_mm=1.0)]
        rec = ARSTelemetryRecord(design=_design(), sar_series=sar, rainfall_series=rain)
        assert rec.sar_series[1].quality_flag == DataQualityFlag.STEP_OUT_OF_BOUNDS
        assert rec.sar_series[1].los_displacement_cm == 15.1
        assert rec.sar_series[1].is_reliable is False

    def test_flags_low_coherence(self):
        obs = SARObservation(
            structure_id="ARS-0001",
            timestamp=datetime.now(UTC),
            los_displacement_cm=0.5,
            coherence=0.3,
            incidence_angle_deg=35.0,
        )
        assert obs.quality_flag == DataQualityFlag.LOW_COHERENCE
        assert obs.is_reliable is False

    def test_naive_timestamp_becomes_utc(self):
        naive = datetime(2024, 1, 1, 12, 0, 0)
        obs = SARObservation(
            structure_id="ARS-0001",
            timestamp=naive,
            los_displacement_cm=0.1,
            coherence=0.9,
            incidence_angle_deg=35.0,
        )
        assert obs.timestamp.tzinfo is not None

    def test_rejects_coherence_out_of_unit_interval(self):
        with pytest.raises(ValidationError):
            SARObservation(
                structure_id="ARS-0001",
                timestamp=datetime.now(UTC),
                los_displacement_cm=0.1,
                coherence=1.5,
                incidence_angle_deg=35.0,
            )


class TestRainfallObservation:
    def test_clamps_extreme_rainfall(self):
        obs = RainfallObservation(
            structure_id="ARS-0001", timestamp=datetime.now(UTC), rainfall_mm=5000.0
        )
        assert obs.rainfall_mm == 1000.0

    def test_rejects_negative_rainfall(self):
        with pytest.raises(ValidationError):
            RainfallObservation(
                structure_id="ARS-0001", timestamp=datetime.now(UTC), rainfall_mm=-1.0
            )


class TestARSTelemetryRecord:
    def test_rejects_mismatched_structure_ids(self):
        design = _design()
        bad_sar = SARObservation(
            structure_id="ARS-9999",
            timestamp=datetime.now(UTC),
            los_displacement_cm=0.1,
            coherence=0.9,
            incidence_angle_deg=35.0,
        )
        rain = RainfallObservation(
            structure_id="ARS-0001", timestamp=datetime.now(UTC), rainfall_mm=10.0
        )
        with pytest.raises(ValidationError):
            ARSTelemetryRecord(design=design, sar_series=[bad_sar], rainfall_series=[rain])

    def test_coherent_fraction(self):
        design = _design()
        ts = datetime.now(UTC)
        sar = [
            SARObservation(
                structure_id="ARS-0001",
                timestamp=ts,
                los_displacement_cm=0.1,
                coherence=0.9,
                incidence_angle_deg=35.0,
            ),
            SARObservation(
                structure_id="ARS-0001",
                timestamp=ts,
                los_displacement_cm=0.1,
                coherence=0.2,
                incidence_angle_deg=35.0,
            ),
        ]
        rain = [RainfallObservation(structure_id="ARS-0001", timestamp=ts, rainfall_mm=10.0)]
        record = ARSTelemetryRecord(design=design, sar_series=sar, rainfall_series=rain)
        assert record.coherent_fraction == pytest.approx(0.5)


class TestTelemetryBatch:
    def test_rejects_edge_referencing_unknown_structure(self):
        design = _design()
        ts = datetime.now(UTC)
        record = ARSTelemetryRecord(
            design=design,
            sar_series=[
                SARObservation(
                    structure_id="ARS-0001",
                    timestamp=ts,
                    los_displacement_cm=0.1,
                    coherence=0.9,
                    incidence_angle_deg=35.0,
                )
            ],
            rainfall_series=[
                RainfallObservation(structure_id="ARS-0001", timestamp=ts, rainfall_mm=10.0)
            ],
        )
        with pytest.raises(ValidationError):
            TelemetryBatch(
                batch_id=uuid4(),
                records=[record],
                edges=[GraphEdge(source_id="ARS-0001", target_id="ARS-9999", weight=0.5)],
            )


class TestIsReliableMatchesDatasetRule:
    """Same rule as the training dataset (review finding 10 / review-2 item)."""

    def _obs(self, d, coh):
        return SARObservation(
            structure_id="ARS-0001",
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            los_displacement_cm=d,
            coherence=coh,
            incidence_angle_deg=35.0,
        )

    def test_out_of_bounds_jump_at_high_coherence_is_not_reliable(self):
        # Review finding 21: was test_clamped_high_coherence_observation_is_not_reliable
        # (cumulative clamp). Same guarantee under the per-interval rule: a
        # -14 cm jump at coherence 0.95 is flagged and never reliable.
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        first = self._obs(0.1, 0.95)
        jump = self._obs(-14.0, 0.95).model_copy(update={"timestamp": ts + timedelta(days=12)})
        rec = ARSTelemetryRecord(
            design=_design(),
            sar_series=[first, jump],
            rainfall_series=[
                RainfallObservation(structure_id="ARS-0001", timestamp=ts, rainfall_mm=1.0)
            ],
        )
        obs = rec.sar_series[1]
        assert obs.quality_flag == DataQualityFlag.STEP_OUT_OF_BOUNDS
        assert obs.los_displacement_cm == -14.0  # kept, not clamped
        assert obs.is_reliable is False

    def test_coherent_fraction_excludes_clamped(self):
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        rec = ARSTelemetryRecord(
            design=_design(),
            sar_series=[self._obs(0.1, 0.9), self._obs(-14.0, 0.95)],
            rainfall_series=[
                RainfallObservation(structure_id="ARS-0001", timestamp=ts, rainfall_mm=1.0)
            ],
        )
        assert rec.coherent_fraction == pytest.approx(0.5)

    def test_threshold_is_the_shared_constant(self):
        from src.data.preprocessing import COHERENCE_THRESHOLD

        assert self._obs(0.1, COHERENCE_THRESHOLD).is_reliable is True
        below = self._obs(0.1, COHERENCE_THRESHOLD - 1e-6)
        assert below.is_reliable is False
        assert below.quality_flag == DataQualityFlag.LOW_COHERENCE
