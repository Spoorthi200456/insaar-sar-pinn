"""
Review finding 21. The +/-10 cm data bound must apply to the change between
consecutive acquisitions, not to cumulative displacement (which is relative to
the FIRST acquisition and legitimately exceeds 10 cm in subsiding aquifers over
a multi-year series). An implausible single-interval jump is flagged
STEP_OUT_OF_BOUNDS (value kept, never used as a target); steady long-term
motion is kept intact. See docs/math/data_quality_bounds.md.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from src.data.dataset import ARSDataset
from src.data.schemas import (
    ARSDesignParameters,
    ARSTelemetryRecord,
    DataQualityFlag,
    RainfallObservation,
    SARObservation,
)

T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _record(values_cm, coherence=0.9):
    design = ARSDesignParameters(
        structure_id="S",
        latitude=28.6,
        longitude=77.2,
        height_m=4.0,
        storage_volume_m3=4000.0,
        catchment_area_km2=2.0,
        design_recharge_capacity_m3_per_day=50.0,
    )
    sar = [
        SARObservation(
            structure_id="S",
            timestamp=T0 + timedelta(days=12 * k),
            los_displacement_cm=float(v),
            coherence=coherence,
            incidence_angle_deg=38.0,
        )
        for k, v in enumerate(values_cm)
    ]
    rain = [RainfallObservation(structure_id="S", timestamp=T0, rainfall_mm=0.0)]
    return ARSTelemetryRecord(design=design, sar_series=sar, rainfall_series=rain)


def test_long_steady_subsidence_beyond_10_cm_is_kept():
    values = -0.6 * np.arange(60)  # 0.6 cm per 12 days -> -35.4 cm after ~2 years
    rec = _record(values)
    assert all(o.quality_flag == DataQualityFlag.OK for o in rec.sar_series)
    assert all(o.is_reliable for o in rec.sar_series)
    assert [o.los_displacement_cm for o in rec.sar_series] == pytest.approx(list(values))
    sample = ARSDataset([rec], sequence_length=60)[0]
    assert sample.coherence_mask.all()
    assert np.allclose(sample.displacement_cm.numpy(), values, atol=1e-5)


def test_single_interval_jump_is_flagged_and_not_a_target():
    values = [0.0, 0.1, -0.1, 0.2, 0.0, 12.5, 0.1, 0.2]  # one 12.5 cm spike in 12 days
    rec = _record(values)
    flags = [o.quality_flag for o in sorted(rec.sar_series, key=lambda o: o.timestamp)]
    assert flags[5] == DataQualityFlag.STEP_OUT_OF_BOUNDS
    assert all(f == DataQualityFlag.OK for k, f in enumerate(flags) if k != 5)
    assert rec.sar_series[5].los_displacement_cm == pytest.approx(12.5)  # value kept, not clamped
    sample = ARSDataset([rec], sequence_length=8)[0]
    assert sample.coherence_mask.tolist() == [True] * 5 + [False] + [True] * 2


def test_dataset_applies_the_bound_even_to_unvalidated_records():
    rec = _record([0.0, 0.1, 0.2, 0.3])
    sar = sorted(rec.sar_series, key=lambda o: o.timestamp)
    sar[2] = sar[2].model_copy(update={"los_displacement_cm": -14.0})  # bypasses validators
    rec = rec.model_copy(update={"sar_series": sar})
    sample = ARSDataset([rec], sequence_length=4)[0]
    assert sample.coherence_mask.tolist() == [True, True, False, True]
