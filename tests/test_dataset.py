from __future__ import annotations

import numpy as np
import torch

from src.data.dataset import ARSDataset, build_dataloader, make_collate_fn
from src.data.preprocessing import (
    aggregate_rainfall_to_insar_cadence,
    build_adjacency_matrix,
    clamp_displacement,
    clamp_rainfall,
    interpolate_missing_1d,
    time_deltas_days,
)
from src.data.schemas import DataQualityFlag, SARObservation


class TestPreprocessing:
    def test_interpolate_missing_fills_gaps(self):
        values = np.array([1.0, 0.0, 3.0, 0.0, 5.0])
        mask = np.array([True, False, True, False, True])
        filled, was_interp = interpolate_missing_1d(values, mask)
        assert filled[1] == 2.0
        assert filled[3] == 4.0
        assert was_interp[1] and was_interp[3]
        assert not was_interp[0]

    def test_interpolate_all_missing_returns_zeros_and_flags(self):
        values = np.zeros(4)
        mask = np.zeros(4, dtype=bool)
        filled, was_interp = interpolate_missing_1d(values, mask)
        assert np.allclose(filled, 0.0)
        assert was_interp.all()

    def test_clamp_displacement_enforces_bound(self):
        clamped, was_clamped = clamp_displacement(np.array([15.0, -20.0, 3.0]))
        assert np.all(np.abs(clamped) < 10.0)
        assert was_clamped[0] and was_clamped[1] and not was_clamped[2]

    def test_clamp_rainfall_enforces_bound(self):
        clamped, was_clamped = clamp_rainfall(np.array([5000.0, -1.0, 50.0]))
        assert clamped[0] == 1000.0
        assert clamped[1] == 0.0
        assert was_clamped[0]

    def test_time_deltas_days_monotonic(self):
        ts = np.array([0, 7 * 86400, 21 * 86400], dtype=np.float64)
        deltas = time_deltas_days(ts)
        assert deltas.shape == (3,)
        assert deltas[1] == 7.0
        assert deltas[2] == 14.0

    def test_adjacency_matrix_is_symmetric_and_self_free(self):
        coords = np.array([[12.97, 77.59], [12.98, 77.60], [20.0, 80.0]])
        adj = build_adjacency_matrix(coords, connectivity_radius_km=10.0)
        assert np.allclose(adj, adj.T)
        assert np.allclose(np.diag(adj), 0.0)
        assert adj[0, 2] == 0.0  # far apart -> no edge

    def test_rainfall_binned_causally_and_rain_after_last_acquisition_dropped(self):
        # Acquisitions at t = 0, 10, 20; step k must get rain in (t_{k-1}, t_k].
        insar_ts = np.array([0.0, 10.0, 20.0])
        rain_ts = np.array([0.0, 5.0, 10.0, 11.0, 25.0])
        rain_mm = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        binned = aggregate_rainfall_to_insar_cadence(rain_ts, rain_mm, insar_ts)
        # t=0 -> step 0; t=5,10 -> step 1 (t=10 is ON acquisition 1, so it counts
        # for it); t=11 -> step 2 (it fell after acquisition 1, so it cannot
        # have caused the displacement observed at acquisition 1).
        assert np.allclose(binned, [1.0, 5.0, 4.0])
        # Rain at t=25 fell after the last acquisition: it is DROPPED (not
        # flagged, not returned) and must NOT be added to the last bin.
        assert binned[-1] == 4.0
        assert binned.sum() == rain_mm.sum() - 5.0


class TestARSDataset:
    def test_item_shapes(self, synthetic_records):
        dataset = ARSDataset(synthetic_records, sequence_length=16)
        sample = dataset[0]
        assert sample.displacement_cm.shape == (16,)
        assert sample.rainfall_mm.shape == (16,)
        assert sample.time_delta_days.shape == (16,)
        assert sample.coherence_mask.shape == (16,)
        assert sample.design_features.shape == (3,)

    def test_left_padding_when_series_shorter_than_sequence_length(self, synthetic_records):
        dataset = ARSDataset(synthetic_records, sequence_length=64)
        sample = dataset[0]
        assert sample.displacement_cm.shape == (64,)
        # first entries should be padding -> masked False
        assert sample.coherence_mask[0].item() is False

    def test_clamped_observation_is_not_a_reliable_target(self, small_records):
        record = small_records[0]
        sar = sorted(record.sar_series, key=lambda o: o.timestamp)
        k = 3
        # A real -14 cm event at high coherence: the schema stores -9.999 and
        # flags it OUT_OF_BOUNDS_CLAMPED, so it must not be fitted as truth.
        sar[k] = SARObservation(
            structure_id=sar[k].structure_id,
            timestamp=sar[k].timestamp,
            los_displacement_cm=-14.0,
            coherence=0.9,
        )
        assert sar[k].quality_flag == DataQualityFlag.OUT_OF_BOUNDS_CLAMPED
        record = record.model_copy(update={"sar_series": sar})

        sample = ARSDataset([record], sequence_length=len(sar))[0]
        assert sample.coherence_mask[k].item() is False

    def test_len_matches_records(self, synthetic_records):
        dataset = ARSDataset(synthetic_records, sequence_length=16)
        assert len(dataset) == len(synthetic_records)


class TestCollateAndDataLoader:
    def test_collate_builds_adjacency(self, synthetic_records):
        dataset = ARSDataset(synthetic_records, sequence_length=16)
        samples = [dataset[i] for i in range(len(dataset))]
        collate_fn = make_collate_fn()
        batch = collate_fn(samples)
        n = len(samples)
        assert batch["adjacency"].shape == (n, n)
        assert torch.allclose(batch["adjacency"], batch["adjacency"].T)

    def test_dataloader_end_to_end(self, synthetic_records):
        loader = build_dataloader(
            synthetic_records, batch_size=4, sequence_length=16, shuffle=False, num_workers=0
        )
        batch = next(iter(loader))
        assert batch["displacement_cm"].shape[1] == 16
        assert batch["adjacency"].shape[0] == batch["displacement_cm"].shape[0]
