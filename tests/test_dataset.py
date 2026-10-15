from __future__ import annotations

import numpy as np
import torch

from src.data.dataset import ARSDataset, build_dataloader, make_collate_fn
from src.data.preprocessing import (
    build_adjacency_matrix,
    clamp_displacement,
    clamp_rainfall,
    interpolate_missing_1d,
    time_deltas_days,
)


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
