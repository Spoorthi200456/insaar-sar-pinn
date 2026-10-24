"""
Multi-modal data harmonization utilities (Module 3 of the blueprint).

Pure-numpy functions so they are trivially unit-testable without a torch
dependency, then wrapped into tensors by `dataset.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COHERENCE_THRESHOLD = 0.7
MAX_ABS_DISPLACEMENT_CM = 10.0
MAX_RAINFALL_MM = 1000.0
EARTH_RADIUS_M = 6_371_000.0
# Review finding 21: the +/-10 cm data bound applies to the LOS change between
# an acquisition and the last accepted one, not to cumulative displacement
# (which is relative to the first acquisition and legitimately exceeds 10 cm in
# subsiding aquifers). PROVISIONAL value, docs/open_questions.md OQ17.
MAX_INTERVAL_CHANGE_CM = 10.0


@dataclass(frozen=True)
class NormalizationStats:
    mean: float
    std: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        std = self.std if self.std > 1e-8 else 1.0
        return (x - self.mean) / std

    def invert(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std if self.std > 1e-8 else 1.0) + self.mean


def compute_normalization_stats(
    x: np.ndarray, mask: np.ndarray | None = None
) -> NormalizationStats:
    """Mean/std over valid (unmasked) entries only."""
    if mask is not None:
        valid = x[mask.astype(bool)]
    else:
        valid = x
    if valid.size == 0:
        return NormalizationStats(mean=0.0, std=1.0)
    return NormalizationStats(mean=float(np.mean(valid)), std=float(np.std(valid)))


def interpolate_missing_1d(
    values: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate missing entries of a 1D series (e.g. loss of InSAR
    coherence). Values outside the first/last valid index are filled by
    edge-value carry-forward/back. Returns (filled_values, was_interpolated).
    """
    values = values.astype(np.float64).copy()
    valid_mask = valid_mask.astype(bool)
    was_interpolated = ~valid_mask

    if valid_mask.sum() == 0:
        # Nothing observed at all: fall back to zeros, flag everything.
        return np.zeros_like(values), np.ones_like(values, dtype=bool)

    idx = np.arange(len(values))
    values[~valid_mask] = np.interp(idx[~valid_mask], idx[valid_mask], values[valid_mask])
    return values, was_interpolated


def clamp_displacement(values_cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Enforce |d_i(t)| < 10 cm; returns (clamped_values, was_clamped)."""
    was_clamped = np.abs(values_cm) >= MAX_ABS_DISPLACEMENT_CM
    clamped = np.clip(values_cm, -MAX_ABS_DISPLACEMENT_CM + 1e-3, MAX_ABS_DISPLACEMENT_CM - 1e-3)
    return clamped, was_clamped


def flag_interval_jumps(values_cm: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """
    True where |d_k - d_ref| > MAX_INTERVAL_CHANGE_CM, with d_ref the last usable,
    not-flagged value before k (time-sorted input). Comparing with the last
    ACCEPTED value means an isolated spike flags only itself. A persistent step
    (e.g. a phase-unwrapping error) flags every later value too: conservative,
    documented in docs/math/data_quality_bounds.md.
    """
    values = np.asarray(values_cm, dtype=np.float64)
    usable = np.asarray(usable, dtype=bool)
    jump = np.zeros(len(values), dtype=bool)
    ref = None
    for k in range(len(values)):
        if not usable[k]:
            continue
        if ref is not None and abs(values[k] - ref) > MAX_INTERVAL_CHANGE_CM:
            jump[k] = True
        else:
            ref = values[k]
    return jump


def clamp_rainfall(values_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    was_clamped = values_mm > MAX_RAINFALL_MM
    clamped = np.clip(values_mm, 0.0, MAX_RAINFALL_MM)
    return clamped, was_clamped


def time_deltas_days(timestamps_epoch_s: np.ndarray) -> np.ndarray:
    """
    Convert a monotonically increasing sequence of unix timestamps (seconds)
    into per-step time gaps (days). The first acquisition has no preceding
    interval, so its delta is 0 (review finding 22: it used to be a copy of the
    second gap). Δt_k > 0 therefore means "there is an interval (t_{k-1}, t_k]
    between two real acquisitions"; docs/math/time_intervals.md.
    """
    ts = np.asarray(timestamps_epoch_s, dtype=np.float64)
    if len(ts) < 2:
        return np.zeros_like(ts)
    return np.concatenate([[0.0], np.diff(ts) / 86400.0])


def aggregate_rainfall_to_insar_cadence(
    rainfall_timestamps_epoch_s: np.ndarray,
    rainfall_mm: np.ndarray,
    insar_timestamps_epoch_s: np.ndarray,
) -> np.ndarray:
    """
    Aggregate (sum) finer-cadence rainfall onto the coarser InSAR acquisition
    intervals, as described in Module 3 ("Rainfall & Meteorology").

    Step k receives the rain that fell in the *causal* interval
    (t_{k-1}, t_k] ending at acquisition k (step 0 receives everything up to
    and including t_0). Rain after the last acquisition is dropped: it cannot
    have influenced any observed displacement.
    """
    insar_ts = np.asarray(insar_timestamps_epoch_s, dtype=np.float64)
    rain_ts = np.asarray(rainfall_timestamps_epoch_s, dtype=np.float64)
    rain_mm = np.asarray(rainfall_mm, dtype=np.float64)

    bin_idx = np.searchsorted(insar_ts, rain_ts, side="left")
    in_range = bin_idx < len(insar_ts)
    return np.bincount(
        bin_idx[in_range], weights=rain_mm[in_range], minlength=len(insar_ts)
    ).astype(np.float64)


def build_adjacency_matrix(
    coordinates: np.ndarray,
    connectivity_radius_km: float = 5.0,
) -> np.ndarray:
    """
    Fallback geographic-proximity adjacency (haversine-based) used when no
    explicit hydraulic connectivity graph is supplied. a_ij in [0, 1],
    decaying with distance, zero beyond `connectivity_radius_km`.
    """
    n = coordinates.shape[0]
    adjacency = np.zeros((n, n), dtype=np.float64)
    lat = np.radians(coordinates[:, 0])
    lon = np.radians(coordinates[:, 1])
    r_earth_km = 6371.0

    for i in range(n):
        dlat = lat - lat[i]
        dlon = lon - lon[i]
        a = np.sin(dlat / 2) ** 2 + np.cos(lat[i]) * np.cos(lat) * np.sin(dlon / 2) ** 2
        dist_km = 2 * r_earth_km * np.arcsin(np.clip(np.sqrt(a), 0, 1))
        weight = np.where(
            dist_km <= connectivity_radius_km,
            np.exp(-dist_km / max(connectivity_radius_km, 1e-6)),
            0.0,
        )
        adjacency[i] = weight

    np.fill_diagonal(adjacency, 0.0)
    # Symmetrize (undirected hydraulic connectivity).
    adjacency = np.maximum(adjacency, adjacency.T)
    return adjacency


def pairwise_distance_m(coordinates_deg: np.ndarray) -> np.ndarray:
    """(N, 2) lat/lon in degrees -> (N, N) great-circle (haversine) distances [m]."""
    lat = np.radians(coordinates_deg[:, 0])[:, None]
    lon = np.radians(coordinates_deg[:, 1])[:, None]
    a = (
        np.sin((lat - lat.T) / 2) ** 2
        + np.cos(lat) * np.cos(lat.T) * np.sin((lon - lon.T) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_M * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
