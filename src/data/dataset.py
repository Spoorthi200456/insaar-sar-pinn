"""
PyTorch `Dataset` / `DataLoader` for the InSAR-PINN ARS pipeline.

Each item is one `ARSTelemetryRecord` (one structure's full time series).
Per-item work (interpolation, clamping, rainfall aggregation, normalization,
time-delta computation) happens inside `__getitem__`, so it is parallelized
"for free" across `DataLoader(num_workers=...)` worker processes -- this is
the practical form of "asynchronous temporal alignment" for a map-style
dataset: alignment work for structure i does not block alignment work for
structure j, and results are prefetched ahead of the training step.

The custom `collate_fn` then builds the *batch-level* hydraulic-connectivity
graph (adjacency matrix) required by the GAT layers in `src/models/layers.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import (
    COHERENCE_THRESHOLD,
    NormalizationStats,
    aggregate_rainfall_to_insar_cadence,
    build_adjacency_matrix,
    clamp_displacement,
    clamp_rainfall,
    compute_normalization_stats,
    interpolate_missing_1d,
    time_deltas_days,
)
from src.data.schemas import ARSTelemetryRecord, DataQualityFlag, GraphEdge

# Observations whose stored value is not the true measurement (clamped to the
# +/-10 cm bound by the schema, or explicitly missing) must never be used as
# a data-misfit target.
_UNRELIABLE_FLAGS = (DataQualityFlag.OUT_OF_BOUNDS_CLAMPED, DataQualityFlag.MISSING)


@dataclass
class ARSSample:
    """Fixed-length (T timesteps) tensors for a single ARS structure."""

    structure_id: str
    displacement_cm: torch.Tensor          # (T,)
    rainfall_mm: torch.Tensor              # (T,)
    time_delta_days: torch.Tensor          # (T,)
    coherence_mask: torch.Tensor           # (T,) bool, True == reliable observation
    design_features: torch.Tensor          # (Q,)
    coordinates: torch.Tensor              # (2,) lat, lon
    design_recharge_capacity: torch.Tensor  # scalar
    incidence_angle_deg: torch.Tensor      # scalar [deg], mean over the series' acquisitions


class ARSDataset(Dataset):
    """
    Map-style dataset over a list of `ARSTelemetryRecord`.

    Parameters
    ----------
    records: telemetry records, one per structure (already Pydantic-validated
        upstream -- this class assumes clean, schema-conformant input and
        focuses purely on tensor construction / temporal harmonization).
    sequence_length: T, the fixed number of InSAR acquisition steps per
        sample. Series longer than T are truncated to the most recent T
        steps; shorter series are left-padded (padding is captured in
        `coherence_mask=False` so it is excluded from the data-misfit loss).
    design_feature_normalizers: optional pre-fit normalization stats (e.g.
        fit on a training split and reused, unmodified, for validation/test).
    """

    DESIGN_FEATURE_NAMES = (
        "height_m",
        "storage_volume_m3",
        "catchment_area_km2",
    )

    def __init__(
        self,
        records: list[ARSTelemetryRecord],
        sequence_length: int = 52,
        design_feature_normalizers: dict[str, NormalizationStats] | None = None,
    ) -> None:
        if not records:
            raise ValueError("ARSDataset requires at least one telemetry record")
        self.records = records
        self.sequence_length = sequence_length
        self._design_stats = design_feature_normalizers or self._fit_design_stats(records)

    # ------------------------------------------------------------------ #
    @classmethod
    def _fit_design_stats(cls, records: list[ARSTelemetryRecord]) -> dict[str, NormalizationStats]:
        stats: dict[str, NormalizationStats] = {}
        for name in cls.DESIGN_FEATURE_NAMES:
            values = np.array([getattr(r.design, name) for r in records], dtype=np.float64)
            stats[name] = compute_normalization_stats(values)
        return stats

    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------ #
    def __getitem__(self, idx: int) -> ARSSample:
        record = self.records[idx]

        sar_sorted = sorted(record.sar_series, key=lambda o: o.timestamp)
        rain_sorted = sorted(record.rainfall_series, key=lambda o: o.timestamp)

        insar_ts = np.array([o.timestamp.timestamp() for o in sar_sorted], dtype=np.float64)
        disp_raw = np.array([o.los_displacement_cm for o in sar_sorted], dtype=np.float64)
        coherence = np.array([o.coherence for o in sar_sorted], dtype=np.float64)
        flag_ok = np.array([o.quality_flag not in _UNRELIABLE_FLAGS for o in sar_sorted], dtype=bool)
        reliable = (coherence >= COHERENCE_THRESHOLD) & flag_ok

        disp_clamped, _ = clamp_displacement(disp_raw)
        disp_filled, was_interp = interpolate_missing_1d(disp_clamped, reliable)

        rain_ts = np.array([o.timestamp.timestamp() for o in rain_sorted], dtype=np.float64)
        rain_mm_raw = np.array([o.rainfall_mm for o in rain_sorted], dtype=np.float64)
        rain_mm_clamped, _ = clamp_rainfall(rain_mm_raw)
        rain_aligned = aggregate_rainfall_to_insar_cadence(rain_ts, rain_mm_clamped, insar_ts)

        deltas = time_deltas_days(insar_ts)

        # Anomaly rule from Module 3: no rainfall in an interval => recharge
        # should be near zero; we surface this as an additional reliability
        # signal by *not* masking it out (the loss/reg terms handle it), but
        # we do mark interpolated points as unreliable for the data term.
        reliable_final = reliable & (~was_interp)

        disp_t, rain_t, delta_t, mask_t = self._fixed_length(
            disp_filled, rain_aligned, deltas, reliable_final
        )

        design_vec = np.array(
            [getattr(record.design, name) for name in self.DESIGN_FEATURE_NAMES],
            dtype=np.float64,
        )
        design_vec_norm = np.array(
            [
                self._design_stats[name].apply(np.array([val]))[0]
                for name, val in zip(self.DESIGN_FEATURE_NAMES, design_vec)
            ],
            dtype=np.float32,
        )

        return ARSSample(
            structure_id=record.design.structure_id,
            displacement_cm=torch.tensor(disp_t, dtype=torch.float32),
            rainfall_mm=torch.tensor(rain_t, dtype=torch.float32),
            time_delta_days=torch.tensor(delta_t, dtype=torch.float32),
            coherence_mask=torch.tensor(mask_t, dtype=torch.bool),
            design_features=torch.tensor(design_vec_norm, dtype=torch.float32),
            coordinates=torch.tensor(
                [record.design.latitude, record.design.longitude], dtype=torch.float32
            ),
            design_recharge_capacity=torch.tensor(
                record.design.design_recharge_capacity_m3_per_day, dtype=torch.float32
            ),
            incidence_angle_deg=torch.tensor(
                float(np.mean([o.incidence_angle_deg for o in sar_sorted])), dtype=torch.float32
            ),
        )

    # ------------------------------------------------------------------ #
    def _fixed_length(
        self,
        disp: np.ndarray,
        rain: np.ndarray,
        delta: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        t = len(disp)
        length = self.sequence_length
        if t >= length:
            sl = slice(t - length, t)
            return disp[sl], rain[sl], delta[sl], mask[sl]

        pad = length - t
        disp_p = np.concatenate([np.zeros(pad), disp])
        rain_p = np.concatenate([np.zeros(pad), rain])
        delta_p = np.concatenate([np.zeros(pad), delta])
        mask_p = np.concatenate([np.zeros(pad, dtype=bool), mask])
        return disp_p, rain_p, delta_p, mask_p


class ARSCollateFn:
    """
    Batches `ARSSample`s into stacked tensors and builds the (B, B)
    hydraulic-connectivity adjacency matrix `a_ij` used by the GAT layers.

    This is a plain class (not a closure returned by a function) on purpose:
    `DataLoader(num_workers > 0)` has to send `collate_fn` to each worker
    process. On Windows -- which has no `fork()` and always uses `spawn` --
    and on Python 3.14+ even on Linux/macOS (which switched the default
    start method to `forkserver`), that means `collate_fn` is *pickled* and
    sent over, not just referenced. A nested function (a closure) can't be
    pickled at all (`Can't pickle local object ...`), so `num_workers > 0`
    used to crash on exactly those platforms. A class instance defined at
    module scope, holding only plain-picklable attributes (a dict/float
    here), pickles fine everywhere, so this works identically on Linux
    `fork`, Windows `spawn`, and Python 3.14's `forkserver`.
    """

    def __init__(
        self,
        edge_lookup: dict[tuple[str, str], float] | None = None,
        connectivity_radius_km: float = 5.0,
    ) -> None:
        self.edge_lookup = edge_lookup or {}
        self.connectivity_radius_km = connectivity_radius_km

    def __call__(self, batch: list[ARSSample]) -> dict[str, torch.Tensor | list[str]]:
        structure_ids = [s.structure_id for s in batch]
        coords = torch.stack([s.coordinates for s in batch]).numpy()

        if self.edge_lookup:
            n = len(batch)
            adjacency = np.zeros((n, n), dtype=np.float32)
            for i, sid_i in enumerate(structure_ids):
                for j, sid_j in enumerate(structure_ids):
                    if i != j:
                        adjacency[i, j] = self.edge_lookup.get((sid_i, sid_j), 0.0)
        else:
            adjacency = build_adjacency_matrix(coords, self.connectivity_radius_km).astype(
                np.float32
            )

        return {
            "structure_ids": structure_ids,
            "displacement_cm": torch.stack([s.displacement_cm for s in batch]),
            "rainfall_mm": torch.stack([s.rainfall_mm for s in batch]),
            "time_delta_days": torch.stack([s.time_delta_days for s in batch]),
            "coherence_mask": torch.stack([s.coherence_mask for s in batch]),
            "design_features": torch.stack([s.design_features for s in batch]),
            "coordinates": torch.stack([s.coordinates for s in batch]),
            "design_recharge_capacity": torch.stack(
                [s.design_recharge_capacity for s in batch]
            ),
            "incidence_angle_deg": torch.stack([s.incidence_angle_deg for s in batch]),
            "adjacency": torch.from_numpy(adjacency),
        }


def make_collate_fn(
    explicit_edges: list[GraphEdge] | None = None, connectivity_radius_km: float = 5.0
) -> ARSCollateFn:
    """
    Returns a picklable `collate_fn` callable (see `ARSCollateFn`) that
    batches `ARSSample`s into stacked tensors and additionally builds the
    (B, B) hydraulic-connectivity adjacency matrix `a_ij` used by the GAT
    layers.

    If `explicit_edges` (structure_id -> structure_id, weight) is supplied
    (e.g. from a hydrogeological survey), those weights are used; otherwise
    a geographic-proximity fallback graph is built from coordinates.
    """
    edge_lookup: dict[tuple[str, str], float] = {}
    if explicit_edges:
        for e in explicit_edges:
            edge_lookup[(e.source_id, e.target_id)] = e.weight
            edge_lookup[(e.target_id, e.source_id)] = e.weight
    return ARSCollateFn(edge_lookup=edge_lookup, connectivity_radius_km=connectivity_radius_km)


def build_dataloader(
    records: list[ARSTelemetryRecord],
    batch_size: int = 32,
    sequence_length: int = 52,
    shuffle: bool = True,
    num_workers: int = 4,
    edges: list[GraphEdge] | None = None,
) -> DataLoader:
    dataset = ARSDataset(records, sequence_length=sequence_length)
    collate_fn = make_collate_fn(explicit_edges=edges)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
