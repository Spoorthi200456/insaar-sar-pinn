"""
PyTorch `Dataset` / `DataLoader` for the InSAR-PINN ARS pipeline.

Each item is one `ARSTelemetryRecord` (one structure's full time series).
Per-item work (interpolation, clamping, rainfall aggregation, normalization,
time-delta computation) happens inside `__getitem__`, so it is parallelized
"for free" across `DataLoader(num_workers=...)` worker processes -- this is
the practical form of "asynchronous temporal alignment" for a map-style
dataset: alignment work for structure i does not block alignment work for
structure j, and results are prefetched ahead of the training step.

Batches are cut out of ONE fixed structure graph (`src/data/graph.py`,
review finding 4): seed structures plus their neighbourhood, with only seeds
contributing to the loss, so physics never depends on batch composition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from src.data.graph import DEFAULT_CONNECTIVITY_RADIUS_KM, StructureGraph
from src.data.preprocessing import (
    COHERENCE_THRESHOLD,
    NormalizationStats,
    aggregate_rainfall_to_insar_cadence,
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
M2_PER_KM2 = 1.0e6
# ARSNet default gat_num_layers (2) + 1: see GraphBatchSampler.
DEFAULT_NEIGHBOUR_HOPS = 3

_UNRELIABLE_FLAGS = (DataQualityFlag.OUT_OF_BOUNDS_CLAMPED, DataQualityFlag.MISSING)


@dataclass
class ARSSample:
    """Fixed-length (T timesteps) tensors for a single ARS structure."""

    structure_id: str
    displacement_cm: torch.Tensor          # (T,)
    rainfall_mm: torch.Tensor              # (T,)
    time_delta_days: torch.Tensor          # (T,)
    coherence_mask: torch.Tensor           # (T,) bool, True == reliable observation
    valid_step_mask: torch.Tensor          # (T,) bool, True == real acquisition (not padding)
    design_features: torch.Tensor          # (Q,)
    coordinates: torch.Tensor              # (2,) lat, lon
    design_recharge_capacity: torch.Tensor  # scalar [m³/day]
    recharge_area_m2: torch.Tensor          # scalar A [m²] (finding 6)
    catchment_area_m2: torch.Tensor         # scalar A_catch [m²]
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
        _, _, _, real_t = self._fixed_length(
            disp_filled, rain_aligned, deltas, np.ones(len(sar_sorted), dtype=bool)
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
            valid_step_mask=torch.tensor(real_t, dtype=torch.bool),
            design_features=torch.tensor(design_vec_norm, dtype=torch.float32),
            coordinates=torch.tensor(
                [record.design.latitude, record.design.longitude], dtype=torch.float32
            ),
            design_recharge_capacity=torch.tensor(
                record.design.design_recharge_capacity_m3_per_day, dtype=torch.float32
            ),
            recharge_area_m2=torch.tensor(
                record.design.effective_recharge_area_m2, dtype=torch.float32
            ),
            catchment_area_m2=torch.tensor(
                record.design.catchment_area_km2 * M2_PER_KM2, dtype=torch.float32
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
    Batches `ARSSample`s into stacked tensors and cuts the batch's adjacency
    matrices out of ONE fixed `StructureGraph` (review finding 4).

    * `graph` given (training): the sub-matrices are indexed from it, so a
      structure's neighbours never depend on batch composition. Items may be
      `(sample, is_seed)` pairs from `GraphBatchSampler`; only seeds get
      `loss_node_mask = True`.
    * `graph` is None (inference / full-batch demo): the batch is taken to BE
      the whole network, and the graph is built from it. Only valid when every
      structure of the network is in the batch.

    Batch keys: `adjacency` = dimensionless attention prior (GAT input),
    `physics_adjacency` = a_ij [1/m²] (∇·(T∇h) in the loss).

    A plain module-level class (not a closure) so `DataLoader(num_workers > 0)`
    can pickle it under `spawn` (Windows) and `forkserver` (Python 3.14+).
    """

    def __init__(
        self,
        graph: StructureGraph | None = None,
        edges: list[tuple[str, str, float]] | None = None,
        connectivity_radius_km: float = DEFAULT_CONNECTIVITY_RADIUS_KM,
    ) -> None:
        self.graph = graph
        self.edges = edges or []
        self.connectivity_radius_km = connectivity_radius_km

    def __call__(self, batch: list) -> dict[str, torch.Tensor | list[str]]:
        if batch and isinstance(batch[0], tuple):
            samples = [b[0] for b in batch]
            seeds = [bool(b[1]) for b in batch]
        else:
            samples, seeds = list(batch), [True] * len(batch)
        structure_ids = [s.structure_id for s in samples]

        graph = self.graph
        if graph is None:
            coords = torch.stack([s.coordinates for s in samples]).double().numpy()
            graph = StructureGraph.from_coordinates(
                structure_ids, coords, self.edges, self.connectivity_radius_km
            )
        physics, attention = graph.submatrices(structure_ids)

        return {
            "structure_ids": structure_ids,
            "displacement_cm": torch.stack([s.displacement_cm for s in samples]),
            "rainfall_mm": torch.stack([s.rainfall_mm for s in samples]),
            "time_delta_days": torch.stack([s.time_delta_days for s in samples]),
            "coherence_mask": torch.stack([s.coherence_mask for s in samples]),
            "valid_step_mask": torch.stack([s.valid_step_mask for s in samples]),
            "design_features": torch.stack([s.design_features for s in samples]),
            "coordinates": torch.stack([s.coordinates for s in samples]),
            "design_recharge_capacity": torch.stack(
                [s.design_recharge_capacity for s in samples]
            ),
            "recharge_area_m2": torch.stack([s.recharge_area_m2 for s in samples]),
            "catchment_area_m2": torch.stack([s.catchment_area_m2 for s in samples]),
            "incidence_angle_deg": torch.stack([s.incidence_angle_deg for s in samples]),
            "adjacency": torch.from_numpy(attention.astype(np.float32)),
            "physics_adjacency": torch.from_numpy(physics.astype(np.float32)),
            "loss_node_mask": torch.tensor(seeds, dtype=torch.bool),
        }


def make_collate_fn(
    explicit_edges: list[GraphEdge] | None = None,
    connectivity_radius_km: float = DEFAULT_CONNECTIVITY_RADIUS_KM,
    graph: StructureGraph | None = None,
) -> ARSCollateFn:
    """
    Returns a picklable collate callable (see `ARSCollateFn`). Pass `graph`
    (the fixed network) whenever the batch is a subset of the network. Without
    it, the batch itself is treated as the whole network: `explicit_edges`
    (e.g. from a hydrogeological survey) if given, else the radius fallback.
    """
    edges = [(e.source_id, e.target_id, e.weight) for e in (explicit_edges or [])]
    return ARSCollateFn(graph=graph, edges=edges, connectivity_radius_km=connectivity_radius_km)


class _SeededDataset(Dataset):
    """Wraps `ARSDataset` so `GraphBatchSampler` can tag items as seed / neighbour."""

    def __init__(self, base: ARSDataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, key: tuple[int, bool]) -> tuple[ARSSample, bool]:
        idx, is_seed = key
        return self.base[idx], is_seed


class GraphBatchSampler(Sampler):
    """
    Yields batches of `(index, is_seed)`: `batch_size` seed structures (each
    structure is a seed exactly once per epoch) plus every structure within
    `hops` edges of a seed on the fixed graph. With `hops` = number of GAT
    layers + 1, a seed's embedding, its K and R, and the neighbour heads in its
    ∇·(T∇h) are exactly those of a full-network forward pass.
    """

    def __init__(self, graph: StructureGraph, batch_size: int, shuffle: bool, hops: int) -> None:
        self.graph = graph
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.hops = hops

    def __len__(self) -> int:
        n = len(self.graph.structure_ids)
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.graph.structure_ids)
        order = torch.randperm(n).tolist() if self.shuffle else list(range(n))
        for start in range(0, n, self.batch_size):
            seeds = order[start : start + self.batch_size]
            nodes = self.graph.neighbour_closure(seeds, self.hops)
            seed_set = set(seeds)
            yield [(i, i in seed_set) for i in nodes]


def build_dataloader(
    records: list[ARSTelemetryRecord],
    batch_size: int = 32,
    sequence_length: int = 52,
    shuffle: bool = True,
    num_workers: int = 4,
    edges: list[GraphEdge] | None = None,
    connectivity_radius_km: float = DEFAULT_CONNECTIVITY_RADIUS_KM,
    neighbour_hops: int = DEFAULT_NEIGHBOUR_HOPS,
) -> DataLoader:
    """
    Minibatches cut from ONE fixed graph over all `records` (review finding 4).
    `neighbour_hops` must be >= (ARSNet GAT layers + 1); the default matches
    ARSNet's default of 2 GAT layers.
    """
    dataset = ARSDataset(records, sequence_length=sequence_length)
    graph = StructureGraph.from_records(records, edges, connectivity_radius_km)
    return DataLoader(
        _SeededDataset(dataset),
        batch_sampler=GraphBatchSampler(graph, batch_size, shuffle, neighbour_hops),
        num_workers=num_workers,
        collate_fn=ARSCollateFn(graph=graph),
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
