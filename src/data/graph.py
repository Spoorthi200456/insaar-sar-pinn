"""
One fixed structure network (review finding 4, CLAUDE.md hard rule 3).

Built ONCE from all structures (and optional surveyed edges). Every batch is
cut out of it, so a structure's neighbours, and therefore its physics, never
depend on which other structures happen to share a minibatch.

Two matrices, kept separate on purpose:

* `physics_adjacency` a_ij [1/m²]: finite-volume stencil weights for
  ∇·(T∇h) ≈ Σ_j a_ij (T_i+T_j)/2 (h_j − h_i)  [m/day].
  a_ij = c_ij / d_ij², the regular-grid limit of w_ij / (d_ij · A_i) with face
  width w = d and cell area A = d² (MODFLOW conductance form; Wang & Anderson
  1982, *Introduction to Groundwater Modeling* ch. 3). c_ij ∈ [0, 1] is the
  dimensionless connectivity (surveyed edge weight, or 1 within the radius
  fallback). Diagonal 0. Derivation: docs/math/graph_stencil.md.
* `attention_prior` [dimensionless, 0..1]: prior used by the GAT only
  (surveyed edge weight, or exp(−d/r) within the radius fallback). Not physics.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from src.data.preprocessing import build_adjacency_matrix, pairwise_distance_m

DEFAULT_CONNECTIVITY_RADIUS_KM = 5.0  # PROVISIONAL (docs/open_questions.md OQ11)
MIN_EDGE_LENGTH_M = 1.0  # two connected structures closer than this are a data error


@dataclass
class StructureGraph:
    structure_ids: list[str]
    physics_adjacency: np.ndarray   # (N, N) a_ij [1/m²]
    attention_prior: np.ndarray     # (N, N) dimensionless
    index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {sid: i for i, sid in enumerate(self.structure_ids)}
        if len(self.index) != len(self.structure_ids):
            raise ValueError("structure_ids must be unique")

    # ------------------------------------------------------------------ #
    @classmethod
    def from_coordinates(
        cls,
        structure_ids: Sequence[str],
        coordinates_deg: np.ndarray,                       # (N, 2) lat, lon
        edges: Iterable[tuple[str, str, float]] | None = None,
        connectivity_radius_km: float = DEFAULT_CONNECTIVITY_RADIUS_KM,
    ) -> StructureGraph:
        ids = list(structure_ids)
        n = len(ids)
        coords = np.asarray(coordinates_deg, dtype=np.float64).reshape(n, 2)
        dist_m = pairwise_distance_m(coords)
        edges = list(edges or [])

        if edges:
            pos = {sid: i for i, sid in enumerate(ids)}
            connectivity = np.zeros((n, n))
            for src, dst, w in edges:
                if src in pos and dst in pos and src != dst:
                    connectivity[pos[src], pos[dst]] = w
                    connectivity[pos[dst], pos[src]] = w
            attention = connectivity.copy()
        else:
            attention = build_adjacency_matrix(coords, connectivity_radius_km)
            connectivity = (attention > 0).astype(np.float64)

        np.fill_diagonal(connectivity, 0.0)
        connected = connectivity > 0
        if np.any(dist_m[connected] < MIN_EDGE_LENGTH_M):
            raise ValueError("connected structures closer than 1 m: check coordinates")
        physics = np.zeros((n, n))
        physics[connected] = connectivity[connected] / dist_m[connected] ** 2
        return cls(structure_ids=ids, physics_adjacency=physics, attention_prior=attention)

    @classmethod
    def from_records(
        cls,
        records,                     # list[ARSTelemetryRecord]
        edges=None,                  # list[GraphEdge] | None
        connectivity_radius_km: float = DEFAULT_CONNECTIVITY_RADIUS_KM,
    ) -> StructureGraph:
        ids = [r.design.structure_id for r in records]
        coords = np.array([[r.design.latitude, r.design.longitude] for r in records])
        edge_tuples = [(e.source_id, e.target_id, e.weight) for e in (edges or [])]
        return cls.from_coordinates(ids, coords, edge_tuples, connectivity_radius_km)

    # ------------------------------------------------------------------ #
    def neighbour_closure(self, seeds: Sequence[int], hops: int) -> list[int]:
        """Seeds first (in the given order), then every node within `hops` edges, sorted."""
        support = (self.physics_adjacency > 0) | (self.attention_prior > 0)
        seen = set(seeds)
        frontier = set(seeds)
        for _ in range(hops):
            nxt = set()
            for i in frontier:
                nxt.update(np.flatnonzero(support[i]).tolist())
            frontier = nxt - seen
            seen |= frontier
        return list(seeds) + sorted(seen - set(seeds))

    def submatrices(self, structure_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        idx = [self.index[sid] for sid in structure_ids]
        sel = np.ix_(idx, idx)
        return self.physics_adjacency[sel], self.attention_prior[sel]
