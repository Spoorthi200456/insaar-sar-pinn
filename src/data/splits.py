"""
Train / validation / test split by structure (review finding 24).

Structures are coupled through the fixed structure graph (lateral flow in the
PDE, graph attention), so splitting individual structures at random would leak
information across splits: a held-out structure's data would still enter its
training neighbours' forward pass. The unit of splitting is therefore a
CONNECTED COMPONENT of the graph (a group of hydraulically linked structures);
no edge crosses splits and no structure is in more than one split.

Components are shuffled with a fixed seed and assigned greedily to the split
whose target size (fraction x n_structures) is furthest from being met. With
`stratify` (e.g. planted working/failing labels in a synthetic experiment) this
is done separately within each class, so every class is represented in every
split when there are enough components. See docs/math/validation_split.md.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.data.graph import DEFAULT_CONNECTIVITY_RADIUS_KM, StructureGraph

SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_FRACTIONS = (0.7, 0.15, 0.15)


@dataclass(frozen=True)
class StructureSplit:
    train: list[str]
    validation: list[str]
    test: list[str]

    def as_dict(self) -> dict[str, list[str]]:
        return {"train": self.train, "validation": self.validation, "test": self.test}


def connected_components(graph: StructureGraph) -> list[list[int]]:
    support = (graph.physics_adjacency > 0) | (graph.attention_prior > 0)
    support = support | support.T
    n = len(graph.structure_ids)
    seen = np.zeros(n, dtype=bool)
    components = []
    for start in range(n):
        if seen[start]:
            continue
        stack, comp = [start], []
        seen[start] = True
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in np.flatnonzero(support[i]):
                if not seen[j]:
                    seen[j] = True
                    stack.append(int(j))
        components.append(sorted(comp))
    return components


def _assign(components: list[list[int]], fractions, rng) -> list[list[int]]:
    """Greedy: shuffled components, each to the split with the largest unmet target."""
    order = rng.permutation(len(components))
    total = sum(len(c) for c in components)
    targets = np.asarray(fractions, dtype=float) * total
    sizes = np.zeros(len(fractions))
    parts: list[list[int]] = [[] for _ in fractions]
    for k in order:
        comp = components[k]
        empty = [s for s in range(len(fractions)) if sizes[s] == 0 and targets[s] > 0]
        # make sure every split with a positive target gets something first
        s = empty[0] if empty and len(order) >= len(fractions) else int(np.argmax(targets - sizes))
        parts[s].extend(comp)
        sizes[s] += len(comp)
    return parts


def split_structures(
    records,
    edges=None,
    connectivity_radius_km: float = DEFAULT_CONNECTIVITY_RADIUS_KM,
    fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = 0,
    stratify: dict[str, str] | None = None,
) -> StructureSplit:
    if len(fractions) != 3 or abs(sum(fractions) - 1.0) > 1e-9 or min(fractions) < 0:
        raise ValueError("fractions must be three non-negative numbers summing to 1")
    graph = StructureGraph.from_records(records, edges, connectivity_radius_km)
    components = connected_components(graph)
    rng = np.random.default_rng(seed)

    if stratify is None:
        groups = {"all": components}
    else:
        groups: dict[str, list[list[int]]] = {}
        for comp in components:
            labels = Counter(stratify[graph.structure_ids[i]] for i in comp)
            label = sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
            groups.setdefault(label, []).append(comp)

    parts: list[list[int]] = [[], [], []]
    for label in sorted(groups):
        for s, members in enumerate(_assign(groups[label], fractions, rng)):
            parts[s].extend(members)

    ids = graph.structure_ids
    return StructureSplit(*[[ids[i] for i in sorted(p)] for p in parts])
