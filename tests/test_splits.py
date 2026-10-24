"""
Review finding 24: train / validation / test split by STRUCTURE with a fixed seed.

No structure may appear in more than one split, and because structures are
coupled through the physics graph (lateral flow, graph attention), no graph
edge may cross splits either: whole connected groups of structures go to one
split. Model selection uses the validation split only; the test split is never
seen during training. See docs/math/validation_split.md.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.training.train import make_synthetic_records

R_EARTH_M = 6_371_000.0


def _records_with_clusters(n=20, seed=4):
    """n structures; 0-2 and 3-5 form two 1.5 km clusters (connected components of size 3)."""
    recs = make_synthetic_records(n_structures=n, n_timesteps=8, seed=seed)
    dlon = math.degrees(1500.0 / (R_EARTH_M * math.cos(math.radians(13.0))))
    out = []
    for i, r in enumerate(recs):
        if i < 6:
            lat, lon = (13.0, 77.0 + i * dlon) if i < 3 else (13.4, 77.6 + (i - 3) * dlon)
            r = r.model_copy(
                update={"design": r.design.model_copy(update={"latitude": lat, "longitude": lon})}
            )
        out.append(r)
    return out


class TestSplitStructures:
    def test_disjoint_complete_and_reproducible(self):
        from src.data.splits import split_structures

        recs = _records_with_clusters()
        ids = {r.design.structure_id for r in recs}
        s1 = split_structures(recs, seed=0)
        s2 = split_structures(recs, seed=0)
        parts = [set(s1.train), set(s1.validation), set(s1.test)]
        assert all(parts), "every split must be non-empty"
        assert parts[0] | parts[1] | parts[2] == ids
        assert sum(len(p) for p in parts) == len(ids)  # no structure in two splits
        assert (s1.train, s1.validation, s1.test) == (s2.train, s2.validation, s2.test)
        others = [split_structures(recs, seed=k) for k in range(1, 6)]
        assert any(set(o.test) != parts[2] for o in others)  # the seed matters

    def test_connected_structures_stay_in_one_split(self):
        from src.data.graph import StructureGraph
        from src.data.splits import split_structures

        recs = _records_with_clusters()
        s = split_structures(recs, seed=0)
        where = {sid: name for name in ("train", "validation", "test") for sid in getattr(s, name)}
        g = StructureGraph.from_records(recs)
        support = (g.physics_adjacency > 0) | (g.attention_prior > 0)
        i, j = np.nonzero(support)
        assert len(i) > 0
        for a, b in zip(i, j, strict=True):
            assert where[g.structure_ids[a]] == where[g.structure_ids[b]]

    def test_fractions_roughly_respected(self):
        from src.data.splits import split_structures

        recs = make_synthetic_records(n_structures=40, n_timesteps=6, seed=1)
        s = split_structures(recs, fractions=(0.7, 0.15, 0.15), seed=0)
        assert len(s.train) == 28 and len(s.validation) == 6 and len(s.test) == 6

    def test_stratified_split_puts_each_class_in_every_split(self):
        from src.data.splits import split_structures

        recs = make_synthetic_records(n_structures=30, n_timesteps=6, seed=2)
        labels = {
            r.design.structure_id: ("failing" if k % 3 == 0 else "working")
            for k, r in enumerate(recs)
        }
        s = split_structures(recs, seed=0, stratify=labels)
        for part in (s.train, s.validation, s.test):
            assert {labels[sid] for sid in part} == {"failing", "working"}


class TestTrainingUsesTheSplit:
    def test_only_train_structures_reach_the_optimizer_and_val_selects_checkpoint(
        self, monkeypatch, tmp_path
    ):
        import src.training.train as tr

        recs = _records_with_clusters()
        seen = []
        real_loader = tr.build_dataloader

        def spy(records, **kw):
            seen.append({r.design.structure_id for r in records})
            return real_loader(records, **kw)

        monkeypatch.setattr(tr, "build_dataloader", spy)
        ck = tmp_path / "ck.pt"
        tr.train(
            recs,
            epochs=2,
            batch_size=4,
            sequence_length=8,
            checkpoint_path=ck,
            device="cpu",
            num_workers=0,
        )
        saved = torch.load(ck, weights_only=False)
        split = saved["split"]
        assert seen == [set(split["train"])]  # only training structures are batched
        assert not (set(split["train"]) & (set(split["validation"]) | set(split["test"])))
        assert "val_loss" in saved and np.isfinite(saved["val_loss"])
        assert saved["selected_on"] == "validation"
        assert np.isfinite(saved["test_loss"])

    def test_validation_and_test_use_training_normalisation(self, monkeypatch, tmp_path):
        import src.training.train as tr

        recs = _records_with_clusters()
        tr.train(
            recs,
            epochs=1,
            batch_size=4,
            sequence_length=8,
            checkpoint_path=tmp_path / "ck.pt",
            device="cpu",
            num_workers=0,
        )
        saved = torch.load(tmp_path / "ck.pt", weights_only=False)
        train_ids = set(saved["split"]["train"])
        heights = np.array([r.design.height_m for r in recs if r.design.structure_id in train_ids])
        stats = saved["design_stats"]["height_m"]
        assert stats["mean"] == pytest.approx(float(heights.mean()))
        assert stats["std"] == pytest.approx(float(heights.std()))
