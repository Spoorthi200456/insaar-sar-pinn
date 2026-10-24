"""
Review finding 4 (CLAUDE.md hard rule 3): physics must not depend on minibatch
composition. One fixed structure graph is built from ALL records once; a batch
is a set of seed structures plus their neighbourhood on that fixed graph, and
only seeds contribute to the loss.

Also the physics weights a_ij must carry 1/m² so that
sum_j a_ij (T_i+T_j)/2 (h_j - h_i) is in m/day (docs/math/graph_stencil.md).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.data.dataset import build_dataloader
from src.models.losses import CompositeARSLoss, LossWeights, pde_residual
from src.models.network import ARSNet
from src.training.train import make_synthetic_records

R_EARTH_M = 6_371_000.0


def _clustered_records(n=7, spacing_km=2.0, seed=5):
    """n structures on a line of latitude, spacing_km apart: a connected chain at 5 km radius."""
    records = make_synthetic_records(n_structures=n, n_timesteps=12, seed=seed)
    dlon = math.degrees(spacing_km * 1000.0 / (R_EARTH_M * math.cos(math.radians(13.0))))
    out = []
    for i, r in enumerate(records):
        design = r.design.model_copy(update={"latitude": 13.0, "longitude": 77.0 + i * dlon})
        out.append(r.model_copy(update={"design": design}))
    return out


def _per_structure_outputs(records, batch_size, shuffle_seed):
    torch.manual_seed(0)
    model = ARSNet(design_feature_dim=3).eval()
    torch.manual_seed(shuffle_seed)
    loader = build_dataloader(
        records, batch_size=batch_size, sequence_length=12, shuffle=True, num_workers=0
    )
    k, residual_rms = {}, {}
    with torch.no_grad():
        for batch in loader:
            out = model(
                design_features=batch["design_features"],
                displacement_cm=batch["displacement_cm"],
                rainfall_mm=batch["rainfall_mm"],
                time_delta_days=batch["time_delta_days"],
                adjacency=batch["adjacency"],
                incidence_angle_deg=batch["incidence_angle_deg"],
                coherence_mask=batch["coherence_mask"],
            )
            n = len(batch["structure_ids"])
            seeds = batch.get("loss_node_mask", torch.ones(n, dtype=torch.bool))
            physics_adj = batch.get("physics_adjacency", batch["adjacency"])
            r, valid = pde_residual(
                out.predicted_head_change,
                batch["time_delta_days"],
                out.storativity,
                out.hydraulic_conductivity,
                out.recharge_rate,
                physics_adj,
            )
            for i, sid in enumerate(batch["structure_ids"]):
                if seeds[i]:
                    assert sid not in k, f"{sid} was a seed twice in one epoch"
                    k[sid] = out.hydraulic_conductivity[i].item()
                    residual_rms[sid] = r[i][valid[i]].pow(2).mean().sqrt().item()
    return k, residual_rms


class TestPhysicsIndependentOfBatchComposition:
    def test_k_and_residual_identical_for_any_batch_size_and_shuffle(self):
        records = _clustered_records()
        ref_k, ref_r = _per_structure_outputs(records, batch_size=len(records), shuffle_seed=0)
        assert len(ref_k) == len(records)
        for batch_size, shuffle_seed in [(1, 1), (3, 2), (3, 3), (5, 4)]:
            k, r = _per_structure_outputs(records, batch_size, shuffle_seed)
            assert k.keys() == ref_k.keys()
            for sid in ref_k:
                assert k[sid] == pytest.approx(ref_k[sid], rel=1e-5, abs=1e-12), (batch_size, sid)
                assert r[sid] == pytest.approx(ref_r[sid], rel=1e-5, abs=1e-12), (batch_size, sid)

    def test_chain_is_actually_connected(self):
        # Guard against the test above passing trivially on an edgeless graph.
        records = _clustered_records()
        loader = build_dataloader(
            records, batch_size=len(records), sequence_length=12, shuffle=False, num_workers=0
        )
        batch = next(iter(loader))
        degree = (batch["physics_adjacency"] > 0).sum(dim=1)
        assert (degree >= 1).all()


class TestPhysicsWeightUnits:
    def test_weight_is_inverse_distance_squared_in_m2(self):
        records = _clustered_records(n=2, spacing_km=2.0)
        loader = build_dataloader(
            records, batch_size=2, sequence_length=12, shuffle=False, num_workers=0
        )
        batch = next(iter(loader))
        lat = math.radians(13.0)
        lon = [math.radians(r.design.longitude) for r in records]
        # haversine, independent of the code under test
        a = math.cos(lat) ** 2 * math.sin((lon[1] - lon[0]) / 2) ** 2
        d_m = 2 * R_EARTH_M * math.asin(math.sqrt(a))
        assert d_m == pytest.approx(2000.0, rel=1e-3)
        a_ij = batch["physics_adjacency"].double()
        assert a_ij[0, 1].item() == pytest.approx(1.0 / d_m**2, rel=1e-4)  # [1/m²]
        assert a_ij[0, 0].item() == 0.0 and torch.allclose(a_ij, a_ij.T)

    def test_divergence_of_linear_head_drop_is_flux_difference(self):
        # Two cells 2 km apart, T = 100 m²/day, heads 1 m and 0 m: the FV flux
        # T*(h_j-h_i)/d through a face of width d into a cell of area d² gives
        # T*(h_j-h_i)/d² = 100 * (-1) / 4e6 = -2.5e-5 m/day for the high cell.
        records = _clustered_records(n=2, spacing_km=2.0)
        batch = next(
            iter(
                build_dataloader(
                    records, batch_size=2, sequence_length=12, shuffle=False, num_workers=0
                )
            )
        )
        h = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float64)
        dt = torch.tensor([[12.0, 12.0], [12.0, 12.0]], dtype=torch.float64)
        r, _ = pde_residual(
            h,
            dt,
            torch.tensor(0.0, dtype=torch.float64),
            torch.full((2,), 100.0, dtype=torch.float64),
            torch.zeros(2, 2, dtype=torch.float64),
            batch["physics_adjacency"].double(),
        )
        # r = S dh/dt - div - R with S = 0, R = 0 -> r = -div
        assert -r[0, 1].item() == pytest.approx(-2.5e-5, rel=2e-3)
        assert -r[1, 1].item() == pytest.approx(+2.5e-5, rel=2e-3)


class TestLossOnlyOnSeedNodes:
    def test_non_seed_rows_do_not_contribute(self):
        torch.manual_seed(0)
        b, t = 3, 5
        args = dict(
            predicted_displacement_cm=torch.zeros(b, t),
            observed_displacement_cm=torch.zeros(b, t),
            predicted_head_change=torch.zeros(b, t),
            hydraulic_conductivity=torch.ones(b),
            recharge_rate=torch.zeros(b, t),
            design_recharge_capacity=torch.full((b,), 100.0),
            rainfall_mm=torch.full((b, t), 10.0),
            adjacency=torch.zeros(b, b),
            time_delta_days=torch.full((b, t), 12.0),
            storativity=torch.tensor(0.02),
            recharge_area_m2=torch.full((b,), 100.0),
            catchment_area_m2=torch.full((b,), 1e6),
            observation_mask=torch.ones(b, t, dtype=torch.bool),
        )
        loss_fn = CompositeARSLoss(LossWeights())
        clean = loss_fn(**args).total.item()
        # node 2 is a neighbour-only (non-seed) node with garbage in every term
        args["observed_displacement_cm"][2] = 50.0
        args["recharge_rate"][2] = 10.0
        masked = loss_fn(**args, node_mask=torch.tensor([True, True, False])).total.item()
        assert masked == pytest.approx(clean, abs=1e-9)
        assert loss_fn(**args).total.item() > clean + 1.0


def test_graph_module_builds_from_all_records_once():
    from src.data.graph import StructureGraph

    records = _clustered_records()
    g = StructureGraph.from_records(records)
    assert g.structure_ids == [r.design.structure_id for r in records]
    assert g.physics_adjacency.shape == (len(records), len(records))
    assert np.all(np.diag(g.physics_adjacency) == 0)
