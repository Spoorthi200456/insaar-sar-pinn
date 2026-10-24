"""
Known-answer experiment (review 2, step 4): experiments/synthetic_recovery/.

Sanity tests check that the independent simulator is correct by itself. The
recovery test trains ARSNet on simulated data and asserts the thresholds
pre-registered in experiments/synthetic_recovery/README.md (fixed before
any training run; not tuned).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.synthetic_recovery import run_recovery as rr
from experiments.synthetic_recovery.simulator import (
    TruthConfig,
    integrate_head,
    lateral_operator,
    los_cm,
    oracle_eta,
    simulate,
)
from src.data.dataset import ARSDataset
from src.models.coupling import head_to_los_cm


@pytest.fixture(scope="module")
def sim():
    return simulate(TruthConfig())


class TestSimulatorSanity:
    def test_mass_balance_without_sinks(self, sim):
        # L = 0 and no lateral flow: S * h_K = sum_k R_k dt_k exactly
        cfg = sim.cfg
        h = integrate_head(
            cfg, sim.recharge, sim.dt_days, np.zeros((cfg.n_structures,) * 2), leakance=0.0
        )
        stored = cfg.storativity * h[:, -1]
        recharged = (sim.recharge[:, 1:] * sim.dt_days[1:]).sum(axis=1)
        assert np.allclose(stored, recharged, rtol=1e-10)

    def test_lateral_flow_conserves_water(self, sim):
        g = lateral_operator(sim.a_ij, sim.transmissivity)
        assert np.allclose(g.sum(axis=0), 0.0, atol=1e-18)  # symmetric stencil, zero row sums
        assert (sim.a_ij > 0).sum() >= 12  # the two 3-structure clusters

    def test_independent_coupling_matches_model_coupling(self, sim):
        ours = los_cm(sim.head, sim.cfg.skeletal_storage, sim.incidence_deg)
        model = head_to_los_cm(
            torch.tensor(sim.head),
            torch.tensor(sim.cfg.skeletal_storage, dtype=torch.float64),
            torch.tensor(sim.incidence_deg),
        ).numpy()
        assert np.allclose(ours, model, rtol=1e-12)
        assert np.allclose(ours, sim.displacement_clean_cm)

    def test_dataset_rain_binning_matches_simulator(self, sim):
        records = rr.to_records(sim)
        sample = ARSDataset(records, sequence_length=sim.cfg.n_acquisitions)[0]
        assert np.allclose(sample.rainfall_mm.numpy(), sim.rain_per_step_mm, atol=1e-3)
        assert np.allclose(sample.time_delta_days.numpy(), sim.dt_days, atol=1e-4)

    def test_oracle_recovers_eta_exactly_without_noise(self, sim):
        clean = replace(sim, displacement_cm=sim.displacement_clean_cm)
        assert np.allclose(oracle_eta(clean), sim.eta, atol=1e-8)

    def test_signal_is_above_noise_for_working_structures(self, sim):
        amp = np.abs(sim.displacement_clean_cm).max(axis=1)
        assert np.all(amp[~sim.failing] > 3 * sim.cfg.noise_cm)


@pytest.mark.slow
def test_recovery_meets_preregistered_thresholds(sim):
    floor = rr.oracle_floor(sim)
    a = rr.run_scenario("A_insar_only", sim)
    b = rr.run_scenario("B_oracle_S_Sk", sim, oracle_constants=True)
    msg = (
        f"A: spearman {a.spearman:.3f} auroc {a.auroc:.3f}; "
        f"B: median rel err {b.median_rel_err_working:.3f} (oracle floor {floor:.3f})"
    )
    assert a.spearman >= rr.MIN_SPEARMAN_A, msg
    assert a.auroc >= rr.MIN_AUROC_A, msg
    assert b.median_rel_err_working <= max(rr.MAX_MEDIAN_REL_ERR_B_FLOOR, 2 * floor), msg
