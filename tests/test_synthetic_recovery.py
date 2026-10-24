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
from scipy.stats import spearmanr

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


class TestIdentifiability:
    """
    What InSAR alone can and cannot determine (finding-2 plan §7; README
    "Identifiability"). These are exact properties of the physics, checked on
    the independent simulator, not of the network.
    """

    def test_insar_cannot_separate_skeletal_storage_from_recharge(self, sim):
        # d = 100 cos(theta) S_k h and h is linear in R, so (S_k, R) -> (c S_k, R / c)
        # leaves every InSAR observation unchanged.
        c = 10.0
        g = lateral_operator(sim.a_ij, sim.transmissivity)
        h = integrate_head(sim.cfg, sim.recharge / c, sim.dt_days, g)
        d = los_cm(h, sim.cfg.skeletal_storage * c, sim.incidence_deg)
        assert np.allclose(d, sim.displacement_clean_cm, rtol=1e-12, atol=1e-14)

    def test_head_cannot_separate_storativity_from_recharge(self, sim):
        # (S, L, T, R) -> c (S, L, T, R) leaves the head unchanged, so even
        # well heads do not give absolute R without S.
        c = 10.0
        g = lateral_operator(sim.a_ij, sim.transmissivity * c)
        cfg = replace(sim.cfg, storativity=sim.cfg.storativity * c)
        h = integrate_head(cfg, sim.recharge * c, sim.dt_days, g, leakance=sim.cfg.leakance * c)
        assert np.allclose(h, sim.head, rtol=1e-12, atol=1e-14)

    def test_ranking_survives_a_wrong_global_skeletal_storage(self, sim):
        # With S_k wrong by a common factor, the best possible estimate of eta
        # is scaled by 1/factor, but the ORDER of structures is still recovered.
        wrong = replace(sim, cfg=replace(sim.cfg, skeletal_storage=sim.cfg.skeletal_storage / 10))
        eta_hat = oracle_eta(wrong)
        working = ~sim.failing
        assert np.median(eta_hat[working] / sim.eta[working]) == pytest.approx(10.0, rel=0.1)
        assert spearmanr(eta_hat, sim.eta).statistic > 0.9


# Official pre-registered outcome: run 1b FAILED (experiments/synthetic_recovery/
# README.md; reproducible at commit a249eae with --protocol preregistered). The
# tests below check the CURRENT training code (protocol "current-code-run3",
# revised after seeing run 1 by the author's decisions) against the SAME
# pre-registered thresholds.


@pytest.mark.slow
def test_current_code_ranks_structures_from_insar_alone(sim):
    a = rr.run_scenario("A_insar_only", sim, protocol="current-code-run3")
    msg = f"A: spearman {a.spearman:.3f} auroc {a.auroc:.3f}"
    assert a.spearman >= rr.MIN_SPEARMAN_A, msg
    assert a.auroc >= rr.MIN_AUROC_A, msg


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Scenario B (S and S_k fixed to truth) misses the pre-registered threshold under the "
        "current code: median relative error of mean recharge 0.42 > max(0.30, 2 x oracle "
        "0.061) in run 3. The model underfits (InSAR RMSE 0.375 cm vs 0.30 cm noise; head at "
        "69 % of true amplitude); cause not established. Absolute recharge is therefore NOT "
        "recovered even with known aquifer constants. See experiments/synthetic_recovery/"
        "README.md (run 3). strict=True: if this starts passing, revisit the README."
    ),
)
def test_current_code_recovers_absolute_recharge_with_known_constants(sim):
    floor = rr.oracle_floor(sim)
    b = rr.run_scenario("B_oracle_S_Sk", sim, oracle_constants=True, protocol="current-code-run3")
    assert b.median_rel_err_working <= max(
        rr.MAX_MEDIAN_REL_ERR_B_FLOOR, 2 * floor
    ), f"B: median rel err {b.median_rel_err_working:.3f} (oracle floor {floor:.3f})"


# --------------------------------------------------------------------------- #
# Held-out structures (review 3, run 4): trained on the training split,
# scored ONLY on test structures the model never saw. Same thresholds.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def heldout_sim():
    from experiments.synthetic_recovery import heldout

    return simulate(heldout.HELDOUT_CONFIG)


class TestHeldOutSplit:
    def test_test_structures_never_reach_training(self, heldout_sim):
        from experiments.synthetic_recovery import heldout

        _, split = heldout.make_split(heldout_sim)
        train, val, test = set(split.train), set(split.validation), set(split.test)
        assert not (train & val) and not (train & test) and not (val & test)
        assert len(test) == 10
        fail = {f"SYN-{i:02d}" for i in range(64) if heldout_sim.failing[i]}
        assert 0 < len(test & fail) < len(test)  # both classes present


@pytest.mark.slow
def test_heldout_ranks_unseen_structures_from_insar_alone(heldout_sim):
    from experiments.synthetic_recovery import heldout

    a = heldout.run_heldout("A_insar_only", heldout_sim)
    msg = f"held-out A: spearman {a.spearman:.3f} auroc {a.auroc:.3f} (n_test {a.n_test})"
    assert a.spearman >= rr.MIN_SPEARMAN_A, msg
    assert a.auroc >= rr.MIN_AUROC_A, msg


@pytest.mark.slow
def test_heldout_recovers_absolute_recharge_with_known_constants(heldout_sim):
    from experiments.synthetic_recovery import heldout

    b = heldout.run_heldout("B_oracle_S_Sk", heldout_sim, oracle_constants=True)
    limit = max(rr.MAX_MEDIAN_REL_ERR_B_FLOOR, 2 * b.oracle_floor_test)
    assert (
        b.median_rel_err_working <= limit
    ), f"held-out B: median rel err {b.median_rel_err_working:.3f} > {limit:.3f}"
