"""
Review 4: building blocks of the A/B/C method comparison
(experiments/method_comparison/README.md). Fast tests only; the comparison
itself is run by experiments/method_comparison/compare.py.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from experiments.synthetic_recovery.simulator import (
    TruthConfig,
    integrate_head,
    simulate,
)


@pytest.fixture(scope="module")
def sim():
    return simulate(replace(TruthConfig(), n_structures=24, n_failing=9, seed=3))


def _isolated(sim):
    return np.flatnonzero((sim.a_ij > 0).sum(axis=1) == 0)


class TestThresholdTruth:
    def test_default_is_the_linear_truth(self, sim):
        p, dt = sim.rain_per_step_mm, sim.dt_days
        expected = np.zeros_like(sim.recharge)
        expected[:, 1:] = sim.eta[:, None] * p[None, 1:] / 1000.0 / dt[None, 1:]
        assert np.allclose(sim.recharge, expected, rtol=1e-12)

    def test_threshold_truth_infiltrates_only_above_the_rain_rate(self):
        cfg = replace(
            TruthConfig(), n_structures=8, n_failing=3, seed=3, recharge_threshold_mm_per_day=2.0
        )
        s = simulate(cfg)
        rate = s.rain_per_step_mm[1:] / s.dt_days[1:]  # mm/day
        expected = s.eta[:, None] * np.maximum(rate - 2.0, 0.0)[None, :] / 1000.0
        assert np.allclose(s.recharge[:, 1:], expected, rtol=1e-12)
        assert (rate < 2.0).any() and (rate > 2.0).any()


class TestOLSBaseline:
    def test_unit_response_is_the_linear_reservoir(self, sim):
        from experiments.method_comparison.methods import unit_response

        tau = 60.0
        g = unit_response(sim.rain_per_step_mm, sim.dt_days, tau)
        cfg = replace(sim.cfg, storativity=1.0, recession_days=tau)  # S = 1, L = 1/tau
        u = np.zeros((1, len(g)))
        u[0, 1:] = sim.rain_per_step_mm[1:] / 1000.0 / sim.dt_days[1:]
        h = integrate_head(cfg, u, sim.dt_days, np.zeros((1, 1)))
        assert np.allclose(g, h[0], rtol=1e-12)

    def test_recovers_amplitude_exactly_without_noise(self, sim):
        from experiments.method_comparison.methods import ols_amplitude, unit_response

        g = unit_response(sim.rain_per_step_mm, sim.dt_days, sim.cfg.recession_days)
        cos_t = np.cos(np.radians(sim.incidence_deg))
        for i in _isolated(sim):
            c = ols_amplitude(sim.displacement_clean_cm[i], np.ones(len(g), bool), cos_t[i], g)
            truth = sim.eta[i] * sim.cfg.skeletal_storage / sim.cfg.storativity
            assert c == pytest.approx(truth, rel=1e-9)

    def test_ignores_unreliable_epochs(self, sim):
        from experiments.method_comparison.methods import ols_amplitude, unit_response

        g = unit_response(sim.rain_per_step_mm, sim.dt_days, 60.0)
        i = _isolated(sim)[0]
        d = sim.displacement_clean_cm[i].copy()
        reliable = np.ones(len(d), bool)
        reliable[7] = False
        d_bad = d.copy()
        d_bad[7] += 50.0
        cos_t = np.cos(np.radians(sim.incidence_deg[i]))
        assert ols_amplitude(d_bad, reliable, cos_t, g) == pytest.approx(
            ols_amplitude(d, reliable, cos_t, g), rel=1e-12
        )

    def test_recession_time_found_on_training_structures(self, sim):
        from experiments.method_comparison.methods import TAU_GRID_DAYS, fit_tau

        clean = replace(
            sim, displacement_cm=sim.displacement_clean_cm, coherence=np.ones_like(sim.coherence)
        )
        tau = fit_tau(clean, list(_isolated(sim)))
        nearest = TAU_GRID_DAYS[np.argmin(np.abs(np.log(TAU_GRID_DAYS / 60.0)))]
        assert tau == pytest.approx(nearest)
        assert len(TAU_GRID_DAYS) == 41 and TAU_GRID_DAYS[0] == 10.0 and TAU_GRID_DAYS[-1] == 1000.0


class TestTestTimeFitting:
    def test_fits_one_structure_without_truth_and_keeps_global_constants(self, sim):
        import inspect

        from experiments.method_comparison.methods import B_STEPS, test_time_fit
        from experiments.synthetic_recovery.run_recovery import to_records
        from src.data.dataset import ARSDataset
        from src.models.network import ARSNet

        assert B_STEPS == 300
        assert not any(
            "truth" in p or "eta" in p or "recharge" in p
            for p in inspect.signature(test_time_fit).parameters
        )
        records = to_records(sim)
        stats = ARSDataset(records[:10], sequence_length=sim.cfg.n_acquisitions)._design_stats
        torch.manual_seed(0)
        base = ARSNet(design_feature_dim=3)
        before = {k: v.clone() for k, v in base.state_dict().items()}
        fitted, loss_start, loss_end = test_time_fit(
            base, records[12], stats, sim.cfg.n_acquisitions, steps=40
        )
        assert loss_end < loss_start  # it really fits the structure
        for name in ("storativity_logit", "skeletal_storage_raw", "leakance_raw"):
            assert torch.equal(fitted.state_dict()[name], before[name])  # frozen
        assert all(torch.equal(v, base.state_dict()[k]) for k, v in before.items())  # copy


class TestProtocolSplits:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_each_seed_holds_out_at_least_30_structures_of_both_kinds(self, seed):
        from experiments.method_comparison.compare import make_split, simulate_seed

        s = simulate_seed(seed, "linear")
        _, split = make_split(s, seed)
        test = set(split.test)
        assert len(test) >= 30
        assert not (test & set(split.train)) and not (test & set(split.validation))
        failing = {f"SYN-{i:02d}" for i in range(s.cfg.n_structures) if s.failing[i]}
        assert 0 < len(test & failing) < len(test)
