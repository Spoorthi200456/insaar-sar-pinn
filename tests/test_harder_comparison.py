"""
Review 5: the harder comparison's field-like simulator
(experiments/harder_comparison, protocol in its README, committed before the code).

These tests check that the simulator really contains the behaviours the protocol
lists, and that it is NOT method C's model.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.harder_comparison import compare as HC
from experiments.harder_comparison.simulator import (
    CLOGGING,
    FAILED,
    WORKING,
    FieldConfig,
    simulate_field,
)
from src.data.dataset import ARSDataset
from src.models import ols_baseline as ols

SMALL = replace(FieldConfig(), n_working=10, n_failed=6, n_clogging=6, seed=11)


@pytest.fixture(scope="module")
def sim():
    return simulate_field(SMALL)


def test_simulator_shares_no_code_with_method_c_or_the_old_simulator():
    src = Path("experiments/harder_comparison/simulator.py").read_text(encoding="utf-8")
    imports = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert all("synthetic_recovery" not in ln for ln in imports)
    assert all("ols_baseline" not in ln and "src." not in ln for ln in imports)


def test_rain_differs_between_structures_but_is_regionally_correlated(sim):
    r = sim.rain_per_step_mm[:, 1:]
    assert not np.allclose(r, r[0])
    corr = np.corrcoef(r)
    assert np.median(corr[np.triu_indices(len(r), 1)]) > 0.3  # shared storms
    assert sim.rain_multiplier.std() > 0.1  # a real spatial multiplier


def test_kinds_and_clogging_follow_the_protocol(sim):
    spin = sim.cfg.spinup_days
    k = sim.infiltration_capacity
    start, end = k[:, spin], k[:, -1]
    w, f, c = sim.kind == WORKING, sim.kind == FAILED, sim.kind == CLOGGING
    assert np.allclose(end[w], start[w])  # working: constant
    assert np.all(start[f] <= 0.05 * 0.3 + 1e-12)  # failed: <= 5 % of k0 <= 0.3 m/day
    ratio = end[c] / start[c]
    last = spin + int(sim.t_days[-1]) - 1  # last simulated day
    assert np.all(np.abs(k[c, last] / start[c] - ratio) < 0.01)
    assert np.all((ratio > 0.02) & (ratio < 0.16))  # 3-15 % left (one-day rounding)
    assert np.all(np.diff(k[c, spin:], axis=1) <= 0)  # monotone decline


def test_recharge_is_limited_by_infiltration_capacity(sim):
    limit = sim.infiltration_capacity * sim.area_m2[:, None] / sim.cfg.cell_area_m2
    assert np.all(sim.recharge_daily <= limit + 1e-15)
    assert np.all(sim.recharge_daily >= 0)


def test_thresholds_and_recession_times_are_structure_specific(sim):
    assert sim.threshold_mm.min() >= 2.0 and sim.threshold_mm.max() <= 15.0
    assert sim.threshold_mm.std() > 1.0
    assert sim.tau_days.min() >= 30.0 and sim.tau_days.max() <= 150.0
    assert sim.tau_days.max() / sim.tau_days.min() > 1.5


def test_pumping_draws_the_head_down_in_rabi():
    base = simulate_field(SMALL)
    none = simulate_field(replace(SMALL, pumping_range_m3_per_day=(0.0, 0.0)))
    p = base.pumped
    assert p.sum() == round(0.3 * 10) + 2 * round(0.3 * 6)
    # Identical except the pumping sink. ABSOLUTE head: a sink can only lower it,
    # at every node (lateral flow spreads the drawdown). Head relative to t0 can
    # rise, because t0 (1 January) is itself inside a pumping season.
    diff = base.head_absolute - none.head_absolute
    assert np.all(diff <= 1e-12)
    assert diff[p].min() < -0.2  # pumped nodes: tens of cm of drawdown
    assert diff[p].min() < diff[~p].min()
    assert np.allclose(base.recharge_daily, none.recharge_daily)  # pumping is not recharge


def test_insar_is_the_elastic_coupling_of_the_simulated_head(sim):
    cos_t = np.cos(np.radians(sim.incidence_deg))[:, None]
    assert np.allclose(sim.displacement_clean_cm, 100 * cos_t * sim.cfg.skeletal_storage * sim.head)
    assert np.all(sim.head[:, 0] == 0)


def test_working_signal_is_well_above_noise(sim):
    amp = np.ptp(sim.displacement_clean_cm, axis=1)
    assert np.median(amp[sim.kind == WORKING]) > 3 * sim.cfg.noise_cm


def test_dataset_rain_matches_the_simulated_intervals(sim):
    recs = HC.to_records(sim)
    ds = ARSDataset(recs, sequence_length=sim.cfg.n_acquisitions)
    for i in (0, 5, 17):
        assert np.allclose(ds[i].rainfall_mm.double().numpy(), sim.rain_per_step_mm[i], atol=1e-3)
        assert np.allclose(ds[i].time_delta_days.numpy(), sim.dt_days)


def test_simulator_is_not_method_c_model(sim):
    """
    On NOISE-FREE data C's best fit (tau grid-searched on the same structures)
    leaves a large part of the signal unexplained here. On the review-4
    simulator, which is C's model, the same fit explains essentially all of it.
    """
    from experiments.synthetic_recovery.simulator import TruthConfig, simulate

    def unexplained(d_clean, rain, dt, cos_t):
        series = [
            ols.OLSSeries(d_clean[i], np.ones(d_clean.shape[1], bool), cos_t[i], rain[i], dt)
            for i in range(len(d_clean))
        ]
        tau = ols.fit_tau(series)
        out = []
        for s in series:
            res = ols.squared_residual(s, tau)
            out.append(res / max(float(np.sum(s.displacement_cm**2)), 1e-12))
        return float(np.median(out))

    w = sim.kind == WORKING
    here = unexplained(
        sim.displacement_clean_cm[w],
        sim.rain_per_step_mm[w],
        sim.dt_days,
        np.cos(np.radians(sim.incidence_deg[w])),
    )
    old = simulate(replace(TruthConfig(), seed=11))
    n = old.cfg.n_structures
    review4 = unexplained(
        old.displacement_clean_cm,
        np.repeat(old.rain_per_step_mm[None, :], n, axis=0),
        old.dt_days,
        np.cos(np.radians(old.incidence_deg)),
    )
    assert review4 < 0.02
    assert here > 0.10


def test_late_window_is_the_last_third(sim):
    late = sim.late_steps()
    t = sim.t_days
    assert not late[0]
    assert np.all(t[late] >= (2 / 3) * t[-1])
    assert np.all(t[~late][1:] < (2 / 3) * t[-1])


def test_comparison_rule_needs_one_metric_and_no_reverse_win():
    def run(b, c):
        return {
            "B": {"rho_amount": b[0], "auroc_now": b[1]},
            "C": {"rho_amount": c[0], "auroc_now": c[1]},
        }

    # B clearly better on AUROC, C not clearly better on rho -> B clearly better
    runs = [run((0.8, 0.9), (0.8, 0.8)) for _ in range(5)]
    assert HC.overall_clearly_better(runs, "B", "C")["clearly_better"]
    # B clearly better on AUROC but C clearly better on rho -> not
    runs = [run((0.7, 0.9), (0.8, 0.8)) for _ in range(5)]
    assert not HC.overall_clearly_better(runs, "B", "C")["clearly_better"]
    # +0.05 in only 3 of 5 seeds -> not
    runs = [run((0.8, 0.9), (0.8, 0.8)) for _ in range(3)] + [
        run((0.8, 0.79), (0.8, 0.8)) for _ in range(2)
    ]
    assert not HC.overall_clearly_better(runs, "B", "C")["clearly_better"]
