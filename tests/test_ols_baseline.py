"""
Method C as a library (review 5): `src/models/ols_baseline.py`.

Review 4 ran method C from experiments/ with ONE regional rain series. The API
must handle structures with their own rain gauges, so every function here works
per structure. Derivation: docs/math/ols_baseline.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.ols_baseline import (
    OLSSeries,
    fit_amplitude,
    fit_tau,
    mean_relative_recharge,
    predicted_displacement_cm,
    unit_response,
)

TAU = 60.0
S, S_K = 0.05, 5e-3


def _schedule(k=40, seed=0):
    rng = np.random.default_rng(seed)
    dt = rng.choice([6.0, 12.0, 24.0], size=k)
    dt[0] = 0.0
    return dt


def _series(eta, rain, dt, theta=35.0, noise=0.0, seed=0):
    g = unit_response(rain, dt, TAU)
    d = 100.0 * np.cos(np.radians(theta)) * (eta * S_K / S) * g
    d = d + np.random.default_rng(seed).normal(0, noise, size=d.shape)
    return OLSSeries(
        displacement_cm=d,
        reliable=np.ones(len(d), dtype=bool),
        cos_theta=float(np.cos(np.radians(theta))),
        rain_per_step_mm=rain,
        dt_days=dt,
    )


def _rain(dt, scale, seed):
    rng = np.random.default_rng(seed)
    return scale * rng.gamma(0.6, 8.0, size=len(dt)) * dt / 12.0


def test_amplitude_is_eta_sk_over_s_with_each_structures_own_rain():
    dt = _schedule()
    for eta, scale, seed in [(0.5, 1.0, 1), (0.5, 3.0, 2), (0.1, 0.5, 3)]:
        c, _ = fit_amplitude(_series(eta, _rain(dt, scale, seed), dt), TAU)
        assert c == pytest.approx(eta * S_K / S, rel=1e-9)


def test_tau_is_recovered_from_structures_with_different_rain():
    dt = _schedule()
    series = [_series(0.4, _rain(dt, 0.5 + i, 10 + i), dt) for i in range(5)]
    tau = fit_tau(series)
    grid = np.logspace(1.0, 3.0, 41)
    assert tau == pytest.approx(grid[np.argmin(np.abs(np.log(grid / TAU)))])


def test_relative_recharge_ranks_by_flux_not_by_efficiency():
    # Same efficiency eta, three times the rain -> three times the recharge.
    dt = _schedule()
    wet = _series(0.3, _rain(dt, 3.0, 5), dt)
    dry = OLSSeries(**{**wet.__dict__, "rain_per_step_mm": wet.rain_per_step_mm / 3.0})
    dry = _series(0.3, dry.rain_per_step_mm, dt)
    c_wet, _ = fit_amplitude(wet, TAU)
    c_dry, _ = fit_amplitude(dry, TAU)
    assert c_wet == pytest.approx(c_dry, rel=1e-9)  # same efficiency
    ratio = mean_relative_recharge(wet, c_wet) / mean_relative_recharge(dry, c_dry)
    assert ratio == pytest.approx(3.0, rel=1e-9)


def test_relative_recharge_equals_true_flux_times_the_common_factor():
    dt = _schedule()
    rain = _rain(dt, 1.0, 7)
    s = _series(0.45, rain, dt)
    c, _ = fit_amplitude(s, TAU)
    m = dt > 0
    true_mean_flux = np.mean(0.45 * rain[m] / 1000.0 / dt[m])  # m/day
    assert mean_relative_recharge(s, c) == pytest.approx(true_mean_flux * S_K / S, rel=1e-9)


def test_predicted_displacement_reproduces_noise_free_data():
    dt = _schedule()
    s = _series(0.2, _rain(dt, 1.0, 9), dt)
    c, g = fit_amplitude(s, TAU)
    assert np.allclose(predicted_displacement_cm(s, c, g), s.displacement_cm, atol=1e-12)


def test_unreliable_epochs_have_no_influence():
    dt = _schedule()
    s = _series(0.3, _rain(dt, 1.0, 4), dt)
    d = s.displacement_cm.copy()
    rel = s.reliable.copy()
    d[10], rel[10] = 1e3, False
    c, _ = fit_amplitude(OLSSeries(d, rel, s.cos_theta, s.rain_per_step_mm, s.dt_days), TAU)
    assert c == pytest.approx(0.3 * S_K / S, rel=1e-9)


def test_review4_implementation_uses_this_module():
    from experiments.method_comparison import methods as M
    from src.models import ols_baseline as C

    assert M.unit_response is C.unit_response
    assert M.ols_amplitude is C.ols_amplitude
    assert np.array_equal(M.TAU_GRID_DAYS, C.TAU_GRID_DAYS)
