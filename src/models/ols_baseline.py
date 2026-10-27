"""
Method C: ordinary least squares on a linear-reservoir model. No neural network.

Derivation: docs/math/ols_baseline.md. In review 4 it ranked unseen structures as
well as an oracle and better than the physics-informed network fitted to each
structure (experiments/method_comparison/README.md), so it is the default
assessment method of the API (author decision, review 5).

Per structure:
    u_k = P_k / (1000 dt_k)                               rain rate [m/day]
    g_k = (g_{k-1} + dt_k u_k) / (1 + dt_k / tau)         unit response [m], g_0 = 0
    d_k = 100 cos(theta) c g_k                            LOS [cm]
    c   = sum w d / sum w^2,  w_k = 100 cos(theta) g_k    over reliable epochs
c = eta S_k / S is the only fitted number; tau is one regional value chosen by
least squares over a set of structures (no ground truth is used).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

TAU_GRID_DAYS = np.logspace(1.0, 3.0, 41)  # 10 ... 1000 days
CM_PER_M = 100.0
MM_PER_M = 1000.0


@dataclass(frozen=True)
class OLSSeries:
    """One structure's inputs to method C, each (K,) except cos_theta."""

    displacement_cm: np.ndarray
    reliable: np.ndarray  # bool: coherent, unflagged, real acquisition
    cos_theta: float
    rain_per_step_mm: np.ndarray  # rain in the interval ending at step k [mm]
    dt_days: np.ndarray  # 0 at the first acquisition and on padding


def unit_response(p_step_mm: np.ndarray, dt_days: np.ndarray, tau_days: float) -> np.ndarray:
    """
    Head of a linear reservoir with unit storativity driven by u_k = P_k/(1000 dt_k)
    [m/day], backward Euler with h_0 = 0:  g_k = (g_{k-1} + dt_k u_k) / (1 + dt_k / tau).
    Steps with dt = 0 (the reference acquisition, padding) keep g unchanged.
    """
    p = np.asarray(p_step_mm, dtype=float)
    dt = np.asarray(dt_days, dtype=float)
    g = np.zeros(len(p))
    for k in range(1, len(p)):
        if dt[k] <= 0:
            g[k] = g[k - 1]
            continue
        u = p[k] / MM_PER_M / dt[k]
        g[k] = (g[k - 1] + dt[k] * u) / (1.0 + dt[k] / tau_days)
    return g


def ols_amplitude(d_cm: np.ndarray, reliable: np.ndarray, cos_theta: float, g: np.ndarray) -> float:
    """Least-squares c in d_k = 100 cos(theta) c g_k over reliable epochs: c = sum w d / sum w^2."""
    w = CM_PER_M * cos_theta * np.asarray(g, dtype=float)
    m = np.asarray(reliable, dtype=bool)
    denom = float(np.sum(w[m] ** 2))
    if denom <= 0:
        return 0.0
    return float(np.sum(w[m] * np.asarray(d_cm, dtype=float)[m]) / denom)


def fit_amplitude(s: OLSSeries, tau_days: float) -> tuple[float, np.ndarray]:
    """(c, g) for one structure, using that structure's own rain and schedule."""
    g = unit_response(s.rain_per_step_mm, s.dt_days, tau_days)
    return ols_amplitude(s.displacement_cm, s.reliable, s.cos_theta, g), g


def squared_residual(s: OLSSeries, tau_days: float) -> float:
    """Sum over reliable epochs of (d - 100 cos(theta) c g)^2 at the best c [cm^2]."""
    c, g = fit_amplitude(s, tau_days)
    r = (np.asarray(s.displacement_cm, dtype=float) - CM_PER_M * s.cos_theta * c * g)[
        np.asarray(s.reliable, dtype=bool)
    ]
    return float(np.sum(r**2))


def fit_tau(series: Sequence[OLSSeries], grid: np.ndarray = TAU_GRID_DAYS) -> float:
    """The one regional tau minimising the total squared OLS residual over `series`."""
    best_tau, best_sse = float(grid[0]), np.inf
    for tau in grid:
        sse = sum(squared_residual(s, float(tau)) for s in series)
        if sse < best_sse:
            best_tau, best_sse = float(tau), sse
    return best_tau


def predicted_displacement_cm(s: OLSSeries, c: float, g: np.ndarray) -> np.ndarray:
    return CM_PER_M * s.cos_theta * c * np.asarray(g, dtype=float)


def mean_relative_recharge(s: OLSSeries, c: float) -> float:
    """
    Mean over intervals (dt > 0) of c * u_k [m/day x the unknown common factor S/S_k].
    Under C's model R_k = eta u_k = (S/S_k) c u_k, so this ranks structures by their
    mean recharge flux even when rain differs between structures. With one shared rain
    series it orders structures exactly as c does.
    """
    dt = np.asarray(s.dt_days, dtype=float)
    m = dt > 0
    if not m.any():
        return 0.0
    u = np.asarray(s.rain_per_step_mm, dtype=float)[m] / MM_PER_M / dt[m]
    return float(c * np.mean(u))
