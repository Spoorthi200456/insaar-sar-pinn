"""
Methods B and C of the review-4 comparison (protocol: README.md in this folder).

B  test_time_fit: per-structure physics-informed fitting at assessment time.
C  unit_response / ols_amplitude / fit_tau: ordinary least squares on a
   linear-reservoir model, no neural network. Derivation: docs/math/ols_baseline.md.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from src.models.losses import CompositeARSLoss, LossWeights
from src.training.train import GLOBAL_SCALARS, full_batch

# ---- method C (fixed by the protocol) ----
TAU_GRID_DAYS = np.logspace(1.0, 3.0, 41)  # 10 ... 1000 days
CM_PER_M = 100.0

# ---- method B (fixed by the protocol) ----
B_STEPS = 300
B_LR = 1e-3


# --------------------------------------------------------------------------- #
# C: ordinary least squares, no network
# --------------------------------------------------------------------------- #


def unit_response(p_step_mm: np.ndarray, dt_days: np.ndarray, tau_days: float) -> np.ndarray:
    """
    Head of a linear reservoir with unit storativity driven by u_k = P_k/(1000 dt_k)
    [m/day], backward Euler with h_0 = 0:  g_k = (g_{k-1} + dt_k u_k) / (1 + dt_k / tau).
    Steps with dt = 0 (the reference acquisition, padding) keep g unchanged... at 0.
    """
    p = np.asarray(p_step_mm, dtype=float)
    dt = np.asarray(dt_days, dtype=float)
    g = np.zeros(len(p))
    for k in range(1, len(p)):
        if dt[k] <= 0:
            g[k] = g[k - 1]
            continue
        u = p[k] / 1000.0 / dt[k]
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


def _series(data, i):
    """(d, reliable, cos_theta) of structure i from a Simulation-like object."""
    return (
        np.asarray(data.displacement_cm[i], dtype=float),
        np.asarray(data.coherence[i]) >= 0.7,
        float(np.cos(np.radians(data.incidence_deg[i]))),
    )


def fit_tau(data, indices) -> float:
    """
    The recession time scale tau (one value for all structures) minimising the
    total squared OLS residual over the given (TRAINING) structures, searched
    over TAU_GRID_DAYS. `data` exposes displacement_cm, coherence, incidence_deg,
    rain_per_step_mm and dt_days (a Simulation, or SeriesTable below).
    """
    best_tau, best_sse = float(TAU_GRID_DAYS[0]), np.inf
    for tau in TAU_GRID_DAYS:
        g = unit_response(data.rain_per_step_mm, data.dt_days, float(tau))
        sse = 0.0
        for i in indices:
            d, rel, cos_t = _series(data, i)
            c = ols_amplitude(d, rel, cos_t, g)
            r = (d - CM_PER_M * cos_t * c * g)[rel]
            sse += float(np.sum(r**2))
        if sse < best_sse:
            best_tau, best_sse = float(tau), sse
    return best_tau


class SeriesTable:
    """Method C's inputs taken from the SAME dataset tensors the network sees."""

    def __init__(self, samples):
        self.displacement_cm = np.stack([s.displacement_cm.double().numpy() for s in samples])
        # reliable epochs = the dataset's target mask (coherence, flags, interval bound)
        self.coherence = np.stack([s.coherence_mask.numpy().astype(float) for s in samples])
        self.incidence_deg = np.array([float(s.incidence_angle_deg) for s in samples])
        self.rain_per_step_mm = samples[0].rainfall_mm.double().numpy()
        self.dt_days = samples[0].time_delta_days.double().numpy()
        for s in samples:  # one regional rain series and one acquisition schedule
            assert np.allclose(s.rainfall_mm.double().numpy(), self.rain_per_step_mm)
            assert np.allclose(s.time_delta_days.double().numpy(), self.dt_days)


def method_c_scores(train_samples, test_samples) -> tuple[np.ndarray, float]:
    train = SeriesTable(train_samples)
    tau = fit_tau(train, range(len(train_samples)))
    test = SeriesTable(test_samples)
    g = unit_response(test.rain_per_step_mm, test.dt_days, tau)
    scores = np.array([ols_amplitude(*_series(test, i), g) for i in range(len(test_samples))])
    return scores, tau


# --------------------------------------------------------------------------- #
# B: per-structure physics-informed fitting
# --------------------------------------------------------------------------- #


def _forward(model, b):
    return model(
        design_features=b["design_features"],
        displacement_cm=b["displacement_cm"],
        rainfall_mm=b["rainfall_mm"],
        time_delta_days=b["time_delta_days"],
        adjacency=b["adjacency"],
        incidence_angle_deg=b["incidence_angle_deg"],
        coherence_mask=b["coherence_mask"],
    )


def structure_loss(loss_fn, out, b):
    """The composite loss of one batch: physics + InSAR data + regularisers. No truth."""
    return loss_fn(
        predicted_displacement_cm=out.predicted_displacement_cm,
        observed_displacement_cm=b["displacement_cm"],
        predicted_head_change=out.predicted_head_change,
        hydraulic_conductivity=out.hydraulic_conductivity,
        recharge_rate=out.recharge_rate,
        design_recharge_capacity=b["design_recharge_capacity"],
        rainfall_mm=b["rainfall_mm"],
        adjacency=b["physics_adjacency"],
        time_delta_days=b["time_delta_days"],
        storativity=out.storativity,
        leakance=out.leakance,
        recharge_area_m2=b["recharge_area_m2"],
        catchment_area_m2=b["catchment_area_m2"],
        observation_mask=b["coherence_mask"],
        valid_mask=b["valid_step_mask"],
    )


def test_time_fit(model, record, design_stats, sequence_length, steps=B_STEPS, lr=B_LR):
    """
    Fit a COPY of the trained model to one structure's own data (method B).
    The global constants S, S_k, L stay frozen at the trained values; all other
    weights are optimised with Adam on that structure's composite loss, in eval
    mode (no dropout). Returns (fitted_model, loss_before, loss_after).
    """
    fitted = copy.deepcopy(model)
    fitted.eval()
    for name, p in fitted.named_parameters():
        p.requires_grad_(name not in GLOBAL_SCALARS)
    batch = full_batch([record], sequence_length, design_stats)
    loss_fn = CompositeARSLoss(LossWeights())
    params = [p for p in fitted.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    with torch.no_grad():
        loss_before = float(structure_loss(loss_fn, _forward(fitted, batch), batch).total)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = structure_loss(loss_fn, _forward(fitted, batch), batch).total
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
    with torch.no_grad():
        loss_after = float(structure_loss(loss_fn, _forward(fitted, batch), batch).total)
    return fitted, loss_before, loss_after


def mean_recharge(model, records, design_stats, sequence_length) -> np.ndarray:
    """Score = mean predicted recharge over steps 1..K-1, per structure (records order)."""
    b = full_batch(records, sequence_length, design_stats)
    model.eval()
    with torch.no_grad():
        r = _forward(model, b).recharge_rate[:, 1:].mean(dim=1).double().numpy()
    order = {sid: i for i, sid in enumerate(b["structure_ids"])}
    return np.array([r[order[rec.design.structure_id]] for rec in records])
