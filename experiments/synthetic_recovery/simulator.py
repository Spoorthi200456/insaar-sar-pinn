"""
Independent forward simulator for the known-answer experiment (review 2, step 4).

Pure NumPy. It shares NO code with the neural network or its loss: its own
haversine, its own stencil weights, its own implicit time stepping, its own
LOS coupling. If the model and the simulator agree, it is because both encode
the same physics, not because they call the same function.

Physics (CLAUDE.md form (b) with leakage, docs/math/*.md):
    S dh/dt = div(T grad h) - L h + R           [m/day]
    discretised backward Euler on a fixed structure graph:
    ((S/dt + L) I - G) h_k = (S/dt) h_{k-1} + R_k,   h_0 = 0
    (G h)_i = sum_j a_ij (T_i+T_j)/2 (h_j - h_i),   a_ij = 1/d_ij^2 within 5 km
    d_LOS [cm] = 100 cos(theta_i) S_k h + noise     (+ = toward satellite)

Planted truth per structure i: infiltration fraction eta_i, so
    R_i,k = eta_i * P_k / 1000 / dt_k     [m/day]
with P_k the rain [mm] in the causal interval (t_{k-1}, t_k].
"Working" structures have eta in [0.3, 0.6]; "failing" (clogged) ones in [0, 0.05].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

R_EARTH_M = 6_371_000.0
RADIUS_M = 5_000.0


@dataclass(frozen=True)
class TruthConfig:
    n_structures: int = 16
    n_failing: int = 6
    n_acquisitions: int = 40
    seed: int = 0
    storativity: float = 0.05  # S [-]
    skeletal_storage: float = 5e-3  # S_k [-]  (compliant alluvium: signal > noise)
    recession_days: float = 60.0  # S / L [day]
    noise_cm: float = 0.3  # InSAR noise at good coherence [cm]
    noise_low_coherence_cm: float = 1.0  # extra-noisy low-coherence epochs [cm]
    low_coherence_fraction: float = 0.15
    revisit_days: tuple[float, ...] = (6.0, 12.0, 24.0)
    revisit_probs: tuple[float, ...] = (0.25, 0.5, 0.25)

    @property
    def leakance(self) -> float:  # L [1/day]
        return self.storativity / self.recession_days


@dataclass
class Simulation:
    cfg: TruthConfig
    lat: np.ndarray  # (N,) deg
    lon: np.ndarray  # (N,) deg
    t_days: np.ndarray  # (K,) acquisition times since start [day]
    dt_days: np.ndarray  # (K,) dt_k, dt_0 = 0 (no interval before the first; as the dataset)
    rain_days: np.ndarray  # (D,) time of each daily rain record [day]
    rain_mm: np.ndarray  # (D,) daily rain [mm]
    rain_per_step_mm: np.ndarray  # (K,) P_k [mm], causal (t_{k-1}, t_k]
    eta: np.ndarray  # (N,) planted infiltration fraction
    failing: np.ndarray  # (N,) bool
    transmissivity: np.ndarray  # (N,) T [m2/day]
    incidence_deg: np.ndarray  # (N,)
    a_ij: np.ndarray  # (N, N) [1/m2]
    recharge: np.ndarray  # (N, K) R [m/day]
    head: np.ndarray  # (N, K) h [m]
    displacement_clean_cm: np.ndarray  # (N, K)
    displacement_cm: np.ndarray  # (N, K) with noise
    coherence: np.ndarray  # (N, K)
    recharge_area_m2: np.ndarray  # (N,)
    design_capacity_m3_per_day: np.ndarray  # (N,)
    height_m: np.ndarray
    catchment_km2: np.ndarray
    extras: dict = field(default_factory=dict)


def haversine_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    a = np.sin((la - la.T) / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin((lo - lo.T) / 2) ** 2
    return 2 * R_EARTH_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def stencil(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    d = haversine_m(lat, lon)
    a = np.zeros_like(d)
    near = (d > 0) & (d <= RADIUS_M)
    a[near] = 1.0 / d[near] ** 2
    return a


def lateral_operator(a: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Matrix G with (G h)_i = sum_j a_ij (T_i+T_j)/2 (h_j - h_i)  [m/day per m]."""
    w = a * 0.5 * (t[:, None] + t[None, :])
    return w - np.diag(w.sum(axis=1))


def integrate_head(
    cfg: TruthConfig,
    recharge: np.ndarray,
    dt: np.ndarray,
    g: np.ndarray,
    leakance: float | None = None,
) -> np.ndarray:
    """Backward Euler of S dh/dt = G h - L h + R, h_0 = 0. recharge (N, K) -> head (N, K)."""
    n, k = recharge.shape
    s = cfg.storativity
    leak = cfg.leakance if leakance is None else leakance
    h = np.zeros((n, k))
    eye = np.eye(n)
    for step in range(1, k):
        m = (s / dt[step] + leak) * eye - g
        h[:, step] = np.linalg.solve(m, (s / dt[step]) * h[:, step - 1] + recharge[:, step])
    return h


def los_cm(head_m: np.ndarray, skeletal_storage: float, incidence_deg: np.ndarray) -> np.ndarray:
    """Independent re-implementation of d_LOS = 100 cos(theta) S_k h."""
    return 100.0 * np.cos(incidence_deg * np.pi / 180.0)[:, None] * skeletal_storage * head_m


def _daily_rain(rng: np.random.Generator, n_days: int) -> np.ndarray:
    """Monsoon-like daily rain: June-September wet (p=0.6, mean 15 mm), else dry (p=0.05, 5 mm)."""
    doy = np.arange(n_days) % 365
    wet = (doy >= 151) & (doy < 273)
    p = np.where(wet, 0.6, 0.05)
    mean = np.where(wet, 15.0, 5.0)
    return np.where(rng.random(n_days) < p, rng.exponential(mean), 0.0)


def simulate(cfg: TruthConfig = TruthConfig()) -> Simulation:
    rng = np.random.default_rng(cfg.seed)
    n, k = cfg.n_structures, cfg.n_acquisitions

    # --- geometry: two clusters of 3 (1.5 km spacing) + scattered structures
    lat = np.empty(n)
    lon = np.empty(n)
    for c, (la0, lo0) in enumerate([(13.0, 77.0), (13.3, 77.6)]):
        for j in range(3):
            lat[3 * c + j] = la0
            lon[3 * c + j] = lo0 + j * 1500.0 / (R_EARTH_M * np.cos(np.radians(la0))) * 180 / np.pi
    lat[6:] = 12.4 + rng.uniform(0, 1.2, n - 6)
    lon[6:] = 76.4 + rng.uniform(0, 1.6, n - 6)
    a = stencil(lat, lon)

    # --- acquisitions with irregular revisit, daily rain after t0
    gaps = rng.choice(cfg.revisit_days, size=k - 1, p=cfg.revisit_probs)
    t = np.concatenate([[0.0], np.cumsum(gaps)])
    dt = np.concatenate([[0.0], gaps])  # first acquisition: no interval (as the dataset)
    n_days = int(t[-1]) + 1
    rain_days = np.arange(n_days) + 0.5  # noon of each day after t0
    rain_mm = _daily_rain(rng, n_days)
    idx = np.searchsorted(t, rain_days, side="left")  # interval (t_{k-1}, t_k]
    keep = idx < k
    p_step = np.bincount(idx[keep], weights=rain_mm[keep], minlength=k)

    # --- planted truths
    failing = np.zeros(n, dtype=bool)
    failing[rng.choice(n, size=cfg.n_failing, replace=False)] = True
    eta = np.where(failing, rng.uniform(0.0, 0.05, n), rng.uniform(0.3, 0.6, n))
    transmissivity = np.exp(rng.uniform(np.log(20.0), np.log(200.0), n))
    theta = rng.uniform(29.0, 46.0, n)

    recharge = np.zeros((n, k))
    recharge[:, 1:] = eta[:, None] * p_step[None, 1:] / 1000.0 / dt[None, 1:]
    g = lateral_operator(a, transmissivity)
    head = integrate_head(cfg, recharge, dt, g)
    d_clean = los_cm(head, cfg.skeletal_storage, theta)

    low = rng.random((n, k)) < cfg.low_coherence_fraction
    low[:, 0] = False
    coherence = np.where(low, rng.uniform(0.3, 0.65, (n, k)), rng.uniform(0.75, 0.98, (n, k)))
    noise = np.where(low, cfg.noise_low_coherence_cm, cfg.noise_cm) * rng.standard_normal((n, k))
    d_obs = d_clean + noise

    # --- design data: designed for eta_design in [0.3, 0.6] at 30-90 % utilisation
    area = rng.uniform(2000.0, 8000.0, n)
    height = rng.uniform(2.0, 8.0, n)
    eta_design = rng.uniform(0.3, 0.6, n)
    mean_flux_design = eta_design * np.mean(p_step[1:] / dt[1:]) / 1000.0
    capacity = mean_flux_design * area / rng.uniform(0.3, 0.9, n)
    catchment = rng.uniform(1.0, 20.0, n)

    return Simulation(
        cfg=cfg,
        lat=lat,
        lon=lon,
        t_days=t,
        dt_days=dt,
        rain_days=rain_days,
        rain_mm=rain_mm,
        rain_per_step_mm=p_step,
        eta=eta,
        failing=failing,
        transmissivity=transmissivity,
        incidence_deg=theta,
        a_ij=a,
        recharge=recharge,
        head=head,
        displacement_clean_cm=d_clean,
        displacement_cm=d_obs,
        coherence=coherence,
        recharge_area_m2=area,
        design_capacity_m3_per_day=capacity,
        height_m=height,
        catchment_km2=catchment,
    )


def oracle_eta(sim: Simulation) -> np.ndarray:
    """
    Best-case estimate of eta with PERFECT knowledge of S, S_k, L, T, theta and
    the graph: h is linear in eta, so d = Phi eta + noise and least squares on
    the reliable (coherence >= 0.7) epochs gives eta_hat. This is the noise
    floor no model can beat on average.
    """
    cfg = sim.cfg
    n, k = sim.recharge.shape
    g = lateral_operator(sim.a_ij, sim.transmissivity)
    unit_r = np.zeros((n, k))
    unit_r[:, 1:] = sim.rain_per_step_mm[None, 1:] / 1000.0 / sim.dt_days[None, 1:]
    cols = []
    for j in range(n):
        r = np.zeros((n, k))
        r[j] = unit_r[j]
        cols.append(
            los_cm(integrate_head(cfg, r, sim.dt_days, g), cfg.skeletal_storage, sim.incidence_deg)
        )
    reliable = sim.coherence >= 0.7
    phi = np.stack([c[reliable] for c in cols], axis=1)  # (n_obs, N)
    eta_hat, *_ = np.linalg.lstsq(phi, sim.displacement_cm[reliable], rcond=None)
    return eta_hat
