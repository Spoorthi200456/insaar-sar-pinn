"""
Field-like simulator for the harder comparison (review 5; protocol: README.md here).

Pure NumPy. It imports NOTHING from experiments/synthetic_recovery/simulator.py,
src/models/ols_baseline.py or the network, and it is deliberately NOT method C's
model:
- daily time step with a 365-day spin-up (the aquifer is not at rest at t0);
- rain differs per structure (regional storms, patchy local rain, a smooth
  spatial multiplier);
- rain -> recharge through a daily pond bucket with a structure-specific runoff
  threshold, storage cap, evaporation and an infiltration-capacity limit;
- working / failed / progressively clogging structures;
- seasonal pumping near some structures (a sink, not recharge);
- structure-specific recession times and lateral flow on the structure graph.

Aquifer, backward Euler with dt = 1 day (form (b), CLAUDE.md):
    S (h_d - h_{d-1}) = sum_j a_ij (T_i+T_j)/2 (h_j - h_i) - L_i h_i + R_i,d - Q_i,d
    a_ij = 1/d_ij^2 within 5 km; L_i = S / tau_i
InSAR: d_k = 100 cos(theta) S_k (h(t_k) - h(t_0)) + noise   [cm]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

R_EARTH_M = 6_371_000.0
RADIUS_M = 5_000.0
WORKING, FAILED, CLOGGING = 0, 1, 2
KIND_NAMES = ("working", "failed", "clogging")


@dataclass(frozen=True)
class FieldConfig:
    seed: int = 0
    n_working: int = 70
    n_failed: int = 45
    n_clogging: int = 45
    n_acquisitions: int = 60
    spinup_days: int = 365
    revisit_days: tuple[float, ...] = (6.0, 12.0, 24.0)
    revisit_probs: tuple[float, ...] = (0.25, 0.5, 0.25)
    storativity: float = 0.05  # S [-]
    skeletal_storage: float = 5e-3  # S_k [-]
    cell_area_m2: float = 2e5  # area a structure's recharge spreads over [m2]
    tau_range_days: tuple[float, float] = (30.0, 150.0)  # log-uniform
    k0_range_m_per_day: tuple[float, float] = (0.1, 0.3)
    failed_fraction_range: tuple[float, float] = (0.0, 0.05)
    clogging_final_range: tuple[float, float] = (0.03, 0.15)
    threshold_range_mm: tuple[float, float] = (2.0, 15.0)
    runoff_coeff_range: tuple[float, float] = (0.1, 0.3)
    catchment_range_km2: tuple[float, float] = (0.2, 3.0)
    area_range_m2: tuple[float, float] = (2000.0, 8000.0)
    height_range_m: tuple[float, float] = (2.0, 8.0)
    evaporation_mm_per_day: float = 5.0
    pumped_fraction: float = 0.3
    pumping_range_m3_per_day: tuple[float, float] = (50.0, 300.0)
    rain_field_sigma: float = 0.35
    rain_corr_km: float = 20.0
    local_dry_prob: float = 0.3
    local_lognormal_sigma: float = 0.5
    noise_cm: float = 0.3
    noise_low_coherence_cm: float = 1.0
    low_coherence_fraction: float = 0.15

    @property
    def n_structures(self) -> int:
        return self.n_working + self.n_failed + self.n_clogging


@dataclass
class FieldSimulation:
    cfg: FieldConfig
    lat: np.ndarray  # (N,)
    lon: np.ndarray  # (N,)
    kind: np.ndarray  # (N,) WORKING / FAILED / CLOGGING
    pumped: np.ndarray  # (N,) bool
    t_days: np.ndarray  # (K,) acquisition times since t0 [day], integers
    dt_days: np.ndarray  # (K,) dt_0 = 0
    rain_mm: np.ndarray  # (N, D) daily rain at each structure, days 0..D-1 after t0
    rain_per_step_mm: np.ndarray  # (N, K) rain binned to (t_{k-1}, t_k]
    recharge_daily: np.ndarray  # (N, D_total) R [m/day] incl. spin-up
    recharge_interval: np.ndarray  # (N, K) interval-mean R [m/day]; column 0 = 0
    infiltration_capacity: np.ndarray  # (N, D_total) k [m/day]
    head: np.ndarray  # (N, K) h(t_k) - h(t_0) [m]
    head_absolute: np.ndarray  # (N, K) h(t_k) above the start of the spin-up [m]
    displacement_clean_cm: np.ndarray  # (N, K)
    displacement_cm: np.ndarray  # (N, K)
    coherence: np.ndarray  # (N, K)
    incidence_deg: np.ndarray  # (N,)
    tau_days: np.ndarray  # (N,)
    threshold_mm: np.ndarray  # (N,)
    rain_multiplier: np.ndarray  # (N,)
    transmissivity: np.ndarray  # (N,)
    area_m2: np.ndarray
    height_m: np.ndarray
    catchment_km2: np.ndarray
    design_capacity_m3_per_day: np.ndarray

    @property
    def failing_now(self) -> np.ndarray:
        return self.kind != WORKING

    def late_steps(self) -> np.ndarray:
        """Intervals k >= 1 ending in the last third of the record."""
        t = self.t_days
        late = t >= t[0] + (2.0 / 3.0) * (t[-1] - t[0])
        late[0] = False
        return late


def _distance_m(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]
    a = np.sin((la - la.T) / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin((lo - lo.T) / 2) ** 2
    return 2 * R_EARTH_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _wet_season(doy: np.ndarray) -> np.ndarray:
    return (doy >= 151) & (doy < 273)  # 1 June - 30 September


def _rabi(doy: np.ndarray) -> np.ndarray:
    return (doy >= 304) | (doy < 90)  # 1 November - 31 March


def _uniform(rng, bounds, n):
    return rng.uniform(bounds[0], bounds[1], n)


def simulate_field(cfg: FieldConfig) -> FieldSimulation:
    rng = np.random.default_rng(cfg.seed)
    n, k = cfg.n_structures, cfg.n_acquisitions

    # --- geometry, kinds, pumping
    lat = rng.uniform(12.4, 13.6, n)
    lon = rng.uniform(76.4, 78.0, n)
    dist = _distance_m(lat, lon)
    kind = np.concatenate(
        [
            np.full(cfg.n_working, WORKING),
            np.full(cfg.n_failed, FAILED),
            np.full(cfg.n_clogging, CLOGGING),
        ]
    )[rng.permutation(n)]
    pumped = np.zeros(n, dtype=bool)
    for kd in (WORKING, FAILED, CLOGGING):
        members = np.flatnonzero(kind == kd)
        n_p = int(round(cfg.pumped_fraction * len(members)))
        pumped[rng.choice(members, size=n_p, replace=False)] = True

    # --- acquisitions (integer days after t0)
    gaps = rng.choice(cfg.revisit_days, size=k - 1, p=cfg.revisit_probs)
    t = np.concatenate([[0.0], np.cumsum(gaps)])
    dt = np.concatenate([[0.0], gaps])
    n_days = int(t[-1])  # rain/recharge days 0..t_{K-1}-1 after t0
    spin = cfg.spinup_days
    total = spin + n_days
    doy = np.arange(total) % 365  # day 0 of the spin-up is 1 January

    # --- rain: regional storms x local patchiness x smooth spatial multiplier
    wet = _wet_season(doy)
    regional_wet = rng.random(total) < np.where(wet, 0.6, 0.05)
    regional_mm = np.where(regional_wet, rng.exponential(np.where(wet, 15.0, 5.0)), 0.0)
    cov = np.exp(-dist / (cfg.rain_corr_km * 1000.0))
    z = np.linalg.cholesky(cov + 1e-9 * np.eye(n)) @ rng.standard_normal(n)
    sig = cfg.rain_field_sigma
    multiplier = np.exp(sig * z - sig**2 / 2)
    s_loc = cfg.local_lognormal_sigma
    local = np.exp(s_loc * rng.standard_normal((n, total)) - s_loc**2 / 2)
    local_dry = rng.random((n, total)) < cfg.local_dry_prob
    rain_all = np.where(local_dry, 0.0, regional_mm[None, :] * multiplier[:, None] * local)

    # --- structures and their pond buckets
    area = _uniform(rng, cfg.area_range_m2, n)
    height = _uniform(rng, cfg.height_range_m, n)
    v_max = area * height
    catchment_km2 = _uniform(rng, cfg.catchment_range_km2, n)
    runoff_c = _uniform(rng, cfg.runoff_coeff_range, n)
    threshold = _uniform(rng, cfg.threshold_range_mm, n)
    k0 = _uniform(rng, cfg.k0_range_m_per_day, n)
    failed_frac = _uniform(rng, cfg.failed_fraction_range, n)
    final_frac = _uniform(rng, cfg.clogging_final_range, n)
    t_clog = n_days / np.log(1.0 / final_frac)  # k(t_{K-1}) = final_frac * k0

    days_after_t0 = np.arange(total) - spin  # negative during spin-up
    cap = np.repeat(k0[:, None], total, axis=1)
    cap[kind == FAILED] *= failed_frac[kind == FAILED, None]
    clog = kind == CLOGGING
    decay = np.exp(-np.clip(days_after_t0, 0, None)[None, :] / t_clog[clog, None])
    cap[clog] = k0[clog, None] * decay

    evap_m3 = cfg.evaporation_mm_per_day / 1000.0 * area
    volume = np.zeros(n)
    infiltration = np.zeros((n, total))  # [m3/day]
    for d in range(total):
        p = rain_all[:, d]
        inflow = (
            runoff_c * np.maximum(p - threshold, 0.0) / 1000.0 * catchment_km2 * 1e6
            + p / 1000.0 * area
        )
        volume = np.minimum(volume + inflow, v_max)  # overflow is lost
        i_d = np.minimum(volume, cap[:, d] * area)  # infiltration-capacity limit
        volume -= i_d
        volume -= np.minimum(volume, evap_m3)  # evaporation while water is present
        infiltration[:, d] = i_d
    recharge = infiltration / cfg.cell_area_m2  # [m/day] into the aquifer node

    q_pump = _uniform(rng, cfg.pumping_range_m3_per_day, n) * pumped
    pumping = q_pump[:, None] / cfg.cell_area_m2 * _rabi(doy)[None, :]  # [m/day]

    # --- aquifer: daily backward Euler with lateral flow and per-structure leakance
    transmissivity = np.exp(rng.uniform(np.log(20.0), np.log(200.0), n))
    a = np.zeros_like(dist)
    near = (dist > 0) & (dist <= RADIUS_M)
    a[near] = 1.0 / dist[near] ** 2
    w = a * 0.5 * (transmissivity[:, None] + transmissivity[None, :])
    lateral = w - np.diag(w.sum(axis=1))
    tau = np.exp(rng.uniform(np.log(cfg.tau_range_days[0]), np.log(cfg.tau_range_days[1]), n))
    s = cfg.storativity
    system = s * np.eye(n) + np.diag(s / tau) - lateral
    inv = np.linalg.inv(system)
    h = np.zeros(n)
    h_end = np.zeros((n, total))
    for d in range(total):
        h = inv @ (s * h + recharge[:, d] - pumping[:, d])
        h_end[:, d] = h

    # --- sample at acquisitions: h(t_k) = head at the end of day t_k - 1
    acq_day = spin + t.astype(int) - 1
    head = h_end[:, acq_day] - h_end[:, acq_day[:1]]
    theta = rng.uniform(29.0, 46.0, n)
    d_clean = 100.0 * np.cos(np.radians(theta))[:, None] * cfg.skeletal_storage * head
    low = rng.random((n, k)) < cfg.low_coherence_fraction
    low[:, 0] = False
    coherence = np.where(low, rng.uniform(0.3, 0.65, (n, k)), rng.uniform(0.75, 0.98, (n, k)))
    noise = np.where(low, cfg.noise_low_coherence_cm, cfg.noise_cm) * rng.standard_normal((n, k))

    # --- per-interval rain and true recharge: interval k holds days t_{k-1} .. t_k - 1
    rain_after = rain_all[:, spin:]
    rain_step = np.zeros((n, k))
    r_int = np.zeros((n, k))
    for step in range(1, k):
        lo, hi = int(t[step - 1]), int(t[step])
        rain_step[:, step] = rain_after[:, lo:hi].sum(axis=1)
        r_int[:, step] = recharge[:, spin + lo : spin + hi].mean(axis=1)

    capacity = area * k0 * rng.uniform(0.7, 1.3, n)

    return FieldSimulation(
        cfg=cfg,
        lat=lat,
        lon=lon,
        kind=kind,
        pumped=pumped,
        t_days=t,
        dt_days=dt,
        rain_mm=rain_after,
        rain_per_step_mm=rain_step,
        recharge_daily=recharge,
        recharge_interval=r_int,
        infiltration_capacity=cap,
        head=head,
        head_absolute=h_end[:, acq_day],
        displacement_clean_cm=d_clean,
        displacement_cm=d_clean + noise,
        coherence=coherence,
        incidence_deg=theta,
        tau_days=tau,
        threshold_mm=threshold,
        rain_multiplier=multiplier,
        transmissivity=transmissivity,
        area_m2=area,
        height_m=height,
        catchment_km2=catchment_km2,
        design_capacity_m3_per_day=capacity,
    )
