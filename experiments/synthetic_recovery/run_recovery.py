"""
Known-answer experiment: can ARSNet recover recharge that we planted?

    .venv/Scripts/python experiments/synthetic_recovery/run_recovery.py

1. `simulator.simulate()` (pure NumPy, independent of the network) creates
   InSAR LOS displacement, rainfall and design data from known per-structure
   recharge, and known S, S_k, L, T and incidence angles.
2. ARSNet is trained on those records exactly as on real data (same dataset,
   graph, loss, optimiser family), fixed seed, CPU.
3. Recovered mean recharge per structure is compared with the truth.

Scenarios (see README.md for the protocol and the pre-registered thresholds):
  A  InSAR only, every parameter learned.
  B  as A, but S and S_k fixed to their true values ("oracle constants").
  C  hard-rock aquifer: S_k = 2e-5, signal far below noise (detectability).
  D  as A, plus piezometer heads at every structure (boundary-head term).

Writes results.csv (per structure, per scenario) and results.json next to
this file, and prints a summary.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.synthetic_recovery.simulator import (  # noqa: E402
    Simulation,
    TruthConfig,
    oracle_eta,
    simulate,
)
from src.data.dataset import ARSDataset, make_collate_fn  # noqa: E402
from src.data.graph import StructureGraph  # noqa: E402
from src.data.schemas import (  # noqa: E402
    ARSDesignParameters,
    ARSTelemetryRecord,
    RainfallObservation,
    SARObservation,
)
from src.models.losses import CompositeARSLoss, LossWeights  # noqa: E402
from src.models.network import ARSNet  # noqa: E402

# ---- pre-registered thresholds (README.md "Thresholds"; fixed before any run) ----
MIN_SPEARMAN_A = 0.70
MIN_AUROC_A = 0.85
MAX_MEDIAN_REL_ERR_B_FLOOR = 0.30  # B passes if median rel. error <= max(0.30, 2 x oracle)

# ---- training protocol (fixed before any run) ----
SEED = 0
EPOCHS = 600
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 5.0
WELL_NOISE_M = 0.05

T0 = datetime(2024, 1, 1, tzinfo=timezone.utc)


@dataclass
class ScenarioResult:
    name: str
    spearman: float
    auroc: float
    median_rel_err_working: float
    s_fit: float
    s_k_fit: float
    leakance_fit: float
    r_true_mean: np.ndarray
    r_fit_mean: np.ndarray
    seconds: float


def seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def to_records(sim: Simulation) -> list[ARSTelemetryRecord]:
    records = []
    for i in range(sim.cfg.n_structures):
        sid = f"SYN-{i:02d}"
        design = ARSDesignParameters(
            structure_id=sid,
            latitude=float(sim.lat[i]),
            longitude=float(sim.lon[i]),
            height_m=float(sim.height_m[i]),
            storage_volume_m3=float(sim.recharge_area_m2[i] * sim.height_m[i]),
            catchment_area_km2=float(sim.catchment_km2[i]),
            design_recharge_capacity_m3_per_day=float(sim.design_capacity_m3_per_day[i]),
            recharge_area_m2=float(sim.recharge_area_m2[i]),
        )
        sar = [
            SARObservation(
                structure_id=sid,
                timestamp=T0 + timedelta(days=float(t)),
                los_displacement_cm=float(sim.displacement_cm[i, k]),
                coherence=float(sim.coherence[i, k]),
                incidence_angle_deg=float(sim.incidence_deg[i]),
            )
            for k, t in enumerate(sim.t_days)
        ]
        rain = [
            RainfallObservation(
                structure_id=sid, timestamp=T0 + timedelta(days=float(d)), rainfall_mm=float(p)
            )
            for d, p in zip(sim.rain_days, sim.rain_mm, strict=True)
        ]
        records.append(ARSTelemetryRecord(design=design, sar_series=sar, rainfall_series=rain))
    return records


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties count 1/2."""
    pos, neg = score[positive], score[~positive]
    greater = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(greater / (len(pos) * len(neg)))


def _freeze_to_truth(model: ARSNet, cfg: TruthConfig) -> None:
    with torch.no_grad():
        model.storativity_logit.copy_(torch.logit(torch.tensor(cfg.storativity)))
        frac = (np.log10(cfg.skeletal_storage) + 6.0) / 4.0
        model.skeletal_storage_raw.copy_(torch.logit(torch.tensor(frac, dtype=torch.float32)))
    model.storativity_logit.requires_grad_(False)
    model.skeletal_storage_raw.requires_grad_(False)


def run_scenario(
    name: str,
    sim: Simulation,
    oracle_constants: bool = False,
    with_wells: bool = False,
    epochs: int = EPOCHS,
) -> ScenarioResult:
    seed_everything(SEED)
    t_start = time.time()
    records = to_records(sim)
    k = sim.cfg.n_acquisitions
    dataset = ARSDataset(records, sequence_length=k)
    batch = make_collate_fn(graph=StructureGraph.from_records(records))(
        [dataset[i] for i in range(len(dataset))]
    )

    model = ARSNet(design_feature_dim=3)
    if oracle_constants:
        _freeze_to_truth(model, sim.cfg)
    loss_fn = CompositeARSLoss(LossWeights())
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    boundary = {}
    if with_wells:
        rng = np.random.default_rng(SEED + 1)
        h_obs = sim.head + WELL_NOISE_M * rng.standard_normal(sim.head.shape)
        boundary = dict(
            boundary_head=torch.tensor(h_obs, dtype=torch.float32),
            boundary_mask=batch["valid_step_mask"].clone(),
        )

    def forward():
        return model(
            design_features=batch["design_features"],
            displacement_cm=batch["displacement_cm"],
            rainfall_mm=batch["rainfall_mm"],
            time_delta_days=batch["time_delta_days"],
            adjacency=batch["adjacency"],
            incidence_angle_deg=batch["incidence_angle_deg"],
            coherence_mask=batch["coherence_mask"],
        )

    def loss_of(out):
        return loss_fn(
            predicted_displacement_cm=out.predicted_displacement_cm,
            observed_displacement_cm=batch["displacement_cm"],
            predicted_head_change=out.predicted_head_change,
            hydraulic_conductivity=out.hydraulic_conductivity,
            recharge_rate=out.recharge_rate,
            design_recharge_capacity=batch["design_recharge_capacity"],
            rainfall_mm=batch["rainfall_mm"],
            adjacency=batch["physics_adjacency"],
            time_delta_days=batch["time_delta_days"],
            storativity=out.storativity,
            leakance=out.leakance,
            recharge_area_m2=batch["recharge_area_m2"],
            catchment_area_m2=batch["catchment_area_m2"],
            observation_mask=batch["coherence_mask"],
            valid_mask=batch["valid_step_mask"],
            **boundary,
        )

    for _ in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = loss_of(forward()).total
        if not torch.isfinite(loss):
            raise RuntimeError(f"[{name}] non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        out = forward()
    # mean recharge over steps 1..K-1 (step 0 has no preceding interval)
    r_fit = out.recharge_rate[:, 1:].mean(dim=1).double().numpy()
    r_true = sim.recharge[:, 1:].mean(axis=1)
    working = ~sim.failing
    rel = np.abs(r_fit[working] - r_true[working]) / r_true[working]
    return ScenarioResult(
        name=name,
        spearman=float(spearmanr(r_fit, r_true).statistic),
        auroc=auroc(r_fit, working),
        median_rel_err_working=float(np.median(rel)),
        s_fit=float(out.storativity),
        s_k_fit=float(out.skeletal_storage),
        leakance_fit=float(out.leakance),
        r_true_mean=r_true,
        r_fit_mean=r_fit,
        seconds=time.time() - t_start,
    )


def oracle_floor(sim: Simulation) -> float:
    eta_hat = oracle_eta(sim)
    working = ~sim.failing
    return float(np.median(np.abs(eta_hat[working] - sim.eta[working]) / sim.eta[working]))


def main() -> None:
    base = TruthConfig()
    sim = simulate(base)
    sim_hard = simulate(replace(base, skeletal_storage=2e-5))
    floor = oracle_floor(sim)
    floor_hard = oracle_floor(sim_hard)
    print(
        f"oracle median rel. error of eta (working): alluvial {floor:.3f}, "
        f"hard rock {floor_hard:.3f}"
    )

    results = [
        run_scenario("A_insar_only", sim),
        run_scenario("B_oracle_S_Sk", sim, oracle_constants=True),
        run_scenario("C_hard_rock", sim_hard),
        run_scenario("D_with_wells", sim, with_wells=True),
    ]

    rows = [
        "scenario,structure,failing,eta_true,R_true_mean_m_per_day,R_fit_mean_m_per_day,"
        "ratio_fit_over_true"
    ]
    summary = {
        "oracle_median_rel_err": floor,
        "oracle_median_rel_err_hard_rock": floor_hard,
        "thresholds": {
            "spearman_A": MIN_SPEARMAN_A,
            "auroc_A": MIN_AUROC_A,
            "median_rel_err_B": max(MAX_MEDIAN_REL_ERR_B_FLOOR, 2 * floor),
        },
        "truth": {"S": base.storativity, "S_k": base.skeletal_storage, "L": base.leakance},
        "scenarios": {},
    }
    for res in results:
        s = sim_hard if res.name.startswith("C") else sim
        for i in range(base.n_structures):
            rows.append(
                f"{res.name},SYN-{i:02d},{int(s.failing[i])},{s.eta[i]:.4f},"
                f"{res.r_true_mean[i]:.6e},{res.r_fit_mean[i]:.6e},"
                f"{res.r_fit_mean[i] / res.r_true_mean[i]:.4f}"
            )
        summary["scenarios"][res.name] = {
            "spearman": res.spearman,
            "auroc": res.auroc,
            "median_rel_err_working": res.median_rel_err_working,
            "S_fit": res.s_fit,
            "S_k_fit": res.s_k_fit,
            "L_fit": res.leakance_fit,
            "seconds": round(res.seconds, 1),
        }
        print(
            f"{res.name:15s} spearman {res.spearman:+.3f}  auroc {res.auroc:.3f}  "
            f"median rel err (working) {res.median_rel_err_working:.3f}  "
            f"S {res.s_fit:.4f}  S_k {res.s_k_fit:.2e}  L {res.leakance_fit:.2e}  "
            f"({res.seconds:.0f} s)"
        )
    (HERE / "results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (HERE / "results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {HERE / 'results.csv'} and results.json")


if __name__ == "__main__":
    main()
