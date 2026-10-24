"""
Known-answer test on structures the model has NEVER seen (review 3, run 4).

    .venv/Scripts/python experiments/synthetic_recovery/heldout.py

Same independent simulator and physics as run_recovery.py, but with 64
structures (24 failing). They are split 70/15/15 by connected structure group,
stratified by working/failing, seed 0 (src/data/splits.py), and handled exactly
as src/training/train.py does:
  - the model is trained on the TRAINING structures only (full batch),
    with design normalisation fitted on them and train.build_optimizer;
  - every 10 epochs the validation loss is computed; the best epoch is kept;
  - the kept model is run ONCE on the TEST structures, whose data it has
    never seen, and recovery is scored on those structures only.
Scenario D gives well heads (absolute, +87.3 m datum) for TRAINING structures
only; the test structures are scored from InSAR alone, as at deployment.

The pre-registered thresholds of README.md are applied unchanged to the test
structures. Writes results_heldout.{csv,json}.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.synthetic_recovery import run_recovery as rr  # noqa: E402
from experiments.synthetic_recovery.simulator import (  # noqa: E402
    Simulation,
    TruthConfig,
    oracle_eta,
    simulate,
)
from src.data.splits import split_structures  # noqa: E402
from src.models.losses import CompositeARSLoss, LossWeights  # noqa: E402
from src.models.network import ARSNet  # noqa: E402
from src.training.train import build_optimizer, full_batch  # noqa: E402

# ---- protocol (declared and committed before run 4 was made) ----
HELDOUT_CONFIG = replace(TruthConfig(), n_structures=64, n_failing=24)
SPLIT_FRACTIONS = (0.7, 0.15, 0.15)
SPLIT_SEED = 0
VAL_EVERY = 10


@dataclass
class HeldOutResult:
    name: str
    n_train: int
    n_validation: int
    n_test: int
    test_ids: list[str]
    spearman: float
    auroc: float
    median_rel_err_working: float
    oracle_floor_test: float
    best_epoch: int
    s_fit: float
    s_k_fit: float
    leakance_fit: float
    r_true_mean: np.ndarray
    r_fit_mean: np.ndarray
    test_data_rmse_cm: float
    seconds: float


def make_split(sim: Simulation):
    records = rr.to_records(sim)
    labels = {
        f"SYN-{i:02d}": ("failing" if sim.failing[i] else "working")
        for i in range(sim.cfg.n_structures)
    }
    split = split_structures(records, fractions=SPLIT_FRACTIONS, seed=SPLIT_SEED, stratify=labels)
    return records, split


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


def _loss(loss_fn, out, b, extra=None):
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
        **(extra or {}),
    )


def run_heldout(
    name: str,
    sim: Simulation,
    oracle_constants: bool = False,
    with_wells: bool = False,
    epochs: int = rr.EPOCHS,
) -> HeldOutResult:
    rr.seed_everything(rr.SEED)
    t_start = time.time()
    records, split = make_split(sim)
    by_id = {r.design.structure_id: r for r in records}
    k = sim.cfg.n_acquisitions
    train_recs = [by_id[s] for s in split.train]
    from src.data.dataset import ARSDataset

    stats = ARSDataset(train_recs, sequence_length=k)._design_stats  # training split only
    tb = full_batch(train_recs, k, stats)
    vb = full_batch([by_id[s] for s in split.validation], k, stats)
    xb = full_batch([by_id[s] for s in split.test], k, stats)

    model = ARSNet(design_feature_dim=3)
    if oracle_constants:
        rr._freeze_to_truth(model, sim.cfg)
    loss_fn = CompositeARSLoss(LossWeights())
    opt = build_optimizer(model, lr=rr.LR, weight_decay=rr.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    extra = None
    if with_wells:  # wells at TRAINING structures only
        rng = np.random.default_rng(rr.SEED + 1)
        idx = [int(s.split("-")[1]) for s in tb["structure_ids"]]
        h = sim.head[idx] + rr.WELL_NOISE_M * rng.standard_normal((len(idx), k))
        extra = dict(
            well_head_m=torch.tensor(h + rr.WELL_DATUM_OFFSET_M, dtype=torch.float32),
            well_mask=tb["valid_step_mask"].clone(),
        )

    best_val, best_state, best_epoch = float("inf"), None, -1
    params = [p for p in model.parameters() if p.requires_grad]
    for epoch in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = _loss(loss_fn, _forward(model, tb), tb, extra).total
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, rr.GRAD_CLIP)
        opt.step()
        sched.step()
        if epoch % VAL_EVERY == VAL_EVERY - 1 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                val = float(_loss(loss_fn, _forward(model, vb), vb).total)
            if val < best_val:
                best_val, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out = _forward(model, xb)  # the ONLY time the test structures are touched
    test_idx = np.array([int(s.split("-")[1]) for s in xb["structure_ids"]])
    r_fit = out.recharge_rate[:, 1:].mean(dim=1).double().numpy()
    r_true = sim.recharge[test_idx, 1:].mean(axis=1)
    working = ~sim.failing[test_idx]
    rel = np.abs(r_fit[working] - r_true[working]) / r_true[working]
    eta_hat = oracle_eta(sim)[test_idx]
    eta = sim.eta[test_idx]
    reliable = xb["coherence_mask"].numpy()
    d_fit = out.predicted_displacement_cm.numpy()
    return HeldOutResult(
        name=name,
        n_train=len(split.train),
        n_validation=len(split.validation),
        n_test=len(split.test),
        test_ids=list(xb["structure_ids"]),
        spearman=float(spearmanr(r_fit, r_true).statistic),
        auroc=rr.auroc(r_fit, working),
        median_rel_err_working=float(np.median(rel)),
        oracle_floor_test=float(np.median(np.abs(eta_hat[working] - eta[working]) / eta[working])),
        best_epoch=best_epoch,
        s_fit=float(out.storativity),
        s_k_fit=float(out.skeletal_storage),
        leakance_fit=float(out.leakance),
        r_true_mean=r_true,
        r_fit_mean=r_fit,
        test_data_rmse_cm=float(
            np.sqrt(np.mean((d_fit - sim.displacement_cm[test_idx])[reliable] ** 2))
        ),
        seconds=time.time() - t_start,
    )


def main() -> None:
    sim = simulate(HELDOUT_CONFIG)
    sim_hard = simulate(replace(HELDOUT_CONFIG, skeletal_storage=2e-5))
    results = [
        run_heldout("A_insar_only", sim),
        run_heldout("B_oracle_S_Sk", sim, oracle_constants=True),
        run_heldout("C_hard_rock", sim_hard),
        run_heldout("D_wells_train_only", sim, with_wells=True),
    ]
    rows = ["scenario,structure,failing,eta_true,R_true_mean_m_per_day,R_fit_mean_m_per_day"]
    summary = {
        "protocol": "held-out-run4",
        "config": {
            "n_structures": 64,
            "n_failing": 24,
            "fractions": SPLIT_FRACTIONS,
            "split_seed": SPLIT_SEED,
        },
        "thresholds": {
            "spearman_A": rr.MIN_SPEARMAN_A,
            "auroc_A": rr.MIN_AUROC_A,
            "median_rel_err_B": "max(0.30, 2 x oracle floor on test)",
        },
        "scenarios": {},
    }
    for res in results:
        s = sim_hard if res.name.startswith("C") else sim
        for sid, rt, rf in zip(res.test_ids, res.r_true_mean, res.r_fit_mean, strict=True):
            i = int(sid.split("-")[1])
            rows.append(f"{res.name},{sid},{int(s.failing[i])},{s.eta[i]:.4f},{rt:.6e},{rf:.6e}")
        summary["scenarios"][res.name] = {
            "n_train": res.n_train,
            "n_validation": res.n_validation,
            "n_test": res.n_test,
            "spearman": res.spearman,
            "auroc": res.auroc,
            "median_rel_err_working": res.median_rel_err_working,
            "oracle_floor_test": res.oracle_floor_test,
            "best_epoch": res.best_epoch,
            "S_fit": res.s_fit,
            "S_k_fit": res.s_k_fit,
            "L_fit": res.leakance_fit,
            "test_data_rmse_cm": res.test_data_rmse_cm,
            "seconds": round(res.seconds, 1),
        }
        print(
            f"{res.name:18s} n={res.n_train}/{res.n_validation}/{res.n_test}  "
            f"spearman {res.spearman:+.3f}  auroc {res.auroc:.3f}  "
            f"median rel err {res.median_rel_err_working:.3f} (oracle {res.oracle_floor_test:.3f})  "
            f"best epoch {res.best_epoch}  S {res.s_fit:.4f} S_k {res.s_k_fit:.2e} "
            f"L {res.leakance_fit:.2e}  test RMSE {res.test_data_rmse_cm:.3f} cm  "
            f"({res.seconds:.0f} s)"
        )
    (HERE / "results_heldout.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (HERE / "results_heldout.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
