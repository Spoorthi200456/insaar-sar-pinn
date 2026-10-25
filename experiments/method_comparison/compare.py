"""
Review-4 method comparison on held-out structures (protocol: README.md here).

    .venv/Scripts/python experiments/method_comparison/compare.py

A  trained network applied to unseen structures (amortised)
B  per-structure physics-informed test-time fitting (methods.test_time_fit)
C  ordinary least squares on a linear-reservoir model, no network
Two truths ("linear", "threshold") x seeds 0-4. Writes results.json / results.csv.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.method_comparison import methods as M  # noqa: E402
from experiments.synthetic_recovery import run_recovery as rr  # noqa: E402
from experiments.synthetic_recovery.simulator import (  # noqa: E402
    TruthConfig,
    oracle_eta,
    simulate,
)
from src.data.dataset import ARSDataset  # noqa: E402
from src.data.splits import split_structures  # noqa: E402
from src.models.losses import CompositeARSLoss, LossWeights  # noqa: E402
from src.models.network import ARSNet  # noqa: E402
from src.training.train import build_optimizer, full_batch  # noqa: E402

# ---- protocol constants (README "Protocol", committed before any run) ----
SEEDS = (0, 1, 2, 3, 4)
N_STRUCTURES = 160
N_FAILING = 60
FRACTIONS = (0.6, 0.2, 0.2)
THRESHOLD_MM_PER_DAY = 2.0
TRUTHS = ("linear", "threshold")
A_EPOCHS = 600
VAL_EVERY = 10
MIN_TEST = 30
PASS_RHO, PASS_AUROC = 0.70, 0.85
CLEARLY_BETTER_DRHO, CLEARLY_BETTER_WINS = 0.05, 4


def simulate_seed(seed: int, truth: str):
    thr = THRESHOLD_MM_PER_DAY if truth == "threshold" else 0.0
    cfg = replace(
        TruthConfig(),
        seed=seed,
        n_structures=N_STRUCTURES,
        n_failing=N_FAILING,
        recharge_threshold_mm_per_day=thr,
    )
    return simulate(cfg)


def make_split(sim, seed: int):
    records = rr.to_records(sim)
    labels = {
        r.design.structure_id: ("failing" if sim.failing[i] else "working")
        for i, r in enumerate(records)
    }
    return records, split_structures(records, fractions=FRACTIONS, seed=seed, stratify=labels)


def train_network(records_train, records_val, stats, k, seed):
    """Method A training, as heldout.py / train.py: best epoch by validation loss."""
    rr.seed_everything(seed)
    tb = full_batch(records_train, k, stats)
    vb = full_batch(records_val, k, stats)
    model = ARSNet(design_feature_dim=3)
    loss_fn = CompositeARSLoss(LossWeights())
    opt = build_optimizer(model, lr=rr.LR, weight_decay=rr.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A_EPOCHS)
    best, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(A_EPOCHS):
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = M.structure_loss(loss_fn, M._forward(model, tb), tb).total
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), rr.GRAD_CLIP)
        opt.step()
        sched.step()
        if epoch % VAL_EVERY == VAL_EVERY - 1 or epoch == A_EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                v = float(M.structure_loss(loss_fn, M._forward(model, vb), vb).total)
            if v < best:
                best, best_state, best_epoch = v, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def run_seed(seed: int, truth: str) -> dict:
    t0 = time.time()
    sim = simulate_seed(seed, truth)
    records, split = make_split(sim, seed)
    if len(split.test) < MIN_TEST:
        raise RuntimeError(f"seed {seed}: only {len(split.test)} test structures (< {MIN_TEST})")
    by_id = {r.design.structure_id: r for r in records}
    index = {r.design.structure_id: i for i, r in enumerate(records)}
    k = sim.cfg.n_acquisitions
    train_r = [by_id[s] for s in split.train]
    val_r = [by_id[s] for s in split.validation]
    test_r = [by_id[s] for s in split.test]
    train_ds = ARSDataset(train_r, sequence_length=k)
    stats = train_ds._design_stats

    test_idx = np.array([index[s] for s in split.test])
    r_true = sim.recharge[test_idx, 1:].mean(axis=1)
    working = ~sim.failing[test_idx]

    model, best_epoch = train_network(train_r, val_r, stats, k, seed)
    t_a = time.time()
    score_a = M.mean_recharge(model, test_r, stats, k)

    torch.manual_seed(seed)
    score_b, b_loss_drop = [], []
    for rec in test_r:
        fitted, l0, l1 = M.test_time_fit(model, rec, stats, k)
        score_b.append(M.mean_recharge(fitted, [rec], stats, k)[0])
        b_loss_drop.append(l1 / l0 if l0 > 0 else np.nan)
    score_b = np.array(score_b)
    t_b = time.time()

    test_ds = ARSDataset(test_r, sequence_length=k, design_feature_normalizers=stats)
    score_c, tau = M.method_c_scores(
        [train_ds[i] for i in range(len(train_ds))], [test_ds[i] for i in range(len(test_ds))]
    )
    t_c = time.time()

    oracle = oracle_eta(sim)[test_idx]

    def metrics(score):
        return {
            "spearman": float(spearmanr(score, r_true).statistic),
            "auroc": rr.auroc(np.asarray(score, dtype=float), working),
        }

    out = {
        "seed": seed,
        "truth": truth,
        "n_train": len(split.train),
        "n_validation": len(split.validation),
        "n_test": len(split.test),
        "n_test_failing": int((~working).sum()),
        "A": metrics(score_a),
        "B": metrics(score_b),
        "C": metrics(score_c),
        "oracle": metrics(oracle),
        "A_best_epoch": best_epoch,
        "C_tau_days": tau,
        "B_median_loss_ratio_after_over_before": float(np.nanmedian(b_loss_drop)),
        "seconds": {
            "A_train": round(t_a - t0, 1),
            "B": round(t_b - t_a, 1),
            "C": round(t_c - t_b, 1),
        },
        "per_structure": [
            {
                "structure": s,
                "failing": bool(sim.failing[index[s]]),
                "r_true": float(rt),
                "A": float(a),
                "B": float(b),
                "C": float(c),
                "oracle": float(o),
            }
            for s, rt, a, b, c, o in zip(
                split.test, r_true, score_a, score_b, score_c, oracle, strict=True
            )
        ],
    }
    return out


def summarise(runs: list[dict]) -> dict:
    summary = {}
    for method in ("A", "B", "C", "oracle"):
        rho = np.array([r[method]["spearman"] for r in runs])
        auc = np.array([r[method]["auroc"] for r in runs])
        summary[method] = {
            "spearman_mean": float(rho.mean()),
            "spearman_sd": float(rho.std(ddof=1)),
            "spearman_min": float(rho.min()),
            "auroc_mean": float(auc.mean()),
            "auroc_sd": float(auc.std(ddof=1)),
            "auroc_min": float(auc.min()),
            "passes": bool(rho.mean() >= PASS_RHO and auc.mean() >= PASS_AUROC),
        }

    def clearly_better(x, y):
        rx = np.array([r[x]["spearman"] for r in runs])
        ry = np.array([r[y]["spearman"] for r in runs])
        wins = int((rx > ry).sum())
        diff = float(rx.mean() - ry.mean())
        return {
            "mean_rho_difference": diff,
            "seeds_won": wins,
            "clearly_better": bool(diff >= CLEARLY_BETTER_DRHO and wins >= CLEARLY_BETTER_WINS),
        }

    summary["B_vs_C"] = clearly_better("B", "C")
    summary["A_vs_C"] = clearly_better("A", "C")
    summary["B_vs_A"] = clearly_better("B", "A")
    return summary


def main() -> None:
    results = {"protocol": "experiments/method_comparison/README.md", "truths": {}}
    rows = ["truth,seed,structure,failing,r_true,A,B,C,oracle"]
    for truth in TRUTHS:
        runs = []
        for seed in SEEDS:
            r = run_seed(seed, truth)
            runs.append(r)
            print(
                f"[{truth}] seed {seed}  n_test {r['n_test']} ({r['n_test_failing']} failing)  "
                + "  ".join(
                    f"{m}: rho {r[m]['spearman']:+.3f} auc {r[m]['auroc']:.3f}"
                    for m in ("A", "B", "C", "oracle")
                )
                + f"  tau_C {r['C_tau_days']:.0f} d  {r['seconds']}",
                flush=True,
            )
            for p in r["per_structure"]:
                rows.append(
                    f"{truth},{seed},{p['structure']},{int(p['failing'])},"
                    f"{p['r_true']:.6e},{p['A']:.6e},{p['B']:.6e},{p['C']:.6e},"
                    f"{p['oracle']:.6e}"
                )
        s = summarise(runs)
        results["truths"][truth] = {
            "summary": s,
            "runs": [{k: v for k, v in r.items() if k != "per_structure"} for r in runs],
        }
        print(f"[{truth}] SUMMARY " + json.dumps(s), flush=True)
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (HERE / "results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()


def c_only(seed: int, truth: str) -> dict:
    """Method C alone for one seed/truth (same data, split and code path as run_seed)."""
    sim = simulate_seed(seed, truth)
    records, split = make_split(sim, seed)
    by_id = {r.design.structure_id: r for r in records}
    index = {r.design.structure_id: i for i, r in enumerate(records)}
    k = sim.cfg.n_acquisitions
    train_ds = ARSDataset([by_id[s] for s in split.train], sequence_length=k)
    test_ds = ARSDataset(
        [by_id[s] for s in split.test],
        sequence_length=k,
        design_feature_normalizers=train_ds._design_stats,
    )
    score, tau = M.method_c_scores(
        [train_ds[i] for i in range(len(train_ds))], [test_ds[i] for i in range(len(test_ds))]
    )
    test_idx = np.array([index[s] for s in split.test])
    r_true = sim.recharge[test_idx, 1:].mean(axis=1)
    working = ~sim.failing[test_idx]
    return {
        "spearman": float(spearmanr(score, r_true).statistic),
        "auroc": rr.auroc(score, working),
        "tau": tau,
    }
