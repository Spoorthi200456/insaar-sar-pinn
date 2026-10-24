"""
Reproducible demo of ARSNet on SYNTHETIC data (CPU, a few minutes).

    .venv/Scripts/python scripts/demo_synthetic.py

What it does
- Generates the existing synthetic records (`make_synthetic_records`, seed 0).
- Trains ARSNet with `CompositeARSLoss`, using the same optimiser settings as
  `src/training/train.py` (AdamW lr 3e-4, wd 1e-5, cosine schedule, grad clip
  5.0). Two differences from train.py: a fixed epoch count, and ONE full batch
  holding every structure (the fixed network graph of review finding 4; since
  review 2, minibatches in train.py give identical physics anyway).
- Saves docs/demo/loss_curves.png (one panel per loss term vs epoch) and
  docs/demo/loss_history.csv (the same numbers as a table).
- Prints the learned S, S_k and L next to the generator's planted values, the
  number of graph edges, and the predicted T and R for 3 structures.

What it is NOT
- Not a recovery test: that is experiments/synthetic_recovery/ (which, under
  its pre-registered protocol, FAILS - see its README). The smoke generator here
  uses the model's own coupling (review finding 2), so a good fit here says
  nothing about real aquifers.
- Does not modify anything under src/.
"""

from __future__ import annotations

import copy
import csv
import os
import random
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

warnings.filterwarnings("ignore", message=r".*protected namespace.*")

from src.data.dataset import ARSDataset, make_collate_fn  # noqa: E402
from src.data.splits import split_structures  # noqa: E402
from src.models.losses import CompositeARSLoss, LossWeights  # noqa: E402
from src.models.network import ARSNet  # noqa: E402
from src.training.train import (  # noqa: E402
    build_optimizer,
    make_synthetic_records_with_truth,
)

TAG = "[SYNTHETIC DATA]"
SEED = 0
N_STRUCTURES = 24
N_TIMESTEPS = 52
EPOCHS = 300
LR = 3e-4
CONNECTIVITY_RADIUS_KM = 5.0  # make_collate_fn default
TIME_BUDGET_S = 300.0
OUT_DIR = ROOT / "docs" / "demo"

# (as_dict key, panel title, weight attribute or None for the weighted total)
TERMS = [
    ("pde_residual", "PDE residual  r = S∂h/∂t − ∇·(T∇h) + Lh − R  [(m/day)²]", "pde"),
    ("data_misfit", "Data misfit (displacement) [cm²]", "data"),
    ("boundary_condition", "Boundary / initial condition", "boundary"),
    ("design_capacity_penalty", "Design-capacity penalty", "design_capacity"),
    ("non_negativity_penalty", "Non-negativity penalty", "non_negativity"),
    ("mass_balance_penalty", "Mass-balance penalty", "mass_balance"),
    ("conductivity_smoothness", "T smoothness on graph", "conductivity_smoothness"),
    ("well_head_misfit", "Well-head misfit [m²] (no wells in this demo)", "well"),
    ("total", "Total (weighted sum)", None),
]

# Reference palette (dataviz skill): light surface, ink tokens, categorical slot 1.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"
SERIES_2 = "#c2410c"


def seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def full_batch(records, design_stats=None) -> dict:
    dataset = ARSDataset(
        records, sequence_length=N_TIMESTEPS, design_feature_normalizers=design_stats
    )
    collate = make_collate_fn(connectivity_radius_km=CONNECTIVITY_RADIUS_KM)
    return collate([dataset[i] for i in range(len(dataset))])


def forward(model: ARSNet, batch: dict):
    return model(
        design_features=batch["design_features"],
        displacement_cm=batch["displacement_cm"],
        rainfall_mm=batch["rainfall_mm"],
        time_delta_days=batch["time_delta_days"],
        adjacency=batch["adjacency"],
        incidence_angle_deg=batch["incidence_angle_deg"],
        coherence_mask=batch["coherence_mask"],
    )


def loss_of(loss_fn: CompositeARSLoss, out, batch: dict):
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
    )


def plot_loss_curves(
    history: dict[str, list[float]], weights: LossWeights, path: Path, val_total: list[float]
) -> None:
    epochs = np.arange(1, len(history["total"]) + 1)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), facecolor=SURFACE)
    for ax, (key, title, weight_attr) in zip(axes.flat, TERMS):
        y = np.asarray(history[key])
        ax.set_facecolor(SURFACE)
        ax.plot(
            epochs,
            y,
            color=SERIES_1,
            linewidth=1.5,
            solid_capstyle="round",
            label="training structures",
        )
        if key == "total":
            ax.plot(
                epochs,
                val_total,
                color=SERIES_2,
                linewidth=1.5,
                label="validation structures (unseen)",
            )
            ax.legend(fontsize=7, frameon=False)
        subtitle = (
            "weighted sum of all terms"
            if weight_attr is None
            else (f"unweighted; weight w = {getattr(weights, weight_attr):g}")
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=9.5, color=INK, loc="left")
        if np.all(y == 0):
            ax.text(
                0.5,
                0.8,
                "0 at every epoch\n(term inactive in this run)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color=INK_2,
            )
        elif np.all(y > 0) and y.max() / y.min() > 100:
            ax.set_yscale("log")
            ax.set_ylabel("log scale", fontsize=8, color=INK_2)
        ax.set_xlabel("epoch", fontsize=8, color=INK_2)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    fig.suptitle(
        f"SYNTHETIC DATA — ARSNet loss terms vs epoch  "
        f"({N_STRUCTURES} structures × {N_TIMESTEPS} steps, seed {SEED}, full batch, CPU)",
        fontsize=12,
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.text(
        0.01,
        0.005,
        "SYNTHETIC DATA. Trained on the training structures only; validation = unseen structures. "
        "Terms unweighted in physical units; total dimensionless. Not a recovery test.",
        fontsize=8,
        color=INK_2,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    t_start = time.time()
    seed_everything(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records, truth = make_synthetic_records_with_truth(
        n_structures=N_STRUCTURES, n_timesteps=N_TIMESTEPS, seed=SEED
    )
    # Review 3: the demo now trains on the TRAINING split only and reports on
    # structures it never saw (split by connected group, fixed seed; finding 24).
    split = split_structures(records, seed=SEED)
    by_id = {r.design.structure_id: r for r in records}
    truth_by_id = {t["structure_id"]: t for t in truth}
    train_recs = [by_id[i] for i in split.train]
    stats = ARSDataset(train_recs, sequence_length=N_TIMESTEPS)._design_stats
    batch = full_batch(train_recs, stats)
    val_batch = full_batch([by_id[i] for i in split.validation], stats)
    test_batch = full_batch([by_id[i] for i in split.test], stats)

    model = ARSNet(design_feature_dim=3)
    weights = LossWeights()
    loss_fn = CompositeARSLoss(weights)
    optimizer = build_optimizer(model, lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history: dict[str, list[float]] = {key: [] for key, _, _ in TERMS}
    s_history: list[float] = []
    val_total: list[float] = []
    best_val, best_state, best_epoch = float("inf"), None, -1
    print(
        f"{TAG} training ARSNet on {len(split.train)} of {N_STRUCTURES} structures "
        f"(validation {len(split.validation)}, test {len(split.test)}, never seen in training), "
        f"{N_TIMESTEPS} steps, {EPOCHS} epochs, seed {SEED}, CPU"
    )
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = forward(model, batch)
        breakdown = loss_of(loss_fn, out, batch)
        if not torch.isfinite(breakdown.total):
            raise RuntimeError(f"{TAG} non-finite loss at epoch {epoch}: {breakdown.as_dict()}")
        breakdown.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        scheduler.step()
        for key, value in breakdown.as_dict().items():
            history[key].append(value)
        s_history.append(float(out.storativity.detach()))
        model.eval()
        with torch.no_grad():
            v = float(loss_of(loss_fn, forward(model, val_batch), val_batch).total)
        val_total.append(v)
        if v < best_val:
            best_val, best_state, best_epoch = v, copy.deepcopy(model.state_dict()), epoch
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(
                f"{TAG} epoch {epoch:4d}  train total {history['total'][-1]:.5f}  "
                f"validation total {v:.5f}  data {history['data_misfit'][-1]:.5f}"
            )
    train_seconds = time.time() - t_start

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_test = forward(model, test_batch)  # the only time the test structures are used
        test = loss_of(loss_fn, out_test, test_batch).as_dict()

    png = OUT_DIR / "loss_curves.png"
    csv_path = OUT_DIR / "loss_history.csv"
    plot_loss_curves(history, weights, png, val_total)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "# SYNTHETIC DATA - loss terms per epoch on TRAINING structures (unweighted; "
                "total is weighted); validation_total on unseen validation structures"
            ]
        )
        writer.writerow(
            ["epoch", *[key for key, _, _ in TERMS], "validation_total", "storativity_S"]
        )
        for i in range(EPOCHS):
            writer.writerow(
                [
                    i + 1,
                    *[f"{history[key][i]:.8g}" for key, _, _ in TERMS],
                    f"{val_total[i]:.8g}",
                    f"{s_history[i]:.8g}",
                ]
            )

    test_ids = test_batch["structure_ids"]
    r_fit = out_test.recharge_rate[:, 1:].mean(dim=1).numpy()
    r_true = np.array([np.mean(truth_by_id[i]["recharge_m_per_day"][1:]) for i in test_ids])
    print()
    print(f"{TAG} ===== RESULTS (test structures never seen in training) =====")
    print(
        f"{TAG} training time        : {train_seconds:.1f} s on CPU "
        f"({'within' if train_seconds < TIME_BUDGET_S else 'OVER'} the {TIME_BUDGET_S:.0f} s budget)"
    )
    print(
        f"{TAG} split (seed {SEED})     : train {len(split.train)}, validation "
        f"{len(split.validation)}, test {len(split.test)}; best validation epoch {best_epoch}"
    )
    print(
        f"{TAG} learned S / S_k / L  : {float(out_test.storativity):.4f} / "
        f"{float(out_test.skeletal_storage):.2e} / {float(out_test.leakance):.2e} 1/day "
        f"(generator: {truth[0]['storativity']:.4f} / {truth[0]['skeletal_storage']:.2e} / "
        f"{truth[0]['leakance_per_day']:.2e})"
    )
    print(f"{TAG} test losses          : " + ", ".join(f"{k}={v:.4g}" for k, v in test.items()))
    print(
        f"{TAG} test InSAR misfit    : RMSE {np.sqrt(test['data_misfit']):.3f} cm "
        f"(generator noise 0.05 cm)"
    )
    print(f"{TAG}   {'structure':<10} {'R fit [m/day]':>14} {'R planted':>12} {'ratio':>7}")
    for sid, rf, rt in zip(test_ids, r_fit, r_true, strict=True):
        print(f"{TAG}   {sid:<10} {rf:>14.4e} {rt:>12.4e} {rf / rt:>7.3f}")
    print(f"{TAG} saved plot  : {png.relative_to(ROOT).as_posix()}")
    print(f"{TAG} saved table : {csv_path.relative_to(ROOT).as_posix()}")
    print(
        f"{TAG} caveat: {len(split.test)} test structures are too few for a ranking "
        f"statistic; see experiments/synthetic_recovery/heldout.py (run 4)."
    )


if __name__ == "__main__":
    main()
