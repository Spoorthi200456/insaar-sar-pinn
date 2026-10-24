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
    ("total", "Total (weighted sum)", None),
]

# Reference palette (dataviz skill): light surface, ink tokens, categorical slot 1.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"


def seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))


def full_batch(records) -> dict:
    dataset = ARSDataset(records, sequence_length=N_TIMESTEPS)
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


def plot_loss_curves(history: dict[str, list[float]], weights: LossWeights, path: Path) -> None:
    epochs = np.arange(1, len(history["total"]) + 1)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), facecolor=SURFACE)
    for ax, (key, title, weight_attr) in zip(axes.flat, TERMS):
        y = np.asarray(history[key])
        ax.set_facecolor(SURFACE)
        ax.plot(epochs, y, color=SERIES_1, linewidth=1.5, solid_capstyle="round")
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
        "SYNTHETIC DATA. Not a recovery test (see experiments/synthetic_recovery/). Terms "
        "unweighted in physical units; the total is dimensionless (review finding 29).",
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
    batch = full_batch(records)

    adjacency = batch["physics_adjacency"]
    n_edges = int(torch.triu(adjacency > 0, diagonal=1).sum().item())
    n_pairs = N_STRUCTURES * (N_STRUCTURES - 1) // 2

    model = ARSNet(design_feature_dim=3)
    weights = LossWeights()
    loss_fn = CompositeARSLoss(weights)
    optimizer = build_optimizer(model, lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history: dict[str, list[float]] = {key: [] for key, _, _ in TERMS}
    s_history: list[float] = []
    print(
        f"{TAG} training ARSNet: {N_STRUCTURES} structures x {N_TIMESTEPS} steps, "
        f"{EPOCHS} epochs, seed {SEED}, full batch, CPU"
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
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(
                f"{TAG} epoch {epoch:4d}  total {history['total'][-1]:.5f}  "
                f"pde {history['pde_residual'][-1]:.3e}  data {history['data_misfit'][-1]:.5f}"
            )
    train_seconds = time.time() - t_start

    model.eval()
    with torch.no_grad():
        out = forward(model, batch)
        final = loss_of(loss_fn, out, batch).as_dict()

    png = OUT_DIR / "loss_curves.png"
    csv_path = OUT_DIR / "loss_history.csv"
    plot_loss_curves(history, weights, png)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["# SYNTHETIC DATA - ARSNet loss terms per epoch (unweighted; total is weighted)"]
        )
        writer.writerow(["epoch", *[key for key, _, _ in TERMS], "storativity_S"])
        for i in range(EPOCHS):
            writer.writerow(
                [i + 1, *[f"{history[key][i]:.8g}" for key, _, _ in TERMS], f"{s_history[i]:.8g}"]
            )

    print()
    print(f"{TAG} ===== RESULTS =====")
    print(
        f"{TAG} training time        : {train_seconds:.1f} s on CPU "
        f"({'within' if train_seconds < TIME_BUDGET_S else 'OVER'} the {TIME_BUDGET_S:.0f} s budget)"
    )
    print(
        f"{TAG} learned S / S_k / L  : {float(out.storativity):.4f} / "
        f"{float(out.skeletal_storage):.2e} / {float(out.leakance):.2e} 1/day "
        f"(generator: {truth[0]['storativity']:.4f} / {truth[0]['skeletal_storage']:.2e} / "
        f"{truth[0]['leakance_per_day']:.2e}) - fitted coefficients; only R*S_k/S is "
        f"identifiable from InSAR"
    )
    print(
        f"{TAG} graph edges          : {n_edges} of {n_pairs} possible pairs "
        f"(undirected, radius {CONNECTIVITY_RADIUS_KM:g} km)"
    )
    print(f"{TAG} final eval losses    : " + ", ".join(f"{k}={v:.4g}" for k, v in final.items()))
    print(
        f"{TAG} predicted T and R for 3 structures (T [m²/day]; R [m/day], "
        f"generator mean R in the last column):"
    )
    print(
        f"{TAG}   {'structure':<10} {'T [m²/day]':>11} {'R mean [m/day]':>15} "
        f"{'R min':>10} {'R max':>10} {'graph degree':>13} {'true R mean':>12}"
    )
    degree = (adjacency > 0).sum(dim=1)
    for i in range(3):
        sid = batch["structure_ids"][i]
        r = out.recharge_rate[i]
        print(
            f"{TAG}   {sid:<10} {float(out.hydraulic_conductivity[i]):>11.4f} "
            f"{float(r.mean()):>15.4e} {float(r.min()):>10.3e} {float(r.max()):>10.3e} "
            f"{int(degree[i]):>13d} {np.mean(truth[i]['recharge_m_per_day'][1:]):>12.4e}"
        )
    print(f"{TAG} saved plot  : {png.relative_to(ROOT).as_posix()}")
    print(f"{TAG} saved table : {csv_path.relative_to(ROOT).as_posix()}")
    print(
        f"{TAG} caveat: not a recovery test - see experiments/synthetic_recovery/README.md "
        f"(the pre-registered known-answer test FAILS)."
    )


if __name__ == "__main__":
    main()
