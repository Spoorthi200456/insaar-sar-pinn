"""
Training entrypoint.

    python -m src.training.train --data path/to/records.json --epochs 50

Expects a JSON file containing a list of objects each parseable as
`ARSTelemetryRecord` (see `src/data/schemas.py`). Synthetic data can be
generated with `--synthetic N` for local smoke-testing without real
telemetry.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

import structlog
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.dataset import ARSDataset, build_dataloader, make_collate_fn
from src.data.graph import StructureGraph
from src.data.schemas import (
    ARSDesignParameters,
    ARSTelemetryRecord,
    RainfallObservation,
    SARObservation,
)
from src.data.splits import DEFAULT_FRACTIONS, StructureSplit, split_structures
from src.models.coupling import head_to_los_cm
from src.models.losses import CompositeARSLoss, LossWeights
from src.models.network import ARSNet

log = structlog.get_logger(__name__)


# Planted truths of the smoke-test generator (PROVISIONAL, illustrative values,
# not calibrated to any real aquifer).
SYNTH_STORATIVITY = 0.05  # S [dimensionless]
SYNTH_SKELETAL_STORAGE = 1e-3  # S_k [dimensionless] (compliant alluvium end, so signal > noise)
SYNTH_DRAINAGE_TIMESCALE_DAYS = 60.0  # S / L: head recession time scale [day]


def make_synthetic_records_with_truth(
    n_structures: int, n_timesteps: int = 52, seed: int = 0, noise_cm: float = 0.05
) -> tuple[list[ARSTelemetryRecord], list[dict]]:
    """
    Synthetic telemetry plus the planted truths it was generated from.

    Per structure i, weekly steps k (Δt = 7 days):
      R_i,k = eta_i * P_k / 1000 / Δt               recharge [m/day]; eta_i = infiltrating fraction
      S (h_k - h_{k-1}) / Δt = R_k - L h_k          backward Euler, h_0 = 0 [m]
      d_k   = head_to_los_cm(h_k - h_0, S_k, theta_i) + N(0, noise_cm)   [cm]
    with L = S / SYNTH_DRAINAGE_TIMESCALE_DAYS [1/day] (a linear-reservoir drain
    so head does not rise forever) and theta_i ~ U(29, 46) deg. The coupling is
    the model's own `head_to_los_cm` (review finding 2), so generator and model
    cannot drift apart. Returns (records, truths), truths[i] a dict.
    """
    rng = random.Random(seed)
    records: list[ARSTelemetryRecord] = []
    truths: list[dict] = []
    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    dt_days = 7.0
    s = SYNTH_STORATIVITY
    leakance = s / SYNTH_DRAINAGE_TIMESCALE_DAYS

    for i in range(n_structures):
        structure_id = f"ARS-{i:04d}"
        design = ARSDesignParameters(
            structure_id=structure_id,
            name=f"Recharge Structure {i}",
            latitude=12.9 + rng.uniform(-0.5, 0.5),
            longitude=77.5 + rng.uniform(-0.5, 0.5),
            height_m=rng.uniform(2.0, 15.0),
            storage_volume_m3=rng.uniform(500.0, 20000.0),
            catchment_area_km2=rng.uniform(0.5, 25.0),
            design_recharge_capacity_m3_per_day=rng.uniform(10.0, 500.0),
        )
        eta = rng.uniform(0.05, 0.5)
        theta = rng.uniform(29.0, 46.0)

        timestamps, rain, recharge, head, coherence = [], [], [], [], []
        h = 0.0
        for t in range(n_timesteps):
            timestamps.append(base_time + timedelta(days=dt_days * t))
            p_mm = max(0.0, rng.gauss(40.0, 30.0))
            r = eta * p_mm / 1000.0 / dt_days
            if t > 0:  # h_0 = 0 is the reference acquisition
                h = (h + dt_days * r / s) / (1.0 + dt_days * leakance / s)
            rain.append(p_mm)
            recharge.append(r)
            head.append(h)
            coherence.append(min(1.0, max(0.0, rng.gauss(0.85, 0.15))))

        d_clean = head_to_los_cm(
            torch.tensor([head], dtype=torch.float64),
            torch.tensor(SYNTH_SKELETAL_STORAGE, dtype=torch.float64),
            torch.tensor([theta], dtype=torch.float64),
        )[0].tolist()
        sar_series = [
            SARObservation(
                structure_id=structure_id,
                timestamp=ts,
                los_displacement_cm=d + (rng.gauss(0.0, noise_cm) if noise_cm > 0 else 0.0),
                coherence=c,
                incidence_angle_deg=theta,
            )
            for ts, d, c in zip(timestamps, d_clean, coherence, strict=False)
        ]
        rain_series = [
            RainfallObservation(structure_id=structure_id, timestamp=ts, rainfall_mm=p_mm)
            for ts, p_mm in zip(timestamps, rain, strict=False)
        ]
        records.append(
            ARSTelemetryRecord(design=design, sar_series=sar_series, rainfall_series=rain_series)
        )
        truths.append(
            {
                "structure_id": structure_id,
                "head_m": head,
                "recharge_m_per_day": recharge,
                "infiltration_fraction": eta,
                "incidence_angle_deg": theta,
                "storativity": s,
                "skeletal_storage": SYNTH_SKELETAL_STORAGE,
                "leakance_per_day": leakance,
            }
        )
    return records, truths


def make_synthetic_records(
    n_structures: int, n_timesteps: int = 52, seed: int = 0
) -> list[ARSTelemetryRecord]:
    """Synthetic telemetry for smoke tests / demos (see `make_synthetic_records_with_truth`)."""
    return make_synthetic_records_with_truth(n_structures, n_timesteps, seed)[0]


# Author decision 1 (review 2, finding 32), made AFTER the failed pre-registered
# known-answer run 1: the global physical constants are single bounded scalars
# that must be able to cross their whole range (raw in [-4, 4]) within a run.
# Adam moves a parameter at most ~lr per step, so at the network learning rate
# (3e-4 .. 1e-3) they barely move (S_k: 0.36 of the ~2.5 needed in run 1b).
GLOBAL_SCALARS = ("storativity_logit", "skeletal_storage_raw", "leakance_raw")
GLOBAL_SCALAR_LR = 0.02


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> AdamW:
    """AdamW with the network weights at `lr` and the global scalars at GLOBAL_SCALAR_LR
    (no weight decay on them: decay would pull S, S_k, L toward an arbitrary value)."""
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    scalars = [p for n, p in named.items() if n in GLOBAL_SCALARS]
    others = [p for n, p in named.items() if n not in GLOBAL_SCALARS]
    return AdamW(
        [
            {"params": others, "lr": lr, "weight_decay": weight_decay},
            {"params": scalars, "lr": GLOBAL_SCALAR_LR, "weight_decay": 0.0},
        ]
    )


def structures_without_well_data(
    records: list[ARSTelemetryRecord], sequence_length: int = 52
) -> list[str]:
    """Structure ids with no well reading matched to any acquisition in the window used."""
    dataset = ARSDataset(records, sequence_length=sequence_length)
    return [
        dataset[i].structure_id for i in range(len(dataset)) if not bool(dataset[i].has_well_data)
    ]


def load_records_from_json(path: Path) -> list[ARSTelemetryRecord]:
    raw = json.loads(path.read_text())
    return [ARSTelemetryRecord.model_validate(item) for item in raw]


def resolve_device(requested: str | None) -> str:
    """
    Picks the compute device, with diagnostics -- so "why is it still on
    CPU?" has an answer printed right there instead of requiring a
    debugging session. Pass `--device cuda` to force GPU and get a loud,
    explicit error (instead of a silent CPU fallback) if it isn't actually
    available.
    """
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "Requested --device cuda but torch.cuda.is_available() is False.\n"
                f"  torch version: {torch.__version__}\n"
                f"  torch.version.cuda: {torch.version.cuda}\n"
                "This almost always means the installed PyTorch build has no CUDA support "
                "(e.g. a CPU-only wheel got installed) rather than a code problem -- run:\n"
                "  pip uninstall torch torchvision torchaudio -y\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu126\n"
                "(pick the CUDA tag matching your driver at "
                "https://pytorch.org/get-started/locally/), "
                "then re-run this command."
            )
        return "cuda"

    if requested == "cpu":
        return "cpu"

    # No explicit --device: auto-detect, but explain the result either way.
    if torch.cuda.is_available():
        log.info(
            "training.device_auto_detected",
            device="cuda",
            gpu_name=torch.cuda.get_device_name(0),
        )
        return "cuda"

    log.warning(
        "training.device_auto_detected",
        device="cpu",
        reason="torch.cuda.is_available() returned False",
        torch_version=torch.__version__,
        torch_cuda_build=str(torch.version.cuda),
        hint="If you have an NVIDIA GPU, your installed torch build likely has no CUDA "
        "support -- see the --device cuda error message for the fix, or pass --device cuda "
        "to see it now.",
    )
    return "cpu"


def _batch_loss(model, loss_fn, batch, device):
    """Forward pass + composite loss for one collated batch."""
    out = model(
        design_features=batch["design_features"].to(device),
        displacement_cm=batch["displacement_cm"].to(device),
        rainfall_mm=batch["rainfall_mm"].to(device),
        time_delta_days=batch["time_delta_days"].to(device),
        adjacency=batch["adjacency"].to(device),
        incidence_angle_deg=batch["incidence_angle_deg"].to(device),
        coherence_mask=batch["coherence_mask"].to(device),
    )
    return loss_fn(
        predicted_displacement_cm=out.predicted_displacement_cm,
        observed_displacement_cm=batch["displacement_cm"].to(device),
        predicted_head_change=out.predicted_head_change,
        hydraulic_conductivity=out.hydraulic_conductivity,
        recharge_rate=out.recharge_rate,
        design_recharge_capacity=batch["design_recharge_capacity"].to(device),
        rainfall_mm=batch["rainfall_mm"].to(device),
        adjacency=batch["physics_adjacency"].to(device),
        node_mask=batch["loss_node_mask"].to(device),
        valid_mask=batch["valid_step_mask"].to(device),
        time_delta_days=batch["time_delta_days"].to(device),
        storativity=out.storativity,
        leakance=out.leakance,
        recharge_area_m2=batch["recharge_area_m2"].to(device),
        catchment_area_m2=batch["catchment_area_m2"].to(device),
        observation_mask=batch["coherence_mask"].to(device),
        well_head_m=batch["well_head_m"].to(device),
        well_mask=batch["well_mask"].to(device),
    )


def full_batch(records, sequence_length, design_stats):
    """One batch holding every structure in `records`, normalised with `design_stats`.
    The graph is built from `records` alone: valid for a split, because no graph
    edge crosses splits (src/data/splits.py)."""
    dataset = ARSDataset(
        records, sequence_length=sequence_length, design_feature_normalizers=design_stats
    )
    collate = make_collate_fn(graph=StructureGraph.from_records(records))
    return collate([dataset[i] for i in range(len(dataset))])


def evaluate(model, loss_fn, records, sequence_length, design_stats, device) -> float:
    if not records:
        return float("nan")
    model.eval()
    with torch.no_grad():
        batch = full_batch(records, sequence_length, design_stats)
        return float(_batch_loss(model, loss_fn, batch, device).total.detach().cpu())


def train(
    records: list[ARSTelemetryRecord],
    epochs: int = 50,
    batch_size: int = 16,
    sequence_length: int = 52,
    lr: float = 3e-4,
    checkpoint_path: Path = Path("checkpoints/ars_net.pt"),
    device: str | None = None,
    num_workers: int = 2,
    split_seed: int = 0,
    split_fractions: tuple[float, float, float] = DEFAULT_FRACTIONS,
) -> None:
    device = resolve_device(device)
    log.info("training.start", n_records=len(records), device=device, epochs=epochs)
    no_wells = structures_without_well_data(records, sequence_length)
    if no_wells:
        # Author decision 4: say so. Without wells, only a relative ranking of
        # recharge is identifiable (experiments/synthetic_recovery/README.md).
        log.warning(
            "training.no_well_data",
            n_without=len(no_wells),
            n_total=len(records),
            structure_ids=no_wells[:20],
            note="InSAR only for these structures: absolute recharge not identifiable",
        )

    # Review finding 24: split by connected group of structures, fixed seed.
    split = split_structures(records, fractions=split_fractions, seed=split_seed)
    by_id = {r.design.structure_id: r for r in records}
    train_records = [by_id[s] for s in split.train]
    val_records = [by_id[s] for s in split.validation]
    test_records = [by_id[s] for s in split.test]
    if not val_records or not test_records:
        log.warning(
            "training.no_validation_split",
            reason="too few independent structure groups to hold any out",
            note="checkpoint selected on TRAINING loss; nothing is held out",
        )
        train_records, val_records, test_records = list(records), [], []
        split = StructureSplit([r.design.structure_id for r in records], [], [])
    log.info(
        "training.split",
        seed=split_seed,
        n_train=len(train_records),
        n_validation=len(val_records),
        n_test=len(test_records),
    )

    loader = build_dataloader(
        records=train_records,
        batch_size=min(batch_size, len(train_records)),
        sequence_length=sequence_length,
        shuffle=True,
        num_workers=num_workers,
    )
    design_stats = loader.dataset.base._design_stats  # fitted on the training split only

    model = ARSNet(design_feature_dim=3).to(device)
    loss_fn = CompositeARSLoss(LossWeights())
    optimizer = build_optimizer(model, lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    selected_on = "validation" if val_records else "training"
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            breakdown = _batch_loss(model, loss_fn, batch, device)
            breakdown.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(breakdown.total.detach().cpu()))

        scheduler.step()
        train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        val_loss = evaluate(model, loss_fn, val_records, sequence_length, design_stats, device)
        log.info(
            "training.epoch",
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=scheduler.get_last_lr()[0],
        )

        score = val_loss if val_records else train_loss
        if score < best_loss:
            best_loss = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "loss": best_loss,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "selected_on": selected_on,
                    "split": split.as_dict(),
                    "split_seed": split_seed,
                    "design_stats": {
                        k: {"mean": v.mean, "std": v.std} for k, v in design_stats.items()
                    },
                    "model_config": {"design_feature_dim": 3},
                },
                checkpoint_path,
            )
            log.info(
                "training.checkpoint_saved",
                path=str(checkpoint_path),
                loss=best_loss,
                selected_on=selected_on,
            )

    # The test split is evaluated ONCE, with the selected checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = evaluate(model, loss_fn, test_records, sequence_length, design_stats, device)
    checkpoint["test_loss"] = test_loss
    torch.save(checkpoint, checkpoint_path)
    log.info(
        "training.complete",
        best_loss=best_loss,
        selected_on=selected_on,
        test_loss=test_loss,
        n_test=len(test_records),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ARSNet on ARS telemetry")
    parser.add_argument("--data", type=Path, default=None, help="Path to JSON telemetry records")
    parser.add_argument(
        "--synthetic", type=int, default=0, help="Generate N synthetic structures instead"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=52)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ars_net.pt"))
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Force a device. Omit to auto-detect (uses CUDA if available, else CPU).",
    )
    args = parser.parse_args()

    if args.synthetic > 0:
        records = make_synthetic_records(args.synthetic)
    elif args.data is not None:
        records = load_records_from_json(args.data)
    else:
        raise SystemExit("Provide either --data <path> or --synthetic <N>")

    train(
        records,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        lr=args.lr,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
