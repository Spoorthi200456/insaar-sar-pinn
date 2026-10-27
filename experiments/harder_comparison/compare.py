"""
Harder A/B/C comparison on a simulator that is NOT method C's model (review 5).
Protocol: README.md in this folder (committed before this code existed).

    .venv/Scripts/python experiments/harder_comparison/compare.py

A  trained network applied to unseen structures          (review 4 code)
B  per-structure physics-informed test-time fitting      (review 4 code)
C  ordinary least squares, no network, own rain per structure (src/models/ols_baseline.py)
Seeds 0-4. Writes results.json / results.csv.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.harder_comparison.simulator import (  # noqa: E402
    CLOGGING,
    FAILED,
    WORKING,
    FieldConfig,
    FieldSimulation,
    simulate_field,
)
from experiments.method_comparison import methods as M  # noqa: E402
from experiments.method_comparison.compare import train_network  # noqa: E402
from experiments.synthetic_recovery.run_recovery import auroc  # noqa: E402
from src.data.dataset import ARSDataset  # noqa: E402
from src.data.schemas import (  # noqa: E402
    ARSDesignParameters,
    ARSTelemetryRecord,
    RainfallObservation,
    SARObservation,
)
from src.data.splits import split_structures  # noqa: E402
from src.models import ols_baseline as ols  # noqa: E402
from src.training.train import full_batch  # noqa: E402

# ---- protocol constants (README "Protocol", committed before any run) ----
SEEDS = (0, 1, 2, 3, 4)
FRACTIONS = (0.6, 0.2, 0.2)
MIN_TEST = 30
N_STEP = 16  # raise N in steps of 16 if any seed has fewer than MIN_TEST test structures
PASS_RHO, PASS_AUROC = 0.70, 0.85
CLEARLY_BETTER_DIFF, CLEARLY_BETTER_WINS = 0.05, 4
HEADLINE = ("rho_amount", "auroc_now")
METHODS = ("A", "B", "C")
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def config_for(seed: int, extra_steps: int = 0) -> FieldConfig:
    base = FieldConfig()
    if extra_steps == 0:
        return replace(base, seed=seed)
    n = base.n_structures + N_STEP * extra_steps
    w = int(round(n * base.n_working / base.n_structures))
    f = int(round(n * base.n_failed / base.n_structures))
    return replace(base, seed=seed, n_working=w, n_failed=f, n_clogging=n - w - f)


def to_records(sim: FieldSimulation) -> list[ARSTelemetryRecord]:
    """Each structure gets ITS OWN daily rain gauge (rain day j at t0 + j + 12 h)."""
    records = []
    for i in range(sim.cfg.n_structures):
        sid = f"FLD-{i:03d}"
        design = ARSDesignParameters(
            structure_id=sid,
            latitude=float(sim.lat[i]),
            longitude=float(sim.lon[i]),
            height_m=float(sim.height_m[i]),
            storage_volume_m3=float(sim.area_m2[i] * sim.height_m[i]),
            catchment_area_km2=float(sim.catchment_km2[i]),
            design_recharge_capacity_m3_per_day=float(sim.design_capacity_m3_per_day[i]),
            recharge_area_m2=float(sim.area_m2[i]),
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
                structure_id=sid,
                timestamp=T0 + timedelta(days=j + 0.5),
                rainfall_mm=float(p),
            )
            for j, p in enumerate(sim.rain_mm[i])
        ]
        records.append(ARSTelemetryRecord(design=design, sar_series=sar, rainfall_series=rain))
    return records


def make_split(sim: FieldSimulation, records, seed: int):
    labels = {
        r.design.structure_id: ("failing" if sim.failing_now[i] else "working")
        for i, r in enumerate(records)
    }
    return split_structures(records, fractions=FRACTIONS, seed=seed, stratify=labels)


def check_split_sizes() -> int:
    """Protocol rule: the smallest N-raise (in steps of 16) giving >= 30 test structures."""
    for extra in range(0, 10):
        sizes = []
        for seed in SEEDS:
            sim = simulate_field(config_for(seed, extra))
            sizes.append(len(make_split(sim, to_records(sim), seed).test))
        print(f"N = {config_for(0, extra).n_structures}: test sizes {sizes}", flush=True)
        if min(sizes) >= MIN_TEST:
            return extra
    raise RuntimeError("could not reach the minimum test size")


def recharge_series(model, records, stats, k) -> np.ndarray:
    """Predicted R [m/day], (n, K), in records order."""
    b = full_batch(records, k, stats)
    model.eval()
    with torch.no_grad():
        r = M._forward(model, b).recharge_rate.double().numpy()
    order = {sid: i for i, sid in enumerate(b["structure_ids"])}
    return np.stack([r[order[rec.design.structure_id]] for rec in records])


def ols_series(sample) -> ols.OLSSeries:
    return ols.OLSSeries(
        displacement_cm=sample.displacement_cm.double().numpy(),
        reliable=(sample.coherence_mask & sample.valid_step_mask).numpy(),
        cos_theta=float(np.cos(np.radians(float(sample.incidence_angle_deg)))),
        rain_per_step_mm=sample.rainfall_mm.double().numpy(),
        dt_days=sample.time_delta_days.double().numpy(),
    )


def method_c(train_samples, test_samples, late: np.ndarray):
    """C's amount and now scores (README table) and its regional tau."""
    tau = ols.fit_tau([ols_series(s) for s in train_samples])
    amount, now = [], []
    for s in test_samples:
        series = ols_series(s)
        c, _ = ols.fit_amplitude(series, tau)
        u = np.zeros(len(series.dt_days))
        m = series.dt_days > 0
        u[m] = series.rain_per_step_mm[m] / 1000.0 / series.dt_days[m]
        amount.append(ols.mean_relative_recharge(series, c))
        now.append(c * float(np.mean(u[late])))
    return np.array(amount), np.array(now), tau


def _rho(x, y) -> float:
    return float(spearmanr(x, y).statistic)


def metrics(amount, now, truth_amount, truth_late, kind, pumped) -> dict:
    working = kind == WORKING
    clog_or_work = (kind == CLOGGING) | working
    fail_or_work = (kind == FAILED) | working
    return {
        "rho_amount": _rho(amount, truth_amount),
        "auroc_now": auroc(np.asarray(now, dtype=float), working),
        "auroc_now_clogging_vs_working": auroc(
            np.asarray(now, dtype=float)[clog_or_work], working[clog_or_work]
        ),
        "auroc_now_failed_vs_working": auroc(
            np.asarray(now, dtype=float)[fail_or_work], working[fail_or_work]
        ),
        "rho_now_vs_true_late": _rho(now, truth_late),
        "rho_amount_pumped": _rho(amount[pumped], truth_amount[pumped]),
        "rho_amount_unpumped": _rho(amount[~pumped], truth_amount[~pumped]),
    }


def run_seed(seed: int, extra: int = 0, methods=METHODS) -> dict:
    t0 = time.time()
    sim = simulate_field(config_for(seed, extra))
    records = to_records(sim)
    split = make_split(sim, records, seed)
    if len(split.test) < MIN_TEST:
        raise RuntimeError(f"seed {seed}: only {len(split.test)} test structures")
    by_id = {r.design.structure_id: r for r in records}
    index = {r.design.structure_id: i for i, r in enumerate(records)}
    k = sim.cfg.n_acquisitions
    train_r = [by_id[s] for s in split.train]
    val_r = [by_id[s] for s in split.validation]
    test_r = [by_id[s] for s in split.test]
    train_ds = ARSDataset(train_r, sequence_length=k)
    stats = train_ds._design_stats
    test_ds = ARSDataset(test_r, sequence_length=k, design_feature_normalizers=stats)

    idx = np.array([index[s] for s in split.test])
    late = sim.late_steps()
    truth_amount = sim.recharge_interval[idx, 1:].mean(axis=1)
    truth_late = sim.recharge_interval[idx][:, late].mean(axis=1)
    kind, pumped = sim.kind[idx], sim.pumped[idx]

    out = {
        "seed": seed,
        "n_structures": sim.cfg.n_structures,
        "n_train": len(split.train),
        "n_validation": len(split.validation),
        "n_test": len(split.test),
        "n_test_by_kind": {
            "working": int((kind == WORKING).sum()),
            "failed": int((kind == FAILED).sum()),
            "clogging": int((kind == CLOGGING).sum()),
        },
        "n_test_pumped": int(pumped.sum()),
        "late_steps": int(late.sum()),
        "seconds": {},
    }
    scores = {}

    t = time.time()
    c_amount, c_now, tau = method_c(
        [train_ds[i] for i in range(len(train_ds))],
        [test_ds[i] for i in range(len(test_ds))],
        late,
    )
    scores["C"] = (c_amount, c_now)
    out["C_tau_days"] = tau
    out["seconds"]["C"] = round(time.time() - t, 1)

    if "A" in methods or "B" in methods:
        t = time.time()
        model, best_epoch = train_network(train_r, val_r, stats, k, seed)
        out["A_best_epoch"] = best_epoch
        out["seconds"]["A_train"] = round(time.time() - t, 1)
        series_a = recharge_series(model, test_r, stats, k)
        scores["A"] = (series_a[:, 1:].mean(axis=1), series_a[:, late].mean(axis=1))
    if "B" in methods:
        t = time.time()
        torch.manual_seed(seed)
        series_b, ratio = [], []
        for rec in test_r:
            fitted, l0, l1 = M.test_time_fit(model, rec, stats, k)
            series_b.append(recharge_series(fitted, [rec], stats, k)[0])
            ratio.append(l1 / l0 if l0 > 0 else np.nan)
        series_b = np.stack(series_b)
        scores["B"] = (series_b[:, 1:].mean(axis=1), series_b[:, late].mean(axis=1))
        out["B_median_loss_ratio_after_over_before"] = float(np.nanmedian(ratio))
        out["seconds"]["B"] = round(time.time() - t, 1)

    for m, (amount, now) in scores.items():
        out[m] = metrics(amount, now, truth_amount, truth_late, kind, pumped)
    out["seconds"]["total"] = round(time.time() - t0, 1)
    out["per_structure"] = [
        {
            "structure": s,
            "kind": ["working", "failed", "clogging"][int(kind[j])],
            "pumped": bool(pumped[j]),
            "true_amount": float(truth_amount[j]),
            "true_late": float(truth_late[j]),
            **{f"{m}_amount": float(scores[m][0][j]) for m in scores},
            **{f"{m}_now": float(scores[m][1][j]) for m in scores},
        }
        for j, s in enumerate(split.test)
    ]
    return out


def clearly_better(runs, x, y, metric) -> dict:
    vx = np.array([r[x][metric] for r in runs])
    vy = np.array([r[y][metric] for r in runs])
    wins = int((vx > vy).sum())
    diff = float(vx.mean() - vy.mean())
    return {
        "mean_difference": diff,
        "seeds_won": wins,
        "clearly_better": bool(diff >= CLEARLY_BETTER_DIFF and wins >= CLEARLY_BETTER_WINS),
    }


def overall_clearly_better(runs, x, y) -> dict:
    """README rule 3: X clearly better on >= 1 headline metric; Y not so on the other."""
    per = {m: clearly_better(runs, x, y, m) for m in HEADLINE}
    rev = {m: clearly_better(runs, y, x, m) for m in HEADLINE}
    verdict = False
    for m in HEADLINE:
        other = [o for o in HEADLINE if o != m][0]
        if per[m]["clearly_better"] and not rev[other]["clearly_better"]:
            verdict = True
    return {"per_metric": per, "clearly_better": verdict}


def summarise(runs: list[dict]) -> dict:
    present = [m for m in METHODS if m in runs[0]]
    summary = {}
    for m in present:
        entry = {}
        for metric in runs[0][m]:
            v = np.array([r[m][metric] for r in runs], dtype=float)
            entry[metric] = {
                "mean": float(np.nanmean(v)),
                "sd": float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0,
                "min": float(np.nanmin(v)),
                "per_seed": [float(x) for x in v],
            }
        entry["passes"] = bool(
            entry["rho_amount"]["mean"] >= PASS_RHO and entry["auroc_now"]["mean"] >= PASS_AUROC
        )
        summary[m] = entry
    pairs = [("B", "C"), ("C", "B"), ("A", "C"), ("B", "A")]
    for x, y in pairs:
        if x in present and y in present:
            summary[f"{x}_vs_{y}"] = overall_clearly_better(runs, x, y)
    return summary


def c_only(extra: int = 0) -> dict:
    """Method C's full protocol (all seeds), same data and code path as main()."""
    runs = [run_seed(seed, extra, methods=("C",)) for seed in SEEDS]
    return {"summary": summarise(runs), "runs": runs}


def main() -> None:
    extra = check_split_sizes()
    results = {"protocol": "experiments/harder_comparison/README.md", "n_raise_steps": extra}
    runs = []
    rows = [
        "seed,structure,kind,pumped,true_amount,true_late,"
        + ",".join(f"{m}_{s}" for m in METHODS for s in ("amount", "now"))
    ]
    for seed in SEEDS:
        r = run_seed(seed, extra)
        runs.append(r)
        print(
            f"seed {seed}  n_test {r['n_test']} {r['n_test_by_kind']} pumped {r['n_test_pumped']}  "
            + "  ".join(
                f"{m}: rho {r[m]['rho_amount']:+.3f} auc {r[m]['auroc_now']:.3f} "
                f"(clog {r[m]['auroc_now_clogging_vs_working']:.3f})"
                for m in METHODS
            )
            + f"  tau_C {r['C_tau_days']:.0f} d  {r['seconds']}",
            flush=True,
        )
        for p in r["per_structure"]:
            rows.append(
                f"{seed},{p['structure']},{p['kind']},{int(p['pumped'])},"
                f"{p['true_amount']:.6e},{p['true_late']:.6e},"
                + ",".join(f"{p[f'{m}_{s}']:.6e}" for m in METHODS for s in ("amount", "now"))
            )
    results["summary"] = summarise(runs)
    results["runs"] = [{k: v for k, v in r.items() if k != "per_structure"} for r in runs]
    print("SUMMARY " + json.dumps(results["summary"]), flush=True)
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (HERE / "results.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
