"""
Author decision 1 (review 2, finding 32): the global physical constants S, S_k and
L get their own, faster learning rate in the training code, so they can cross
their bounded range within a training run. Introduced AFTER the failed
pre-registered known-answer run 1 (experiments/synthetic_recovery/README.md).
"""

from __future__ import annotations

import pytest

from src.models.network import ARSNet


def test_optimizer_gives_global_scalars_their_own_learning_rate():
    from src.training.train import GLOBAL_SCALAR_LR, GLOBAL_SCALARS, build_optimizer

    model = ARSNet(design_feature_dim=3)
    opt = build_optimizer(model, lr=3e-4, weight_decay=1e-5)
    by_id = {id(p): g for g in opt.param_groups for p in g["params"]}
    named = dict(model.named_parameters())
    assert set(GLOBAL_SCALARS) == {"storativity_logit", "skeletal_storage_raw", "leakance_raw"}
    for name in GLOBAL_SCALARS:
        group = by_id[id(named[name])]
        assert group["lr"] == pytest.approx(GLOBAL_SCALAR_LR)
        assert group["weight_decay"] == 0.0
    other = by_id[id(named["k_head.0.weight"])]
    assert other["lr"] == pytest.approx(3e-4)
    assert other["weight_decay"] == pytest.approx(1e-5)
    assert sum(len(g["params"]) for g in opt.param_groups) == len(named)  # nothing dropped


def test_learning_rate_lets_skeletal_storage_cross_its_range():
    # Adam moves a parameter at most ~lr per step; under a cosine schedule the
    # total travel over E epochs is ~lr*E/2. raw in [-4, 4] spans ~98 % of the
    # bounded range, so reaching either end from 0 in the experiment's 600
    # epochs needs lr >= 4/300.
    from src.training.train import GLOBAL_SCALAR_LR

    assert GLOBAL_SCALAR_LR * 600 / 2 >= 4.0


def test_train_uses_build_optimizer(monkeypatch, tmp_path, small_records):
    import src.training.train as tr

    calls = []
    real = tr.build_optimizer

    def spy(model, lr, weight_decay):
        calls.append((lr, weight_decay))
        return real(model, lr=lr, weight_decay=weight_decay)

    monkeypatch.setattr(tr, "build_optimizer", spy)
    tr.train(
        small_records,
        epochs=1,
        batch_size=2,
        sequence_length=8,
        checkpoint_path=tmp_path / "ck.pt",
        device="cpu",
        num_workers=0,
    )
    assert calls == [(3e-4, 1e-5)]
