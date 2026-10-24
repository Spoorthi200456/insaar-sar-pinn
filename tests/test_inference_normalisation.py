"""
Review finding 33 (found while fixing 24). The inference engine normalised
design features with statistics fitted on the REQUEST batch, so one structure's
inputs depended on which other structures were submitted with it, and did not
match training. The engine must use the training split's statistics saved in
the checkpoint.
"""

from __future__ import annotations

import pytest
import torch

from src.backend.inference_engine import InferenceEngine
from src.training.train import make_synthetic_records, train


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("ck") / "ars.pt"
    recs = make_synthetic_records(n_structures=12, n_timesteps=8, seed=5)
    train(
        recs,
        epochs=1,
        batch_size=4,
        sequence_length=8,
        checkpoint_path=path,
        device="cpu",
        num_workers=0,
    )
    return path


def test_design_features_do_not_depend_on_request_composition(checkpoint):
    eng = InferenceEngine(onnx_path=None, torch_checkpoint_path=checkpoint, sequence_length=8)
    others = make_synthetic_records(n_structures=4, n_timesteps=8, seed=9)
    target = others[0]
    alone = eng._build_batch_tensors([target], [])["design_features"][0]
    crowd = eng._build_batch_tensors(others, [])["design_features"][0]
    assert torch.allclose(alone, crowd)


def test_engine_uses_the_training_statistics(checkpoint):
    saved = torch.load(checkpoint, weights_only=False)["design_stats"]["height_m"]
    eng = InferenceEngine(onnx_path=None, torch_checkpoint_path=checkpoint, sequence_length=8)
    rec = make_synthetic_records(n_structures=1, n_timesteps=8, seed=9)[0]
    got = eng._build_batch_tensors([rec], [])["design_features"][0, 0].item()
    expected = (rec.design.height_m - saved["mean"]) / saved["std"]
    assert got == pytest.approx(expected, rel=1e-5)
