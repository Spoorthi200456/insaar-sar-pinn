"""
Review finding 17: the ONNX export path (used by the deployed API) had no
test. Round trip: export -> onnxruntime -> outputs equal eager PyTorch.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

ort = pytest.importorskip("onnxruntime")

from src.backend.export_onnx import (  # noqa: E402
    ARSNetONNXWrapper,
    build_dummy_inputs,
    export_to_onnx,
)
from src.models.network import ARSNet  # noqa: E402


def test_onnx_round_trip_matches_eager(tmp_path):
    torch.manual_seed(0)
    model = ARSNet(design_feature_dim=3).eval()
    path = export_to_onnx(
        model, tmp_path / "ars.onnx", sequence_length=8, num_structures=4, design_feature_dim=3
    )
    inputs = build_dummy_inputs(4, 8, 3)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {k: v.numpy() for k, v in inputs.items()})
    with torch.no_grad():
        eager = ARSNetONNXWrapper(model)(*inputs.values())
    names = [
        "hydraulic_conductivity",
        "recharge_rate",
        "predicted_head_change",
        "predicted_displacement_cm",
    ]
    for name, a, b in zip(names, onnx_out, eager, strict=True):
        assert a.shape == tuple(b.shape), name
        assert np.allclose(a, b.numpy(), rtol=1e-4, atol=1e-7), name
    # the input names the inference engine feeds must exist in the graph
    assert {i.name for i in session.get_inputs()} == set(inputs)
