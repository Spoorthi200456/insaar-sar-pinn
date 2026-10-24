"""
Export a trained `ARSNet` checkpoint to ONNX, then optionally to a
dynamically-quantized INT8 ONNX artifact for CPU deployment.

    python -m src.backend.export_onnx \
        --checkpoint checkpoints/ars_net.pt \
        --output artifacts/ars_net.onnx \
        --sequence-length 52 \
        --num-structures 32 \
        --quantize int8

FP16 acceleration is handled at *inference* time by selecting the CUDA or
TensorRT execution provider in `inference_engine.py` (ONNX Runtime applies
FP16 kernels on those providers automatically for compatible ops); this
script additionally supports `--fp16-graph` to convert the exported graph's
initializers/weights to float16 directly for GPU deployments where a
half-precision graph is preferred over provider-side casting.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.models.network import ARSNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_path: Path, device: str = "cpu") -> ARSNet:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("model_config", {"design_feature_dim": 3})
    model = ARSNet(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_dummy_inputs(
    num_structures: int, sequence_length: int, design_feature_dim: int, device: str = "cpu"
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(42)
    adjacency = torch.rand(num_structures, num_structures, generator=generator)
    adjacency = (adjacency + adjacency.T) / 2
    adjacency.fill_diagonal_(0.0)

    return {
        "design_features": torch.randn(num_structures, design_feature_dim, generator=generator),
        "displacement_cm": torch.randn(num_structures, sequence_length, generator=generator) * 0.5,
        "rainfall_mm": torch.rand(num_structures, sequence_length, generator=generator) * 50.0,
        "time_delta_days": torch.full((num_structures, sequence_length), 7.0),
        "adjacency": adjacency,
        "incidence_angle_deg": torch.rand(num_structures, generator=generator) * 17.0 + 29.0,
        "coherence_mask": torch.ones(num_structures, sequence_length, dtype=torch.bool),
    }


class ARSNetONNXWrapper(torch.nn.Module):
    """
    Thin wrapper flattening `ARSNetOutput` into a plain tensor tuple, since
    `torch.onnx.export` requires tensor (or tuple-of-tensor) outputs, not
    a dataclass.
    """

    def __init__(self, model: ARSNet) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        design_features: torch.Tensor,
        displacement_cm: torch.Tensor,
        rainfall_mm: torch.Tensor,
        time_delta_days: torch.Tensor,
        adjacency: torch.Tensor,
        incidence_angle_deg: torch.Tensor,
        coherence_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(
            design_features=design_features,
            displacement_cm=displacement_cm,
            rainfall_mm=rainfall_mm,
            time_delta_days=time_delta_days,
            adjacency=adjacency,
            incidence_angle_deg=incidence_angle_deg,
            coherence_mask=coherence_mask,
        )
        return (
            out.hydraulic_conductivity,
            out.recharge_rate,
            out.predicted_head_change,
            out.predicted_displacement_cm,
        )


def export_to_onnx(
    model: ARSNet,
    output_path: Path,
    sequence_length: int,
    num_structures: int,
    design_feature_dim: int,
    opset: int = 17,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ARSNetONNXWrapper(model).eval()
    dummy = build_dummy_inputs(num_structures, sequence_length, design_feature_dim)

    input_names = list(dummy.keys())
    output_names = [
        "hydraulic_conductivity",
        "recharge_rate",
        "predicted_head_change",
        "predicted_displacement_cm",
    ]
    dynamic_axes = {
        "design_features": {0: "num_structures"},
        "displacement_cm": {0: "num_structures", 1: "sequence_length"},
        "rainfall_mm": {0: "num_structures", 1: "sequence_length"},
        "time_delta_days": {0: "num_structures", 1: "sequence_length"},
        "adjacency": {0: "num_structures", 1: "num_structures"},
        "incidence_angle_deg": {0: "num_structures"},
        "coherence_mask": {0: "num_structures", 1: "sequence_length"},
        "hydraulic_conductivity": {0: "num_structures"},
        "recharge_rate": {0: "num_structures", 1: "sequence_length"},
        "predicted_head_change": {0: "num_structures", 1: "sequence_length"},
        "predicted_displacement_cm": {0: "num_structures", 1: "sequence_length"},
    }

    logger.info("Exporting ARSNet to ONNX at %s (opset=%d)", output_path, opset)
    # `dynamo=False` pins the legacy TorchScript-based exporter: it has no
    # `onnxscript` dependency and handles our Python-level `for` loops
    # (unrolled at trace time, which is correct since `sequence_length` is
    # fixed) more predictably than the newer dynamo/FX exporter path. Torch
    # versions that predate the `dynamo` kwarg simply ignore it via the
    # try/except below.
    try:
        torch.onnx.export(
            wrapper,
            tuple(dummy[name] for name in input_names),
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
    except TypeError:
        # Older torch (<2.1) does not know the `dynamo` kwarg at all.
        torch.onnx.export(
            wrapper,
            tuple(dummy[name] for name in input_names),
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )
    logger.info("ONNX export complete: %s", output_path)
    return output_path


def quantize_dynamic_int8(onnx_path: Path, output_path: Path | None = None) -> Path:
    """
    CPU-targeted dynamic INT8 quantization via onnxruntime's built-in
    quantizer.

    The unrolled LSTMCell/attention loops in `ARSNet` produce many small
    Gemm/MatMul nodes whose output types the ORT quantizer's shape-inference
    pass can fail to recover on its own; we first run ONNX's own
    (shape-inference-with-data-propagation) pass and write an intermediate
    "inferred" graph, then quantize *that* -- this is the standard fix for
    the `Unable to find data type for weight_name=...` failure mode on
    graphs like this one. `DefaultTensorType=FLOAT` is passed as a second
    line of defense for any node the inference pass still can't resolve.
    """
    import onnx
    from onnx import TensorProto
    from onnx.shape_inference import infer_shapes
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_path = output_path or onnx_path.with_name(onnx_path.stem + ".int8.onnx")

    inferred_path = onnx_path.with_name(onnx_path.stem + ".shape_inferred.onnx")
    model = onnx.load(str(onnx_path))
    inferred = infer_shapes(model, check_type=True, strict_mode=False, data_prop=True)
    onnx.save(inferred, str(inferred_path))

    logger.info("Quantizing %s -> INT8 at %s", inferred_path, output_path)
    quantize_dynamic(
        model_input=str(inferred_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
        per_channel=False,
        reduce_range=True,
        extra_options={"DefaultTensorType": TensorProto.FLOAT},
    )
    logger.info("INT8 quantization complete: %s", output_path)
    return output_path


def convert_graph_to_fp16(onnx_path: Path, output_path: Path | None = None) -> Path:
    """GPU-targeted FP16 graph conversion (weights + compatible ops cast to float16)."""
    import onnx
    from onnxconverter_common import float16

    output_path = output_path or onnx_path.with_name(onnx_path.stem + ".fp16.onnx")
    model = onnx.load(str(onnx_path))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(output_path))
    logger.info("FP16 graph conversion complete: %s", output_path)
    return output_path


def try_export_tensorrt_engine(
    onnx_path: Path, output_path: Path, fp16: bool = True
) -> Path | None:
    """
    Best-effort TensorRT engine build from the exported ONNX graph. Requires
    `tensorrt` + a CUDA-capable build host; returns None (and logs a clear
    reason) when unavailable rather than raising, so this script remains
    runnable end-to-end on CPU-only CI/build machines. The FastAPI backend
    does not depend on this artifact -- it uses ONNX Runtime's own
    TensorrtExecutionProvider, which JIT-builds/caches the engine on first
    inference from the plain ONNX graph.
    """
    try:
        import tensorrt as trt  # noqa: F401
    except ImportError:
        logger.warning(
            "tensorrt package not available in this environment; skipping standalone "
            "TensorRT engine build. ONNX Runtime's TensorrtExecutionProvider will still "
            "build/cache an engine at inference time if the runtime host has TensorRT installed."
        )
        return None

    import tensorrt as trt

    logger_trt = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger_trt)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger_trt)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("TensorRT ONNX parse error: %s", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX graph for TensorRT engine build")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(serialized_engine)
    logger.info("TensorRT engine written to %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export ARSNet to optimized ONNX / TensorRT artifacts"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=52)
    parser.add_argument(
        "--num-structures", type=int, default=32, help="Trace batch size (dynamic afterward)"
    )
    parser.add_argument("--design-feature-dim", type=int, default=3)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--quantize", choices=["none", "int8"], default="none")
    parser.add_argument("--fp16-graph", action="store_true")
    parser.add_argument("--tensorrt-engine", type=Path, default=None)
    args = parser.parse_args()

    model = load_model_from_checkpoint(args.checkpoint)
    onnx_path = export_to_onnx(
        model,
        args.output,
        args.sequence_length,
        args.num_structures,
        args.design_feature_dim,
        args.opset,
    )

    if args.quantize == "int8":
        quantize_dynamic_int8(onnx_path)
    if args.fp16_graph:
        convert_graph_to_fp16(onnx_path)
    if args.tensorrt_engine is not None:
        try_export_tensorrt_engine(onnx_path, args.tensorrt_engine)


if __name__ == "__main__":
    main()
