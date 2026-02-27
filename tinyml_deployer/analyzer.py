"""Analyze TFLite model files for MCU deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tensorflow as tf

from tinyml_deployer.targets import MCUTarget, get_target

ARENA_OVERHEAD_FACTOR = 1.5


@dataclass
class TensorInfo:
    """Metadata for a single tensor."""

    name: str
    shape: list[int]
    dtype: str
    size_bytes: int


@dataclass
class ModelAnalysis:
    """Complete analysis results for a TFLite model on a target MCU."""

    model_path: str
    model_size_bytes: int
    target_name: str
    inputs: list[TensorInfo]
    outputs: list[TensorInfo]
    operators: list[str]
    total_macs: int
    tensor_arena_bytes: int
    flash_usage_bytes: int
    estimated_latency_ms: float
    compatible: bool
    unsupported_ops: list[str]
    warnings: list[str]
    fits_in_flash: bool
    fits_in_ram: bool


def _tensor_info_from_detail(detail: dict) -> TensorInfo:
    """Convert an interpreter tensor detail dict to TensorInfo."""
    shape = list(detail["shape"])
    dtype = np.dtype(detail["dtype"])
    size_bytes = int(np.prod(shape)) * dtype.itemsize
    return TensorInfo(
        name=detail["name"],
        shape=shape,
        dtype=str(dtype),
        size_bytes=size_bytes,
    )


def _estimate_macs_for_op(
    op_name: str,
    input_tensors: list[dict],
    output_tensors: list[dict],
) -> int:
    """Estimate multiply-accumulate operations for a single operator."""
    if not output_tensors:
        return 0

    output_shape = output_tensors[0]["shape"]

    if op_name == "CONV_2D" and len(input_tensors) >= 2:
        kernel_shape = input_tensors[1]["shape"]
        if len(output_shape) == 4 and len(kernel_shape) == 4:
            out_h, out_w, out_c = output_shape[1], output_shape[2], output_shape[3]
            k_h, k_w, c_in = kernel_shape[1], kernel_shape[2], kernel_shape[3]
            return int(out_h * out_w * out_c * k_h * k_w * c_in)
        return 0

    if op_name == "DEPTHWISE_CONV_2D" and len(input_tensors) >= 2:
        kernel_shape = input_tensors[1]["shape"]
        if len(output_shape) == 4 and len(kernel_shape) == 4:
            out_h, out_w, out_c = output_shape[1], output_shape[2], output_shape[3]
            k_h, k_w = kernel_shape[1], kernel_shape[2]
            return int(out_h * out_w * out_c * k_h * k_w)
        return 0

    if op_name == "FULLY_CONNECTED" and len(input_tensors) >= 2:
        weight_shape = input_tensors[1]["shape"]
        if len(weight_shape) == 2:
            return int(weight_shape[0] * weight_shape[1])
        return 0

    return 0


def _estimate_total_macs(
    interpreter: tf.lite.Interpreter,
    ops_details: list[dict],
) -> int:
    """Estimate total MACs across all operators in the model."""
    tensor_details = {d["index"]: d for d in interpreter.get_tensor_details()}
    total = 0
    for op in ops_details:
        input_indices = [i for i in op.get("inputs", []) if i >= 0]
        output_indices = [i for i in op.get("outputs", []) if i >= 0]
        input_tensors = [tensor_details[i] for i in input_indices if i in tensor_details]
        output_tensors = [tensor_details[i] for i in output_indices if i in tensor_details]
        total += _estimate_macs_for_op(op["op_name"], input_tensors, output_tensors)
    return total


def _estimate_tensor_arena(interpreter: tf.lite.Interpreter) -> int:
    """Estimate tensor arena size as sum of all tensor sizes with overhead."""
    total = 0
    for detail in interpreter.get_tensor_details():
        shape = detail["shape"]
        itemsize = np.dtype(detail["dtype"]).itemsize
        total += int(np.prod(shape)) * itemsize
    return int(total * ARENA_OVERHEAD_FACTOR)


def analyze_model(model_path: str, target_name: str) -> ModelAnalysis:
    """Analyze a TFLite model for deployment to a target MCU.

    Args:
        model_path: Path to the .tflite model file.
        target_name: Name of the target MCU (e.g. "esp32", "stm32f4").

    Returns:
        ModelAnalysis with all results, warnings, and compatibility status.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the target name is not recognized.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    target = get_target(target_name)
    model_size_bytes = path.stat().st_size
    warnings: list[str] = []

    # Load model
    interpreter = tf.lite.Interpreter(model_path=str(path))
    interpreter.allocate_tensors()

    # Extract input/output info
    inputs = [_tensor_info_from_detail(d) for d in interpreter.get_input_details()]
    outputs = [_tensor_info_from_detail(d) for d in interpreter.get_output_details()]

    # Extract operators via private API (available since TF 2.5)
    ops_details: list[dict] = []
    operators: list[str] = []
    try:
        ops_details = interpreter._get_ops_details()  # noqa: SLF001
        operators = [op["op_name"] for op in ops_details]
    except (AttributeError, KeyError):
        warnings.append("Could not extract operator list (TF version may be too old)")

    # Check operator compatibility
    supported_set = set(target.supported_ops)
    unsupported_ops = sorted({op for op in operators if op not in supported_set})
    compatible = len(unsupported_ops) == 0

    if unsupported_ops:
        warnings.append(
            f"Unsupported ops for {target.name}: {', '.join(unsupported_ops)}"
        )

    # Memory estimates
    tensor_arena_bytes = _estimate_tensor_arena(interpreter)
    flash_usage_bytes = model_size_bytes

    fits_in_flash = flash_usage_bytes <= target.flash_kb * 1024
    fits_in_ram = tensor_arena_bytes <= target.ram_kb * 1024

    if not fits_in_flash:
        warnings.append(
            f"Model size ({flash_usage_bytes // 1024} KB) exceeds "
            f"{target.name} flash ({target.flash_kb} KB)"
        )
    if not fits_in_ram:
        warnings.append(
            f"Estimated arena ({tensor_arena_bytes // 1024} KB) exceeds "
            f"{target.name} RAM ({target.ram_kb} KB)"
        )

    # MAC and latency estimation
    total_macs = _estimate_total_macs(interpreter, ops_details) if ops_details else 0

    estimated_latency_ms = 0.0
    if total_macs > 0 and target.cycles_per_mac > 0:
        ops_per_second = target.clock_mhz * 1e6 / target.cycles_per_mac
        estimated_latency_ms = total_macs / ops_per_second * 1000
    elif total_macs == 0 and ops_details:
        warnings.append("Could not estimate MACs; latency estimate unavailable")

    return ModelAnalysis(
        model_path=str(path),
        model_size_bytes=model_size_bytes,
        target_name=target.name,
        inputs=inputs,
        outputs=outputs,
        operators=operators,
        total_macs=total_macs,
        tensor_arena_bytes=tensor_arena_bytes,
        flash_usage_bytes=flash_usage_bytes,
        estimated_latency_ms=estimated_latency_ms,
        compatible=compatible,
        unsupported_ops=unsupported_ops,
        warnings=warnings,
        fits_in_flash=fits_in_flash,
        fits_in_ram=fits_in_ram,
    )
