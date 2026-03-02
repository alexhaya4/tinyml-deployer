"""Post-training quantization for TFLite models."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import tensorflow as tf


@dataclass
class QuantizationResult:
    """Results from quantizing a TFLite model."""

    original_size: int
    quantized_size: int
    compression_ratio: float
    quantization_type: str
    output_path: str


def _default_representative_dataset(
    input_shape: list[int],
    num_samples: int = 100,
) -> Callable[[], object]:
    """Create a representative dataset generator from random samples."""

    def generator() -> object:
        for _ in range(num_samples):
            yield [np.random.uniform(-1.0, 1.0, size=input_shape).astype(np.float32)]

    return generator


def _run_dummy_inference(interpreter: tf.lite.Interpreter) -> None:
    """Run a single forward pass so all tensor buffers are populated."""
    input_detail = interpreter.get_input_details()[0]
    dummy = np.zeros(input_detail["shape"], dtype=input_detail["dtype"])
    interpreter.set_tensor(input_detail["index"], dummy)
    interpreter.invoke()


def _rebuild_keras_model(interpreter: tf.lite.Interpreter) -> tf.keras.Model:
    """Rebuild a Keras Sequential model from TFLite interpreter weights.

    Uses the interpreter's ops details to find FULLY_CONNECTED layers and
    extract their weight/bias tensors. The interpreter must already have
    allocate_tensors() and invoke() called.
    """
    input_shape = list(interpreter.get_input_details()[0]["shape"])

    # Get FC layers from ops details
    try:
        ops = interpreter._get_ops_details()  # noqa: SLF001
    except AttributeError:
        raise RuntimeError(
            "Cannot extract ops details from this TFLite model. "
            "TF version may be too old."
        )

    fc_ops = [op for op in ops if op["op_name"] == "FULLY_CONNECTED"]
    if not fc_ops:
        raise ValueError(
            "No FULLY_CONNECTED layers found. "
            "Only Dense/fully-connected models are supported for re-quantization."
        )

    # For each FC op: inputs = [input_tensor, weight_tensor, bias_tensor]
    # TFLite stores weights as (output_units, input_features).
    layers_data: list[dict[str, np.ndarray | None]] = []
    for op in fc_ops:
        op_inputs = list(op.get("inputs", []))
        weight_idx = op_inputs[1] if len(op_inputs) > 1 else -1
        bias_idx = op_inputs[2] if len(op_inputs) > 2 else -1

        weight: np.ndarray | None = None
        bias: np.ndarray | None = None

        if weight_idx >= 0:
            try:
                weight = interpreter.get_tensor(weight_idx).copy()
            except ValueError:
                pass

        if bias_idx >= 0:
            try:
                bias = interpreter.get_tensor(bias_idx).copy()
            except ValueError:
                pass

        layers_data.append({"weight": weight, "bias": bias})

    # Build matching Keras Sequential model
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(shape=input_shape[1:]))

    for i, ld in enumerate(layers_data):
        w = ld["weight"]
        if w is None:
            raise ValueError(f"Could not read weight tensor for FC layer {i}")

        # TFLite weight shape: (output_units, input_features)
        units = w.shape[0]
        is_last = i == len(layers_data) - 1
        activation = None if is_last else "relu"
        has_bias = ld["bias"] is not None

        model.add(tf.keras.layers.Dense(
            units,
            activation=activation,
            use_bias=has_bias,
        ))

    model.build(input_shape=[None] + input_shape[1:])

    # Copy weights from interpreter into the Keras layers
    dense_layers = [
        layer for layer in model.layers
        if isinstance(layer, tf.keras.layers.Dense)
    ]
    for i, layer in enumerate(dense_layers):
        w = layers_data[i]["weight"]
        b = layers_data[i]["bias"]
        # Keras Dense expects kernel shape (input_features, output_units)
        weights_to_set: list[np.ndarray] = [w.T]
        if b is not None and layer.use_bias:
            weights_to_set.append(b)
        layer.set_weights(weights_to_set)

    return model


def _convert_from_tflite(
    model_content: bytes,
    quantization_type: str,
    representative_dataset: Callable[[], object] | None,
) -> bytes:
    """Re-convert a TFLite model through a saved-model round-trip.

    TFLite flatbuffers cannot be fed directly into TFLiteConverter, so we
    rebuild a Keras model from the interpreter weights, export it as a
    SavedModel, then convert with the requested quantization options.
    """
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    _run_dummy_inference(interpreter)

    input_shape = list(interpreter.get_input_details()[0]["shape"])

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_dir = os.path.join(tmpdir, "saved_model")

        keras_model = _rebuild_keras_model(interpreter)
        keras_model.export(saved_model_dir)

        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if quantization_type == "int8":
            if representative_dataset is None:
                representative_dataset = _default_representative_dataset(input_shape)
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        elif quantization_type == "float16":
            converter.target_spec.supported_types = [tf.float16]

        # dynamic: DEFAULT optimizations only (already set)

        return converter.convert()


def quantize_model(
    input_path: str,
    output_path: str | None = None,
    quantization_type: str = "int8",
    representative_dataset: Callable[[], object] | None = None,
) -> QuantizationResult:
    """Quantize a TFLite model using post-training quantization.

    Args:
        input_path: Path to the source .tflite file.
        output_path: Where to write the quantized model. Defaults to
            ``<name>_quant.tflite`` next to the original.
        quantization_type: One of "int8", "float16", or "dynamic".
        representative_dataset: Optional callable that yields representative
            input samples (required for best int8 accuracy; a random default
            is generated when *None*).

    Returns:
        QuantizationResult with size and compression information.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        ValueError: If *quantization_type* is not recognized.
    """
    valid_types = {"int8", "float16", "dynamic"}
    if quantization_type not in valid_types:
        raise ValueError(
            f"Unknown quantization type '{quantization_type}'. "
            f"Choose from: {', '.join(sorted(valid_types))}"
        )

    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Model file not found: {input_path}")

    if output_path is None:
        output_path = str(src.with_name(f"{src.stem}_quant{src.suffix}"))

    original_size = src.stat().st_size
    model_content = src.read_bytes()

    quantized_bytes = _convert_from_tflite(
        model_content, quantization_type, representative_dataset
    )

    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(quantized_bytes)

    quantized_size = dst.stat().st_size
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0.0

    return QuantizationResult(
        original_size=original_size,
        quantized_size=quantized_size,
        compression_ratio=compression_ratio,
        quantization_type=quantization_type,
        output_path=str(dst),
    )
