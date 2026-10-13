"""Numerical comparison of actual TFLite model predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf


@dataclass
class EvaluationResult:
    samples: int
    output_values: int
    mean_absolute_error: float
    max_absolute_error: float
    root_mean_squared_error: float
    max_allowed_error: float
    passed: bool


def load_samples(path: str) -> np.ndarray:
    """Load preprocessed samples from a non-pickled NPY array, leading axis N."""
    samples = np.load(path, allow_pickle=False, mmap_mode="r")
    if not isinstance(samples, np.ndarray) or samples.ndim < 1 or len(samples) == 0:
        raise ValueError("Dataset must be a non-empty NPY array with a sample axis.")
    if samples.dtype.kind != "f":
        raise ValueError(
            "Dataset must contain floating-point values in model input units."
        )
    if not np.isfinite(samples).all():
        raise ValueError("Dataset contains NaN or infinite values.")
    return samples


def interpreter_for(content: bytes) -> tf.lite.Interpreter:
    interpreter = tf.lite.Interpreter(
        model_content=content,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    if (
        len(interpreter.get_input_details()) != 1
        or len(interpreter.get_output_details()) != 1
    ):
        raise ValueError("Prediction validation requires one input and one output.")
    return interpreter


def encode_input(sample: np.ndarray, detail: dict) -> np.ndarray:
    expected = tuple(int(d) for d in detail["shape"])
    value = np.asarray(sample)
    if value.dtype.kind != "f" or not np.isfinite(value).all():
        raise ValueError(
            "Samples must contain finite floating-point values in input units."
        )
    if value.shape != expected:
        raise ValueError(
            f"Sample shape {value.shape} does not match model input {expected}."
        )
    dtype = np.dtype(detail["dtype"])
    if dtype == np.float32:
        converted = value.astype(np.float32)
        if not np.isfinite(converted).all():
            raise ValueError("Sample exceeds float32 input range.")
        return converted
    if dtype not in (np.dtype("int8"), np.dtype("uint8"), np.dtype("int16")):
        raise ValueError(f"Unsupported inference input dtype: {dtype}")
    scale, zero = detail["quantization"]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            "Integer input requires a positive per-tensor quantization scale."
        )
    limits = np.iinfo(dtype)
    return np.clip(np.rint(value / scale) + zero, limits.min, limits.max).astype(dtype)


def decode_output(value: np.ndarray, detail: dict) -> np.ndarray:
    if np.issubdtype(value.dtype, np.integer):
        scale, zero = detail["quantization"]
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                "Integer output requires a positive per-tensor quantization scale."
            )
        return (value.astype(np.float64) - zero) * scale
    return value.astype(np.float64)


def predict(interpreter: tf.lite.Interpreter, sample: np.ndarray) -> np.ndarray:
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.set_tensor(inp["index"], encode_input(sample, inp))
    interpreter.invoke()
    result = decode_output(interpreter.get_tensor(out["index"]), out)
    if not np.isfinite(result).all():
        raise ValueError("Model produced a non-finite prediction.")
    return result


def compare_models(
    reference: bytes, candidate: bytes, samples: np.ndarray, max_abs_error: float
) -> EvaluationResult:
    if not np.isfinite(max_abs_error) or max_abs_error < 0:
        raise ValueError("max_abs_error must be finite and non-negative.")
    if len(samples) == 0:
        raise ValueError("Validation requires at least one sample.")
    ref, quant = interpreter_for(reference), interpreter_for(candidate)
    count = 0
    absolute_sum = squared_sum = maximum = 0.0
    for sample in samples:
        a, b = predict(ref, sample), predict(quant, sample)
        if a.shape != b.shape:
            raise ValueError("Conversion changed the model output shape.")
        delta = np.abs(a - b)
        count += delta.size
        absolute_sum += float(delta.sum())
        squared_sum += float(np.square(delta).sum())
        maximum = max(maximum, float(delta.max()))
    if count == 0:
        raise ValueError("Model has an empty output tensor.")
    return EvaluationResult(
        len(samples),
        count,
        absolute_sum / count,
        maximum,
        float(np.sqrt(squared_sum / count)),
        max_abs_error,
        maximum <= max_abs_error,
    )


def evaluate_models(
    reference: str, candidate: str, data: str, max_abs_error: float
) -> EvaluationResult:
    return compare_models(
        Path(reference).read_bytes(),
        Path(candidate).read_bytes(),
        load_samples(data),
        max_abs_error,
    )
