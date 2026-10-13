"""Quantize source graphs and enforce a supplied numerical error budget."""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from tinyml_deployer.evaluation import EvaluationResult, compare_models, interpreter_for
from tinyml_deployer.io_utils import write_new_file


@dataclass
class QuantizationResult:
    original_size: int
    quantized_size: int
    compression_ratio: float
    quantization_type: str
    output_path: str
    evaluation: EvaluationResult


@contextmanager
def _source_graph(path: Path) -> Iterator[str]:
    if path.is_dir() and (path / "saved_model.pb").is_file():
        yield str(path)
        return
    with tempfile.TemporaryDirectory(prefix="tinyml_source_") as directory:
        saved = Path(directory) / "saved_model"
        if path.suffix.lower() == ".keras":
            model = tf.keras.models.load_model(str(path), compile=False, safe_mode=True)
            # Preserve explicitly fixed batch dimensions during Keras export.
            signature = [
                tf.TensorSpec(t.shape, t.dtype, name=t.name.split(":")[0])
                for t in model.inputs
            ]
            model.export(str(saved), input_signature=signature)
        elif path.suffix.lower() == ".onnx":
            from tinyml_deployer.converters import _check_onnx_deps

            _check_onnx_deps()
            import onnx
            import onnx2tf

            onnx.checker.check_model(str(path))
            onnx2tf.convert(
                input_onnx_file_path=str(path),
                output_folder_path=str(saved),
                non_verbose=True,
                not_use_onnxsim=True,
            )
            if not (saved / "saved_model.pb").is_file():
                raise ValueError(
                    "ONNX converter did not produce the required SavedModel."
                )
        else:
            raise ValueError(
                "Quantization requires an original .keras, .onnx or SavedModel source. "
                "Existing .tflite models can be analyzed and deployed, not reconstructed."
            )
        yield str(saved)


def quantize_model(
    input_path: str,
    output_path: str | None = None,
    quantization_type: str = "int8",
    representative_dataset: Callable[[], object] | None = None,
    *,
    validation_samples: np.ndarray | None = None,
    max_abs_error: float | None = None,
) -> QuantizationResult:
    if quantization_type not in {"int8", "float16", "dynamic"}:
        raise ValueError(f"Unknown quantization type: {quantization_type}")
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Model file not found: {input_path}")
    if validation_samples is None or max_abs_error is None:
        raise ValueError(
            "Supply validation samples and an explicit maximum absolute error."
        )
    if not np.isfinite(validation_samples).all() or len(validation_samples) == 0:
        raise ValueError("Validation samples must be non-empty and finite.")
    if not np.isfinite(max_abs_error) or max_abs_error < 0:
        raise ValueError("Maximum absolute error must be finite and non-negative.")
    if quantization_type == "int8" and representative_dataset is None:
        raise ValueError("INT8 quantization requires representative calibration data.")
    dst = (
        Path(output_path) if output_path else src.with_name(src.stem + "_quant.tflite")
    )
    if dst.exists():
        raise FileExistsError(f"Output already exists: {dst}")
    with _source_graph(src) as saved:
        reference = tf.lite.TFLiteConverter.from_saved_model(saved).convert()
        # Validate sample shapes and input count before running calibration.
        baseline = interpreter_for(reference)
        expected = tuple(baseline.get_input_details()[0]["shape"])
        converter = tf.lite.TFLiteConverter.from_saved_model(saved)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if quantization_type == "int8":

            def checked_calibration() -> Iterator[list[np.ndarray]]:
                count = 0
                for item in representative_dataset():
                    if not isinstance(item, (list, tuple)) or len(item) != 1:
                        raise ValueError(
                            "Calibration must yield one input tensor per sample."
                        )
                    sample = np.asarray(item[0])
                    if (
                        sample.shape != expected
                        or sample.dtype != np.float32
                        or not np.isfinite(sample).all()
                    ):
                        raise ValueError(
                            f"Calibration requires finite float32 tensors with shape {expected}."
                        )
                    count += 1
                    yield [sample]
                if count == 0:
                    raise ValueError("Calibration dataset is empty.")

            converter.representative_dataset = checked_calibration
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif quantization_type == "float16":
            converter.target_spec.supported_types = [tf.float16]
        candidate = converter.convert()
    evaluation = compare_models(reference, candidate, validation_samples, max_abs_error)
    if not evaluation.passed:
        raise ValueError(
            f"Quantization error {evaluation.max_absolute_error:.8g} exceeds "
            f"the allowed {max_abs_error:.8g}; no model was written."
        )
    write_new_file(dst, candidate)
    return QuantizationResult(
        len(reference),
        len(candidate),
        len(reference) / len(candidate),
        quantization_type,
        str(dst),
        evaluation,
    )
