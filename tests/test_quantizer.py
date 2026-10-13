"""Real source conversion and numerical quality gates."""

from pathlib import Path

import numpy as np
import pytest

from tinyml_deployer.evaluation import interpreter_for, predict
from tinyml_deployer.quantizer import quantize_model


@pytest.mark.parametrize(
    "mode,budget", [("dynamic", 1e-5), ("float16", 1e-3), ("int8", 0.03)]
)
def test_preserves_nonlinear_predictions(
    source_model, validation_samples, calibration_samples, tmp_path, mode, budget
):
    path, model = source_model
    out = tmp_path / "model.tflite"
    result = quantize_model(
        path,
        str(out),
        mode,
        lambda: ([sample] for sample in calibration_samples),
        validation_samples=validation_samples,
        max_abs_error=budget,
    )
    assert result.evaluation.passed
    interpreter = interpreter_for(out.read_bytes())
    for sample in validation_samples:
        np.testing.assert_allclose(
            predict(interpreter, sample), model(sample).numpy(), atol=budget, rtol=0
        )


def test_error_gate_does_not_write(source_model, validation_samples, tmp_path):
    out = tmp_path / "rejected.tflite"
    with pytest.raises(ValueError, match="exceeds"):
        quantize_model(
            source_model[0],
            str(out),
            "float16",
            validation_samples=validation_samples,
            max_abs_error=0,
        )
    assert not out.exists()


def test_int8_requires_real_calibration(source_model, validation_samples, tmp_path):
    with pytest.raises(ValueError, match="calibration"):
        quantize_model(
            source_model[0],
            str(tmp_path / "q.tflite"),
            validation_samples=validation_samples,
            max_abs_error=0.03,
        )


def test_rejects_reconstruction(sine_model_path, validation_samples, tmp_path):
    with pytest.raises(ValueError, match="original"):
        quantize_model(
            sine_model_path,
            str(tmp_path / "q.tflite"),
            "dynamic",
            validation_samples=validation_samples,
            max_abs_error=0.01,
        )


def test_saved_model_supported(source_model, validation_samples, tmp_path):
    saved = tmp_path / "saved"
    source_model[1].export(str(saved))
    result = quantize_model(
        str(saved),
        quantization_type="dynamic",
        validation_samples=validation_samples,
        max_abs_error=1e-5,
    )
    assert Path(result.output_path) == tmp_path / "saved_quant.tflite"
    assert Path(result.output_path).exists()


def test_existing_output_preserved(source_model, validation_samples, tmp_path):
    out = tmp_path / "existing.tflite"
    out.write_bytes(b"existing user data")
    with pytest.raises(FileExistsError):
        quantize_model(
            source_model[0],
            str(out),
            "dynamic",
            validation_samples=validation_samples,
            max_abs_error=0.01,
        )
    assert out.read_bytes() == b"existing user data"


def test_bad_calibration_shape_fails(source_model, validation_samples, tmp_path):
    with pytest.raises((ValueError, RuntimeError)):
        quantize_model(
            source_model[0],
            str(tmp_path / "q.tflite"),
            "int8",
            lambda: ([np.ones((2, 3), np.float32)] for _ in range(2)),
            validation_samples=validation_samples,
            max_abs_error=0.03,
        )
    assert not (tmp_path / "q.tflite").exists()
