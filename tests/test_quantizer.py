"""Tests for tinyml_deployer.quantizer."""

from __future__ import annotations

from pathlib import Path

import pytest
import tensorflow as tf

from tinyml_deployer.quantizer import QuantizationResult, quantize_model


class TestQuantizeModel:
    """Test quantize_model() across all quantization modes."""

    @pytest.mark.parametrize("qtype", ["int8", "float16", "dynamic"])
    def test_produces_output_file(
        self, sine_model_path: str, tmp_output: str, qtype: str
    ) -> None:
        out_path = str(Path(tmp_output) / f"model_{qtype}.tflite")
        result = quantize_model(
            sine_model_path,
            output_path=out_path,
            quantization_type=qtype,
        )
        assert Path(result.output_path).exists()

    @pytest.mark.parametrize("qtype", ["int8", "float16", "dynamic"])
    def test_output_is_valid_tflite(
        self, sine_model_path: str, tmp_output: str, qtype: str
    ) -> None:
        out_path = str(Path(tmp_output) / f"model_{qtype}.tflite")
        result = quantize_model(
            sine_model_path,
            output_path=out_path,
            quantization_type=qtype,
        )
        # Loading with the TFLite interpreter should not raise
        interpreter = tf.lite.Interpreter(model_path=result.output_path)
        interpreter.allocate_tensors()
        assert len(interpreter.get_input_details()) >= 1

    @pytest.mark.parametrize("qtype", ["int8", "float16", "dynamic"])
    def test_result_fields(
        self, sine_model_path: str, tmp_output: str, qtype: str
    ) -> None:
        out_path = str(Path(tmp_output) / f"model_{qtype}.tflite")
        result = quantize_model(
            sine_model_path,
            output_path=out_path,
            quantization_type=qtype,
        )
        assert isinstance(result, QuantizationResult)
        assert result.original_size > 0
        assert result.quantized_size > 0
        assert result.compression_ratio > 0
        assert result.quantization_type == qtype
        assert result.output_path == out_path

    def test_default_output_path(self, sine_model_path: str) -> None:
        result = quantize_model(sine_model_path, quantization_type="dynamic")
        assert "_quant" in result.output_path
        # Clean up
        Path(result.output_path).unlink(missing_ok=True)

    def test_invalid_type_raises(self, sine_model_path: str) -> None:
        with pytest.raises(ValueError, match="Unknown quantization type"):
            quantize_model(sine_model_path, quantization_type="bad")

    def test_file_not_found(self, tmp_output: str) -> None:
        with pytest.raises(FileNotFoundError):
            quantize_model("nonexistent.tflite", output_path=str(Path(tmp_output) / "out.tflite"))
