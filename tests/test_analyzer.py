"""Tests for tinyml_deployer.analyzer."""

from __future__ import annotations

import pytest

from tinyml_deployer.analyzer import ModelAnalysis, TensorInfo, analyze_model


class TestAnalyzeModel:
    """Test analyze_model() on the sine wave example."""

    def test_returns_model_analysis(self, sine_model_path: str) -> None:
        result = analyze_model(sine_model_path, "esp32")
        assert isinstance(result, ModelAnalysis)

    def test_model_size_positive(self, sine_model_path: str) -> None:
        result = analyze_model(sine_model_path, "esp32")
        assert result.model_size_bytes > 0

    def test_has_inputs_and_outputs(self, sine_model_path: str) -> None:
        result = analyze_model(sine_model_path, "esp32")
        assert len(result.inputs) >= 1
        assert len(result.outputs) >= 1
        assert isinstance(result.inputs[0], TensorInfo)
        assert isinstance(result.outputs[0], TensorInfo)

    def test_input_tensor_info(self, sine_model_path: str) -> None:
        result = analyze_model(sine_model_path, "esp32")
        inp = result.inputs[0]
        assert len(inp.shape) > 0
        assert inp.dtype == "float32"
        assert inp.size_bytes > 0

    def test_has_operators(self, sine_model_path: str) -> None:
        result = analyze_model(sine_model_path, "esp32")
        assert len(result.operators) > 0
        assert "FULLY_CONNECTED" in result.operators

    def test_memory_estimates_positive(self, sine_model_path: str) -> None:
        result = analyze_model(sine_model_path, "esp32")
        assert result.tensor_arena_bytes > 0
        assert result.flash_usage_bytes > 0

    def test_total_macs_positive(self, sine_model_path: str) -> None:
        result = analyze_model(sine_model_path, "esp32")
        assert result.total_macs > 0

    def test_fits_in_flash_and_ram(self, sine_model_path: str) -> None:
        result = analyze_model(sine_model_path, "esp32")
        assert result.fits_in_flash is True
        assert result.fits_in_ram is True

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            analyze_model("nonexistent.tflite", "esp32")

    def test_invalid_target(self, sine_model_path: str) -> None:
        with pytest.raises(ValueError, match="Unknown target"):
            analyze_model(sine_model_path, "bad_target")
