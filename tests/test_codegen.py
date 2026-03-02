"""Tests for tinyml_deployer.codegen."""

from __future__ import annotations

from pathlib import Path

import pytest

from tinyml_deployer.analyzer import analyze_model
from tinyml_deployer.codegen import GeneratedFile, generate_inference_wrapper, generate_model_data


class TestGenerateModelData:
    """Test model_data.h and model_data.c generation."""

    def test_produces_two_files(self, sine_model_path: str, tmp_output: str) -> None:
        files = generate_model_data(sine_model_path, tmp_output)
        assert len(files) == 2
        assert all(isinstance(f, GeneratedFile) for f in files)

    def test_header_content(self, sine_model_path: str, tmp_output: str) -> None:
        generate_model_data(sine_model_path, tmp_output)
        header = (Path(tmp_output) / "model_data.h").read_text()
        assert "#ifndef MODEL_DATA_H" in header
        assert "extern const unsigned char model_tflite[]" in header
        assert "extern const unsigned int model_tflite_len" in header

    def test_source_content(self, sine_model_path: str, tmp_output: str) -> None:
        generate_model_data(sine_model_path, tmp_output)
        source = (Path(tmp_output) / "model_data.c").read_text()
        assert '#include "model_data.h"' in source
        assert "const unsigned char model_tflite[]" in source
        assert "0x" in source  # hex byte values

    def test_source_has_correct_length(self, sine_model_path: str, tmp_output: str) -> None:
        model_size = Path(sine_model_path).stat().st_size
        generate_model_data(sine_model_path, tmp_output)
        source = (Path(tmp_output) / "model_data.c").read_text()
        assert f"model_tflite_len = {model_size}" in source


class TestGenerateInferenceWrapper:
    """Test inference.h and inference.c generation."""

    @pytest.fixture
    def analysis(self, sine_model_path: str):
        return analyze_model(sine_model_path, "esp32")

    def test_produces_two_files(self, analysis, tmp_output: str) -> None:
        files = generate_inference_wrapper(analysis, "ESP-IDF", tmp_output)
        assert len(files) == 2
        assert all(isinstance(f, GeneratedFile) for f in files)

    def test_header_has_api_declarations(self, analysis, tmp_output: str) -> None:
        generate_inference_wrapper(analysis, "ESP-IDF", tmp_output)
        header = (Path(tmp_output) / "inference.h").read_text()
        assert "int model_init(void)" in header
        assert "int model_run(" in header
        assert "void model_free(void)" in header
        assert "TENSOR_ARENA_SIZE" in header
        assert "MODEL_INPUT_SIZE" in header
        assert "MODEL_OUTPUT_SIZE" in header

    def test_source_includes_tflite_headers(self, analysis, tmp_output: str) -> None:
        generate_inference_wrapper(analysis, "ESP-IDF", tmp_output)
        source = (Path(tmp_output) / "inference.c").read_text()
        assert "micro_interpreter.h" in source
        assert '#include "model_data.h"' in source

    def test_esp_includes_esp_log(self, analysis, tmp_output: str) -> None:
        generate_inference_wrapper(analysis, "ESP-IDF", tmp_output)
        source = (Path(tmp_output) / "inference.c").read_text()
        assert "esp_log.h" in source

    def test_stm32_omits_esp_log(self, analysis, tmp_output: str) -> None:
        generate_inference_wrapper(analysis, "STM32CubeAI", tmp_output)
        source = (Path(tmp_output) / "inference.c").read_text()
        assert "esp_log.h" not in source

    def test_custom_arena_size(self, analysis, tmp_output: str) -> None:
        generate_inference_wrapper(analysis, "ESP-IDF", tmp_output, arena_size=8192)
        header = (Path(tmp_output) / "inference.h").read_text()
        assert "#define TENSOR_ARENA_SIZE 8192" in header
