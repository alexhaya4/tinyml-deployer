"""Tests for tinyml_deployer.deployer."""

from __future__ import annotations

from pathlib import Path

import pytest

from tinyml_deployer.deployer import DeployResult, deploy_model


class TestDeployESP32:
    """Test deploy_model() for ESP-IDF targets."""

    def test_returns_deploy_result(self, sine_model_path: str, tmp_output: str) -> None:
        out_dir = str(Path(tmp_output) / "esp_proj")
        result = deploy_model(sine_model_path, "esp32", out_dir)
        assert isinstance(result, DeployResult)
        assert result.target_name == "esp32"
        assert result.analysis is not None

    def test_creates_expected_files(self, sine_model_path: str, tmp_output: str) -> None:
        out_dir = Path(tmp_output) / "esp_proj"
        deploy_model(sine_model_path, "esp32", str(out_dir))

        expected = [
            out_dir / "CMakeLists.txt",
            out_dir / "sdkconfig.defaults",
            out_dir / "README.md",
            out_dir / "main" / "CMakeLists.txt",
            out_dir / "main" / "main.c",
            out_dir / "main" / "model_data.h",
            out_dir / "main" / "model_data.c",
            out_dir / "main" / "inference.h",
            out_dir / "main" / "inference.c",
        ]
        for path in expected:
            assert path.exists(), f"Missing: {path}"

    def test_main_c_has_app_main(self, sine_model_path: str, tmp_output: str) -> None:
        out_dir = Path(tmp_output) / "esp_proj"
        deploy_model(sine_model_path, "esp32", str(out_dir))
        main_c = (out_dir / "main" / "main.c").read_text()
        assert "app_main" in main_c
        assert "model_init" in main_c

    def test_file_count(self, sine_model_path: str, tmp_output: str) -> None:
        out_dir = str(Path(tmp_output) / "esp_proj")
        result = deploy_model(sine_model_path, "esp32", out_dir)
        assert len(result.files) == 9


class TestDeploySTM32:
    """Test deploy_model() for STM32 targets."""

    def test_returns_deploy_result(self, sine_model_path: str, tmp_output: str) -> None:
        out_dir = str(Path(tmp_output) / "stm_proj")
        result = deploy_model(sine_model_path, "stm32f4", out_dir)
        assert isinstance(result, DeployResult)
        assert result.target_name == "stm32f4"

    def test_creates_expected_files(self, sine_model_path: str, tmp_output: str) -> None:
        out_dir = Path(tmp_output) / "stm_proj"
        deploy_model(sine_model_path, "stm32f4", str(out_dir))

        expected = [
            out_dir / "Makefile",
            out_dir / "README.md",
            out_dir / "model_data.h",
            out_dir / "model_data.c",
            out_dir / "inference.h",
            out_dir / "inference.c",
            out_dir / "Core" / "Inc" / "main.h",
            out_dir / "Core" / "Src" / "main.c",
        ]
        for path in expected:
            assert path.exists(), f"Missing: {path}"

    def test_makefile_has_cortex_flag(self, sine_model_path: str, tmp_output: str) -> None:
        out_dir = Path(tmp_output) / "stm_proj"
        deploy_model(sine_model_path, "stm32f4", str(out_dir))
        makefile = (out_dir / "Makefile").read_text()
        assert "cortex-m4" in makefile

    def test_stm32h7_uses_cortex_m7(self, sine_model_path: str, tmp_output: str) -> None:
        out_dir = Path(tmp_output) / "stm_h7_proj"
        deploy_model(sine_model_path, "stm32h7", str(out_dir))
        makefile = (out_dir / "Makefile").read_text()
        assert "cortex-m7" in makefile

    def test_file_count(self, sine_model_path: str, tmp_output: str) -> None:
        out_dir = str(Path(tmp_output) / "stm_proj")
        result = deploy_model(sine_model_path, "stm32f4", out_dir)
        assert len(result.files) == 8
