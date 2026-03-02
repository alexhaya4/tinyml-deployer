"""Tests for the tinyml-deployer CLI commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from tinyml_deployer.cli import cli
from tests.conftest import SINE_MODEL_PATH


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class TestAnalyzeCommand:
    """Test the 'analyze' CLI command."""

    def test_exit_code_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["analyze", SINE_MODEL_PATH, "--target", "esp32"])
        assert result.exit_code == 0, result.output

    def test_output_contains_model_info(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["analyze", SINE_MODEL_PATH])
        assert "Model Info" in result.output
        assert "Memory Estimates" in result.output

    def test_invalid_target_exits_nonzero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["analyze", SINE_MODEL_PATH, "--target", "bad"])
        assert result.exit_code != 0


class TestQuantizeCommand:
    """Test the 'quantize' CLI command."""

    def test_exit_code_zero(self, runner: CliRunner, tmp_path: Path) -> None:
        out = str(tmp_path / "q.tflite")
        result = runner.invoke(cli, [
            "quantize", SINE_MODEL_PATH, "--type", "dynamic", "--output", out,
        ])
        assert result.exit_code == 0, result.output

    def test_output_contains_result(self, runner: CliRunner, tmp_path: Path) -> None:
        out = str(tmp_path / "q.tflite")
        result = runner.invoke(cli, [
            "quantize", SINE_MODEL_PATH, "--type", "dynamic", "--output", out,
        ])
        assert "Quantization Result" in result.output


class TestDeployCommand:
    """Test the 'deploy' CLI command."""

    def test_exit_code_zero_esp32(self, runner: CliRunner, tmp_path: Path) -> None:
        out = str(tmp_path / "proj")
        result = runner.invoke(cli, [
            "deploy", SINE_MODEL_PATH, "--target", "esp32", "--output", out,
        ])
        assert result.exit_code == 0, result.output

    def test_exit_code_zero_stm32(self, runner: CliRunner, tmp_path: Path) -> None:
        out = str(tmp_path / "proj")
        result = runner.invoke(cli, [
            "deploy", SINE_MODEL_PATH, "--target", "stm32f4", "--output", out,
        ])
        assert result.exit_code == 0, result.output

    def test_output_contains_generated_files(self, runner: CliRunner, tmp_path: Path) -> None:
        out = str(tmp_path / "proj")
        result = runner.invoke(cli, [
            "deploy", SINE_MODEL_PATH, "--target", "esp32", "--output", out,
        ])
        assert "Generated Files" in result.output


class TestBenchmarkCommand:
    """Test the 'benchmark' CLI command."""

    def test_exit_code_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, [
            "benchmark", SINE_MODEL_PATH, "--target", "esp32",
        ])
        assert result.exit_code == 0, result.output

    def test_compare_flag(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, [
            "benchmark", SINE_MODEL_PATH, "--compare",
        ])
        assert result.exit_code == 0, result.output

    def test_output_contains_benchmark(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, [
            "benchmark", SINE_MODEL_PATH, "--target", "esp32",
        ])
        assert "Benchmark Result" in result.output

    def test_compare_output_contains_table(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, [
            "benchmark", SINE_MODEL_PATH, "--compare",
        ])
        assert "Target Comparison" in result.output
