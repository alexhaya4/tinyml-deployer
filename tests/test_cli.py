"""CLI commands exercise real conversion and filesystem behavior."""

import json

from click.testing import CliRunner

from tinyml_deployer.cli import cli


def test_analyze(sine_model_path):
    result = CliRunner().invoke(
        cli, ["analyze", sine_model_path, "--target", "esp32s3"]
    )
    assert result.exit_code == 0, result.output
    assert "Static preflight" in result.output


def test_quantize_source(source_model, validation_path, tmp_path):
    output = tmp_path / "quantized.tflite"
    report = tmp_path / "quantization.json"
    result = CliRunner().invoke(
        cli,
        [
            "quantize",
            source_model[0],
            "--type",
            "dynamic",
            "--validation",
            validation_path,
            "--max-abs-error",
            "0.00001",
            "--output",
            str(output),
            "--report",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    assert json.loads(report.read_text())["evaluation"]["passed"]


def test_quantize_requires_validation(source_model):
    result = CliRunner().invoke(cli, ["quantize", source_model[0]])
    assert result.exit_code != 0
    assert "--validation" in result.output


def test_deploy(sine_model_path, validation_path, tmp_path):
    result = CliRunner().invoke(
        cli,
        [
            "deploy",
            sine_model_path,
            "--target",
            "esp32s3",
            "--output",
            str(tmp_path / "firmware"),
            "--arena-size",
            "65536",
            "--validation",
            validation_path,
            "--max-abs-error",
            "0.00001",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "firmware" / "main" / "inference.cpp").exists()


def test_estimate_is_labeled(sine_model_path):
    result = CliRunner().invoke(cli, ["benchmark", sine_model_path])
    assert result.exit_code == 0, result.output
    assert "analytical estimate" in result.output


def test_evaluate(sine_model_path, validation_path, tmp_path):
    report = tmp_path / "evaluation.html"
    result = CliRunner().invoke(
        cli,
        [
            "evaluate",
            sine_model_path,
            sine_model_path,
            "--validation",
            validation_path,
            "--max-abs-error",
            "0",
            "--report",
            str(report),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "max_absolute_error" in report.read_text()


def test_doctor_writes_actual_status(tmp_path):
    report = tmp_path / "doctor.json"
    result = CliRunner().invoke(cli, ["doctor", "--report", str(report)])
    assert result.exit_code in (0, 1), result.output
    data = json.loads(report.read_text())
    assert data["ready"] == all(check["ok"] for check in data["checks"])


def test_profile_invalid_port(tmp_path):
    manifest = tmp_path / "deployment.json"
    manifest.write_text('{"schema_version":1}')
    result = CliRunner().invoke(
        cli,
        [
            "profile",
            "--port",
            "/nonexistent/tinyml-device",
            "--manifest",
            str(manifest),
            "--report",
            str(tmp_path / "result.json"),
            "--timeout",
            "1",
        ],
    )
    assert result.exit_code != 0
    assert not (tmp_path / "result.json").exists()
