"""Emulator provenance and real QEMU execution checks used by firmware CI."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from tinyml_deployer.cli import cli
from tinyml_deployer.emulator import _flash_inputs, emulate_project


def test_existing_report_preserved(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text("existing results")
    result = CliRunner().invoke(
        cli, ["emulate", str(tmp_path), "--report", str(report)]
    )
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert report.read_text() == "existing results"


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_rejected(timeout: float, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Timeout"):
        emulate_project(str(tmp_path), timeout=timeout)


@pytest.mark.parametrize("defect", ["path_escape", "overlap", "overflow", "target"])
def test_unsafe_flash_metadata_rejected(tmp_path: Path, defect: str) -> None:
    # Only metadata validation is tested here; these bytes are never executed.
    build = tmp_path / "build"
    build.mkdir()
    (build / "first.bin").write_bytes(b"input-one")
    (build / "second.bin").write_bytes(b"input-two")
    args = {
        "extra_esptool_args": {"chip": "esp32s3"},
        "flash_settings": {"flash_size": "2MB"},
        "flash_files": {"0x0": "first.bin", "0x10000": "second.bin"},
    }
    if defect == "path_escape":
        (tmp_path / "outside.bin").write_bytes(b"outside")
        args["flash_files"]["0x10000"] = "../outside.bin"
    elif defect == "overlap":
        args["flash_files"] = {"0x0": "first.bin", "0x1": "second.bin"}
    elif defect == "overflow":
        args["flash_files"] = {"0x0": "first.bin", "0x200000": "second.bin"}
    else:
        args["extra_esptool_args"]["chip"] = "esp32"
    (build / "flasher_args.json").write_text(json.dumps(args))
    with pytest.raises(ValueError):
        _flash_inputs(tmp_path, {"target": "esp32s3"})


@pytest.fixture
def built_project() -> Path:
    configured = os.environ.get("TINYML_QEMU_PROJECT")
    if not configured:
        pytest.skip("Set TINYML_QEMU_PROJECT to a real compiled ESP32-S3 project.")
    path = Path(configured).resolve()
    assert (path / "build" / "flasher_args.json").is_file()
    return path


@pytest.fixture
def qemu() -> str:
    executable = shutil.which(
        os.environ.get("TINYML_QEMU_BINARY", "qemu-system-xtensa")
    )
    assert executable, "QEMU is required when a compiled project is supplied"
    return executable


def test_actual_firmware_predictions(
    built_project: Path, qemu: str, tmp_path: Path
) -> None:
    report = tmp_path / "emulation.json"
    original_manifest = (built_project / "deployment.json").read_bytes()
    result = CliRunner().invoke(
        cli, ["emulate", str(built_project), "--qemu", qemu, "--report", str(report)]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(report.read_text())
    assert (
        data["measurement_kind"] == data["validation"]["measurement_kind"] == "emulator"
    )
    assert data["hardware_validation"] == "not_run"
    assert data["hardware_timing_valid"] is False
    assert data["esp_nn_optimized"] is True
    manifest = json.loads(original_manifest)
    assert data["manifest_sha256"] == hashlib.sha256(original_manifest).hexdigest()
    assert (
        data["validation"]["observations"]
        == manifest["samples"] * manifest["repetitions"]
    )
    assert data["validation"]["prediction_agreement_passed"]
    assert (built_project / "deployment.json").read_bytes() == original_manifest
    for entry in data["firmware_files"]:
        content = (built_project / "build" / entry["path"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]


def test_real_firmware_rejects_wrong_manifest(
    built_project: Path, qemu: str, tmp_path: Path
) -> None:
    # Copy actual compiled artifacts and change only the expected model identity.
    copied = tmp_path / "project"
    (copied / "build").mkdir(parents=True)
    for name in ("deployment.json", "sdkconfig", "build/flasher_args.json"):
        shutil.copy2(built_project / name, copied / name)
    args = json.loads((copied / "build/flasher_args.json").read_text())
    for name in args["flash_files"].values():
        target = copied / "build" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built_project / "build" / name, target)
    path = copied / "deployment.json"
    manifest = json.loads(path.read_text())
    manifest["model_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="model_sha256 does not match"):
        emulate_project(str(copied), qemu_binary=qemu)


def test_real_qemu_timeout_fails(built_project: Path, qemu: str) -> None:
    with pytest.raises(RuntimeError, match="timeout"):
        emulate_project(str(built_project), qemu_binary=qemu, timeout=0.001)
