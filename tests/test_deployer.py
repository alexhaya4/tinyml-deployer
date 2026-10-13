"""Validate generated artifacts, safety gates and deployment provenance."""

import json

import pytest

from tinyml_deployer.deployer import deploy_model


def test_deployment_artifacts(sine_model_path, validation_path, tmp_path):
    out = tmp_path / "firmware"
    result = deploy_model(
        sine_model_path,
        "esp32s3",
        str(out),
        arena_size=65536,
        validation_data=validation_path,
        max_abs_error=1e-5,
    )
    assert result.analysis.compatible
    manifest = json.loads((out / "deployment.json").read_text())
    assert manifest["validation"]["hardware"] == "not_run"
    assert manifest["samples"] == 17
    assert len(manifest["model_sha256"]) == 64
    assert (out / "main" / "inference.cpp").exists()
    assert not (out / "main" / "inference.c").exists()
    assert (
        "espressif/esp-tflite-micro" in (out / "main" / "idf_component.yml").read_text()
    )
    assert "run_id" in (out / "main" / "main.cpp").read_text()
    assert not any(
        "@" in line for line in (out / "main" / "main.cpp").read_text().splitlines()
    )


def test_existing_directory_untouched(sine_model_path, validation_path, tmp_path):
    (tmp_path / "keep.txt").write_text("user data")
    with pytest.raises(FileExistsError):
        deploy_model(
            sine_model_path,
            "esp32s3",
            str(tmp_path),
            arena_size=65536,
            validation_data=validation_path,
            max_abs_error=1e-5,
        )
    assert list(tmp_path.glob("keep.txt"))
    assert not (tmp_path / "main").exists()


def test_stm32_never_emits_unfinished_firmware(
    sine_model_path, validation_path, tmp_path
):
    out = tmp_path / "stm"
    with pytest.raises(ValueError, match="STM32"):
        deploy_model(
            sine_model_path,
            "stm32f4",
            str(out),
            arena_size=65536,
            validation_data=validation_path,
            max_abs_error=1e-5,
        )
    assert not out.exists()


def test_bad_input_shape_has_no_partial_project(sine_model_path, tmp_path):
    import numpy as np

    path = tmp_path / "wrong.npy"
    np.save(path, np.ones((2, 1, 3), dtype=np.float32))
    out = tmp_path / "firmware"
    with pytest.raises(ValueError, match="shape"):
        deploy_model(
            sine_model_path,
            "esp32s3",
            str(out),
            arena_size=65536,
            validation_data=str(path),
            max_abs_error=1e-5,
        )
    assert not out.exists()
