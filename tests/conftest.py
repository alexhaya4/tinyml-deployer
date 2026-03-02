"""Shared fixtures for tinyml-deployer tests."""

from __future__ import annotations

from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SINE_MODEL_PATH = str(REPO_ROOT / "examples" / "sine_model" / "sine_model.tflite")

ALL_TARGET_NAMES: list[str] = ["esp32", "esp32s3", "stm32f4", "stm32h7"]
ESP_TARGET_NAMES: list[str] = ["esp32", "esp32s3"]
STM_TARGET_NAMES: list[str] = ["stm32f4", "stm32h7"]


@pytest.fixture
def sine_model_path() -> str:
    """Path to the example sine_model.tflite file."""
    assert Path(SINE_MODEL_PATH).exists(), f"Missing test model: {SINE_MODEL_PATH}"
    return SINE_MODEL_PATH


@pytest.fixture
def tmp_output(tmp_path: Path) -> str:
    """Temporary output directory as a string."""
    return str(tmp_path)
