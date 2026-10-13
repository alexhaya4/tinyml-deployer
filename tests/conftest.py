"""Shared fixtures for tinyml-deployer tests."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SINE_MODEL_PATH = str(REPO_ROOT / "examples" / "sine_model" / "sine_model.tflite")

ALL_TARGET_NAMES: list[str] = [
    "esp32",
    "esp32s3",
    "esp32c3",
    "esp32c6",
    "stm32f4",
    "stm32h7",
]
ESP_TARGET_NAMES: list[str] = ["esp32", "esp32s3", "esp32c3", "esp32c6"]
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


@pytest.fixture(scope="session")
def source_model(tmp_path_factory):
    """A real non-ReLU model that exposes activation reconstruction errors."""
    import numpy as np
    import tensorflow as tf

    root = tmp_path_factory.mktemp("source")
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,), batch_size=1),
            tf.keras.layers.Dense(3, activation="tanh"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.set_weights(
        [
            np.array([[0.7, -1.2, 0.4]], dtype=np.float32),
            np.array([0.1, 0.2, -0.1], dtype=np.float32),
            np.array([[1.1], [-0.5], [0.8]], dtype=np.float32),
            np.array([0.05], dtype=np.float32),
        ]
    )
    path = root / "activation.keras"
    model.save(path)
    return str(path), model


@pytest.fixture
def validation_samples():
    import numpy as np

    return np.linspace(-0.9, 0.9, 17, dtype=np.float32).reshape(-1, 1, 1)


@pytest.fixture
def validation_path(tmp_path, validation_samples):
    import numpy as np

    path = tmp_path / "validation.npy"
    np.save(path, validation_samples)
    return str(path)


@pytest.fixture
def calibration_samples():
    import numpy as np

    return np.linspace(-1, 1, 100, dtype=np.float32).reshape(-1, 1, 1)
