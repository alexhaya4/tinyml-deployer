"""Generate real float32 and INT8 firmware build fixtures, without hardware claims."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from tinyml_deployer.deployer import deploy_model
from tinyml_deployer.quantizer import quantize_model


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(41)
    rng = np.random.default_rng(41)

    inputs = tf.keras.Input(shape=(4,), batch_size=1)
    hidden = tf.keras.layers.Dense(4, activation="tanh")(inputs)
    summed = tf.keras.layers.Add()([hidden, inputs])
    product = tf.keras.layers.Multiply()([summed, hidden])
    outputs = tf.keras.layers.Dense(2, activation="softmax")(product)
    dense = tf.keras.Model(inputs, outputs)
    dense_path = output / "dense.tflite"
    dense_path.write_bytes(tf.lite.TFLiteConverter.from_keras_model(dense).convert())
    dense_data = output / "dense_validation.npy"
    np.save(dense_data, rng.uniform(-1, 1, (8, 1, 4)).astype(np.float32))
    deploy_model(
        str(dense_path),
        "esp32s3",
        str(output / "float32_project"),
        arena_size=65536,
        validation_data=str(dense_data),
        max_abs_error=0.0001,
        repetitions=3,
    )

    cnn = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(8, 8, 1), batch_size=1),
            tf.keras.layers.Conv2D(4, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    source = output / "cnn.keras"
    cnn.save(source)
    # Deterministic integration-test inputs, not a task dataset or device results.
    calibration = rng.uniform(-1, 1, (32, 1, 8, 8, 1)).astype(np.float32)
    validation = rng.uniform(-0.9, 0.9, (8, 1, 8, 8, 1)).astype(np.float32)
    cnn_data = output / "cnn_validation.npy"
    np.save(cnn_data, validation)
    quantized = quantize_model(
        str(source),
        str(output / "cnn_int8.tflite"),
        "int8",
        lambda: ([sample] for sample in calibration),
        validation_samples=validation,
        max_abs_error=0.03,
    )
    deploy_model(
        quantized.output_path,
        "esp32s3",
        str(output / "int8_project"),
        arena_size=65536,
        validation_data=str(cnn_data),
        max_abs_error=0.02,
        repetitions=3,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    prepare(parser.parse_args().output)


if __name__ == "__main__":
    main()
