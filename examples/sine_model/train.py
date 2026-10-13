"""Train a sine function approximation and export source plus held-out inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def train(output: Path, epochs: int) -> None:
    output.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(42)
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 2 * np.pi, (1000, 1)).astype(np.float32)
    y = np.sin(x)
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    # Explicit batches avoid a separate dataset thread pool on small hosts.
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        model.reset_metrics()
        for start in range(0, len(x), 32):
            batch = order[start : start + 32]
            loss = model.train_on_batch(x[batch], y[batch])
        print(f"Epoch {epoch + 1}/{epochs}: training loss {float(loss):.6f}")
    model.save(output / "sine.keras")
    content = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    (output / "sine.tflite").write_bytes(content)
    calibration = rng.uniform(0, 2 * np.pi, (256, 1, 1)).astype(np.float32)
    validation = np.linspace(0, 2 * np.pi, 65, dtype=np.float32).reshape(-1, 1, 1)
    np.save(output / "calibration.npy", calibration)
    np.save(output / "validation.npy", validation)
    predictions = model(validation[:, 0, :], training=False).numpy()
    error = float(np.mean(np.square(predictions - np.sin(validation[:, 0, :]))))
    print(f"Held-out function MSE: {error:.6f}; artifacts saved to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    train(args.output, args.epochs)


if __name__ == "__main__":
    main()
