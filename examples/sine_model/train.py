"""Train a tiny sine wave prediction model and export to TFLite."""

from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> None:
    output_dir = Path(__file__).parent
    output_path = output_dir / "sine_model.tflite"

    # Generate training data: 1000 random samples from 0 to 2*pi
    np.random.seed(42)
    x_train = np.random.uniform(0, 2 * np.pi, 1000).astype(np.float32)
    y_train = np.sin(x_train).astype(np.float32)

    # Build a small 3-layer dense network
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation="relu", input_shape=(1,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    model.compile(optimizer="adam", loss="mse")
    model.summary()

    # Train
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    output_path.write_bytes(tflite_model)
    print(f"\nSaved TFLite model to {output_path} ({len(tflite_model)} bytes)")


if __name__ == "__main__":
    main()
