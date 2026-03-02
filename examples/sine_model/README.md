# Sine Wave Model Demo

A minimal example that trains a tiny neural network to approximate sin(x), exports it as a TFLite model, and analyzes it for MCU deployment.

## Model Architecture

- Input: single float (x value in range 0 to 2*pi)
- Dense layer: 16 neurons, ReLU activation
- Dense layer: 16 neurons, ReLU activation
- Output: single float (predicted sin(x))

Total parameters: ~337 (16 + 16 weights/biases per hidden layer, plus output layer).

## Training

- 1000 random samples from 0 to 2*pi
- Loss: mean squared error
- Optimizer: Adam
- Epochs: 50

## Usage

Train and export the model:

```bash
python examples/sine_model/train.py
```

Analyze the exported model for ESP32:

```bash
tinyml-deployer analyze examples/sine_model/sine_model.tflite --target esp32
```

Try other targets:

```bash
tinyml-deployer analyze examples/sine_model/sine_model.tflite --target stm32f4
tinyml-deployer analyze examples/sine_model/sine_model.tflite --target stm32h7
```
