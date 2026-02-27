# tinyml-deployer

A Python CLI tool for deploying TensorFlow Lite models to ESP32 and STM32 microcontrollers.

## Features

- **Model analysis** - Check model compatibility with target MCU constraints (flash, RAM, supported ops)
- **Quantization** - Quantize models to int8 or float16 for efficient on-device inference
- **Deployment** - Flash optimized models directly to connected microcontrollers
- **Benchmarking** - Measure inference latency and memory usage on real hardware

## Supported Targets

| Target   | Flash (KB) | RAM (KB) | Clock (MHz) | FPU | Framework   |
|----------|-----------|----------|-------------|-----|-------------|
| ESP32    | 4096      | 520      | 240         | No  | ESP-IDF     |
| ESP32-S3 | 8192      | 512      | 240         | No  | ESP-IDF     |
| STM32F4  | 1024      | 192      | 168         | Yes | STM32CubeAI |
| STM32H7  | 2048      | 1024     | 480         | Yes | STM32CubeAI |

## Installation

```bash
pip install -e .
```

Or install from source:

```bash
git clone https://github.com/alexhaya4/tinyml-deployer.git
cd tinyml-deployer
pip install -e .
```

## CLI Usage

```bash
# Show version and help
tinyml-deployer --version
tinyml-deployer --help

# Analyze a model for a specific target
tinyml-deployer analyze model.tflite --target esp32

# Quantize a model to int8
tinyml-deployer quantize model.tflite --dtype int8 --output model_quant.tflite

# Deploy to a connected board
tinyml-deployer deploy model.tflite --target stm32f4 --port /dev/ttyUSB0

# Benchmark inference performance
tinyml-deployer benchmark model.tflite --target esp32s3 --runs 200
```

## Development

```bash
git clone https://github.com/alexhaya4/tinyml-deployer.git
cd tinyml-deployer
pip install -e .
```

## License

MIT License. See [LICENSE](LICENSE) for details.
