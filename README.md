[![PyPI version](https://img.shields.io/pypi/v/tinyml-deployer)](https://pypi.org/project/tinyml-deployer/)
[![Python version](https://img.shields.io/pypi/pyversions/tinyml-deployer)](https://pypi.org/project/tinyml-deployer/)
[![License](https://img.shields.io/pypi/l/tinyml-deployer)](LICENSE)

# tinyml-deployer

A Python CLI tool for deploying TensorFlow Lite models to ESP32 and STM32 microcontrollers.

## Features

- **Model analysis** -- check model compatibility with target MCU constraints (flash, RAM, supported ops)
- **Post-training quantization** -- quantize models to int8, float16, or dynamic range for smaller size
- **C code generation** -- convert models to C byte arrays and generate TFLite Micro inference wrappers
- **Project scaffolding** -- generate complete ESP-IDF or STM32 build projects ready to compile and flash
- **Performance benchmarking** -- estimate latency, throughput, and utilization across all targets

## Supported Targets

| Target   | Arch    | Flash (KB) | RAM (KB) | Clock (MHz) | FPU | Framework   |
|----------|---------|-----------|----------|-------------|-----|-------------|
| ESP32    | Xtensa  | 4096      | 520      | 240         | No  | ESP-IDF     |
| ESP32-S3 | Xtensa  | 8192      | 512      | 240         | No  | ESP-IDF     |
| ESP32-C3 | RISC-V  | 4096      | 400      | 160         | No  | ESP-IDF     |
| ESP32-C6 | RISC-V  | 4096      | 512      | 160         | No  | ESP-IDF     |
| STM32F4  | Cortex-M4 | 1024    | 192      | 168         | Yes | STM32CubeAI |
| STM32H7  | Cortex-M7 | 2048    | 1024     | 480         | Yes | STM32CubeAI |

## Installation

Install from PyPI:

```bash
pip install tinyml-deployer
```

Or install from source for development:

```bash
git clone https://github.com/alexhaya4/tinyml-deployer.git
cd tinyml-deployer
pip install -e ".[dev]"
```

## Quick Start

Full workflow from training a model to generating a deployment project:

```bash
# 1. Train a sine wave model (included example)
python examples/sine_model/train.py

# 2. Analyze the model for your target
tinyml-deployer analyze examples/sine_model/sine_model.tflite --target esp32

# 3. Quantize the model (optional, for larger models)
tinyml-deployer quantize examples/sine_model/sine_model.tflite --type int8

# 4. Compare performance across all targets
tinyml-deployer benchmark examples/sine_model/sine_model.tflite --compare

# 5. Generate a complete ESP-IDF project
tinyml-deployer deploy examples/sine_model/sine_model.tflite --target esp32 --output my_project

# 6. Build and flash (requires ESP-IDF toolchain)
cd my_project
idf.py set-target esp32
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

## CLI Usage

### Analyze

Check a model's compatibility and memory requirements for a specific target:

```
$ tinyml-deployer analyze examples/sine_model/sine_model.tflite --target esp32

Analyzing examples/sine_model/sine_model.tflite for esp32...

                        Model Info
+---------------------------------------------------------+
| Path      | examples/sine_model/sine_model.tflite       |
| Size      | 3.1 KB                                      |
| Input     | serving_default_keras_tensor:0    (float32)  |
| Output    | StatefulPartitionedCall_1:0    (float32)     |
| Operators | DELEGATE, FULLY_CONNECTED                    |
| Total ops | 4                                            |
+---------------------------------------------------------+
                Memory Estimates
+----------------------------------------------+
| Flash usage       | 3.1 KB  (limit: 4096 KB) |
| Fits in flash     | Yes                       |
| Tensor arena      | 2.1 KB  (limit: 520 KB)   |
| Fits in RAM       | Yes                       |
| Estimated MACs    | 288                       |
| Estimated latency | 0.01 ms                   |
+----------------------------------------------+
```

### Quantize

Apply post-training quantization to reduce model size:

```
$ tinyml-deployer quantize examples/sine_model/sine_model.tflite --type int8

Quantizing examples/sine_model/sine_model.tflite with int8 quantization...

                        Quantization Result
+-----------------------------------------------------------------+
| Input model       | examples/sine_model/sine_model.tflite       |
| Output model      | examples/sine_model/sine_model_quant.tflite |
| Quantization type | int8                                        |
| Original size     | 3.1 KB                                      |
| Quantized size    | 3.3 KB                                      |
| Compression ratio | 0.95x                                       |
+-----------------------------------------------------------------+
```

Supported quantization types: `int8`, `float16`, `dynamic`.

Note: very small models (like the 337-parameter sine model) may not shrink after
quantization because the metadata overhead exceeds the savings from reduced
precision. Quantization provides significant compression on larger models.

### Deploy

Generate a complete build project with model data, inference wrapper, and
platform-specific scaffolding:

```
$ tinyml-deployer deploy examples/sine_model/sine_model.tflite --target esp32 --output sine_esp32_project

Deploying examples/sine_model/sine_model.tflite for esp32 to sine_esp32_project...

                             Generated Files
+-----------------------------------------------------------------------+
| File                                   | Description                  |
|----------------------------------------+------------------------------|
| sine_esp32_project/main/model_data.h   | Model byte array header      |
| sine_esp32_project/main/model_data.c   | Model byte array source      |
| sine_esp32_project/main/inference.h    | Inference wrapper header     |
| sine_esp32_project/main/inference.c    | Inference wrapper source     |
| sine_esp32_project/CMakeLists.txt      | Root CMake project file      |
| sine_esp32_project/main/CMakeLists.txt | Main component CMake file    |
| sine_esp32_project/main/main.c         | Application entry point      |
| sine_esp32_project/sdkconfig.defaults  | Default SDK configuration    |
| sine_esp32_project/README.md           | Build and flash instructions |
+-----------------------------------------------------------------------+

Project generated at: sine_esp32_project
```

Supported targets: `esp32`, `esp32s3`, `esp32c3`, `esp32c6` (ESP-IDF projects), `stm32f4`, `stm32h7` (Makefile + HAL stubs).

### Benchmark

Estimate inference performance on a single target or compare across all targets:

```
$ tinyml-deployer benchmark examples/sine_model/sine_model.tflite --compare

Benchmarking examples/sine_model/sine_model.tflite across all targets...

                                     Target Comparison
+---------------------------------------------------------------------------------------------+
| Target  |   Clock | FPU | Latency (ms) | Throughput (inf/s) | Util % | Flash OK | RAM OK   |
|---------+---------+-----+--------------+--------------------+--------+----------+----------|
| stm32h7 | 480 MHz | Yes |       0.0012 |            833,333 | 0.0001 |   Yes    |  Yes     |
| esp32s3 | 240 MHz | No  |       0.0048 |            208,333 | 0.0005 |   Yes    |  Yes     |
| stm32f4 | 168 MHz | Yes |       0.0069 |            145,833 | 0.0007 |   Yes    |  Yes     |
| esp32c6 | 160 MHz | No  |       0.0180 |             55,556 | 0.0018 |   Yes    |  Yes     |
| esp32   | 240 MHz | No  |       0.0120 |             83,333 | 0.0012 |   Yes    |  Yes     |
| esp32c3 | 160 MHz | No  |       0.0216 |             46,296 | 0.0022 |   Yes    |  Yes     |
+---------------------------------------------------------------------------------------------+

Total MACs per inference: 288
```

## Project Structure

```
tinyml_deployer/
    __init__.py       # Package version
    cli.py            # Click CLI entry point with all subcommands
    analyzer.py       # TFLite model analysis (ops, memory, compatibility)
    quantizer.py      # Post-training quantization (int8, float16, dynamic)
    codegen.py        # C source generation (model data + inference wrapper)
    deployer.py       # Full project scaffolding (ESP-IDF, STM32)
    benchmark.py      # Performance estimation across targets
    targets.py        # MCU target definitions and specs
examples/
    sine_model/
        train.py          # Training script for the example model
        sine_model.tflite # Pre-trained 3.1 KB sine wave model
```

## Development

```bash
git clone https://github.com/alexhaya4/tinyml-deployer.git
cd tinyml-deployer
pip install -e ".[dev]"
pytest
```

### Building for PyPI

```bash
python -m build
twine check dist/*
twine upload dist/*   # when ready to publish
```

Dependencies: `tensorflow`, `numpy`, `click`, `rich`. See `pyproject.toml` for the full
list and version constraints.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on
forking the repo, setting up a dev environment, running tests, and submitting pull requests.

## License

MIT License. See [LICENSE](LICENSE) for details.
