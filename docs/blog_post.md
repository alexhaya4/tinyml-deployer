# From Training to MCU: Building a TinyML Deployment Pipeline

**Author: Alex Odhiambo Haya**

You have trained a neural network. It works great on your laptop. Now you need
it running on an ESP32 with 520 KB of RAM and no operating system. What follows
is usually a week of manual work: converting the model, guessing at memory
budgets, writing C boilerplate, and configuring a build system you have never
seen before. I built **tinyml-deployer** to compress that week into a single
command.

This post walks through the tool, the problems it solves, and how it works
under the hood.

## The Problem

Deploying a neural network to a microcontroller today involves a surprisingly
long list of manual steps:

1. **Model conversion.** Export your Keras or PyTorch model to TensorFlow Lite.
   Optionally quantize it to int8 to fit in flash.
2. **Memory estimation.** Open the TFLite file, figure out the tensor shapes,
   estimate the arena size, and check whether it fits in the target's RAM and
   flash. Get it wrong, and the firmware crashes at runtime with a cryptic
   allocator error.
3. **C boilerplate.** Convert the `.tflite` binary into a C byte array. Write
   an inference wrapper that initializes the TFLite Micro interpreter, copies
   input data, invokes the model, and reads the output. This is roughly 150
   lines of repetitive code that varies only in tensor types and arena sizes.
4. **Build system setup.** Create a CMakeLists.txt for ESP-IDF or a Makefile
   for STM32. Configure the SDK, set the correct compiler flags for your
   chip's architecture (Xtensa, RISC-V, or Cortex-M), and wire up the source
   files.
5. **Target selection.** If you are choosing between an ESP32 and an STM32H7,
   you need to estimate how fast each one can run your model. This means
   digging through datasheets for clock speeds and MAC throughput numbers, then
   doing the math by hand.

Each of these steps is well-documented individually. But stitching them
together -- especially when targeting multiple chips -- is tedious, error-prone,
and discourages rapid iteration. You end up spending more time on plumbing than
on the actual ML problem.

## How TinyML Deployer Works

tinyml-deployer is a Python CLI that automates the entire pipeline. It exposes
four commands that correspond to the four stages of deployment:

```
.tflite model
     |
     v
  analyze -----> compatibility check, memory estimates, MAC count
     |
     v
  quantize ----> int8 / float16 / dynamic (optional, shrinks the model)
     |
     v
  benchmark ---> latency, throughput, and utilization for one or all targets
     |
     v
  deploy ------> complete C project: model data, inference wrapper,
                  build system, README with flash instructions
```

Each command is self-contained. You can run `analyze` by itself to check
whether a model fits on a particular chip, or you can run the full pipeline
from analyze through deploy in under 30 seconds.

The tool is organized as a set of Python modules behind a thin Click CLI layer.
The `analyzer` loads the TFLite file and extracts ops, shapes, and memory
estimates. The `quantizer` rebuilds a Keras model from the TFLite weights and
re-converts it with quantization options. The `benchmark` module combines MAC
counts from the analyzer with hardware specs from the target registry. The
`deployer` orchestrates code generation and platform-specific scaffolding.

## Walkthrough

Let's walk through a complete deployment using the included sine wave example --
a small model that learns the `sin(x)` function.

### Step 1: Train the model

```bash
python examples/sine_model/train.py
```

This produces `sine_model.tflite`, a 3.1 KB model with three fully connected
layers and 337 parameters. Tiny, but perfect for demonstrating the workflow.

### Step 2: Analyze for your target

```bash
tinyml-deployer analyze examples/sine_model/sine_model.tflite --target esp32
```

```
                    Model Info
+-----------------------------------------------------------+
| Path      | examples/sine_model/sine_model.tflite         |
| Size      | 3.1 KB                                        |
| Operators | DELEGATE, FULLY_CONNECTED                      |
+-----------------------------------------------------------+
                Memory Estimates
+------------------------------------------------+
| Flash usage       | 3.1 KB  (limit: 4096 KB)   |
| Fits in flash     | Yes                         |
| Tensor arena      | 2.1 KB  (limit: 520 KB)     |
| Fits in RAM       | Yes                         |
| Estimated MACs    | 288                         |
| Estimated latency | 0.01 ms                     |
+------------------------------------------------+
```

The model uses 3.1 KB of flash and an estimated 2.1 KB of RAM -- well within the
ESP32's limits. The analyzer also checks that every operator in the model
(`FULLY_CONNECTED`) is supported by the target runtime.

### Step 3: Quantize (optional)

```bash
tinyml-deployer quantize examples/sine_model/sine_model.tflite --type int8
```

For this tiny model, int8 quantization does not save space (metadata overhead
dominates). But on larger models -- a MobileNet or a keyword spotting network --
int8 quantization typically cuts the model size by 50-75% and can speed up
inference on chips without a floating-point unit.

### Step 4: Deploy

```bash
tinyml-deployer deploy examples/sine_model/sine_model.tflite \
    --target esp32 --output sine_esp32_project
```

This generates a complete ESP-IDF project:

```
sine_esp32_project/
    CMakeLists.txt
    sdkconfig.defaults
    README.md
    main/
        CMakeLists.txt
        main.c
        model_data.h
        model_data.c
        inference.h
        inference.c
```

The project is ready to build. Connect an ESP32, run `idf.py build && idf.py flash`,
and you have a working TFLite Micro inference running on-device.

### Step 5: Benchmark across targets

```bash
tinyml-deployer benchmark examples/sine_model/sine_model.tflite --compare
```

```
                                Target Comparison
+---------------------------------------------------------------------------------+
| Target  |   Clock | FPU | Latency (ms) | Throughput (inf/s) | Flash OK | RAM OK |
|---------+---------+-----+--------------+--------------------+----------+--------|
| stm32h7 | 480 MHz | Yes |       0.0012 |            833,333 |   Yes    |  Yes   |
| esp32s3 | 240 MHz | No  |       0.0048 |            208,333 |   Yes    |  Yes   |
| stm32f4 | 168 MHz | Yes |       0.0069 |            145,833 |   Yes    |  Yes   |
| esp32c6 | 160 MHz | No  |       0.0180 |             55,556 |   Yes    |  Yes   |
| esp32   | 240 MHz | No  |       0.0120 |             83,333 |   Yes    |  Yes   |
| esp32c3 | 160 MHz | No  |       0.0216 |             46,296 |   Yes    |  Yes   |
+---------------------------------------------------------------------------------+
```

One command, six targets, sorted by latency. This makes it easy to choose the
right chip for your latency and power budget without manually computing
anything.

## Under the Hood

A few of the more interesting technical details.

### TFLite model parsing

The analyzer loads the `.tflite` file using `tf.lite.Interpreter` and calls
`allocate_tensors()` to set up the internal state. Input and output tensor
metadata (shapes, dtypes, sizes) come from `get_input_details()` and
`get_output_details()`. Operator information comes from the private
`_get_ops_details()` API, which returns the op name, input tensor indices, and
output tensor indices for every node in the graph. This API is wrapped in a
try/except block so the tool degrades gracefully on older TensorFlow versions.

### MAC estimation

The analyzer estimates multiply-accumulate (MAC) operations per inference by
walking the operator list. For `FULLY_CONNECTED`, MACs equal
`weight_rows * weight_cols`. For `CONV_2D`, it is
`output_height * output_width * output_channels * kernel_height * kernel_width * input_channels`.
These are theoretical counts -- actual cycle counts depend on memory access patterns
and instruction scheduling -- but they give a useful first-order comparison
across targets.

### Memory estimation

Flash usage is simply the `.tflite` file size, since the model weights are
embedded as a C byte array. The tensor arena estimate sums the byte size of
every tensor in the graph and applies a 1.5x overhead factor to account for
alignment padding and interpreter bookkeeping. This is conservative by design:
underestimating the arena leads to runtime crashes that are painful to debug.

### Code generation

The code generator converts the raw `.tflite` bytes into a C source file where
the model is a `const unsigned char[]` array. It then generates an inference
wrapper (`inference.h` and `inference.c`) that provides three functions:
`model_init()`, `model_run()`, and `model_free()`. The wrapper handles
interpreter setup, tensor arena allocation, input/output copying, and cleanup.
Platform-specific details (like `esp_log.h` includes for ESP-IDF) are injected
based on the target's framework.

### Benchmark computation

The benchmark module computes estimated latency from a simple model:
`latency = total_MACs * cycles_per_MAC / clock_frequency`. Each target in the
registry defines its `clock_mhz` and `cycles_per_mac` values, which capture
how many clock cycles the chip needs per multiply-accumulate operation. An
ESP32-S3 with its vector extensions does a MAC in 4 cycles; a basic ESP32
needs 10. Throughput is the reciprocal of latency, and utilization measures
what fraction of the chip's peak compute a model would consume if run
continuously.

## Supported Targets

The tool currently supports six microcontrollers across three architectures:

| Target   | Arch      | Clock   | RAM     | FPU | Best for                                |
|----------|-----------|---------|---------|-----|-----------------------------------------|
| ESP32    | Xtensa    | 240 MHz | 520 KB  | No  | Wi-Fi projects, general-purpose edge AI |
| ESP32-S3 | Xtensa    | 240 MHz | 512 KB  | No  | Vector-accelerated inference, cameras   |
| ESP32-C3 | RISC-V    | 160 MHz | 400 KB  | No  | Low-cost BLE/Wi-Fi sensor nodes         |
| ESP32-C6 | RISC-V    | 160 MHz | 512 KB  | No  | Wi-Fi 6 and Thread/Zigbee devices       |
| STM32F4  | Cortex-M4 | 168 MHz | 192 KB  | Yes | Battery-powered, FPU-accelerated        |
| STM32H7  | Cortex-M7 | 480 MHz | 1024 KB | Yes | Maximum performance, complex models     |

Adding a new target requires only defining an `MCUTarget` dataclass in
`targets.py` with the hardware specs. If the target uses ESP-IDF or
STM32CubeAI, the existing scaffolders handle project generation automatically.

## What's Next

tinyml-deployer is functional today, but there is a lot of room to grow:

- **ONNX support.** Many models are trained in PyTorch and exported as ONNX.
  Adding an ONNX-to-TFLite conversion step (or native ONNX Micro Runtime
  support) would open the tool to a much wider audience.
- **Real hardware benchmarking.** The current benchmarks are analytical
  estimates. A future version could flash a benchmark firmware, measure actual
  inference time over serial, and report real-world numbers.
- **CI/CD integration.** A GitHub Action that runs the analyze and benchmark
  steps on every commit, catching model regressions (size grew, latency
  increased, op became unsupported) before they reach hardware.
- **More targets.** Nordic nRF (Cortex-M33), Raspberry Pi Pico (Cortex-M0+),
  and Kendryte K210 (RISC-V with KPU) are all popular TinyML platforms that
  would be straightforward to add.
- **Operator-level profiling.** Breaking down latency by operator (rather than
  just total MACs) would help identify bottlenecks and guide model
  optimization.

## Getting Started

Install with pip:

```bash
pip install tinyml-deployer
```

Or clone the repository for development:

```bash
git clone https://github.com/alexhaya4/tinyml-deployer.git
cd tinyml-deployer
pip install -e ".[dev]"
```

Run the full test suite:

```bash
pytest
```

The repository includes a pre-trained sine wave model and a training script so
you can try every command immediately. Check the
[README](https://github.com/alexhaya4/tinyml-deployer) for detailed CLI
usage and output examples.

If you are an embedded engineer curious about ML, or an ML engineer who has
never touched a microcontroller, tinyml-deployer is meant to bridge that gap.
The goal is simple: you should be able to go from a trained model to a flashing
firmware in minutes, not days.

Contributions and feedback are welcome on
[GitHub](https://github.com/alexhaya4/tinyml-deployer/issues).
