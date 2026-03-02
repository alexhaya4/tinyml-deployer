# CLAUDE.md -- Claude Code contributor guide

## Project overview

tinyml-deployer is a Python CLI tool that takes TensorFlow Lite models and
prepares them for deployment on ESP32 and STM32 microcontrollers. The workflow
is: analyze a model, optionally quantize it, benchmark it across targets, then
generate a complete C project ready to build and flash.

## Architecture

```
cli.py          Click command group, thin layer that calls into modules
    |
    +-- analyzer.py     Load TFLite via tf.lite.Interpreter, extract ops/memory
    +-- quantizer.py    Round-trip through Keras to re-convert with quantization
    +-- benchmark.py    Compute latency/throughput from analyzer MACs + target specs
    +-- deployer.py     Orchestrate codegen + platform scaffolding
         |
         +-- codegen.py     Generate model_data.h/.c and inference.h/.c
    +-- targets.py      MCUTarget dataclass and the TARGETS registry
```

Each CLI command does a lazy import of its module (`from tinyml_deployer.X import ...`)
inside the command function so startup stays fast.

## Key conventions

- All modules use `from __future__ import annotations` for modern type hints.
- Data classes are used for structured return values (ModelAnalysis,
  QuantizationResult, BenchmarkResult, DeployResult, GeneratedFile).
- The CLI uses Click for argument parsing and Rich for terminal output.
- Private TF APIs (e.g. `interpreter._get_ops_details()`) are wrapped in
  try/except blocks with graceful fallbacks.
- No em-dashes in any text (docstrings, comments, CLI output). Use `--` instead.

## How to add a new MCU target

1. Open `tinyml_deployer/targets.py`.
2. Create a new `MCUTarget` instance with the hardware specs:
   - `name`: lowercase identifier (used in CLI `--target` flag)
   - `flash_kb`, `ram_kb`, `clock_mhz`: hardware limits
   - `fpu`: whether the chip has a floating-point unit
   - `framework`: `"ESP-IDF"` or `"STM32CubeAI"` (controls project scaffolding)
   - `cycles_per_mac`: cycles per multiply-accumulate (lower = faster)
   - `supported_ops`: list of TFLite op names the runtime supports
3. Add the target to the `TARGETS` dict at the bottom of the file.
4. If the target uses a new framework (not ESP-IDF or STM32CubeAI), add a
   corresponding scaffolder in `deployer.py`.

## How to add a new CLI command

1. Add the module in `tinyml_deployer/` with a public function and a result
   dataclass.
2. In `cli.py`, add a new `@cli.command()` function. Use lazy imports inside
   the function body.
3. Follow the existing pattern: print a status line, call the module in a
   try/except block, display results in a Rich table.

## Testing

There are no automated tests yet. Manual testing workflow:

```bash
pip install -e .

# Analyze
tinyml-deployer analyze examples/sine_model/sine_model.tflite --target esp32

# Quantize
tinyml-deployer quantize examples/sine_model/sine_model.tflite --type int8

# Benchmark single target
tinyml-deployer benchmark examples/sine_model/sine_model.tflite --target esp32

# Benchmark all targets
tinyml-deployer benchmark examples/sine_model/sine_model.tflite --compare

# Deploy ESP32
tinyml-deployer deploy examples/sine_model/sine_model.tflite --target esp32 --output test_project

# Deploy STM32
tinyml-deployer deploy examples/sine_model/sine_model.tflite --target stm32f4 --output test_project
```

Generated project directories (`*_project/`) and quantized models
(`*_quant.tflite`) are in `.gitignore`.

## File layout

| File | Purpose |
|------|---------|
| `setup.py` | Package metadata, dependencies, entry point |
| `requirements.txt` | Pinned dependency versions |
| `tinyml_deployer/__init__.py` | Package version (`__version__`) |
| `tinyml_deployer/cli.py` | All Click commands |
| `tinyml_deployer/analyzer.py` | TFLite model analysis |
| `tinyml_deployer/quantizer.py` | Post-training quantization |
| `tinyml_deployer/codegen.py` | C source file generation |
| `tinyml_deployer/deployer.py` | Project scaffolding |
| `tinyml_deployer/benchmark.py` | Performance estimation |
| `tinyml_deployer/targets.py` | MCU target registry |
| `examples/sine_model/train.py` | Example training script |
| `examples/sine_model/sine_model.tflite` | Pre-trained test model |

## Python version

Requires Python 3.10+ (uses `X | Y` union syntax in type hints).
