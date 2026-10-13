# Contributor guide

Use Python 3.11 or 3.12. Package metadata and dependency constraints live in `pyproject.toml`.

## Architecture

The CLI lazily imports command modules. `analyzer.py` reads serialized TFLite operators and tensor metadata; its memory and latency values are estimates. `quantizer.py` converts original Keras, SavedModel or ONNX graphs and gates output on actual prediction comparisons in `evaluation.py`. Do not reconstruct source networks from TFLite layer guesses.

`operators.py` maps serialized built-ins to pinned Micro resolver APIs. `codegen.py` emits C++ with a C-callable checked API. `deployer.py` generates ESP-IDF projects using `templates/profile_main.cpp`. `profiler.py` validates serial measurement records against a deployment manifest. `doctor.py` and `reporting.py` provide environment checks and JSON/HTML results.

## Conventions

- Use future annotations and typed function signatures; use dataclasses for structured results.
- Keep CLI imports lazy and handle errors with Click exceptions.
- No em dashes in code, comments, documentation or terminal output.
- Never substitute random calibration samples or fabricated device measurements for supplied data.
- Separate static preflight, successful compilation and actual device execution in claims and reports.
- Preserve existing output files and project directories.
- A registered operator does not prove every version and dtype is supported by Micro.
- Changes to generated source must compile with the pinned ESP-IDF and Micro component.

## Checks

```bash
python -m pip install -e '.[dev,onnx]'
ruff check tinyml_deployer tests scripts examples
pytest
python -m build
python -m twine check dist/*
python scripts/prepare_firmware_checks.py --output firmware_checks
```

Compile both generated projects with ESP-IDF 5.5.1 for ESP32-S3. Actual board validation is separate and requires device reports. See `CONTRIBUTING.md` and `.github/workflows/ci.yml`.

Adding a target profile does not establish working firmware support. Supply board startup, linker configuration, runtime integration, build checks and device evidence before claiming a new target is supported.
