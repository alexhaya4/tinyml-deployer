# Contributing to tinyml-deployer

Thanks for your interest in contributing! Whether you are fixing a bug, adding
a new MCU target, or improving documentation, your help is welcome.

## Getting started

1. **Fork** the repository on GitHub and clone your fork:

   ```bash
   git clone https://github.com/<your-username>/tinyml-deployer.git
   cd tinyml-deployer
   ```

2. **Install** in development mode with all dev dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

3. **Run the tests** to make sure everything works:

   ```bash
   pytest
   ```

   You can also run tests with coverage:

   ```bash
   pytest --cov=tinyml_deployer --cov-report=term-missing
   ```

## Code style

- **Type hints everywhere.** All functions should have fully annotated
  signatures. Use `from __future__ import annotations` at the top of every
  module.
- **Dataclasses for structured data.** Return values from public APIs should
  be dataclasses (e.g. `ModelAnalysis`, `BenchmarkResult`, `DeployResult`).
- **No em-dashes.** Use `--` (double hyphen) instead of the unicode em-dash
  character in all text: code, docstrings, comments, CLI output, and docs.
- **Lazy imports in CLI commands.** Import modules inside Click command
  functions so that `tinyml-deployer --help` stays fast.
- **Keep it simple.** Avoid over-engineering. Only add abstractions when they
  are clearly needed.

## Adding a new MCU target

1. Open `tinyml_deployer/targets.py`.
2. Create a new `MCUTarget` instance with the hardware specs (`name`,
   `flash_kb`, `ram_kb`, `clock_mhz`, `fpu`, `framework`, `cycles_per_mac`,
   `supported_ops`).
3. Add the target to the `TARGETS` dict at the bottom of the file.
4. Add the target name to `ALL_TARGET_NAMES` (and the appropriate family list)
   in `tests/conftest.py`.
5. Add a test class in `tests/test_targets.py` that verifies the target specs.
6. Run `pytest` to confirm all tests pass.

If the target uses a framework other than `"ESP-IDF"` or `"STM32CubeAI"`, you
will also need to add a scaffolder in `tinyml_deployer/deployer.py`.

## Adding a new CLI command

1. Create a module in `tinyml_deployer/` with a public function and a result
   dataclass.
2. In `tinyml_deployer/cli.py`, add a new `@cli.command()` function. Use lazy
   imports inside the function body.
3. Follow the existing pattern: print a status line, call the module function
   in a try/except block, display results in a Rich table.
4. Add tests in `tests/test_cli.py` using Click's `CliRunner`.

## Submitting a pull request

1. Create a feature branch from `main`:

   ```bash
   git checkout -b my-feature
   ```

2. Make your changes. Write tests for new functionality.
3. Run the full test suite:

   ```bash
   pytest
   ```

4. Commit with a clear, concise message describing the change.
5. Push your branch and open a pull request against `main`.

In your PR description, please include:

- A short summary of what changed and why.
- The type of change (bug fix, new feature, documentation, etc.).
- Confirmation that tests pass and no em-dashes were introduced.

## Reporting issues

Open an issue on GitHub using one of the provided templates:

- **Bug report** -- describe the bug, steps to reproduce, expected behavior,
  target MCU, Python version, and OS.
- **Feature request** -- describe the feature, the use case, and a proposed
  solution if you have one.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Please
read it before participating.

Thank you for helping make tinyml-deployer better!
