"""Run a live-style demo of tinyml-deployer CLI commands.

Each command is printed to the terminal before execution, with short pauses
in between to simulate a human typing. Designed to be recorded with asciinema.
"""

from __future__ import annotations

import subprocess
import sys
import time

COMMANDS = [
    'echo "=== TinyML Deployer Demo ==="',
    "tinyml-deployer --version",
    "tinyml-deployer analyze examples/sine_model/sine_model.tflite --target esp32",
    "tinyml-deployer quantize examples/sine_model/sine_model.tflite --type int8",
    "tinyml-deployer benchmark examples/sine_model/sine_model.tflite --compare",
]

PAUSE_BEFORE_CMD = 1.0  # seconds before printing the next command
PAUSE_AFTER_CMD = 1.5   # seconds after command output before moving on


def run_demo() -> None:
    for cmd in COMMANDS:
        time.sleep(PAUSE_BEFORE_CMD)

        # Print the command like a shell prompt
        print(f"\n$ {cmd}", flush=True)
        time.sleep(0.3)

        result = subprocess.run(cmd, shell=True)

        if result.returncode != 0:
            print(f"[demo] command exited with code {result.returncode}", file=sys.stderr)

        time.sleep(PAUSE_AFTER_CMD)

    print("\n=== Demo complete ===")


if __name__ == "__main__":
    run_demo()
