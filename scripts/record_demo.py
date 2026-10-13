"""Record actual read-only CLI output with asciinema."""

from __future__ import annotations

import shlex
import subprocess
import time

COMMANDS = [
    ["tinyml-deployer", "--version"],
    [
        "tinyml-deployer",
        "analyze",
        "examples/sine_model/sine_model.tflite",
        "--target",
        "esp32s3",
    ],
    [
        "tinyml-deployer",
        "benchmark",
        "examples/sine_model/sine_model.tflite",
        "--compare",
    ],
]


def run_demo() -> None:
    for command in COMMANDS:
        print(f"\n$ {shlex.join(command)}", flush=True)
        subprocess.run(command, check=True)
        time.sleep(1)


if __name__ == "__main__":
    run_demo()
