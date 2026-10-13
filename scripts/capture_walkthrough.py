"""Capture real CLI output for the release walkthrough; never invent observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--firmware-root", type=Path, required=True)
    parser.add_argument("--qemu", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    cli = str(Path(sys.executable).with_name("tinyml-deployer"))
    firmware = args.firmware_root.resolve()
    qemu = str(args.qemu.resolve())
    env = dict(os.environ)
    env.update(
        TF_CPP_MIN_LOG_LEVEL="3",
        TF_NUM_INTEROP_THREADS="1",
        TF_NUM_INTRAOP_THREADS="1",
        NO_COLOR="1",
        COLUMNS="100",
    )
    commands = [
        ("version", [cli, "--version"]),
        (
            "train",
            [
                sys.executable,
                str(root / "examples/sine_model/train.py"),
                "--output",
                "sine",
                "--epochs",
                "50",
            ],
        ),
        (
            "quantize",
            [
                cli,
                "quantize",
                "sine/sine.keras",
                "--type",
                "int8",
                "--calibration",
                "sine/calibration.npy",
                "--validation",
                "sine/validation.npy",
                "--max-abs-error",
                "0.05",
                "--output",
                "sine/sine_int8.tflite",
                "--report",
                "quantization.json",
            ],
        ),
        ("analyze", [cli, "analyze", "sine/sine_int8.tflite", "--target", "esp32s3"]),
        (
            "deploy",
            [
                cli,
                "deploy",
                "sine/sine_int8.tflite",
                "--target",
                "esp32s3",
                "--arena-size",
                "65536",
                "--validation",
                "sine/validation.npy",
                "--max-abs-error",
                "0.01",
                "--repetitions",
                "10",
                "--output",
                "sine_project",
            ],
        ),
    ]
    for kind in ("float32", "int8"):
        commands.append(
            (
                f"emulate-{kind}",
                [
                    cli,
                    "emulate",
                    str(firmware / f"firmware-{kind}-esp32s3"),
                    "--qemu",
                    qemu,
                    "--report",
                    f"{kind}.json",
                ],
            )
        )
    records = []
    for label, command in commands:
        started = datetime.now(UTC).isoformat()
        before = time.monotonic()
        result = subprocess.run(
            command,
            cwd=output,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        record = {
            "label": label,
            "argv": command,
            "started_utc": started,
            "duration_seconds": round(time.monotonic() - before, 3),
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        records.append(record)
        (output / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        print(
            f"{label}: exit {result.returncode}, {record['duration_seconds']} seconds",
            flush=True,
        )
        if result.returncode:
            print(result.stdout + result.stderr)
            raise SystemExit(result.returncode)
    hashes = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            hashes[str(path.relative_to(output))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    (output / "capture-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
