"""Execute generated ESP32-S3 firmware under pinned Espressif QEMU on Linux."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import select
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections import deque
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from tinyml_deployer.profiler import ProfileResult, validate_manifest, validate_records

QEMU_VERSION = "esp_develop_9.2.2_20260417"
MAX_LINE_BYTES = 4096


@dataclass
class EmulationResult:
    measurement_kind: str
    emulator_version: str
    emulator_binary_sha256: str
    hardware_validation: str
    hardware_timing_valid: bool
    esp_nn_optimized: bool
    manifest_sha256: str
    flash_image_sha256: str
    firmware_files: list[dict[str, str | int]]
    validation: ProfileResult
    console_tail: list[str]


def _flash_inputs(project: Path, manifest: dict) -> tuple[dict, list[dict]]:
    build = (project / "build").resolve()
    arguments = json.loads((build / "flasher_args.json").read_text())
    if arguments.get("extra_esptool_args", {}).get("chip") != manifest["target"]:
        raise ValueError("Build target does not match the deployment manifest.")
    settings = arguments["flash_settings"]
    sizes = {f"{n}MB": n * 1024 * 1024 for n in (2, 4, 8, 16)}
    if settings.get("flash_size") not in sizes:
        raise ValueError("ESP32-S3 emulation requires a 2, 4, 8 or 16 MB flash image.")
    limit = sizes[settings["flash_size"]]
    files = []
    for offset, name in arguments["flash_files"].items():
        path = (build / name).resolve()
        if not path.is_relative_to(build) or not path.is_file():
            raise ValueError(
                "Flash inputs must be existing files inside the build directory."
            )
        address = int(offset, 0)
        size = path.stat().st_size
        if address < 0 or size <= 0 or address + size > limit:
            raise ValueError(
                "Flash input is empty or outside the configured flash capacity."
            )
        content = path.read_bytes()
        if len(content) != size:
            raise ValueError(
                "Flash input changed during validation; finish the build and retry."
            )
        files.append(
            {
                "offset": address,
                "path": str(path.relative_to(build)),
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    files.sort(key=lambda entry: entry["offset"])
    if not files or files[0]["offset"] != 0:
        raise ValueError(
            "The flash image must include the ESP32-S3 bootloader at offset zero."
        )
    for previous, current in pairwise(files):
        if previous["offset"] + previous["size_bytes"] > current["offset"]:
            raise ValueError("Flash input regions overlap.")
    return settings, files


def _merge_flash(
    project: Path, output: Path, settings: dict, files: list[dict]
) -> None:
    try:
        installed = importlib.metadata.version("esptool")
    except importlib.metadata.PackageNotFoundError:
        raise RuntimeError(
            "Install the emulation extra: pip install 'tinyml-deployer[emulation]'."
        ) from None
    if installed != "4.12.0":
        raise RuntimeError("The validated flash merger requires esptool 4.12.0.")
    command = [
        sys.executable,
        "-m",
        "esptool",
        "--chip",
        "esp32s3",
        "merge_bin",
        "--fill-flash-size",
        settings["flash_size"],
        "--flash_size",
        settings["flash_size"],
        "--flash_mode",
        settings["flash_mode"],
        "--flash_freq",
        settings["flash_freq"],
        "-o",
        str(output),
    ]
    for entry in files:
        original = project / "build" / entry["path"]
        content = original.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(
                "Flash input changed after validation; finish the build and retry."
            )
        # Snapshot the checked bytes so a concurrent build cannot change the image.
        snapshot = output.parent / (str(entry["offset"]) + ".bin")
        snapshot.write_bytes(content)
        command += [hex(entry["offset"]), str(snapshot)]
    result = subprocess.run(
        command, capture_output=True, text=True, timeout=30, check=False
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Flash image generation failed: {(result.stderr or result.stdout)[-4000:]}"
        )


def _execute(
    qemu: str, image: Path, manifest: dict, timeout: float
) -> tuple[ProfileResult, list[str]]:
    """Run the actual binary and require a fresh, complete protocol transcript."""
    command = [
        qemu,
        "-machine",
        "esp32s3",
        "-display",
        "none",
        "-monitor",
        "none",
        "-serial",
        "stdio",
        "-nic",
        "none",
        "-no-reboot",
        "-snapshot",
        "-drive",
        f"file={image},if=mtd,format=raw",
    ]
    run_id = uuid.uuid4().hex
    rows: list[dict] = []
    pending = bytearray()
    tail: deque[str] = deque(maxlen=80)
    deadline = time.monotonic() + timeout
    next_request = time.monotonic() + 0.5
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    try:
        os.set_blocking(process.stdin.fileno(), False)
        while time.monotonic() < deadline:
            now = time.monotonic()
            if not rows and now >= next_request and process.poll() is None:
                try:
                    process.stdin.write(f"run {run_id}\n".encode("ascii"))
                except BlockingIOError:
                    pass
                next_request = now + 2
            readable, _, _ = select.select(
                [process.stdout], [], [], min(0.2, max(0, deadline - now))
            )
            if readable:
                chunk = os.read(process.stdout.fileno(), MAX_LINE_BYTES)
                if not chunk:
                    raise RuntimeError(
                        "QEMU closed its console before completing inference."
                    )
                pending.extend(chunk)
                while b"\n" in pending:
                    line, _, rest = pending.partition(b"\n")
                    pending = bytearray(rest)
                    if len(line) > MAX_LINE_BYTES:
                        raise ValueError(
                            "QEMU console record exceeds the protocol limit."
                        )
                    decoded = line.decode("utf-8", errors="replace").rstrip("\r")
                    tail.append(decoded)
                    if any(
                        marker in decoded
                        for marker in (
                            "Guru Meditation Error",
                            "abort() was called",
                            "assert failed:",
                        )
                    ):
                        raise RuntimeError(f"Firmware crashed under QEMU: {decoded}")
                    if not line.startswith(b"TINYML "):
                        continue
                    row = json.loads(line[7:])
                    if not isinstance(row, dict) or row.get("run_id") != run_id:
                        raise ValueError(
                            "QEMU returned a malformed or unrelated run record."
                        )
                    if row.get("event") == "error":
                        raise RuntimeError(
                            f"Firmware inference failed with status {row.get('status')}."
                        )
                    rows.append(row)
                    if len(rows) > manifest["samples"] * manifest["repetitions"] + 2:
                        raise ValueError("QEMU returned too many measurement records.")
                    if row.get("event") == "done":
                        validation = validate_records(
                            rows, manifest, run_id, measurement_kind="emulator"
                        )
                        return validation, list(tail)
                if len(pending) > MAX_LINE_BYTES:
                    raise ValueError("QEMU console record exceeds the protocol limit.")
            if process.poll() is not None:
                raise RuntimeError(
                    f"QEMU exited with status {process.returncode} before completion."
                )
        raise TimeoutError(
            "QEMU did not complete inference within the requested timeout."
        )
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        # Keep bounded boot/panic evidence available even when no report is valid.
        detail = "\n".join(tail)[-6000:]
        raise RuntimeError(f"{exc}\nQEMU console tail:\n{detail}") from exc
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()


def emulate_project(
    project_path: str,
    *,
    qemu_binary: str = "qemu-system-xtensa",
    timeout: float = 120.0,
) -> EmulationResult:
    """Validate an already-built project without changing its files or claiming hardware results."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be finite and positive.")
    if sys.platform != "linux":
        raise RuntimeError(
            "The emulator runner is validated on Linux. Use GitHub CI or Linux/WSL2."
        )
    project = Path(project_path).resolve()
    manifest_bytes = (project / "deployment.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    validate_manifest(manifest)
    if manifest["target"] != "esp32s3":
        raise ValueError("Emulation validation currently supports ESP32-S3 only.")
    executable = shutil.which(qemu_binary)
    if not executable:
        raise FileNotFoundError(
            "Espressif QEMU was not found. Install the pinned release and add it to PATH."
        )
    version = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.splitlines()[0]
    if f"({QEMU_VERSION})" not in version:
        raise ValueError(f"Use Espressif QEMU {QEMU_VERSION}; found {version}.")
    settings, files = _flash_inputs(project, manifest)
    optimized = (
        "CONFIG_NN_OPTIMIZED=y" in (project / "sdkconfig").read_text().splitlines()
    )
    with tempfile.TemporaryDirectory(prefix="tinyml_qemu_") as temporary:
        flash = Path(temporary) / "flash.bin"
        _merge_flash(project, flash, settings, files)
        flash_hash = hashlib.sha256(flash.read_bytes()).hexdigest()
        validation, tail = _execute(executable, flash, manifest, timeout)
    return EmulationResult(
        "emulator",
        version,
        hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
        "not_run",
        False,
        optimized,
        hashlib.sha256(manifest_bytes).hexdigest(),
        flash_hash,
        files,
        validation,
        tail,
    )
