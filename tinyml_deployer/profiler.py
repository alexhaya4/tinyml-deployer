"""Collect and validate measurements from generated firmware over serial."""

from __future__ import annotations

import json
import math
import re
import time
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass
class ProfileResult:
    measurement_kind: str
    timing_scope: str
    model_sha256: str
    dataset_sha256: str
    target: str
    idf_version: str
    observations: int
    median_latency_us: float
    p95_latency_us: float
    max_latency_us: float
    max_absolute_error: float
    max_allowed_error: float
    prediction_agreement_passed: bool
    arena_used_bytes: int
    min_free_heap_bytes: int
    records: list[dict]


def validate_manifest(manifest: dict) -> None:
    """Validate identity, limits and run dimensions before opening a device."""
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported deployment manifest version.")
    for key in ("model_sha256", "dataset_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(key, ""))):
            raise ValueError(f"Invalid deployment manifest field: {key}")
    for key in ("samples", "repetitions", "arena_reserved_bytes"):
        if type(manifest.get(key)) is not int or manifest[key] <= 0:
            raise ValueError(f"Invalid deployment manifest field: {key}")
    if manifest["repetitions"] > 10000:
        raise ValueError("Invalid deployment repetition count.")
    limit = manifest.get("max_abs_error")
    if type(limit) not in (float, int) or not math.isfinite(limit) or limit < 0:
        raise ValueError("Invalid deployment error limit.")
    if manifest.get("target") not in {"esp32", "esp32s3", "esp32c3", "esp32c6"}:
        raise ValueError("Unsupported deployment target.")
    if not re.fullmatch(r"\d+\.\d+\.\d+", str(manifest.get("esp_idf_version", ""))):
        raise ValueError("Invalid deployment ESP-IDF version.")


def validate_records(
    records: Iterable[dict],
    manifest: dict,
    run_id: str,
    *,
    measurement_kind: Literal["hardware", "emulator"] = "hardware",
) -> ProfileResult:
    """Reject incomplete, mismatched, duplicate or invalid device records."""
    validate_manifest(manifest)
    if measurement_kind not in {"hardware", "emulator"}:
        raise ValueError("Unknown measurement environment.")
    rows = list(records)
    if (
        len(rows) < 3
        or rows[0].get("event") != "begin"
        or rows[-1].get("event") != "done"
    ):
        raise ValueError("Device run is incomplete.")
    if any(row.get("run_id") != run_id for row in rows):
        raise ValueError("Device run ID mismatch.")
    begin = rows[0]
    for key in ("model_sha256", "dataset_sha256", "target"):
        if begin.get(key) != manifest[key]:
            raise ValueError(f"Device {key} does not match the deployment manifest.")
    if begin.get("idf_version") != "v" + manifest["esp_idf_version"]:
        raise ValueError(
            "Device ESP-IDF version does not match the deployment manifest."
        )
    measurements = rows[1:-1]
    seen: set[tuple[int, int]] = set()
    for row in measurements:
        if row.get("event") != "sample":
            raise ValueError("Device reported an error or an unexpected event.")
        for key in ("sample", "repetition", "arena_used_bytes", "min_free_heap_bytes"):
            if type(row.get(key)) is not int or row[key] < 0:
                raise ValueError(f"Invalid device field: {key}")
        pair = (row["sample"], row["repetition"])
        if (
            pair[0] >= manifest["samples"]
            or pair[1] >= manifest["repetitions"]
            or pair in seen
        ):
            raise ValueError("Duplicate or out-of-range device sample.")
        seen.add(pair)
        for key in ("latency_us", "max_abs_error"):
            value = row.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid device field: {key}")
        if (
            row["arena_used_bytes"] <= 0
            or row["arena_used_bytes"] > manifest["arena_reserved_bytes"]
        ):
            raise ValueError("Reported tensor arena usage is outside its reservation.")
    if len(seen) != manifest["samples"] * manifest["repetitions"]:
        raise ValueError("Device run has missing samples.")
    latency = [row["latency_us"] for row in measurements]
    error = max(row["max_abs_error"] for row in measurements)
    return ProfileResult(
        measurement_kind,
        (
            "Emulated time; not physical device performance. "
            if measurement_kind == "emulator"
            else ""
        )
        + "model_run: input copy, invoke, output copy; excludes warm-up and serial I/O",
        manifest["model_sha256"],
        manifest["dataset_sha256"],
        manifest["target"],
        begin["idf_version"],
        len(measurements),
        float(np.median(latency)),
        float(np.percentile(latency, 95)),
        float(max(latency)),
        error,
        manifest["max_abs_error"],
        error <= manifest["max_abs_error"],
        max(row["arena_used_bytes"] for row in measurements),
        min(row["min_free_heap_bytes"] for row in measurements),
        rows,
    )


def profile_device(
    port: str, manifest_path: str, *, baud: int = 115200, timeout: float = 120.0
) -> ProfileResult:
    import serial

    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be finite and positive.")
    if baud <= 0:
        raise ValueError("Baud must be positive.")
    manifest = json.loads(Path(manifest_path).read_text())
    validate_manifest(manifest)
    run_id = uuid.uuid4().hex
    records: list[dict] = []
    deadline = time.monotonic() + timeout
    next_request = 0.0
    pending = bytearray()
    with serial.Serial(port, baudrate=baud, timeout=0.5, write_timeout=5) as connection:
        connection.reset_input_buffer()
        while time.monotonic() < deadline:
            now = time.monotonic()
            if not records and now >= next_request:
                connection.write(f"run {run_id}\n".encode("ascii"))
                next_request = now + 2.0
            line = connection.read_until(b"\n", size=4096)
            pending.extend(line)
            if len(pending) > 4096:
                raise ValueError("Device record exceeds the 4096-byte protocol limit.")
            if not pending.endswith(b"\n"):
                continue
            line = bytes(pending)
            pending.clear()
            if not line.startswith(b"TINYML "):
                continue
            try:
                row = json.loads(line[7:])
            except (ValueError, UnicodeError):
                raise ValueError("Malformed device measurement record.") from None
            if not isinstance(row, dict) or row.get("run_id") != run_id:
                continue
            if row.get("event") == "error":
                raise RuntimeError(
                    f"Device inference failed with status {row.get('status')}."
                )
            records.append(row)
            if len(records) > manifest["samples"] * manifest["repetitions"] + 2:
                raise ValueError("Device returned too many measurement records.")
            if row.get("event") == "done":
                return validate_records(records, manifest, run_id)
    raise TimeoutError(
        "No complete profiling run received before timeout. Check the port, firmware and console configuration."
    )
