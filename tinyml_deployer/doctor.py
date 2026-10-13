"""Inspect installed toolchains without changing the user's environment."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class DoctorResult:
    python_version: str
    checks: list[dict[str, str | bool]]
    ready: bool


def diagnose(*, onnx: bool = False) -> DoctorResult:
    checks: list[dict[str, str | bool]] = []
    packages = ["tensorflow", "numpy", "click", "rich", "pyserial"]
    if onnx:
        packages += [
            "onnx",
            "onnx2tf",
            "tf-keras",
            "onnxruntime",
            "ai-edge-litert",
            "onnx-graphsurgeon",
            "sng4onnx",
            "psutil",
        ]
    for package in packages:
        try:
            version = importlib.metadata.version(package)
            checks.append({"name": package, "ok": True, "detail": version})
        except importlib.metadata.PackageNotFoundError:
            checks.append(
                {
                    "name": package,
                    "ok": False,
                    "detail": "Install the package runtime dependencies.",
                }
            )
    for module in ["tensorflow", "onnx2tf", "onnxruntime"] if onnx else ["tensorflow"]:
        try:
            importlib.import_module(module)
            checks.append(
                {
                    "name": f"import {module}",
                    "ok": True,
                    "detail": "Runtime import succeeded.",
                }
            )
        except ImportError as exc:
            checks.append({"name": f"import {module}", "ok": False, "detail": str(exc)})
    for tool in ("idf.py", "cmake", "ninja", "xtensa-esp32s3-elf-g++"):
        executable = shutil.which(tool)
        if not executable:
            checks.append(
                {
                    "name": tool,
                    "ok": False,
                    "detail": "Install ESP-IDF 5.5.1 and source its export script.",
                }
            )
            continue
        try:
            result = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            detail = (result.stdout or result.stderr).strip()
            ok = result.returncode == 0
            if tool == "idf.py":
                ok = ok and "v5.5.1" in detail
            checks.append({"name": tool, "ok": ok, "detail": detail})
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append({"name": tool, "ok": False, "detail": str(exc)})
    checks.append(
        {
            "name": "IDF_PATH",
            "ok": bool(os.environ.get("IDF_PATH")),
            "detail": "Set by the ESP-IDF export script.",
        }
    )
    return DoctorResult(
        platform.python_version(), checks, all(bool(c["ok"]) for c in checks)
    )
