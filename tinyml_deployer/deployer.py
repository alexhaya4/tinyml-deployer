"""Generate ESP-IDF firmware from a checked model and real validation inputs."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from tinyml_deployer.analyzer import ModelAnalysis, analyze_model
from tinyml_deployer.codegen import (
    GeneratedFile,
    generate_inference_wrapper,
    generate_model_data,
)
from tinyml_deployer.evaluation import (
    encode_input,
    interpreter_for,
    load_samples,
    predict,
)
from tinyml_deployer.targets import get_target

IDF_VERSION = "5.5.1"
MICRO_VERSION = "1.4.1"
ESP_NN_VERSION = "1.4.1"


@dataclass
class DeployResult:
    target_name: str
    output_dir: str
    files: list[GeneratedFile] = field(default_factory=list)
    analysis: ModelAnalysis | None = None


def deploy_model(
    model_path: str,
    target_name: str,
    output_dir: str,
    *,
    arena_size: int,
    validation_data: str,
    max_abs_error: float,
    repetitions: int = 10,
    flash_budget_bytes: int = 1048576,
) -> DeployResult:
    target = get_target(target_name)
    if target.framework != "ESP-IDF":
        raise ValueError(
            "STM32 firmware generation is unavailable until board startup, linker "
            "and runtime integration are verified. Model analysis remains available."
        )
    if (
        not isinstance(arena_size, int)
        or isinstance(arena_size, bool)
        or not 0 < arena_size < target.ram_kb * 1024
    ):
        raise ValueError(
            "Arena size must be a positive integer below target RAM capacity."
        )
    if (
        not isinstance(repetitions, int)
        or isinstance(repetitions, bool)
        or not 1 <= repetitions <= 10000
    ):
        raise ValueError("Repetitions must be an integer from 1 to 10000.")
    if not np.isfinite(max_abs_error) or max_abs_error < 0:
        raise ValueError("Maximum absolute error must be finite and non-negative.")
    if (
        not isinstance(flash_budget_bytes, int)
        or not 0 < flash_budget_bytes <= target.flash_kb * 1024 - 0x10000
    ):
        raise ValueError(
            "Flash budget must fit the target after the bootloader and partition table."
        )
    if flash_budget_bytes % 0x10000:
        raise ValueError("Application flash budget must be aligned to 65536 bytes.")
    out = Path(output_dir).absolute()
    if out.exists():
        raise FileExistsError(f"Output directory already exists: {out}")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", out.name):
        raise ValueError(
            "Project name must contain only letters, digits, hyphens or underscores."
        )
    analysis = analyze_model(model_path, target_name)
    if not analysis.compatible:
        raise ValueError(
            "Model failed static preflight: " + "; ".join(analysis.warnings)
        )
    samples = load_samples(validation_data)
    model_bytes = Path(model_path).read_bytes()
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    dataset_hash = hashlib.sha256(Path(validation_data).read_bytes()).hexdigest()
    interpreter = interpreter_for(model_bytes)
    detail = interpreter.get_input_details()[0]
    encoded, expected = [], []
    for sample in samples:
        encoded.append(encode_input(sample, detail).tobytes())
        expected.append(predict(interpreter, sample).flatten())
    vector_bytes = sum(map(len, encoded)) + sum(v.size * 4 for v in expected)
    if len(model_bytes) + vector_bytes >= flash_budget_bytes:
        raise ValueError(
            "Model and validation vectors exceed the application flash budget."
        )
    manifest = {
        "schema_version": 1,
        "model_sha256": model_hash,
        "dataset_sha256": dataset_hash,
        "target": target.name,
        "esp_idf_version": IDF_VERSION,
        "micro_version": MICRO_VERSION,
        "esp_nn_version": ESP_NN_VERSION,
        "arena_reserved_bytes": arena_size,
        "app_flash_budget_bytes": flash_budget_bytes,
        "repetitions": repetitions,
        "samples": len(samples),
        "max_abs_error": max_abs_error,
        "validation": {
            "static_preflight": "passed",
            "firmware_build": "not_run",
            "hardware": "not_run",
        },
        "analysis": asdict(analysis),
    }
    # Avoid carrying ephemeral conversion paths into reproducible reports.
    manifest["analysis"]["model_path"] = Path(model_path).name
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tinyml-", dir=out.parent) as staging:
        root = Path(staging) / out.name
        main = root / "main"
        main.mkdir(parents=True)
        generate_model_data(model_path, str(main))
        generate_inference_wrapper(analysis, target.framework, str(main), arena_size)
        (root / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.16)\n"
            "include($ENV{IDF_PATH}/tools/cmake/project.cmake)\n"
            f"project({out.name})\n"
        )
        (main / "CMakeLists.txt").write_text(
            'idf_component_register(SRCS "main.cpp" "model_data.cpp" "inference.cpp"\n'
            '  INCLUDE_DIRS "." REQUIRES esp_timer esp_system app_update)\n'
        )
        (main / "idf_component.yml").write_text(
            f'dependencies:\n  idf: "=={IDF_VERSION}"\n'
            f'  espressif/esp-tflite-micro: "=={MICRO_VERSION}"\n'
            f'  espressif/esp-nn: "=={ESP_NN_VERSION}"\n'
        )
        (root / "sdkconfig.defaults").write_text(
            f'CONFIG_IDF_TARGET="{target.name}"\nCONFIG_ESP_MAIN_TASK_STACK_SIZE=16384\n'
            'CONFIG_PARTITION_TABLE_CUSTOM=y\nCONFIG_PARTITION_TABLE_CUSTOM_FILENAME="partitions.csv"\n'
        )
        (root / "partitions.csv").write_text(
            "# Name, Type, SubType, Offset, Size\n"
            "nvs,data,nvs,0x9000,0x6000\nphy_init,data,phy,0xf000,0x1000\n"
            f"factory,app,factory,0x10000,{hex(flash_budget_bytes)}\n"
        )
        raw = ",\n".join("{" + ",".join(str(v) for v in row) + "}" for row in encoded)
        values = ",\n".join(
            "{" + ",".join(float(v).hex() + "f" for v in row) + "}" for row in expected
        )
        vectors = (
            f"#define VALIDATION_SAMPLES {len(samples)}\n#define PROFILE_REPETITIONS {repetitions}\n"
            f"#define OUTPUT_ELEMENTS {expected[0].size}\n"
            f"#define MAX_ABS_ERROR {float(max_abs_error).hex()}\n"
            f"static const unsigned char inputs[VALIDATION_SAMPLES][MODEL_INPUT_BYTES] = {{\n{raw}\n}};\n"
            f"static const float expected[VALIDATION_SAMPLES][OUTPUT_ELEMENTS] = {{\n{values}\n}};\n"
        )
        (main / "validation_data.h").write_text(vectors)
        template = (
            Path(__file__).parent / "templates" / "profile_main.cpp"
        ).read_text()
        output_info = analysis.outputs[0]
        dtype = {
            "float32": "float",
            "int8": "int8_t",
            "uint8": "uint8_t",
            "int16": "int16_t",
            "int32": "int32_t",
        }[output_info.dtype]
        scale = output_info.scales[0] if output_info.scales else 1.0
        zero = output_info.zero_points[0] if output_info.zero_points else 0
        for key, value in {
            "MODEL_HASH": model_hash,
            "DATASET_HASH": dataset_hash,
            "OUTPUT_CTYPE": dtype,
            "OUTPUT_SCALE": float(scale).hex(),
            "OUTPUT_ZERO": str(zero),
            "TARGET_NAME": target.name,
        }.items():
            template = template.replace("@" + key + "@", value)
        (main / "main.cpp").write_text(template)
        (root / "deployment.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        (root / "README.md").write_text(
            f"# {out.name}\n\nGenerated for {target.name}, ESP-IDF {IDF_VERSION}, "
            f"esp-tflite-micro {MICRO_VERSION}.\n\n"
            f"Build with `idf.py set-target {target.name}` and `idf.py build`.\n"
            "Flash with `idf.py -p PORT flash`. Close the serial monitor before profiling.\n"
            "Run `tinyml-deployer profile --port PORT --manifest deployment.json --report results.json`.\n\n"
            "The firmware waits for a profiling request, performs a warm-up and then measures each "
            "provided sample repeatedly. Results include prediction agreement with the desktop "
            "TFLite reference. This checks implementation agreement, not task accuracy.\n\n"
            "Inspect linker size reports and measured arena/heap usage before integration. "
            "The wrapper uses static storage and requires external serialization. "
            "Validation arrays consume flash. Remove the validation application when integrating "
            "the inference library into an application.\n"
        )
        root.rename(out)
    files = [
        GeneratedFile(str(f), "Generated deployment artifact")
        for f in sorted(out.rglob("*"))
        if f.is_file()
    ]
    return DeployResult(target.name, str(out), files, analysis)
