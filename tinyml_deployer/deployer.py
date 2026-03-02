"""Scaffold complete deployment projects for ESP32 and STM32 targets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tinyml_deployer.analyzer import ModelAnalysis, analyze_model
from tinyml_deployer.codegen import GeneratedFile, generate_inference_wrapper, generate_model_data
from tinyml_deployer.targets import MCUTarget, get_target


@dataclass
class DeployResult:
    """Summary of a generated deployment project."""

    target_name: str
    output_dir: str
    files: list[GeneratedFile] = field(default_factory=list)
    analysis: ModelAnalysis | None = None


# ---------------------------------------------------------------------------
# ESP-IDF project scaffolding
# ---------------------------------------------------------------------------

_ESP_ROOT_CMAKE = """\
cmake_minimum_required(VERSION 3.16)

include($ENV{{IDF_PATH}}/tools/cmake/project.cmake)
project({project_name})
"""

_ESP_MAIN_CMAKE = """\
idf_component_register(
    SRCS "main.c" "model_data.c" "inference.c"
    INCLUDE_DIRS "."
    REQUIRES esp_timer
)
"""

_ESP_MAIN_C = """\
#include <stdio.h>
#include "inference.h"
#include "esp_log.h"

static const char* TAG = "app";

void app_main(void) {{
    ESP_LOGI(TAG, "Initializing TFLite model...");
    if (model_init() != 0) {{
        ESP_LOGE(TAG, "model_init failed");
        return;
    }}

    /* Run a single test inference. */
    {in_ctype} input[MODEL_INPUT_SIZE] = {{ {sample_input} }};
    {out_ctype} output[MODEL_OUTPUT_SIZE] = {{ 0 }};

    if (model_run(input, output) != 0) {{
        ESP_LOGE(TAG, "model_run failed");
        return;
    }}

    ESP_LOGI(TAG, "Inference result: %f", (double)output[0]);
    model_free();
}}
"""

_ESP_SDKCONFIG = """\
# Target chip selection (e.g. esp32, esp32s3, esp32c3, esp32c6).
CONFIG_IDF_TARGET="{idf_target}"
# TFLite Micro needs at least 8 KB of stack for the main task.
CONFIG_ESP_MAIN_TASK_STACK_SIZE=16384
CONFIG_FREERTOS_THREAD_LOCAL_STORAGE_POINTERS=2
"""


def _scaffold_esp32(
    analysis: ModelAnalysis,
    target: MCUTarget,
    output_dir: Path,
) -> list[GeneratedFile]:
    """Create an ESP-IDF project skeleton."""
    files: list[GeneratedFile] = []
    main_dir = output_dir / "main"
    main_dir.mkdir(parents=True, exist_ok=True)

    project_name = output_dir.name

    in_ctype = _dtype_to_c(analysis.inputs[0].dtype)
    out_ctype = _dtype_to_c(analysis.outputs[0].dtype)
    sample_input = "1.0f" if "float" in in_ctype else "1"

    # Root CMakeLists.txt
    path = output_dir / "CMakeLists.txt"
    path.write_text(_ESP_ROOT_CMAKE.format(project_name=project_name))
    files.append(GeneratedFile(str(path), "Root CMake project file"))

    # main/CMakeLists.txt
    path = main_dir / "CMakeLists.txt"
    path.write_text(_ESP_MAIN_CMAKE)
    files.append(GeneratedFile(str(path), "Main component CMake file"))

    # main/main.c
    path = main_dir / "main.c"
    path.write_text(_ESP_MAIN_C.format(
        in_ctype=in_ctype,
        out_ctype=out_ctype,
        sample_input=sample_input,
    ))
    files.append(GeneratedFile(str(path), "Application entry point"))

    # sdkconfig.defaults
    path = output_dir / "sdkconfig.defaults"
    path.write_text(_ESP_SDKCONFIG.format(idf_target=target.name))
    files.append(GeneratedFile(str(path), "Default SDK configuration"))

    # README.md
    files += _write_readme(output_dir, target, analysis, framework="esp-idf")

    return files


# ---------------------------------------------------------------------------
# STM32 project scaffolding
# ---------------------------------------------------------------------------

_STM32_MAKEFILE = """\
# Minimal Makefile for STM32 + TFLite Micro build.
# Adjust TOOLCHAIN_PREFIX and STM32CUBE_PATH for your environment.

TOOLCHAIN_PREFIX ?= arm-none-eabi-
STM32CUBE_PATH   ?= $(HOME)/STM32Cube

CC  = $(TOOLCHAIN_PREFIX)gcc
CXX = $(TOOLCHAIN_PREFIX)g++
AR  = $(TOOLCHAIN_PREFIX)ar

CFLAGS  = -mcpu={cpu} -mthumb -Os -Wall -g
CFLAGS += -ICore/Inc -I.

SRCS = Core/Src/main.c model_data.c inference.c

all:
\t$(CC) $(CFLAGS) $(SRCS) -o firmware.elf -lm

clean:
\trm -f firmware.elf

.PHONY: all clean
"""

_STM32_MAIN_H = """\
#ifndef MAIN_H
#define MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Include the HAL header for your specific chip, e.g.:
   #include "stm32f4xx_hal.h"
   or
   #include "stm32h7xx_hal.h"
*/

void SystemClock_Config(void);
void Error_Handler(void);

#ifdef __cplusplus
}
#endif

#endif  /* MAIN_H */
"""

_STM32_MAIN_C = """\
#include "main.h"
#include "inference.h"
#include <stdio.h>

/* Minimal clock setup stub. Replace with CubeMX-generated code. */
void SystemClock_Config(void) {{
    /* TODO: Configure system clock for your target. */
}}

void Error_Handler(void) {{
    while (1) {{
        /* Stay here on error. */
    }}
}}

int main(void) {{
    /* HAL_Init();            Uncomment after adding HAL sources. */
    /* SystemClock_Config();  Uncomment after adding HAL sources. */

    if (model_init() != 0) {{
        Error_Handler();
    }}

    {in_ctype} input[MODEL_INPUT_SIZE] = {{ {sample_input} }};
    {out_ctype} output[MODEL_OUTPUT_SIZE] = {{ 0 }};

    if (model_run(input, output) != 0) {{
        Error_Handler();
    }}

    /* output[0] now holds the inference result.
       Use a debugger or UART to inspect it. */

    model_free();

    while (1) {{
        /* Main loop. */
    }}
}}
"""


def _cpu_flag(target: MCUTarget) -> str:
    """Return the GCC -mcpu value for an STM32 target."""
    name = target.name.lower()
    if "h7" in name:
        return "cortex-m7"
    return "cortex-m4"


def _scaffold_stm32(
    analysis: ModelAnalysis,
    target: MCUTarget,
    output_dir: Path,
) -> list[GeneratedFile]:
    """Create an STM32 project skeleton."""
    files: list[GeneratedFile] = []
    src_dir = output_dir / "Core" / "Src"
    inc_dir = output_dir / "Core" / "Inc"
    src_dir.mkdir(parents=True, exist_ok=True)
    inc_dir.mkdir(parents=True, exist_ok=True)

    in_ctype = _dtype_to_c(analysis.inputs[0].dtype)
    out_ctype = _dtype_to_c(analysis.outputs[0].dtype)
    sample_input = "1.0f" if "float" in in_ctype else "1"

    # Makefile
    path = output_dir / "Makefile"
    path.write_text(_STM32_MAKEFILE.format(cpu=_cpu_flag(target)))
    files.append(GeneratedFile(str(path), "Build Makefile"))

    # Core/Inc/main.h
    path = inc_dir / "main.h"
    path.write_text(_STM32_MAIN_H)
    files.append(GeneratedFile(str(path), "Main header with HAL stubs"))

    # Core/Src/main.c
    path = src_dir / "main.c"
    path.write_text(_STM32_MAIN_C.format(
        in_ctype=in_ctype,
        out_ctype=out_ctype,
        sample_input=sample_input,
    ))
    files.append(GeneratedFile(str(path), "Application entry point"))

    # README.md
    files += _write_readme(output_dir, target, analysis, framework="stm32")

    return files


# ---------------------------------------------------------------------------
# README generation
# ---------------------------------------------------------------------------

def _write_readme(
    output_dir: Path,
    target: MCUTarget,
    analysis: ModelAnalysis,
    framework: str,
) -> list[GeneratedFile]:
    """Write a README.md with build and flash instructions."""
    model_kb = analysis.model_size_bytes / 1024
    arena_kb = analysis.tensor_arena_bytes / 1024

    if framework == "esp-idf":
        build_steps = (
            "## Build and flash (ESP-IDF)\n"
            "\n"
            "1. Install ESP-IDF (v5.x recommended) and set the IDF_PATH\n"
            "   environment variable.\n"
            "2. Add the TFLite Micro component to `main/`. You can use the\n"
            "   official `tflite-micro` ESP-IDF component from the component\n"
            "   registry or vendor the source directly.\n"
            "3. Build and flash:\n"
            "   ```\n"
            f"   idf.py set-target {target.name}\n"
            "   idf.py build\n"
            "   idf.py -p /dev/ttyUSB0 flash monitor\n"
            "   ```\n"
        )
    else:
        build_steps = (
            "## Build and flash (STM32)\n"
            "\n"
            "1. Install the ARM GCC toolchain (arm-none-eabi-gcc).\n"
            "2. Install STM32CubeMX or STM32CubeIDE and generate the HAL\n"
            "   initialization code for your board. Copy the generated HAL\n"
            "   sources into `Core/`.\n"
            "3. Add TFLite Micro sources to the project and update the\n"
            "   Makefile include paths.\n"
            "4. Build:\n"
            "   ```\n"
            "   make\n"
            "   ```\n"
            "5. Flash using ST-Link or your preferred programmer:\n"
            "   ```\n"
            "   st-flash write firmware.elf 0x8000000\n"
            "   ```\n"
        )

    readme = (
        f"# {output_dir.name}\n"
        "\n"
        f"Auto-generated TFLite Micro deployment project for **{target.name}**.\n"
        "\n"
        "## Model info\n"
        "\n"
        f"- Model size: {model_kb:.1f} KB\n"
        f"- Tensor arena: {arena_kb:.1f} KB\n"
        f"- Input shape: {analysis.inputs[0].shape} ({analysis.inputs[0].dtype})\n"
        f"- Output shape: {analysis.outputs[0].shape} ({analysis.outputs[0].dtype})\n"
        "\n"
        f"{build_steps}"
        "\n"
        "## Project structure\n"
        "\n"
        "- `model_data.h / .c` -- model weights as a C byte array\n"
        "- `inference.h / .c` -- thin wrapper: model_init(), model_run(), model_free()\n"
        "- See the source files for detailed comments.\n"
    )
    path = output_dir / "README.md"
    path.write_text(readme)
    return [GeneratedFile(str(path), "Build and flash instructions")]


# ---------------------------------------------------------------------------
# Helpers shared across scaffolders
# ---------------------------------------------------------------------------

def _dtype_to_c(dtype: str) -> str:
    """Map a numpy dtype string to a C type."""
    mapping = {
        "float32": "float",
        "float16": "float",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "int32": "int32_t",
    }
    return mapping.get(dtype, "float")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deploy_model(
    model_path: str,
    target_name: str,
    output_dir: str,
) -> DeployResult:
    """Generate a complete deployment project for a TFLite model.

    Args:
        model_path: Path to the .tflite model file.
        target_name: Target MCU name (e.g. "esp32", "stm32f4").
        output_dir: Directory where the project will be created.

    Returns:
        DeployResult with the list of all generated files.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the target is not recognized.
    """
    target = get_target(target_name)
    analysis = analyze_model(model_path, target_name)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = DeployResult(
        target_name=target.name,
        output_dir=str(out),
        analysis=analysis,
    )

    is_esp = target.framework == "ESP-IDF"

    # Choose where model_data and inference files land
    code_dir = out / "main" if is_esp else out

    # Generate C source files
    result.files += generate_model_data(model_path, str(code_dir))
    result.files += generate_inference_wrapper(
        analysis,
        target.framework,
        str(code_dir),
        arena_size=analysis.tensor_arena_bytes,
    )

    # Scaffold the platform-specific project
    if is_esp:
        result.files += _scaffold_esp32(analysis, target, out)
    else:
        result.files += _scaffold_stm32(analysis, target, out)

    return result
