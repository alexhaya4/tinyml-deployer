"""Generate deployment-ready C source files from TFLite models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tinyml_deployer.analyzer import ModelAnalysis


@dataclass
class GeneratedFile:
    """Record of a single generated file."""

    path: str
    description: str


def _bytes_to_c_array(data: bytes, var_name: str) -> str:
    """Convert raw bytes to a C byte-array literal."""
    lines: list[str] = []
    for i in range(0, len(data), 12):
        chunk = data[i : i + 12]
        hex_vals = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"    {hex_vals},")
    body = "\n".join(lines)
    return (
        f"const unsigned char {var_name}[] = {{\n"
        f"{body}\n"
        f"}};\n"
        f"const unsigned int {var_name}_len = {len(data)};\n"
    )


def generate_model_data(
    model_path: str,
    output_dir: str,
) -> list[GeneratedFile]:
    """Convert a .tflite file into model_data.h and model_data.c.

    Args:
        model_path: Path to the .tflite model file.
        output_dir: Directory where generated files are written.

    Returns:
        List of GeneratedFile entries for each file created.
    """
    model_bytes = Path(model_path).read_bytes()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- model_data.h --
    header = (
        "#ifndef MODEL_DATA_H\n"
        "#define MODEL_DATA_H\n"
        "\n"
        "#ifdef __cplusplus\n"
        'extern "C" {\n'
        "#endif\n"
        "\n"
        "extern const unsigned char model_tflite[];\n"
        "extern const unsigned int model_tflite_len;\n"
        "\n"
        "#ifdef __cplusplus\n"
        "}\n"
        "#endif\n"
        "\n"
        "#endif  /* MODEL_DATA_H */\n"
    )
    header_path = out / "model_data.h"
    header_path.write_text(header)

    # -- model_data.c --
    c_array = _bytes_to_c_array(model_bytes, "model_tflite")
    source = (
        '#include "model_data.h"\n'
        "\n"
        f"{c_array}"
    )
    source_path = out / "model_data.c"
    source_path.write_text(source)

    return [
        GeneratedFile(str(header_path), "Model byte array header"),
        GeneratedFile(str(source_path), "Model byte array source"),
    ]


def _dtype_to_c(dtype: str) -> str:
    """Map a numpy dtype string to the corresponding C type."""
    mapping = {
        "float32": "float",
        "float16": "float",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "int32": "int32_t",
    }
    return mapping.get(dtype, "float")


def _input_size(analysis: ModelAnalysis) -> int:
    """Total element count of the first input tensor."""
    shape = analysis.inputs[0].shape
    total = 1
    for dim in shape:
        total *= dim
    return total


def _output_size(analysis: ModelAnalysis) -> int:
    """Total element count of the first output tensor."""
    shape = analysis.outputs[0].shape
    total = 1
    for dim in shape:
        total *= dim
    return total


def generate_inference_wrapper(
    analysis: ModelAnalysis,
    target_framework: str,
    output_dir: str,
    arena_size: int | None = None,
) -> list[GeneratedFile]:
    """Generate inference.h and inference.c using TFLite Micro C API.

    Args:
        analysis: Model analysis result from the analyzer.
        target_framework: "ESP-IDF" or "STM32CubeAI".
        output_dir: Directory where generated files are written.
        arena_size: Tensor arena size in bytes. Defaults to the analyzer
            estimate if not provided.

    Returns:
        List of GeneratedFile entries for each file created.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if arena_size is None:
        arena_size = analysis.tensor_arena_bytes

    in_ctype = _dtype_to_c(analysis.inputs[0].dtype)
    out_ctype = _dtype_to_c(analysis.outputs[0].dtype)
    in_count = _input_size(analysis)
    out_count = _output_size(analysis)

    is_esp = target_framework == "ESP-IDF"

    # -- inference.h --
    header = (
        "#ifndef INFERENCE_H\n"
        "#define INFERENCE_H\n"
        "\n"
        "#include <stdint.h>\n"
        "\n"
        "#ifdef __cplusplus\n"
        'extern "C" {\n'
        "#endif\n"
        "\n"
        f"#define MODEL_INPUT_SIZE  {in_count}\n"
        f"#define MODEL_OUTPUT_SIZE {out_count}\n"
        f"#define TENSOR_ARENA_SIZE {arena_size}\n"
        "\n"
        "/* Initialize the TFLite Micro interpreter. Returns 0 on success. */\n"
        "int model_init(void);\n"
        "\n"
        "/* Run inference. Copies input_data into the input tensor,\n"
        "   invokes the interpreter, and copies results to output_data.\n"
        "   Returns 0 on success. */\n"
        f"int model_run(const {in_ctype}* input_data, {out_ctype}* output_data);\n"
        "\n"
        "/* Free interpreter resources. */\n"
        "void model_free(void);\n"
        "\n"
        "#ifdef __cplusplus\n"
        "}\n"
        "#endif\n"
        "\n"
        "#endif  /* INFERENCE_H */\n"
    )
    header_path = out / "inference.h"
    header_path.write_text(header)

    # -- inference.c --
    if is_esp:
        platform_include = '#include "esp_log.h"\n'
        log_tag = '\nstatic const char* TAG = "tflite";\n'
        log_init_ok = '    ESP_LOGI(TAG, "Model interpreter initialized");\n'
        log_init_fail = '    ESP_LOGE(TAG, "Failed to initialize interpreter");\n'
    else:
        platform_include = ""
        log_tag = ""
        log_init_ok = ""
        log_init_fail = ""

    source = (
        '#include "inference.h"\n'
        '#include "model_data.h"\n'
        "\n"
        '#include "tensorflow/lite/micro/micro_interpreter.h"\n'
        '#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"\n'
        '#include "tensorflow/lite/micro/system_setup.h"\n'
        '#include "tensorflow/lite/schema/schema_generated.h"\n'
        f"{platform_include}"
        "\n"
        "#include <string.h>\n"
        f"{log_tag}"
        "\n"
        "static uint8_t tensor_arena[TENSOR_ARENA_SIZE];\n"
        "static tflite::MicroInterpreter* interpreter = nullptr;\n"
        "static TfLiteTensor* input_tensor = nullptr;\n"
        "static TfLiteTensor* output_tensor = nullptr;\n"
        "\n"
        "int model_init(void) {\n"
        "    tflite::InitializeTarget();\n"
        "\n"
        "    const tflite::Model* model = tflite::GetModel(model_tflite);\n"
        "    if (model == nullptr) {\n"
        f"{log_init_fail}"
        "        return -1;\n"
        "    }\n"
        "\n"
        "    static tflite::MicroMutableOpResolver<10> resolver;\n"
        "    /* Add only the ops your model needs. */\n"
        "    resolver.AddFullyConnected();\n"
        "    resolver.AddRelu();\n"
        "    resolver.AddSoftmax();\n"
        "    resolver.AddReshape();\n"
        "    resolver.AddQuantize();\n"
        "    resolver.AddDequantize();\n"
        "    resolver.AddConv2D();\n"
        "    resolver.AddDepthwiseConv2D();\n"
        "    resolver.AddAveragePool2D();\n"
        "    resolver.AddMaxPool2D();\n"
        "\n"
        "    static tflite::MicroInterpreter static_interpreter(\n"
        "        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);\n"
        "    interpreter = &static_interpreter;\n"
        "\n"
        "    if (interpreter->AllocateTensors() != kTfLiteOk) {\n"
        f"{log_init_fail}"
        "        return -1;\n"
        "    }\n"
        "\n"
        "    input_tensor = interpreter->input(0);\n"
        "    output_tensor = interpreter->output(0);\n"
        f"{log_init_ok}"
        "    return 0;\n"
        "}\n"
        "\n"
        f"int model_run(const {in_ctype}* input_data, {out_ctype}* output_data) {{\n"
        f"    memcpy(input_tensor->data.raw, input_data, "
        f"MODEL_INPUT_SIZE * sizeof({in_ctype}));\n"
        "\n"
        "    if (interpreter->Invoke() != kTfLiteOk) {\n"
        "        return -1;\n"
        "    }\n"
        "\n"
        f"    memcpy(output_data, output_tensor->data.raw, "
        f"MODEL_OUTPUT_SIZE * sizeof({out_ctype}));\n"
        "    return 0;\n"
        "}\n"
        "\n"
        "void model_free(void) {\n"
        "    interpreter = nullptr;\n"
        "    input_tensor = nullptr;\n"
        "    output_tensor = nullptr;\n"
        "}\n"
    )
    source_path = out / "inference.c"
    source_path.write_text(source)

    return [
        GeneratedFile(str(header_path), "Inference wrapper header"),
        GeneratedFile(str(source_path), "Inference wrapper source"),
    ]
