"""Generate a C-callable, checked C++ TFLite Micro inference wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tinyml_deployer.analyzer import ModelAnalysis
from tinyml_deployer.operators import RESOLVER_METHODS


@dataclass
class GeneratedFile:
    path: str
    description: str


def _bytes_to_c_array(data: bytes, var_name: str) -> str:
    body = "\n".join(
        "    " + ", ".join(f"0x{b:02x}" for b in data[i : i + 12]) + ","
        for i in range(0, len(data), 12)
    )
    return (
        f"alignas(16) const unsigned char {var_name}[] = {{\n{body}\n}};\n"
        f"const unsigned int {var_name}_len = {len(data)};\n"
    )


def generate_model_data(model_path: str, output_dir: str) -> list[GeneratedFile]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "model_data.h").write_text("""#ifndef MODEL_DATA_H
#define MODEL_DATA_H
#ifdef __cplusplus
extern "C" {
#endif
extern const unsigned char model_tflite[];
extern const unsigned int model_tflite_len;
#ifdef __cplusplus
}
#endif
#endif
""")
    source = '#include "model_data.h"\n' + _bytes_to_c_array(
        Path(model_path).read_bytes(), "model_tflite"
    )
    (out / "model_data.cpp").write_text(source)
    return [
        GeneratedFile(str(out / "model_data.h"), "Model data declarations"),
        GeneratedFile(str(out / "model_data.cpp"), "Aligned model bytes"),
    ]


def generate_inference_wrapper(
    analysis: ModelAnalysis,
    target_framework: str,
    output_dir: str,
    arena_size: int | None = None,
) -> list[GeneratedFile]:
    """Generate a single-input, single-output wrapper; caller serializes access."""
    if len(analysis.inputs) != 1 or len(analysis.outputs) != 1:
        raise ValueError(
            "Firmware generation requires exactly one input and one output."
        )
    if type(arena_size) is not int or arena_size <= 0:
        raise ValueError("Specify a positive tensor arena size in bytes.")
    if not analysis.compatible:
        raise ValueError(
            "Model failed deployment preflight: " + "; ".join(analysis.warnings)
        )
    ops = sorted(set(analysis.operators))
    if not ops or any(op not in RESOLVER_METHODS for op in ops):
        raise ValueError(
            "The model contains an operator without a Micro resolver registration."
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    inp, output = analysis.inputs[0], analysis.outputs[0]
    header = f"""#ifndef INFERENCE_H
#define INFERENCE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {{
#endif
#define MODEL_INPUT_BYTES {inp.size_bytes}
#define MODEL_OUTPUT_BYTES {output.size_bytes}
#define TENSOR_ARENA_SIZE {arena_size}
/* All calls require external serialization. Buffers contain native tensor bytes.
 * See deployment.json for shapes, dtypes and quantization scale/zero point. */
enum ModelStatus {{
    MODEL_OK = 0, MODEL_INVALID_MODEL = -1, MODEL_ALLOCATION_FAILED = -2,
    MODEL_NOT_READY = -3, MODEL_INVALID_BUFFER = -4, MODEL_INVOKE_FAILED = -5
}};
int model_init(void);
int model_run(const void* input_data, size_t input_bytes,
              void* output_data, size_t output_bytes);
size_t model_arena_used_bytes(void);
/* Disable calls to model_run. Static storage remains allocated and reusable. */
void model_free(void);
#ifdef __cplusplus
}}
#endif
#endif
"""
    registrations = "\n".join(
        f"        if (resolver.{RESOLVER_METHODS[op]}() != kTfLiteOk) return MODEL_INVALID_MODEL;"
        for op in ops
    )
    source = """#include "inference.h"
#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include <cstring>

namespace {
alignas(16) uint8_t tensor_arena[TENSOR_ARENA_SIZE];
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;
bool ready = false;

int initialize_once() {
    tflite::InitializeTarget();
    flatbuffers::Verifier verifier(model_tflite, model_tflite_len);
    if (!tflite::VerifyModelBuffer(verifier)) return MODEL_INVALID_MODEL;
    const tflite::Model* model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) return MODEL_INVALID_MODEL;
    static tflite::MicroMutableOpResolver<OP_COUNT> resolver;
REGISTRATIONS
    static tflite::MicroInterpreter instance(model, resolver, tensor_arena, sizeof(tensor_arena));
    if (instance.AllocateTensors() != kTfLiteOk) return MODEL_ALLOCATION_FAILED;
    input_tensor = instance.input(0);
    output_tensor = instance.output(0);
    if (input_tensor == nullptr || output_tensor == nullptr ||
        input_tensor->bytes != MODEL_INPUT_BYTES || output_tensor->bytes != MODEL_OUTPUT_BYTES)
        return MODEL_INVALID_MODEL;
    interpreter = &instance;
    return MODEL_OK;
}
}

int model_init(void) {
    static const int status = initialize_once();
    ready = status == MODEL_OK;
    return status;
}

int model_run(const void* input_data, size_t input_bytes,
              void* output_data, size_t output_bytes) {
    if (!ready) return MODEL_NOT_READY;
    if (input_data == nullptr || output_data == nullptr ||
        input_bytes != MODEL_INPUT_BYTES || output_bytes != MODEL_OUTPUT_BYTES)
        return MODEL_INVALID_BUFFER;
    std::memcpy(input_tensor->data.raw, input_data, input_bytes);
    if (interpreter->Invoke() != kTfLiteOk) return MODEL_INVOKE_FAILED;
    std::memcpy(output_data, output_tensor->data.raw, output_bytes);
    return MODEL_OK;
}

size_t model_arena_used_bytes(void) {
    return ready ? interpreter->arena_used_bytes() : 0;
}

void model_free(void) {
    ready = false;
}
""".replace("OP_COUNT", str(len(ops))).replace("REGISTRATIONS", registrations)
    (out / "inference.h").write_text(header)
    (out / "inference.cpp").write_text(source)
    return [
        GeneratedFile(str(out / "inference.h"), "Checked C API"),
        GeneratedFile(str(out / "inference.cpp"), "C++ Micro interpreter wrapper"),
    ]
