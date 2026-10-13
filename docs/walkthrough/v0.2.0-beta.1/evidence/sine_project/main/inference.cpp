#include "inference.h"
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
    static tflite::MicroMutableOpResolver<1> resolver;
        if (resolver.AddFullyConnected() != kTfLiteOk) return MODEL_INVALID_MODEL;
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
