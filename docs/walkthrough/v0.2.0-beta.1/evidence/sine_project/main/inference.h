#ifndef INFERENCE_H
#define INFERENCE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define MODEL_INPUT_BYTES 1
#define MODEL_OUTPUT_BYTES 1
#define TENSOR_ARENA_SIZE 65536
/* All calls require external serialization. Buffers contain native tensor bytes.
 * See deployment.json for shapes, dtypes and quantization scale/zero point. */
enum ModelStatus {
    MODEL_OK = 0, MODEL_INVALID_MODEL = -1, MODEL_ALLOCATION_FAILED = -2,
    MODEL_NOT_READY = -3, MODEL_INVALID_BUFFER = -4, MODEL_INVOKE_FAILED = -5
};
int model_init(void);
int model_run(const void* input_data, size_t input_bytes,
              void* output_data, size_t output_bytes);
size_t model_arena_used_bytes(void);
/* Disable calls to model_run. Static storage remains allocated and reusable. */
void model_free(void);
#ifdef __cplusplus
}
#endif
#endif
