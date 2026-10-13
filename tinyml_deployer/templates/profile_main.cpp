#include "inference.h"
#include "validation_data.h"
#include "esp_timer.h"
#include "esp_system.h"
#include "esp_app_desc.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cinttypes>

static void run_profile(const char* run_id) {
    static @OUTPUT_CTYPE@ output[OUTPUT_ELEMENTS];
    const int init_status = model_init();
    if (init_status != MODEL_OK) {
        printf("TINYML {\"event\":\"error\",\"run_id\":\"%s\",\"status\":%d}\n", run_id, init_status);
        return;
    }
    printf("TINYML {\"event\":\"begin\",\"run_id\":\"%s\",\"model_sha256\":\"@MODEL_HASH@\","
           "\"dataset_sha256\":\"@DATASET_HASH@\",\"target\":\"@TARGET_NAME@\",\"idf_version\":\"%s\"}\n",
           run_id, esp_app_get_description()->idf_ver);
    for (size_t sample = 0; sample < VALIDATION_SAMPLES; ++sample) {
        // Warm-up is excluded from latency measurements.
        int status = model_run(inputs[sample], MODEL_INPUT_BYTES, output, sizeof(output));
        if (status != MODEL_OK) {
            printf("TINYML {\"event\":\"error\",\"run_id\":\"%s\",\"status\":%d}\n", run_id, status);
            return;
        }
        for (size_t repetition = 0; repetition < PROFILE_REPETITIONS; ++repetition) {
            const int64_t start = esp_timer_get_time();
            status = model_run(inputs[sample], MODEL_INPUT_BYTES, output, sizeof(output));
            const int64_t elapsed = esp_timer_get_time() - start;
            double maximum = 0.0;
            bool finite = true;
            for (size_t i = 0; i < OUTPUT_ELEMENTS; ++i) {
                const double value = (static_cast<double>(output[i]) - @OUTPUT_ZERO@) * @OUTPUT_SCALE@;
                finite = finite && std::isfinite(value);
                maximum = fmax(maximum, fabs(value - expected[sample][i]));
            }
            if (status != MODEL_OK || !finite) {
                printf("TINYML {\"event\":\"error\",\"run_id\":\"%s\",\"status\":%d}\n", run_id, status);
                return;
            }
            printf("TINYML {\"event\":\"sample\",\"run_id\":\"%s\",\"sample\":%u,\"repetition\":%u,"
                   "\"latency_us\":%" PRId64 ",\"max_abs_error\":%.9g,\"arena_used_bytes\":%u,"
                   "\"min_free_heap_bytes\":%u}\n", run_id, static_cast<unsigned>(sample),
                   static_cast<unsigned>(repetition), elapsed, maximum,
                   static_cast<unsigned>(model_arena_used_bytes()),
                   static_cast<unsigned>(esp_get_minimum_free_heap_size()));
            // Yield outside the measured interval so profiling cannot starve the watchdog.
            vTaskDelay(1);
        }
    }
    printf("TINYML {\"event\":\"done\",\"run_id\":\"%s\"}\n", run_id);
    fflush(stdout);
    model_free();
}

extern "C" void app_main(void) {
    char line[80];
    size_t used = 0;
    bool overflow = false;
    while (true) {
        const int ch = getchar();
        if (ch == EOF) {
            clearerr(stdin);
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }
        if (ch == '\r') continue;
        if (ch == '\n') {
            line[used] = '\0';
            if (!overflow && used == 36 && strncmp(line, "run ", 4) == 0 &&
                strspn(line + 4, "0123456789abcdef") == 32) {
                run_profile(line + 4);
            }
            used = 0;
            overflow = false;
        } else if (used < sizeof(line) - 1) {
            line[used++] = static_cast<char>(ch);
        } else {
            overflow = true;
        }
    }
}
