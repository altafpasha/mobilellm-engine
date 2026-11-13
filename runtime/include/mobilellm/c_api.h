#ifndef MOBILELLM_C_API_H
#define MOBILELLM_C_API_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#if defined(_WIN32)
  #if defined(MOBILELLM_BUILD_SHARED)
    #define MOBILELLM_EXPORT __declspec(dllexport)
  #else
    #define MOBILELLM_EXPORT __declspec(dllimport)
  #endif
#else
  #define MOBILELLM_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle definitions
typedef uint64_t mllm_engine_t;
typedef uint64_t mllm_model_t;
typedef uint64_t mllm_session_t;
typedef uint64_t mllm_gen_handle_t;

/**
 * @brief C-compatible status structure.
 *
 * Error Message Ownership & Lifetime Contract:
 * - When code == 0 (success), message points to a static empty string ("").
 * - When code != 0 (error), message points to a null-terminated UTF-8 string describing
 *   the failure. The memory referenced by message is thread-local and remains valid on
 *   the calling thread until the next MobileLLM C-API invocation on that same thread.
 * - Callers requiring error message retention across subsequent MobileLLM API calls
 *   or across thread boundaries must copy the string (e.g. strdup() or equivalent).
 * - Concurrent calls from distinct threads access independent thread-local storage
 *   and do not overwrite each other's error messages.
 */
typedef struct {
    int32_t code;
    const char* message;
} mllm_status_t;

/**
 * @brief C-compatible model loading configuration.
 */
typedef struct {
    const char* model_path;
    int32_t backend; // 0=auto, 1=cpu_neon, 2=vulkan, 3=npu
    uint32_t context_length;
    uint32_t n_threads;
    bool use_mmap;
    bool use_mlock;
    float memory_budget_mb;
} mllm_model_config_t;

/**
 * @brief C-compatible session configuration.
 */
typedef struct {
    uint32_t max_context_tokens;
    bool enable_prefix_caching;
} mllm_session_config_t;

/**
 * @brief C-compatible sampling parameters.
 */
typedef struct {
    float temperature;
    float top_p;
    int32_t top_k;
    float min_p;
    float repetition_penalty;
    uint32_t max_tokens;
} mllm_sampling_params_t;

/**
 * @brief Streaming token chunk passed to token callback.
 */
typedef struct {
    const char* text;
    int32_t token_id;
    bool is_special;
    float logprob;
} mllm_token_chunk_t;

/**
 * @brief Telemetry metrics passed to completion callback.
 */
typedef struct {
    uint32_t prompt_tokens;
    uint32_t completion_tokens;
    double ttft_ms;
    double tps;
    double peak_rss_mb;
    double prompt_tps;
    double prefill_ms;
    double decode_ms;
    double total_ms;
} mllm_metrics_t;

/**
 * @brief Process resident and virtual memory metrics.
 */
typedef struct {
    uint64_t current_rss_bytes;
    uint64_t peak_rss_bytes;
    uint64_t vmsize_bytes;
    uint64_t model_allocated_bytes;
    uint64_t kv_allocated_bytes;
} mllm_process_memory_t;

/**
 * @brief Session context window capacity and token usage.
 */
typedef struct {
    uint32_t context_capacity;
    uint32_t used_tokens;
    uint32_t remaining_capacity;
    bool is_available;
} mllm_context_telemetry_t;

// Callback function signatures
typedef bool (*mllm_token_callback_fn)(const mllm_token_chunk_t* chunk, void* user_data);
typedef void (*mllm_complete_callback_fn)(mllm_status_t status, const mllm_metrics_t* metrics, void* user_data);

// Engine Lifecycle
MOBILELLM_EXPORT mllm_status_t mllm_engine_create(mllm_engine_t* out_engine);
MOBILELLM_EXPORT mllm_status_t mllm_engine_destroy(mllm_engine_t engine);

// Model Management
MOBILELLM_EXPORT mllm_status_t mllm_model_load(mllm_engine_t engine, const mllm_model_config_t* config, mllm_model_t* out_model);
MOBILELLM_EXPORT mllm_status_t mllm_model_unload(mllm_engine_t engine, mllm_model_t model);
MOBILELLM_EXPORT mllm_status_t mllm_model_inspect(const char* model_path, char* json_buffer, size_t max_len);

// Session Management
MOBILELLM_EXPORT mllm_status_t mllm_session_create(mllm_engine_t engine, mllm_model_t model, const mllm_session_config_t* config, mllm_session_t* out_session);
MOBILELLM_EXPORT mllm_status_t mllm_session_destroy(mllm_engine_t engine, mllm_session_t session);

// Generation Operations
MOBILELLM_EXPORT mllm_status_t mllm_generate(
    mllm_engine_t engine,
    mllm_session_t session,
    const char* prompt,
    const mllm_sampling_params_t* sampling,
    mllm_token_callback_fn token_cb,
    mllm_complete_callback_fn complete_cb,
    void* user_data,
    mllm_gen_handle_t* out_handle
);

MOBILELLM_EXPORT mllm_status_t mllm_cancel(mllm_engine_t engine, mllm_gen_handle_t handle);

// Hardware Detection & System Telemetry
MOBILELLM_EXPORT mllm_status_t mllm_get_version(char* buffer, size_t max_len);
MOBILELLM_EXPORT mllm_status_t mllm_detect_hardware(char* json_buffer, size_t max_len);
MOBILELLM_EXPORT mllm_status_t mllm_get_process_memory(mllm_engine_t engine, mllm_process_memory_t* out_memory);
MOBILELLM_EXPORT mllm_status_t mllm_get_session_telemetry(mllm_engine_t engine, mllm_session_t session, mllm_context_telemetry_t* out_telemetry);
MOBILELLM_EXPORT mllm_status_t mllm_set_session_threads(mllm_engine_t engine, mllm_session_t session, uint32_t n_threads);
MOBILELLM_EXPORT const char* mllm_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // MOBILELLM_C_API_H
