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
} mllm_metrics_t;

// Callback function signatures
typedef bool (*mllm_token_callback_fn)(const mllm_token_chunk_t* chunk, void* user_data);
typedef void (*mllm_complete_callback_fn)(mllm_status_t status, const mllm_metrics_t* metrics, void* user_data);

// Engine Lifecycle
MOBILELLM_EXPORT mllm_status_t mllm_engine_create(mllm_engine_t* out_engine);
MOBILELLM_EXPORT mllm_status_t mllm_engine_destroy(mllm_engine_t engine);

// Model Management
MOBILELLM_EXPORT mllm_status_t mllm_model_load(mllm_engine_t engine, const mllm_model_config_t* config, mllm_model_t* out_model);
MOBILELLM_EXPORT mllm_status_t mllm_model_unload(mllm_engine_t engine, mllm_model_t model);

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

#ifdef __cplusplus
}
#endif

#endif // MOBILELLM_C_API_H
