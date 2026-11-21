#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <variant>
#include <functional>
#include <memory>

namespace mobilellm {

/**
 * @brief Standardized status codes across the MobileLLM Engine.
 */
enum class StatusCode : int32_t {
    OK = 0,
    INVALID_ARGUMENT = 1,
    FILE_NOT_FOUND = 2,
    UNSUPPORTED_FORMAT = 3,
    OUT_OF_MEMORY = 4,
    BACKEND_INITIALIZATION_FAILED = 5,
    BACKEND_EXECUTION_ERROR = 6,
    SESSION_NOT_FOUND = 7,
    MODEL_NOT_LOADED = 8,
    CANCELLED = 9,
    CONTEXT_EXCEEDED = 10,
    TIMEOUT = 11,
    NOT_IMPLEMENTED = 12,
    INTERNAL_ERROR = 99
};

/**
 * @brief Represents the result of an operation, containing a code and an optional error message.
 */
struct Status {
    StatusCode code{StatusCode::OK};
    std::string message;

    bool is_ok() const noexcept { return code == StatusCode::OK; }
    explicit operator bool() const noexcept { return is_ok(); }

    static Status ok() { return {StatusCode::OK, ""}; }
    static Status error(StatusCode c, std::string msg) { return {c, std::move(msg)}; }
};

/**
 * @brief Type-safe result container returning either a valid value or an error Status.
 */
template <typename T>
class Result {
public:
    Result(T value) : data_(std::move(value)) {}
    Result(Status status) : data_(std::move(status)) {}
    Result(StatusCode code, std::string msg) : data_(Status::error(code, std::move(msg))) {}

    bool is_ok() const noexcept { return std::holds_alternative<T>(data_); }
    explicit operator bool() const noexcept { return is_ok(); }

    const T& value() const { return std::get<T>(data_); }
    T& value() { return std::get<T>(data_); }

    const Status& status() const {
        if (is_ok()) {
            static const Status s_ok = Status::ok();
            return s_ok;
        }
        return std::get<Status>(data_);
    }

private:
    std::variant<T, Status> data_;
};

// Opaque handles for runtime objects
using ModelHandle = uint64_t;
using SessionHandle = uint64_t;
using GenerationHandle = uint64_t;

constexpr ModelHandle INVALID_MODEL_HANDLE = 0;
constexpr SessionHandle INVALID_SESSION_HANDLE = 0;
constexpr GenerationHandle INVALID_GENERATION_HANDLE = 0;

/**
 * @brief Compute backend target selection.
 */
enum class BackendType : int32_t {
    AUTO = 0,
    CPU_NEON = 1,
    VULKAN = 2,
    VENDOR_NPU = 3,
    OPENCL = 4
};

/**
 * @brief Configuration parameters for loading a model into memory.
 */
struct ModelLoadConfig {
    std::string model_path;
    BackendType preferred_backend{BackendType::AUTO};
    uint32_t context_length{2048};
    uint32_t n_threads{4};
    bool use_mmap{true};
    bool use_mlock{false};
    float memory_budget_mb{0.0f}; // 0 = auto-detect based on device RAM
    int32_t n_gpu_layers{-1};     // -1 = auto (all for GPU, 0 for CPU), >= 0 = explicit layer count
};

/**
 * @brief Configuration parameters for an active conversation session.
 */
struct SessionConfig {
    uint32_t max_context_tokens{2048};
    bool enable_prefix_caching{true};
    uint32_t n_batch{512};
    uint32_t n_ubatch{0}; // 0 = auto (512 on CPU, 256 on Vulkan)
    int32_t flash_attn{-1}; // -1 = auto (true on CPU, false on Vulkan unless overridden), 0 = force false, 1 = force true
};

/**
 * @brief Token sampling hyperparameters.
 */
struct SamplingParams {
    float temperature{0.7f};
    float top_p{0.9f};
    int32_t top_k{40};
    float min_p{0.05f};
    float repetition_penalty{1.1f};
    uint32_t max_tokens{512};
    std::vector<std::string> stop_sequences;
};

/**
 * @brief Generation request payload.
 */
struct GenerationRequest {
    std::string prompt;
    SamplingParams sampling;
};

/**
 * @brief Individual token chunk emitted during streaming generation.
 */
struct TokenChunk {
    std::string text;
    int32_t token_id{0};
    bool is_special{false};
    float logprob{0.0f};
};

/**
 * @brief Optional out-parameter describing how a prompt was evaluated.
 *
 * prefix_cached_tokens == prompt_tokens means the entire prompt was already resident in the KV cache
 * and no token was decoded. Callers MUST treat that as a cache hit and avoid reporting a prefill
 * throughput derived from a near-zero prefill duration.
 */
struct PromptEvalInfo {
    uint32_t prompt_tokens{0};
    uint32_t prefix_cached_tokens{0};
    uint32_t evaluated_tokens{0};
    bool prefix_cache_full_hit{false}; // entire prompt was resident; only a logits refresh decode ran
    double backend_eval_ms{0.0}; // llama.cpp perf counter delta (graph compute only)
    int32_t eval_calls{0};
    double tokenize_ms{0.0};
};

/**
 * @brief Optional out-parameter decomposing one decode iteration.
 *
 * The time is split so that "CPU waiting for GPU" (logits_wait_ms) can be distinguished from
 * "CPU preparing/submitting work" (decode_dispatch_ms) and from host-side sampling (sample_ms).
 * decode_dispatch_ms is asynchronous submit cost: it does not include GPU execution.
 */
struct TokenTiming {
    double logits_wait_ms{0.0};
    double sample_ms{0.0};
    double decode_dispatch_ms{0.0};
};

/**
 * @brief Structured per-generation diagnostics (Agent 14).
 *
 * Populated from monotonic clock measurements plus llama.cpp's own perf counters and log lines.
 * Serialized to JSON by the C API / JNI bridge. Never dumped per token unless verbose logging is on.
 */
struct GenerationDiagnostics {
    // Identity
    std::string backend_requested{"cpu_neon"};
    std::string backend_selected{"cpu_neon"};
    std::string gpu_device;
    std::string gpu_vendor;
    std::string gpu_driver;
    std::string vulkan_gemv_path; // "mat_vec" | "mat_mat" | "" when not applicable
    std::string fallback_reason;  // Reason for automatic fallback to CPU_NEON

    // Offload / graph shape (parsed from llama.cpp logs, not from config intent)
    int32_t model_layers{0};
    int32_t gpu_layers_requested{0};
    int32_t gpu_layers_actual{0};
    int32_t graph_nodes{0};
    int32_t graph_splits_pp{0};
    int32_t graph_splits_tg{0};
    bool gpu_offload_verified{false};

    // Stage timings (monotonic)
    double model_load_ms{0.0};
    double session_create_ms{0.0};
    double prefill_ms{0.0};
    double decode_ms{0.0};
    double decode_compute_ms{0.0};
    double decode_logits_wait_ms{0.0};
    double decode_dispatch_ms{0.0};
    double decode_sample_ms{0.0};
    double callback_ms{0.0};
    double ttft_ms{0.0};

    // Tokens / KV cache
    uint32_t prompt_tokens{0};
    uint32_t generated_tokens{0};
    uint32_t prefix_cached_tokens{0};
    bool prefix_cache_hit{false};
    uint32_t context_capacity{0};
    uint32_t kv_used_tokens{0};
    uint64_t kv_cache_bytes{0};

    // Per-token latency distribution (decode iterations, excludes callback)
    double token_latency_p50_ms{0.0};
    double token_latency_p95_ms{0.0};
    double token_latency_max_ms{0.0};

    // Environment
    uint32_t cpu_threads{0};
    uint32_t cpu_cores{0};
    double current_rss_mb{0.0};
    double peak_rss_mb{0.0};
    int32_t thermal_status{-1}; // -1 = unknown / unsupported
    std::string benchmark_status{"idle"};
    std::string error;
};

/**
 * @brief Telemetry metrics collected over a generation run.
 */
struct GenerationMetrics {
    uint32_t prompt_tokens{0};
    uint32_t completion_tokens{0};
    double time_to_first_token_ms{0.0}; // Legacy public contract: End-to-end TTFT (matches end_to_end_ttft_ms)
    double decode_tokens_per_second{0.0};
    double peak_memory_mb{0.0};
    double prefill_tokens_per_second{0.0};
    double prefill_time_ms{0.0};
    double decode_time_ms{0.0};
    double total_duration_ms{0.0};
    uint32_t threads_used{4};

    // --- Decomposed TTFT and Timing Metrics (PERF-01) ---
    double queue_wait_time_ms{0.0};           // Duration in queue prior to admission: t_dequeue - t_enqueue
    double first_token_sampling_time_ms{0.0}; // Sampling execution time for token 1: t_sample_end - t_sample_start
    double pure_inference_ttft_ms{0.0};       // Pure hardware compute TTFT: prefill_time_ms + first_token_sampling_time_ms
    double end_to_end_ttft_ms{0.0};           // Total end-to-end user-perceived TTFT: t_cb_end - t_enqueue

    // --- Inference-execution vs end-to-end decomposition (Agent 10E) ---
    // decode_time_ms is the end-to-end decode window (prefill_end -> request retirement) and therefore
    // includes the streaming token callback (JNI/UI delivery) for every token. decode_compute_time_ms
    // sums only the backend sample_next_token() durations, isolating actual inference execution.
    double decode_compute_time_ms{0.0};
    double pure_decode_tokens_per_second{0.0}; // completion_tokens / (decode_compute_time_ms / 1000)

    // --- Agent 14: finer decomposition of the decode window ---
    // decode_compute_time_ms == decode_logits_wait_ms + decode_cpu_dispatch_ms + decode_sample_ms
    // + a small residual (session lookup, mutex acquisition, logits buffer copy).
    double decode_logits_wait_ms{0.0};   // blocked on GPU completion (vk fence / backend sync)
    double decode_cpu_dispatch_ms{0.0};  // llama_decode host-side graph build + submit
    double decode_sample_ms{0.0};        // token sampling on host
    double decode_callback_ms{0.0};      // time spent inside the streaming token callback
    double prompt_tokenize_ms{0.0};      // tokenizer only
    double prompt_backend_eval_ms{0.0};  // llama.cpp t_p_eval delta (graph compute only)
    uint32_t prefix_cached_tokens{0};    // prompt tokens served from KV cache without re-eval
    bool prefix_cache_hit{false};        // entire prompt was cached (no decode performed)

    GenerationDiagnostics diagnostics;
};

/**
 * @brief Device hardware capabilities detected at runtime.
 */
struct HardwareProfile {
    std::string device_name;
    std::string soc_platform;
    uint64_t total_ram_bytes{0};
    uint64_t available_ram_bytes{0};
    uint32_t cpu_core_count{0};
    bool neon_supported{false};
    bool vulkan_supported{false};
    std::string vulkan_device_name;
    bool opencl_supported{false};
    std::string opencl_device_name;
    bool npu_supported{false};
};

/**
 * @brief Runtime process memory measurements.
 */
struct ProcessMemoryTelemetry {
    uint64_t current_rss_bytes{0};
    uint64_t peak_rss_bytes{0};
    uint64_t vmsize_bytes{0};
    uint64_t model_allocated_bytes{0};
    uint64_t kv_allocated_bytes{0};
};

/**
 * @brief Telemetry for a session's context window.
 */
struct ContextTelemetry {
    uint32_t context_capacity{0};
    uint32_t used_tokens{0};
    uint32_t remaining_capacity{0};
    bool is_available{false};
};

/**
 * @brief Runtime telemetry & health stats.
 */
struct EngineMetrics {
    uint64_t active_models{0};
    uint64_t active_sessions{0};
    uint64_t resident_memory_bytes{0};
    uint64_t kv_cache_allocated_bytes{0};
    uint64_t kv_cache_free_bytes{0};
};

// Callback signatures
using TokenCallback = std::function<bool(const TokenChunk&)>; // Return false to abort generation
using CompletionCallback = std::function<void(const Status&, const GenerationMetrics&)>;

} // namespace mobilellm
