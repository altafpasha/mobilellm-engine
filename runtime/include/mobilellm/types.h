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
    VENDOR_NPU = 3
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
};

/**
 * @brief Configuration parameters for an active conversation session.
 */
struct SessionConfig {
    uint32_t max_context_tokens{2048};
    bool enable_prefix_caching{true};
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
 * @brief Telemetry metrics collected over a generation run.
 */
struct GenerationMetrics {
    uint32_t prompt_tokens{0};
    uint32_t completion_tokens{0};
    double time_to_first_token_ms{0.0};
    double decode_tokens_per_second{0.0};
    double peak_memory_mb{0.0};
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
    bool npu_supported{false};
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
