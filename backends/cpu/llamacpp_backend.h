#pragma once

#include "backends/common/backend.h"
#include "runtime/src/sampling/sampler.h"
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <memory>

// Forward declarations of llama.cpp structs to avoid leaking headers
struct llama_model;
struct llama_context;
struct ggml_threadpool;

namespace mobilellm {

class LlamaCppBackend : public IBackend {
public:
    LlamaCppBackend();
    ~LlamaCppBackend() override;

    Status initialize(const BackendConfig& config) noexcept override;
    Status shutdown() noexcept override;

    Result<BackendModelHandle> load_model(const ModelLoadConfig& config) noexcept override;
    Status unload_model(BackendModelHandle model) noexcept override;

    Result<BackendSessionHandle> create_session(BackendModelHandle model, const SessionConfig& config) noexcept override;
    Status destroy_session(BackendSessionHandle session) noexcept override;

    Status evaluate_prompt(
        BackendSessionHandle session,
        const std::string& prompt,
        std::vector<int32_t>& out_tokens,
        PromptEvalInfo* out_info = nullptr
    ) noexcept override;

    Result<int32_t> sample_next_token(
        BackendSessionHandle session,
        const SamplingParams& params,
        TokenTiming* out_timing = nullptr
    ) noexcept override;

    GenerationDiagnostics describe_session(BackendSessionHandle session) const noexcept override;

    std::string decode_token(BackendSessionHandle session, int32_t token_id) noexcept override;
    bool is_eos_token(BackendSessionHandle session, int32_t token_id) const noexcept override;

    SessionContextTelemetry get_session_telemetry(BackendSessionHandle session) const noexcept override;
    Status set_threads(BackendSessionHandle session, uint32_t n_threads) noexcept override;
    Status reset_session(BackendSessionHandle session) noexcept override;
    Result<std::vector<float>> get_last_logits(BackendSessionHandle session) const noexcept override;

    static bool is_gpu_supported() noexcept;

    /**
     * @brief Enumerates the ggml backend registry for a GPU adapter description.
     *
     * Static so hardware can be described before a model (or backend instance) exists.
     * Returns an empty string when no GPU backend is registered.
     */
    static std::string probe_gpu_device_description() noexcept;

    std::string gpu_device_description() const noexcept override;

    BackendType type() const noexcept override { return current_backend_; }
    void set_backend_type(BackendType type) noexcept { current_backend_.store(type); }
    std::string name() const noexcept override {
        return (current_backend_ == BackendType::VULKAN) ? "LlamaCpp_Vulkan_GPU" : "LlamaCpp_CPU_NEON";
    }

private:
    std::atomic<BackendType> current_backend_{BackendType::CPU_NEON};
    struct LoadedModel {
        BackendModelHandle handle;
        std::string path;
        llama_model* raw_model{nullptr};
        uint32_t context_length{2048};
        uint32_t n_vocab{0};
        uint32_t n_threads{4};
        int32_t requested_gpu_layers{0};
        int32_t actual_gpu_layers{0};
        BackendType backend_type{BackendType::CPU_NEON};
        double load_ms{0.0};
        // Facts captured from ggml device enumeration and llama.cpp load logs (not config intent).
        std::string gpu_device;
        std::string gpu_vendor;
        std::string gpu_driver;
        int32_t model_layers{0};
        int32_t offloaded_layers{-1};
        int32_t total_offloadable_layers{-1};
        std::string fallback_reason;
    };

    struct SessionContext {
        BackendSessionHandle handle{0};
        BackendModelHandle model_handle{0};
        llama_context* raw_ctx{nullptr};
        struct ggml_threadpool* threadpool{nullptr};
        std::vector<int32_t> history_tokens;
        int32_t last_n_tokens{0};
        std::unique_ptr<Sampler> sampler;
        std::vector<float> reusable_logits; // PERF-03: persistent reusable logits buffer per session
        bool enable_prefix_caching{true};
        double model_load_ms{0.0};
        uint32_t n_threads{4};
        uint32_t kv_capacity_tokens{0};
        int32_t graph_nodes{0};
        int32_t graph_splits_pp{0};
        int32_t graph_splits_tg{0};
        std::string vulkan_matmul_path;
        std::string gpu_device;
        std::string gpu_vendor;
        std::string gpu_driver;
        int32_t model_layers{0};
        int32_t offloaded_layers{-1};
        int32_t total_offloadable_layers{-1};
        std::string fallback_reason;
        mutable std::mutex session_mutex;
    };

    mutable std::mutex mutex_;
    std::atomic<uint64_t> next_model_id_{1};
    std::atomic<uint64_t> next_session_id_{1};

    std::unordered_map<BackendModelHandle, std::shared_ptr<LoadedModel>> models_;
    std::unordered_map<BackendSessionHandle, std::shared_ptr<SessionContext>> sessions_;

    [[maybe_unused]] bool initialized_{false};
};

} // namespace mobilellm
