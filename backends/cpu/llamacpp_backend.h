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
        std::vector<int32_t>& out_tokens
    ) noexcept override;

    Result<int32_t> sample_next_token(
        BackendSessionHandle session,
        const SamplingParams& params
    ) noexcept override;

    std::string decode_token(BackendSessionHandle session, int32_t token_id) noexcept override;
    bool is_eos_token(BackendSessionHandle session, int32_t token_id) const noexcept override;

    BackendType type() const noexcept override { return BackendType::CPU_NEON; }
    std::string name() const noexcept override { return "LlamaCpp_CPU_NEON"; }

private:
    struct LoadedModel {
        BackendModelHandle handle;
        std::string path;
        llama_model* raw_model{nullptr};
        uint32_t context_length{2048};
        uint32_t n_vocab{0};
        uint32_t n_threads{4};
    };

    struct SessionContext {
        BackendSessionHandle handle;
        BackendModelHandle model_handle;
        llama_context* raw_ctx{nullptr};
        std::vector<int32_t> history_tokens;
        int32_t last_n_tokens{0};
    };

    mutable std::mutex mutex_;
    std::atomic<uint64_t> next_model_id_{1};
    std::atomic<uint64_t> next_session_id_{1};

    std::unordered_map<BackendModelHandle, std::shared_ptr<LoadedModel>> models_;
    std::unordered_map<BackendSessionHandle, std::shared_ptr<SessionContext>> sessions_;

    Sampler sampler_;
    [[maybe_unused]] bool initialized_{false};
};

} // namespace mobilellm
