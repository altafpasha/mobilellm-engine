#pragma once

#include "backends/common/backend.h"
#include "runtime/src/sampling/sampler.h"
#include <unordered_map>
#include <atomic>
#include <mutex>
#include <string>

namespace mobilellm {

class MockBackend : public IBackend {
public:
    MockBackend();
    ~MockBackend() override = default;

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
    std::string name() const noexcept override { return "MockBackend"; }

    // Test control hooks
    void set_simulate_load_failure(bool fail) { simulate_load_failure_ = fail; }
    void set_simulate_oom(bool oom) { simulate_oom_ = oom; }

private:
    struct MockModel {
        BackendModelHandle handle;
        std::string path;
        uint32_t context_length;
    };

    struct MockSession {
        BackendSessionHandle handle;
        BackendModelHandle model;
        std::vector<int32_t> prompt_tokens;
        std::vector<int32_t> generated_tokens;
        uint32_t step{0};
    };

    std::mutex mutex_;
    std::atomic<uint64_t> next_model_id_{1};
    std::atomic<uint64_t> next_session_id_{1};
    std::unordered_map<BackendModelHandle, MockModel> models_;
    std::unordered_map<BackendSessionHandle, MockSession> sessions_;

    Sampler sampler_;
    bool simulate_load_failure_{false};
    bool simulate_oom_{false};
};

} // namespace mobilellm

