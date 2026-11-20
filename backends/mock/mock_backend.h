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
    Status set_threads(BackendSessionHandle session, uint32_t n_threads) noexcept override;
    Status reset_session(BackendSessionHandle session) noexcept override;
    Result<std::vector<float>> get_last_logits(BackendSessionHandle session) const noexcept override;

    BackendType type() const noexcept override { return BackendType::CPU_NEON; }
    std::string name() const noexcept override { return "MockBackend"; }

    // Test control hooks
    void set_simulate_load_failure(bool fail) { simulate_load_failure_ = fail; }
    void set_simulate_oom(bool oom) { simulate_oom_ = oom; }
    void set_simulate_eval_failure(bool fail) { simulate_eval_failure_ = fail; }

    // Deterministic per-token compute cost. The mock's real work is negligible (~microseconds),
    // which makes wall-clock throughput measurements dominated by scheduler jitter. Setting a
    // simulated latency gives timing-sensitive tests a controlled, measurable signal.
    void set_simulated_token_latency_us(uint32_t us) noexcept { token_latency_us_ = us; }
    static void set_default_simulated_token_latency_us(uint32_t us) noexcept {
        default_token_latency_us_.store(us);
    }

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
        std::unique_ptr<Sampler> sampler;
        std::vector<float> reusable_logits; // PERF-03: persistent reusable logits buffer for mock
    };

    mutable std::mutex mutex_;
    std::atomic<uint64_t> next_model_id_{1};
    std::atomic<uint64_t> next_session_id_{1};
    std::unordered_map<BackendModelHandle, MockModel> models_;
    std::unordered_map<BackendSessionHandle, MockSession> sessions_;

    bool simulate_load_failure_{false};
    bool simulate_oom_{false};
    bool simulate_eval_failure_{false};
    uint32_t token_latency_us_{0};
    static std::atomic<uint32_t> default_token_latency_us_;
};

class MockVulkanBackend : public MockBackend {
public:
    MockVulkanBackend() = default;
    ~MockVulkanBackend() override = default;

    Status initialize(const BackendConfig& config) noexcept override {
        if (simulate_init_failure_) {
            return Status::error(StatusCode::BACKEND_INITIALIZATION_FAILED, "Simulated Vulkan initialization failure");
        }
        return MockBackend::initialize(config);
    }

    BackendType type() const noexcept override { return BackendType::VULKAN; }
    std::string name() const noexcept override { return "MockVulkanBackend"; }

    void set_simulate_init_failure(bool fail) { simulate_init_failure_ = fail; }

private:
    bool simulate_init_failure_{false};
};

} // namespace mobilellm

