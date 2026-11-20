#include "backends/mock/mock_backend.h"
#include <sstream>
#include <thread>
#include <chrono>

namespace mobilellm {

constexpr int32_t MOCK_EOS_TOKEN = 9999;

std::atomic<uint32_t> MockBackend::default_token_latency_us_{0};

MockBackend::MockBackend() {
    token_latency_us_ = default_token_latency_us_.load();
}

Status MockBackend::initialize(const BackendConfig& /*config*/) noexcept {
    return Status::ok();
}

Status MockBackend::shutdown() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    sessions_.clear();
    models_.clear();
    return Status::ok();
}

Result<BackendModelHandle> MockBackend::load_model(const ModelLoadConfig& config) noexcept {
    if (simulate_oom_) {
        return Status::error(StatusCode::OUT_OF_MEMORY, "Simulated memory allocation failure");
    }
    if (simulate_load_failure_) {
        return Status::error(StatusCode::FILE_NOT_FOUND, "Simulated model loading error");
    }
    if (config.model_path.empty()) {
        return Status::error(StatusCode::INVALID_ARGUMENT, "Model path cannot be empty");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    BackendModelHandle handle = next_model_id_++;
    MockModel model{handle, config.model_path, config.context_length};
    models_[handle] = model;
    return handle;
}

Status MockBackend::unload_model(BackendModelHandle model) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = models_.find(model);
    if (it == models_.end()) {
        return Status::error(StatusCode::MODEL_NOT_LOADED, "Model handle not found");
    }
    models_.erase(it);
    return Status::ok();
}

Result<BackendSessionHandle> MockBackend::create_session(
    BackendModelHandle model,
    const SessionConfig& /*config*/
) noexcept {
    if (simulate_oom_) {
        return Status::error(StatusCode::OUT_OF_MEMORY, "Simulated memory allocation failure in session creation");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (models_.find(model) == models_.end()) {
        return Status::error(StatusCode::MODEL_NOT_LOADED, "Cannot create session for unloaded model");
    }

    BackendSessionHandle session_id = next_session_id_++;
    MockSession session;
    session.handle = session_id;
    session.model = model;
    session.step = 0;
    session.sampler = std::make_unique<Sampler>(42 + static_cast<uint64_t>(session_id));
    session.reusable_logits.resize(257, 0.1f);
    sessions_[session_id] = std::move(session);
    return session_id;
}

Status MockBackend::destroy_session(BackendSessionHandle session) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(session);
    if (it == sessions_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session handle not found");
    }
    sessions_.erase(it);
    return Status::ok();
}

Status MockBackend::evaluate_prompt(
    BackendSessionHandle session,
    const std::string& prompt,
    std::vector<int32_t>& out_tokens,
    PromptEvalInfo* out_info
) noexcept {
    if (out_info) {
        *out_info = PromptEvalInfo{};
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (simulate_eval_failure_) {
        return Status::error(StatusCode::BACKEND_EXECUTION_ERROR, "Simulated prompt evaluation error");
    }

    auto it = sessions_.find(session);
    if (it == sessions_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session not found for prompt evaluation");
    }

    if (prompt.empty()) {
        return Status::error(StatusCode::INVALID_ARGUMENT, "Prompt cannot be empty");
    }

    // Deterministic simple pseudo-tokenization: map each character or word
    out_tokens.clear();
    out_tokens.reserve(prompt.size());
    for (char c : prompt) {
        out_tokens.push_back(static_cast<int32_t>(static_cast<unsigned char>(c)));
    }

    if (out_info) {
        out_info->prompt_tokens = static_cast<uint32_t>(out_tokens.size());
        out_info->evaluated_tokens = static_cast<uint32_t>(out_tokens.size());
        out_info->prefix_cached_tokens = 0;
        out_info->eval_calls = 1;
    }

    it->second.prompt_tokens = out_tokens;
    it->second.generated_tokens.clear();
    it->second.step = 0;
    return Status::ok();
}

Result<int32_t> MockBackend::sample_next_token(
    BackendSessionHandle session,
    const SamplingParams& params,
    TokenTiming* out_timing
) noexcept {
    if (out_timing) {
        *out_timing = TokenTiming{};
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(session);
    if (it == sessions_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session not found for sampling");
    }

    MockSession& s = it->second;
    if (s.step >= params.max_tokens) {
        return MOCK_EOS_TOKEN;
    }

    // PERF-03: Reuse pre-allocated logits buffer for mock sampling
    if (s.reusable_logits.size() != 257) {
        s.reusable_logits.resize(257, 0.1f);
    } else {
        std::fill(s.reusable_logits.begin(), s.reusable_logits.end(), 0.1f);
    }
    // Make letters more likely based on step
    int32_t base_char = 'a' + static_cast<int32_t>(s.step % 26);
    s.reusable_logits[base_char] = 5.0f;
    s.reusable_logits[' '] = 2.0f;

    // EOS condition
    if (s.step >= 50) {
        s.reusable_logits[256] = 10.0f; // EOS
    }

    int32_t chosen = s.sampler ? s.sampler->sample(s.reusable_logits, s.generated_tokens, params) : 0;
    if (chosen == 256) {
        chosen = MOCK_EOS_TOKEN;
    }

    s.generated_tokens.push_back(chosen);
    s.step++;

    // Model a deterministic per-token compute cost (test-only). Busy-wait rather than sleep so the
    // duration is not quantized by the OS scheduler's sleep granularity.
    if (token_latency_us_ > 0) {
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::microseconds(token_latency_us_);
        while (std::chrono::steady_clock::now() < deadline) {
            // spin
        }
    }
    return chosen;
}

std::string MockBackend::decode_token(BackendSessionHandle /*session*/, int32_t token_id) noexcept {
    if (token_id == MOCK_EOS_TOKEN) {
        return "";
    }
    if (token_id >= 0 && token_id < 256) {
        return std::string(1, static_cast<char>(token_id));
    }
    return " [tok_" + std::to_string(token_id) + "] ";
}

bool MockBackend::is_eos_token(BackendSessionHandle /*session*/, int32_t token_id) const noexcept {
    return token_id == MOCK_EOS_TOKEN;
}

GenerationDiagnostics MockBackend::describe_session(BackendSessionHandle session) const noexcept {
    GenerationDiagnostics d;
    d.backend_requested = "mock";
    d.backend_selected = "mock";
    d.cpu_threads = 1;
    d.cpu_cores = 1;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(session);
    if (it == sessions_.end()) {
        d.error = "session_not_found";
        return d;
    }
    d.gpu_offload_verified = false;
    d.gpu_layers_actual = 0;
    return d;
}

Status MockBackend::set_threads(BackendSessionHandle session, uint32_t /*n_threads*/) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(session);
    if (it == sessions_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session handle not found");
    }
    return Status::ok();
}

Status MockBackend::reset_session(BackendSessionHandle session) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(session);
    if (it == sessions_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session handle not found");
    }
    it->second.prompt_tokens.clear();
    it->second.generated_tokens.clear();
    it->second.step = 0;
    if (it->second.sampler) {
        it->second.sampler = std::make_unique<Sampler>(42 + static_cast<uint64_t>(session));
    }
    return Status::ok();
}

Result<std::vector<float>> MockBackend::get_last_logits(BackendSessionHandle session) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(session);
    if (it == sessions_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session handle not found");
    }
    return it->second.reusable_logits;
}

} // namespace mobilellm

