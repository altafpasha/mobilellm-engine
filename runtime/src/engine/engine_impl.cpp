#include "engine_impl.h"
#include "backends/common/backend_registry.h"
#include "backends/mock/mock_backend.h"
#include "backends/cpu/llamacpp_backend.h"
#include <chrono>
#include <iostream>

namespace mobilellm {

std::unique_ptr<IEngine> create_engine() noexcept {
    return std::make_unique<EngineImpl>();
}

EngineImpl::EngineImpl() = default;

EngineImpl::~EngineImpl() {
    shutdown();
}

void EngineImpl::set_backend_override(std::unique_ptr<IBackend> backend) {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    backend_ = std::move(backend);
}

Status EngineImpl::initialize(const EngineConfig& /*config*/) noexcept {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    if (initialized_) {
        return Status::ok();
    }

    // Register standard backends in registry if not present
    BackendRegistry& registry = BackendRegistry::instance();
    if (!registry.has_backend(BackendType::CPU_NEON)) {
#ifdef MOBILELLM_ENABLE_LLAMACPP
        registry.register_backend(BackendType::CPU_NEON, []() {
            return std::make_unique<LlamaCppBackend>();
        });
#else
        registry.register_backend(BackendType::CPU_NEON, []() {
            return std::make_unique<MockBackend>();
        });
#endif
    }

    if (!backend_) {
        backend_ = registry.create(BackendType::AUTO);
        if (!backend_) {
            backend_ = std::make_unique<MockBackend>();
        }
    }

    BackendConfig b_cfg;
    Status b_stat = backend_->initialize(b_cfg);
    if (!b_stat.is_ok()) {
        return b_stat;
    }

    initialized_ = true;
    return Status::ok();
}

Status EngineImpl::shutdown() noexcept {
    // 1. Cancel all active generations
    {
        std::lock_guard<std::mutex> lock(gen_mutex_);
        for (auto& [_, state] : active_generations_) {
            if (state) {
                state->is_cancelled.store(true);
            }
        }
        for (auto& [_, state] : active_generations_) {
            if (state && state->worker.joinable()) {
                state->worker.join();
            }
        }
        active_generations_.clear();
    }

    // 2. Clear sessions and models
    std::lock_guard<std::mutex> lock(engine_mutex_);
    if (backend_) {
        for (auto& [_, s] : sessions_) {
            backend_->destroy_session(s.backend_handle);
        }
        for (auto& [_, m] : models_) {
            backend_->unload_model(m.backend_handle);
        }
        backend_->shutdown();
    }

    sessions_.clear();
    models_.clear();
    initialized_ = false;
    return Status::ok();
}

Result<ModelHandle> EngineImpl::load_model(const ModelLoadConfig& config) noexcept {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    if (!initialized_) {
        Status init_st = initialize(EngineConfig{});
        if (!init_st.is_ok()) return init_st;
    }

    if (!backend_) {
        return Status::error(StatusCode::BACKEND_INITIALIZATION_FAILED, "No compute backend initialized");
    }

    auto b_res = backend_->load_model(config);
    if (!b_res.is_ok()) {
        return b_res.status();
    }

    ModelHandle m_handle = next_model_id_++;
    models_[m_handle] = ModelEntry{m_handle, b_res.value(), config.model_path};
    return m_handle;
}

Status EngineImpl::unload_model(ModelHandle model) noexcept {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    auto it = models_.find(model);
    if (it == models_.end()) {
        return Status::error(StatusCode::MODEL_NOT_LOADED, "Model handle not recognized");
    }

    // Close any sessions using this model
    for (auto s_it = sessions_.begin(); s_it != sessions_.end();) {
        if (s_it->second.model_handle == model) {
            backend_->destroy_session(s_it->second.backend_handle);
            s_it = sessions_.erase(s_it);
        } else {
            ++s_it;
        }
    }

    Status st = backend_->unload_model(it->second.backend_handle);
    models_.erase(it);
    return st;
}

Result<SessionHandle> EngineImpl::create_session(ModelHandle model, const SessionConfig& config) noexcept {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    auto m_it = models_.find(model);
    if (m_it == models_.end()) {
        return Status::error(StatusCode::MODEL_NOT_LOADED, "Model handle not loaded");
    }

    auto b_res = backend_->create_session(m_it->second.backend_handle, config);
    if (!b_res.is_ok()) {
        return b_res.status();
    }

    SessionHandle s_handle = next_session_id_++;
    sessions_[s_handle] = SessionEntry{s_handle, model, b_res.value()};
    return s_handle;
}

Status EngineImpl::destroy_session(SessionHandle session) noexcept {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    auto it = sessions_.find(session);
    if (it == sessions_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session handle not recognized");
    }

    Status st = backend_->destroy_session(it->second.backend_handle);
    sessions_.erase(it);
    return st;
}

Result<GenerationHandle> EngineImpl::generate_async(
    SessionHandle session,
    const GenerationRequest& request,
    TokenCallback token_cb,
    CompletionCallback complete_cb
) noexcept {
    SessionEntry session_entry;
    {
        std::lock_guard<std::mutex> lock(engine_mutex_);
        auto it = sessions_.find(session);
        if (it == sessions_.end()) {
            return Status::error(StatusCode::SESSION_NOT_FOUND, "Session not found");
        }
        session_entry = it->second;
    }

    if (request.prompt.empty()) {
        return Status::error(StatusCode::INVALID_ARGUMENT, "Prompt cannot be empty");
    }

    GenerationHandle gen_handle = next_gen_id_++;
    auto state = std::make_shared<GenerationState>();
    state->handle = gen_handle;
    state->session_handle = session;
    state->is_cancelled.store(false);
    state->is_finished.store(false);

    {
        std::lock_guard<std::mutex> lock(gen_mutex_);
        active_generations_[gen_handle] = state;
    }

    // Launch worker thread
    state->worker = std::thread(
        &EngineImpl::run_generation,
        this,
        state,
        session_entry,
        request,
        token_cb,
        complete_cb
    );

    return gen_handle;
}

void EngineImpl::run_generation(
    std::shared_ptr<GenerationState> state,
    SessionEntry session_entry,
    GenerationRequest request,
    TokenCallback token_cb,
    CompletionCallback complete_cb
) {
    GenerationMetrics metrics;
    auto t_start = std::chrono::high_resolution_clock::now();

    // Prompt evaluation
    std::vector<int32_t> prompt_tokens;
    Status eval_st = backend_->evaluate_prompt(
        session_entry.backend_handle,
        request.prompt,
        prompt_tokens
    );

    if (!eval_st.is_ok()) {
        state->is_finished.store(true);
        if (complete_cb) {
            complete_cb(eval_st, metrics);
        }
        return;
    }

    metrics.prompt_tokens = static_cast<uint32_t>(prompt_tokens.size());

    bool first_token = true;
    uint32_t generated_count = 0;
    Status final_status = Status::ok();

    while (generated_count < request.sampling.max_tokens) {
        if (state->is_cancelled.load(std::memory_order_relaxed)) {
            final_status = Status::error(StatusCode::CANCELLED, "Generation cancelled by user");
            break;
        }

        auto sample_res = backend_->sample_next_token(session_entry.backend_handle, request.sampling);
        if (!sample_res.is_ok()) {
            final_status = sample_res.status();
            break;
        }

        int32_t token_id = sample_res.value();
        if (backend_->is_eos_token(session_entry.backend_handle, token_id)) {
            break;
        }

        if (first_token) {
            auto t_first = std::chrono::high_resolution_clock::now();
            metrics.time_to_first_token_ms = std::chrono::duration<double, std::milli>(t_first - t_start).count();
            first_token = false;
        }

        std::string piece = backend_->decode_token(session_entry.backend_handle, token_id);
        TokenChunk chunk;
        chunk.text = piece;
        chunk.token_id = token_id;
        chunk.is_special = false;
        chunk.logprob = 0.0f;

        generated_count++;
        metrics.completion_tokens = generated_count;

        if (token_cb) {
            bool keep_going = token_cb(chunk);
            if (!keep_going) {
                final_status = Status::error(StatusCode::CANCELLED, "Cancellation requested via token callback");
                break;
            }
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();
    if (total_sec > 0.0 && generated_count > 0) {
        metrics.decode_tokens_per_second = static_cast<double>(generated_count) / total_sec;
    }

    state->is_finished.store(true);
    if (complete_cb) {
        complete_cb(final_status, metrics);
    }
}

Status EngineImpl::cancel(GenerationHandle handle) noexcept {
    std::lock_guard<std::mutex> lock(gen_mutex_);
    auto it = active_generations_.find(handle);
    if (it == active_generations_.end()) {
        return Status::error(StatusCode::INVALID_ARGUMENT, "Generation handle not active");
    }

    it->second->is_cancelled.store(true);
    return Status::ok();
}

EngineMetrics EngineImpl::get_metrics() const noexcept {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    EngineMetrics m;
    m.active_models = models_.size();
    m.active_sessions = sessions_.size();
    m.resident_memory_bytes = 0; // Populated by scheduler / OS
    return m;
}

HardwareProfile EngineImpl::query_hardware() const noexcept {
    HardwareProfile hp;
    hp.device_name = "Host Machine";
    hp.soc_platform = "ARM64/x86_64";
    hp.cpu_core_count = std::thread::hardware_concurrency();
    hp.neon_supported = true;
    hp.vulkan_supported = false;
    hp.npu_supported = false;
    return hp;
}

} // namespace mobilellm

