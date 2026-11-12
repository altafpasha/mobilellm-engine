#include "engine_impl.h"
#include "backends/common/backend_registry.h"
#include "backends/mock/mock_backend.h"
#include "backends/cpu/llamacpp_backend.h"
#include <chrono>
#include <iostream>

#ifdef __ANDROID__
#include <android/log.h>
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "MobileLLMEngine", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "MobileLLMEngine", __VA_ARGS__)
#else
#define LOGE(...)
#define LOGI(...)
#endif

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
    if (scheduler_) {
        scheduler_->stop();
    }
    backend_ = std::move(backend);
    if (initialized_ && backend_) {
        scheduler::SchedulerConfig s_cfg;
        s_cfg.bytes_per_kv_block = 0; // Virtual block tracking: 0 bytes dummy buffer
        scheduler_ = std::make_unique<scheduler::ContinuousBatchScheduler>(backend_, memory_mgr_, s_cfg);
        scheduler_->start();
    }
}

std::shared_ptr<scheduler::MemoryBudgetManager> EngineImpl::memory_manager() const noexcept {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    return memory_mgr_;
}

scheduler::ContinuousBatchScheduler* EngineImpl::scheduler() const noexcept {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    return scheduler_.get();
}

Status EngineImpl::initialize(const EngineConfig& config) noexcept {
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

    // Initialize MemoryBudgetManager and ContinuousBatchScheduler explicitly
    if (!memory_mgr_) {
        memory_mgr_ = std::make_shared<scheduler::MemoryBudgetManager>();
    }
    if (config.max_total_memory_budget_mb > 0.0f) {
        uint64_t ceiling_bytes = static_cast<uint64_t>(config.max_total_memory_budget_mb * 1024ULL * 1024ULL);
        memory_mgr_->recalculate_budget(ceiling_bytes);
    }

    scheduler::SchedulerConfig s_cfg;
    s_cfg.bytes_per_kv_block = 0; // Virtual block tracking: 0 bytes dummy buffer
    scheduler_ = std::make_unique<scheduler::ContinuousBatchScheduler>(backend_, memory_mgr_, s_cfg);
    Status sched_st = scheduler_->start();
    if (!sched_st.is_ok()) {
        return sched_st;
    }

    shutting_down_ = false;
    initialized_ = true;
    return Status::ok();
}

Status EngineImpl::shutdown() noexcept {
    if (shutting_down_.exchange(true)) {
        return Status::ok();
    }

    // 1. Cancel all active requests
    std::vector<GenerationHandle> handles_to_cancel;
    {
        std::lock_guard<std::mutex> lock(active_req_mutex_);
        for (const auto& [handle, req] : active_requests_) {
            handles_to_cancel.push_back(handle);
        }
    }

    if (scheduler_) {
        for (auto h : handles_to_cancel) {
            scheduler_->cancel_request(h);
        }
        // 2. Stop scheduler (joins worker thread and completes remaining requests)
        scheduler_->stop();
        scheduler_.reset();
    }

    // Wait until in-flight requests drain (with timeout)
    {
        std::unique_lock<std::mutex> lock(active_req_mutex_);
        in_flight_cv_.wait_for(lock, std::chrono::milliseconds(500), [this]() {
            return active_requests_.empty();
        });
        active_requests_.clear();
    }

    // 3. Clear sessions and models and release memory reservations
    std::lock_guard<std::mutex> lock(engine_mutex_);
    if (backend_) {
        for (auto& [_, s] : sessions_) {
            if (memory_mgr_) {
                memory_mgr_->deallocate(s.reserved_kv_bytes, scheduler::AllocationType::KV_CACHE);
            }
            backend_->destroy_session(s.backend_handle);
        }
        for (auto& [_, m] : models_) {
            if (memory_mgr_) {
                memory_mgr_->deallocate(m.reserved_memory_bytes, scheduler::AllocationType::MODEL_WEIGHTS);
            }
            backend_->unload_model(m.backend_handle);
        }
        backend_->shutdown();
    }

    sessions_.clear();
    models_.clear();
    initialized_ = false;
    shutting_down_ = false;
    return Status::ok();
}

uint64_t EngineImpl::calculate_model_memory_bytes(const ModelLoadConfig& config, bool& out_is_estimated) const noexcept {
    if (config.memory_budget_mb > 0.0f) {
        out_is_estimated = false;
        return static_cast<uint64_t>(config.memory_budget_mb * 1024ULL * 1024ULL);
    }

    auto meta_res = ModelLoader::inspect_file(config.model_path);
    if (meta_res.is_ok() && meta_res.value().file_size_bytes > 0) {
        out_is_estimated = false;
        return meta_res.value().file_size_bytes;
    }

    std::error_code ec;
    uint64_t fs = std::filesystem::file_size(config.model_path, ec);
    if (!ec && fs > 0) {
        out_is_estimated = false;
        return fs;
    }

    // Fallback nominal estimate for synthetic / mock models without physical disk files
    out_is_estimated = true;
    return 100ULL * 1024ULL * 1024ULL; // 100 MB nominal estimate
}

uint64_t EngineImpl::calculate_session_kv_bytes(
    const ModelEntry& model_entry,
    const SessionConfig& config,
    bool& out_is_estimated
) const noexcept {
    uint32_t ctx_tokens = config.max_context_tokens > 0 ? config.max_context_tokens : 2048;

    if (model_entry.has_metadata && model_entry.metadata.block_count > 0 &&
        model_entry.metadata.head_count > 0 && model_entry.metadata.embedding_length > 0) {
        const auto& meta = model_entry.metadata;
        uint32_t head_dim = meta.embedding_length / meta.head_count;
        // 2 for K and V, FP16 (2 bytes per element)
        uint64_t bytes_per_token = 2ULL * meta.block_count * meta.head_count * head_dim * 2ULL;
        out_is_estimated = false;
        return bytes_per_token * ctx_tokens;
    }

    // Fallback nominal estimate for mobile models without GGUF metadata: ~22.5 KB/token
    out_is_estimated = true;
    return static_cast<uint64_t>(ctx_tokens) * 23040ULL;
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

    // 1. Calculate memory requirement (distinguishing estimate from measured)
    bool is_estimated = false;
    uint64_t model_bytes = calculate_model_memory_bytes(config, is_estimated);

    // 2. Memory budget check and allocation
    if (memory_mgr_) {
        if (!memory_mgr_->can_allocate(model_bytes, scheduler::AllocationType::MODEL_WEIGHTS)) {
            auto status = memory_mgr_->get_status();
            return Status::error(
                StatusCode::OUT_OF_MEMORY,
                "Model memory reservation of " + std::to_string(model_bytes / (1024 * 1024)) +
                " MB (" + (is_estimated ? "estimated" : "measured") +
                ") exceeds available memory budget ceiling of " +
                std::to_string(status.total_budget_ceiling_bytes / (1024 * 1024)) + " MB (current usage: " +
                std::to_string(status.current_usage_bytes / (1024 * 1024)) + " MB)"
            );
        }
        Status alloc_st = memory_mgr_->allocate(model_bytes, scheduler::AllocationType::MODEL_WEIGHTS);
        if (!alloc_st.is_ok()) {
            return alloc_st;
        }
    }

    // 3. Backend load execution
    auto b_res = backend_->load_model(config);
    if (!b_res.is_ok()) {
        // Rollback reservation on failure!
        if (memory_mgr_) {
            memory_mgr_->deallocate(model_bytes, scheduler::AllocationType::MODEL_WEIGHTS);
        }
        return b_res.status();
    }

    ModelHandle m_handle = next_model_id_++;
    ModelEntry entry;
    entry.handle = m_handle;
    entry.backend_handle = b_res.value();
    entry.path = config.model_path;
    entry.reserved_memory_bytes = model_bytes;
    entry.is_memory_estimated = is_estimated;

    auto meta_res = ModelLoader::inspect_file(config.model_path);
    if (meta_res.is_ok()) {
        entry.metadata = meta_res.value();
        entry.has_metadata = true;
    }

    models_[m_handle] = entry;
    return m_handle;
}

Status EngineImpl::unload_model(ModelHandle model) noexcept {
    // 1. Find sessions belonging to this model and cancel active requests
    if (scheduler_) {
        std::vector<GenerationHandle> handles_to_cancel;
        {
            std::lock_guard<std::mutex> lock(active_req_mutex_);
            for (const auto& [h, info] : active_requests_) {
                if (info && info->model_handle == model) {
                    handles_to_cancel.push_back(h);
                }
            }
        }
        for (auto h : handles_to_cancel) {
            scheduler_->cancel_request(h);
        }

        std::unique_lock<std::mutex> lock(active_req_mutex_);
        in_flight_cv_.wait_for(lock, std::chrono::milliseconds(1000), [this, model]() {
            for (const auto& [_, info] : active_requests_) {
                if (info && info->model_handle == model) return false;
            }
            return true;
        });
    }

    // 2. Destroy sessions and release memory reservations
    std::lock_guard<std::mutex> lock(engine_mutex_);
    auto it = models_.find(model);
    if (it == models_.end()) {
        return Status::error(StatusCode::MODEL_NOT_LOADED, "Model handle not recognized");
    }

    for (auto s_it = sessions_.begin(); s_it != sessions_.end();) {
        if (s_it->second.model_handle == model) {
            if (scheduler_ && scheduler_->session_manager()) {
                scheduler_->session_manager()->destroy_session(s_it->first);
            }
            if (memory_mgr_) {
                memory_mgr_->deallocate(s_it->second.reserved_kv_bytes, scheduler::AllocationType::KV_CACHE);
            }
            backend_->destroy_session(s_it->second.backend_handle);
            s_it = sessions_.erase(s_it);
        } else {
            ++s_it;
        }
    }

    if (memory_mgr_) {
        memory_mgr_->deallocate(it->second.reserved_memory_bytes, scheduler::AllocationType::MODEL_WEIGHTS);
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

    // 1. Calculate KV-cache memory requirement (distinguishing estimate from measured)
    bool is_kv_estimated = false;
    uint64_t kv_bytes = calculate_session_kv_bytes(m_it->second, config, is_kv_estimated);

    // 2. Memory budget check and allocation
    if (memory_mgr_) {
        if (!memory_mgr_->can_allocate(kv_bytes, scheduler::AllocationType::KV_CACHE)) {
            auto status = memory_mgr_->get_status();
            return Status::error(
                StatusCode::OUT_OF_MEMORY,
                "Session KV-cache reservation of " + std::to_string(kv_bytes / (1024 * 1024)) +
                " MB (" + (is_kv_estimated ? "estimated" : "measured") +
                ") exceeds available memory budget ceiling of " +
                std::to_string(status.total_budget_ceiling_bytes / (1024 * 1024)) + " MB (current usage: " +
                std::to_string(status.current_usage_bytes / (1024 * 1024)) + " MB)"
            );
        }
        Status alloc_st = memory_mgr_->allocate(kv_bytes, scheduler::AllocationType::KV_CACHE);
        if (!alloc_st.is_ok()) {
            return alloc_st;
        }
    }

    // 3. Backend session creation
    auto b_res = backend_->create_session(m_it->second.backend_handle, config);
    if (!b_res.is_ok()) {
        // Rollback reservation on failure!
        if (memory_mgr_) {
            memory_mgr_->deallocate(kv_bytes, scheduler::AllocationType::KV_CACHE);
        }
        return b_res.status();
    }

    SessionHandle s_handle = next_session_id_++;
    SessionEntry s_entry;
    s_entry.handle = s_handle;
    s_entry.model_handle = model;
    s_entry.backend_handle = b_res.value();
    s_entry.reserved_kv_bytes = kv_bytes;
    s_entry.is_kv_estimated = is_kv_estimated;
    sessions_[s_handle] = s_entry;

    if (scheduler_ && scheduler_->session_manager()) {
        scheduler_->session_manager()->create_session_with_handle(
            s_handle, model, b_res.value(), config
        );
    }

    return s_handle;
}

Status EngineImpl::destroy_session(SessionHandle session) noexcept {
    // 1. Cancel and drain active requests for this session
    if (scheduler_) {
        std::vector<GenerationHandle> handles_to_cancel;
        {
            std::lock_guard<std::mutex> lock(active_req_mutex_);
            for (const auto& [h, info] : active_requests_) {
                if (info && info->session_handle == session) {
                    handles_to_cancel.push_back(h);
                }
            }
        }
        for (auto h : handles_to_cancel) {
            scheduler_->cancel_request(h);
        }

        std::unique_lock<std::mutex> lock(active_req_mutex_);
        in_flight_cv_.wait_for(lock, std::chrono::milliseconds(1000), [this, session]() {
            for (const auto& [_, info] : active_requests_) {
                if (info && info->session_handle == session) return false;
            }
            return true;
        });
    }

    // 2. Destroy session under engine_mutex_
    std::lock_guard<std::mutex> lock(engine_mutex_);
    auto it = sessions_.find(session);
    if (it == sessions_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session handle not recognized");
    }

    if (scheduler_ && scheduler_->session_manager()) {
        scheduler_->session_manager()->destroy_session(session);
    }

    if (memory_mgr_) {
        memory_mgr_->deallocate(it->second.reserved_kv_bytes, scheduler::AllocationType::KV_CACHE);
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
    if (shutting_down_.load()) {
        return Status::error(StatusCode::INTERNAL_ERROR, "Engine is shutting down");
    }
    if (!initialized_.load()) {
        return Status::error(StatusCode::INTERNAL_ERROR, "Engine is not initialized");
    }

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

    if (!scheduler_ || !scheduler_->is_running()) {
        return Status::error(StatusCode::INTERNAL_ERROR, "Scheduler is not initialized or running");
    }

    auto req_info = std::make_shared<ActiveRequestInfo>();
    req_info->session_handle = session;
    req_info->model_handle = session_entry.model_handle;

    auto wrapped_complete = [this, req_info, user_cb = std::move(complete_cb)](const Status& st, const GenerationMetrics& m) {
        req_info->is_finished.store(true);
        {
            std::lock_guard<std::mutex> lock(active_req_mutex_);
            active_requests_.erase(req_info->handle);
        }
        in_flight_cv_.notify_all();

        if (user_cb) {
            user_cb(st, m);
        }
    };

    auto sub_res = scheduler_->submit_request(
        session,
        request,
        scheduler::RequestPriority::NORMAL,
        std::move(token_cb),
        std::move(wrapped_complete)
    );

    if (!sub_res.is_ok()) {
        return sub_res.status();
    }

    GenerationHandle gen_handle = sub_res.value();
    req_info->handle = gen_handle;
    {
        std::lock_guard<std::mutex> lock(active_req_mutex_);
        if (!req_info->is_finished.load()) {
            active_requests_[gen_handle] = req_info;
        }
    }

    return gen_handle;
}

Status EngineImpl::cancel(GenerationHandle handle) noexcept {
    if (!scheduler_) {
        return Status::error(StatusCode::INTERNAL_ERROR, "Scheduler not initialized");
    }
    return scheduler_->cancel_request(handle);
}

EngineMetrics EngineImpl::get_metrics() const noexcept {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    EngineMetrics m;
    m.active_models = models_.size();
    m.active_sessions = sessions_.size();
    if (memory_mgr_) {
        auto status = memory_mgr_->get_status();
        m.resident_memory_bytes = status.current_usage_bytes;
        m.kv_cache_allocated_bytes = status.kv_cache_bytes;
        m.kv_cache_free_bytes = (status.total_budget_ceiling_bytes >= status.current_usage_bytes)
            ? (status.total_budget_ceiling_bytes - status.current_usage_bytes) : 0;
    }
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

