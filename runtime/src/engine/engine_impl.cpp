#include "engine_impl.h"
#include "backends/common/backend_registry.h"
#include "backends/mock/mock_backend.h"
#include "backends/cpu/llamacpp_backend.h"
#include <chrono>
#include <iostream>

#ifdef __ANDROID__
#include <android/log.h>
#include <sys/system_properties.h>
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "MobileLLMEngine", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "MobileLLMEngine", __VA_ARGS__)
#else
#define LOGE(...)
#define LOGI(...)
#endif

namespace mobilellm {

namespace {

std::string architecture_name() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "ARM64";
#elif defined(__arm__) || defined(_M_ARM)
    return "ARMv7";
#elif defined(__x86_64__) || defined(_M_X64)
    return "X86_64";
#elif defined(__i386__) || defined(_M_IX86)
    return "X86";
#else
    return "UNKNOWN";
#endif
}

bool neon_compiled_in() noexcept {
#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
    return true;
#else
    return false;
#endif
}

/**
 * @brief OS-level hardware facts shared by `detect_hardware()` and `EngineImpl::query_hardware()`.
 *
 * Deliberately independent of any engine/backend instance: `mllm_detect_hardware()` must work
 * before a model is loaded, and the values reported here come from the operating system, not from
 * configuration intent.
 */
HardwareProfile detect_os_hardware() noexcept {
    HardwareProfile hp;
    const unsigned cores = std::thread::hardware_concurrency();
    hp.cpu_core_count = cores > 0 ? cores : 1;
    hp.neon_supported = neon_compiled_in();
    hp.npu_supported = false;

#ifdef __ANDROID__
    auto prop = [](const char* key) -> std::string {
        char v[PROP_VALUE_MAX] = {0};
        int len = __system_property_get(key, v);
        return len > 0 ? std::string(v, static_cast<size_t>(len)) : std::string();
    };
    const std::string manufacturer = prop("ro.product.manufacturer");
    const std::string model = prop("ro.product.model");
    if (!model.empty()) {
        hp.device_name = manufacturer.empty() ? model : (manufacturer + " " + model);
    } else {
        hp.device_name = "Android Device";
    }
    std::string soc = prop("ro.soc.model");
    if (soc.empty()) soc = prop("ro.board.platform");
    const std::string soc_manufacturer = prop("ro.soc.manufacturer");
    if (!soc.empty()) {
        hp.soc_platform = soc_manufacturer.empty() ? soc : (soc_manufacturer + " " + soc);
    } else {
        hp.soc_platform = architecture_name();
    }
#else
    // Host builds are not phones: say so explicitly rather than inventing a device name.
    hp.device_name = "Host Machine (" + architecture_name() + ")";
    hp.soc_platform = architecture_name();
#endif

    auto mem = scheduler::MemoryBudgetManager::query_system_memory();
    hp.total_ram_bytes = mem.total_ram_bytes;
    hp.available_ram_bytes = mem.available_ram_bytes;

#ifdef MOBILELLM_ENABLE_LLAMACPP
    hp.vulkan_supported = LlamaCppBackend::is_gpu_supported();
    if (hp.vulkan_supported) {
        hp.vulkan_device_name = LlamaCppBackend::probe_gpu_device_description();
    }
#else
    hp.vulkan_supported = false;
#endif
    return hp;
}

} // namespace

HardwareProfile detect_hardware() noexcept {
    return detect_os_hardware();
}

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

    if (!registry.has_backend(BackendType::VULKAN)) {
#ifdef MOBILELLM_ENABLE_LLAMACPP
        registry.register_backend(
            BackendType::VULKAN,
            []() { return std::make_unique<LlamaCppBackend>(); },
            []() { return LlamaCppBackend::is_gpu_supported(); }
        );
#else
        registry.register_backend(BackendType::VULKAN, []() {
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
        uint32_t kv_heads = meta.head_count_kv > 0 ? meta.head_count_kv : meta.head_count;
        // 2 for K and V, FP16 (2 bytes per element)
        uint64_t bytes_per_token = 2ULL * meta.block_count * kv_heads * head_dim * 2ULL;
        out_is_estimated = false;
        return bytes_per_token * ctx_tokens;
    }

    // Fallback nominal estimate for mobile models without GGUF metadata: ~22.5 KB/token
    out_is_estimated = true;
    return static_cast<uint64_t>(ctx_tokens) * 23040ULL;
}

Result<ModelHandle> EngineImpl::load_model(const ModelLoadConfig& config) noexcept {
    if (config.preferred_backend != BackendType::AUTO) {
        BackendType current_type = BackendType::AUTO;
        {
            std::lock_guard<std::mutex> lock(engine_mutex_);
            if (backend_) {
                current_type = backend_->type();
            }
        }
        if (current_type != BackendType::AUTO && current_type != config.preferred_backend) {
            Status sw_st = switch_backend(config.preferred_backend);
            if (!sw_st.is_ok()) {
                return sw_st;
            }
        }
    }

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
    entry.load_config = config;
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
    s_entry.session_config = config;
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

Status EngineImpl::reset_session(SessionHandle session) noexcept {
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
        in_flight_cv_.wait_for(lock, std::chrono::milliseconds(500), [this, session]() {
            for (const auto& [_, info] : active_requests_) {
                if (info && info->session_handle == session) return false;
            }
            return true;
        });
    }

    // 2. Reset under engine_mutex_
    std::lock_guard<std::mutex> lock(engine_mutex_);
    auto it = sessions_.find(session);
    if (it == sessions_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session handle not recognized");
    }

    if (scheduler_ && scheduler_->session_manager()) {
        scheduler_->session_manager()->reset_session(session);
    }

    return backend_->reset_session(it->second.backend_handle);
}

Result<std::vector<float>> EngineImpl::get_last_logits(SessionHandle session) const noexcept {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    auto it = sessions_.find(session);
    if (it == sessions_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session handle not recognized");
    }
    return backend_->get_last_logits(it->second.backend_handle);
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
    if (is_transitioning_.load()) {
        return Status::error(StatusCode::BACKEND_EXECUTION_ERROR, "Backend transition in progress");
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

    auto wrapped_complete = [this, req_info, session, session_entry, user_cb = std::move(complete_cb)](
                                const Status& st, const GenerationMetrics& m) {
        req_info->is_finished.store(true);
        {
            std::lock_guard<std::mutex> lock(active_req_mutex_);
            active_requests_.erase(req_info->handle);
        }
        in_flight_cv_.notify_all();

        // Merge backend-declared identity / offload evidence into the scheduler-measured
        // diagnostics. Doing it here (on the completion path) keeps the hot decode loop free of
        // any backend introspection, and the handle is still valid: the session is destroyed
        // only by explicit destroy_session()/unload_model().
        GenerationMetrics enriched = m;
        if (backend_) {
            try {
                GenerationDiagnostics backend_diag = backend_->describe_session(
                    session_entry.backend_handle > 0 ? session_entry.backend_handle : session);
                GenerationDiagnostics& d = enriched.diagnostics;
                d.backend_requested = backend_diag.backend_requested;
                d.backend_selected = backend_diag.backend_selected;
                d.gpu_device = backend_diag.gpu_device;
                d.gpu_vendor = backend_diag.gpu_vendor;
                d.gpu_driver = backend_diag.gpu_driver;
                d.vulkan_gemv_path = backend_diag.vulkan_gemv_path;
                d.model_layers = backend_diag.model_layers;
                d.gpu_layers_requested = backend_diag.gpu_layers_requested;
                d.gpu_layers_actual = backend_diag.gpu_layers_actual;
                d.graph_nodes = backend_diag.graph_nodes;
                d.graph_splits_pp = backend_diag.graph_splits_pp;
                d.graph_splits_tg = backend_diag.graph_splits_tg;
                d.gpu_offload_verified = backend_diag.gpu_offload_verified;
                d.model_load_ms = backend_diag.model_load_ms;
                d.context_capacity = backend_diag.context_capacity;
                d.kv_used_tokens = backend_diag.kv_used_tokens;
                d.cpu_threads = backend_diag.cpu_threads;
                d.cpu_cores = backend_diag.cpu_cores;
                if (d.error.empty()) {
                    d.error = backend_diag.error;
                }
            } catch (...) {
                // Diagnostics must never break the inference path.
            }
        }

        {
            std::lock_guard<std::mutex> lock(diagnostics_mutex_);
            last_diagnostics_[session] = enriched.diagnostics;
        }

        if (user_cb) {
            user_cb(st, enriched);
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

Status EngineImpl::switch_backend(BackendType target) noexcept {
    if (shutting_down_.load()) {
        return Status::error(StatusCode::INTERNAL_ERROR, "Engine is shutting down");
    }
    if (!initialized_.load()) {
        return Status::error(StatusCode::INTERNAL_ERROR, "Engine is not initialized");
    }

    // Thread-safe transition guard: reject concurrent transitions
    bool expected = false;
    if (!is_transitioning_.compare_exchange_strong(expected, true)) {
        return Status::error(StatusCode::BACKEND_EXECUTION_ERROR, "Backend transition already in progress");
    }

    struct TransitionGuard {
        std::atomic<bool>& flag;
        ~TransitionGuard() { flag.store(false); }
    } guard{is_transitioning_};

    // Fast-path: already on requested target
    {
        std::lock_guard<std::mutex> lock(engine_mutex_);
        if (backend_ && (backend_->type() == target || target == BackendType::AUTO)) {
            return Status::ok();
        }
    }

    auto& registry = BackendRegistry::instance();
    if (target != BackendType::AUTO && !registry.is_available(target)) {
        return Status::error(StatusCode::BACKEND_INITIALIZATION_FAILED,
                             "Requested backend is not available or supported on this device");
    }

    // 1. Drain or cancel active requests safely
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

        // Wait for outstanding operations to drain (up to 2000 ms)
        {
            std::unique_lock<std::mutex> lock(active_req_mutex_);
            in_flight_cv_.wait_for(lock, std::chrono::milliseconds(2000), [this]() {
                return active_requests_.empty();
            });
        }

        // Stop scheduler before tearing down backend
        scheduler_->stop();
        scheduler_.reset();
    }

    {
        std::lock_guard<std::mutex> lock(active_req_mutex_);
        active_requests_.clear();
    }

    std::lock_guard<std::mutex> lock(engine_mutex_);

    // 2. Tear down existing backend resources in reverse dependency order
    if (backend_) {
        // Sessions and models synchronize with GPU (via llama_synchronize) before destroying raw contexts
        for (auto& [_, s] : sessions_) {
            backend_->destroy_session(s.backend_handle);
        }
        for (auto& [_, m] : models_) {
            backend_->unload_model(m.backend_handle);
        }
        backend_->shutdown();
        backend_.reset();
    }

    // 3. Create target backend from registry
    auto new_backend = registry.create(target);
    if (!new_backend) {
        return Status::error(StatusCode::BACKEND_INITIALIZATION_FAILED,
                             "Failed to create backend: requested backend not available");
    }

    BackendConfig b_cfg;
    Status b_stat = new_backend->initialize(b_cfg);
    if (!b_stat.is_ok()) {
        return b_stat;
    }
    backend_ = std::move(new_backend);

    // 4. Recreate models and sessions on the new backend
    for (auto& [m_id, m_entry] : models_) {
        ModelLoadConfig load_cfg = m_entry.load_config;
        load_cfg.preferred_backend = target;
        if (target == BackendType::CPU_NEON) {
            load_cfg.n_gpu_layers = 0;
        } else if (target == BackendType::VULKAN && load_cfg.n_gpu_layers <= 0) {
            load_cfg.n_gpu_layers = 99;
        }
        auto b_res = backend_->load_model(load_cfg);
        if (!b_res.is_ok()) {
            return b_res.status();
        }
        m_entry.backend_handle = b_res.value();
        m_entry.load_config = load_cfg;
    }

    for (auto& [s_id, s_entry] : sessions_) {
        auto m_it = models_.find(s_entry.model_handle);
        if (m_it == models_.end()) {
            return Status::error(StatusCode::MODEL_NOT_LOADED, "Model entry missing during backend recreation");
        }
        auto b_res = backend_->create_session(m_it->second.backend_handle, s_entry.session_config);
        if (!b_res.is_ok()) {
            return b_res.status();
        }
        s_entry.backend_handle = b_res.value();
    }

    // 5. Recreate ContinuousBatchScheduler with new backend
    scheduler::SchedulerConfig s_cfg;
    s_cfg.bytes_per_kv_block = 0;
    scheduler_ = std::make_unique<scheduler::ContinuousBatchScheduler>(backend_, memory_mgr_, s_cfg);
    Status sched_st = scheduler_->start();
    if (!sched_st.is_ok()) {
        return sched_st;
    }

    // 6. Restore scheduler session entries
    if (scheduler_->session_manager()) {
        for (const auto& [s_id, s_entry] : sessions_) {
            scheduler_->session_manager()->create_session_with_handle(
                s_id, s_entry.model_handle, s_entry.backend_handle, s_entry.session_config
            );
        }
    }

    return Status::ok();
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
    HardwareProfile hp = detect_os_hardware();

    if (!backend_) {
        return hp;
    }

    // The active backend is authoritative about what it is actually executing on.
    hp.vulkan_supported = hp.vulkan_supported || backend_->type() == BackendType::VULKAN;
    if (hp.vulkan_supported) {
        auto desc = backend_->gpu_device_description();
        if (!desc.empty()) {
            hp.vulkan_device_name = desc;
        }
    }
    return hp;
}

ProcessMemoryTelemetry EngineImpl::get_process_memory() const noexcept {
    ProcessMemoryTelemetry t;
    auto proc = scheduler::MemoryBudgetManager::query_process_memory();
    t.current_rss_bytes = proc.current_rss_bytes;
    t.peak_rss_bytes = proc.peak_rss_bytes;
    t.vmsize_bytes = proc.vmsize_bytes;
    if (memory_mgr_) {
        auto status = memory_mgr_->get_status();
        t.model_allocated_bytes = status.model_weights_bytes;
        t.kv_allocated_bytes = status.kv_cache_bytes;
    }
    return t;
}

Result<ContextTelemetry> EngineImpl::get_session_telemetry(SessionHandle session) const noexcept {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    auto s_it = sessions_.find(session);
    if (s_it == sessions_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session not found");
    }
    if (!backend_) {
        return Status::error(StatusCode::BACKEND_INITIALIZATION_FAILED, "Backend not available");
    }
    auto b_telem = backend_->get_session_telemetry(s_it->second.backend_handle);
    ContextTelemetry ct;
    ct.context_capacity = b_telem.context_capacity;
    ct.used_tokens = b_telem.used_tokens;
    ct.remaining_capacity = b_telem.remaining_capacity;
    ct.is_available = b_telem.is_available;
    return ct;
}

Status EngineImpl::set_session_threads(SessionHandle session, uint32_t n_threads) noexcept {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    auto s_it = sessions_.find(session);
    if (s_it == sessions_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session not found");
    }
    if (!backend_) {
        return Status::error(StatusCode::BACKEND_INITIALIZATION_FAILED, "Backend not available");
    }
    return backend_->set_threads(s_it->second.backend_handle, n_threads);
}

GenerationDiagnostics EngineImpl::get_generation_diagnostics(SessionHandle session) const noexcept {
    GenerationDiagnostics d;
    {
        std::lock_guard<std::mutex> lock(diagnostics_mutex_);
        auto it = last_diagnostics_.find(session);
        if (it != last_diagnostics_.end()) {
            d = it->second;
        }
    }
    // Session may not have completed a generation yet: fall back to the static session facts so
    // callers still see the true backend identity and offload evidence.
    if (d.benchmark_status == "idle" && backend_) {
        std::lock_guard<std::mutex> lock(engine_mutex_);
        auto s_it = sessions_.find(session);
        if (s_it != sessions_.end()) {
            GenerationDiagnostics b = backend_->describe_session(s_it->second.backend_handle);
            d.backend_requested = b.backend_requested;
            d.backend_selected = b.backend_selected;
            d.gpu_device = b.gpu_device;
            d.gpu_vendor = b.gpu_vendor;
            d.gpu_driver = b.gpu_driver;
            d.vulkan_gemv_path = b.vulkan_gemv_path;
            d.model_layers = b.model_layers;
            d.gpu_layers_requested = b.gpu_layers_requested;
            d.gpu_layers_actual = b.gpu_layers_actual;
            d.graph_nodes = b.graph_nodes;
            d.graph_splits_pp = b.graph_splits_pp;
            d.graph_splits_tg = b.graph_splits_tg;
            d.gpu_offload_verified = b.gpu_offload_verified;
            d.model_load_ms = b.model_load_ms;
            d.context_capacity = b.context_capacity;
            d.kv_used_tokens = b.kv_used_tokens;
            d.cpu_threads = b.cpu_threads;
            d.cpu_cores = b.cpu_cores;
        }
    }
    return d;
}

} // namespace mobilellm

