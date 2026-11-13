#pragma once

#include "mobilellm/engine.h"
#include "mobilellm/model_loader.h"
#include "backends/common/backend.h"
#include "mobilellm/scheduler/scheduler.h"
#include "mobilellm/scheduler/memory_manager.h"
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <thread>
#include <memory>
#include <filesystem>

namespace mobilellm {

struct ActiveRequestInfo {
    GenerationHandle handle{0};
    SessionHandle session_handle{0};
    ModelHandle model_handle{0};
    std::atomic<bool> is_finished{false};
};

class EngineImpl : public IEngine {
public:
    EngineImpl();
    ~EngineImpl() override;

    Status initialize(const EngineConfig& config) noexcept override;
    Status shutdown() noexcept override;

    Result<ModelHandle> load_model(const ModelLoadConfig& config) noexcept override;
    Status unload_model(ModelHandle model) noexcept override;

    Result<SessionHandle> create_session(ModelHandle model, const SessionConfig& config) noexcept override;
    Status destroy_session(SessionHandle session) noexcept override;

    Result<GenerationHandle> generate_async(
        SessionHandle session,
        const GenerationRequest& request,
        TokenCallback token_cb,
        CompletionCallback complete_cb
    ) noexcept override;

    Status cancel(GenerationHandle handle) noexcept override;

    EngineMetrics get_metrics() const noexcept override;
    HardwareProfile query_hardware() const noexcept override;
    ProcessMemoryTelemetry get_process_memory() const noexcept override;
    Result<ContextTelemetry> get_session_telemetry(SessionHandle session) const noexcept override;
    Status set_session_threads(SessionHandle session, uint32_t n_threads) noexcept override;

    // For test injection & introspection
    void set_backend_override(std::unique_ptr<IBackend> backend);
    std::shared_ptr<scheduler::MemoryBudgetManager> memory_manager() const noexcept;
    scheduler::ContinuousBatchScheduler* scheduler() const noexcept;

private:
    struct ModelEntry {
        ModelHandle handle{0};
        BackendModelHandle backend_handle{0};
        std::string path;
        uint64_t reserved_memory_bytes{0};
        bool is_memory_estimated{false};
        ModelMetadata metadata;
        bool has_metadata{false};
    };

    struct SessionEntry {
        SessionHandle handle{0};
        ModelHandle model_handle{0};
        BackendSessionHandle backend_handle{0};
        uint64_t reserved_kv_bytes{0};
        bool is_kv_estimated{false};
    };

    mutable std::mutex engine_mutex_;
    mutable std::mutex active_req_mutex_;
    std::condition_variable in_flight_cv_;

    std::atomic<uint64_t> next_model_id_{1};
    std::atomic<uint64_t> next_session_id_{1};

    std::unordered_map<ModelHandle, ModelEntry> models_;
    std::unordered_map<SessionHandle, SessionEntry> sessions_;
    std::unordered_map<GenerationHandle, std::shared_ptr<ActiveRequestInfo>> active_requests_;

    std::shared_ptr<IBackend> backend_;
    std::shared_ptr<scheduler::MemoryBudgetManager> memory_mgr_;
    std::unique_ptr<scheduler::ContinuousBatchScheduler> scheduler_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> shutting_down_{false};

    uint64_t calculate_model_memory_bytes(const ModelLoadConfig& config, bool& out_is_estimated) const noexcept;
    uint64_t calculate_session_kv_bytes(const ModelEntry& model_entry, const SessionConfig& config, bool& out_is_estimated) const noexcept;
};

} // namespace mobilellm

