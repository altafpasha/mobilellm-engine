#pragma once

#include "mobilellm/engine.h"
#include "backends/common/backend.h"
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <thread>
#include <memory>

namespace mobilellm {

struct GenerationState {
    GenerationHandle handle{0};
    SessionHandle session_handle{0};
    std::atomic<bool> is_cancelled{false};
    std::atomic<bool> is_finished{false};
    std::thread worker;
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

    // For test injection
    void set_backend_override(std::unique_ptr<IBackend> backend);

private:
    struct ModelEntry {
        ModelHandle handle;
        BackendModelHandle backend_handle;
        std::string path;
    };

    struct SessionEntry {
        SessionHandle handle;
        ModelHandle model_handle;
        BackendSessionHandle backend_handle;
    };

    mutable std::mutex engine_mutex_;
    mutable std::mutex gen_mutex_;

    std::atomic<uint64_t> next_model_id_{1};
    std::atomic<uint64_t> next_session_id_{1};
    std::atomic<uint64_t> next_gen_id_{1};

    std::unordered_map<ModelHandle, ModelEntry> models_;
    std::unordered_map<SessionHandle, SessionEntry> sessions_;
    std::unordered_map<GenerationHandle, std::shared_ptr<GenerationState>> active_generations_;

    std::unique_ptr<IBackend> backend_;
    bool initialized_{false};

    void run_generation(
        std::shared_ptr<GenerationState> state,
        SessionEntry session_entry,
        GenerationRequest request,
        TokenCallback token_cb,
        CompletionCallback complete_cb
    );
};

} // namespace mobilellm

