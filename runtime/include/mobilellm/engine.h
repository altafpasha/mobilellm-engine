#pragma once

#include "mobilellm/types.h"
#include <memory>
#include <string>

namespace mobilellm {

/**
 * @brief Engine initialization configuration.
 */
struct EngineConfig {
    std::string app_data_dir;
    bool enable_telemetry{true};
    float max_total_memory_budget_mb{0.0f}; // 0 = auto-detect
};

/**
 * @brief Canonical C++ interface for the MobileLLM Engine.
 */
class IEngine {
public:
    virtual ~IEngine() = default;

    // Initialization & Lifecycle
    virtual Status initialize(const EngineConfig& config) noexcept = 0;
    virtual Status shutdown() noexcept = 0;

    // Model Management
    virtual Result<ModelHandle> load_model(const ModelLoadConfig& config) noexcept = 0;
    virtual Status unload_model(ModelHandle model) noexcept = 0;

    // Session Management
    virtual Result<SessionHandle> create_session(ModelHandle model, const SessionConfig& config) noexcept = 0;
    virtual Status destroy_session(SessionHandle session) noexcept = 0;
    virtual Status reset_session(SessionHandle session) noexcept = 0;
    virtual Result<std::vector<float>> get_last_logits(SessionHandle session) const noexcept = 0;

    // Generation Operations
    virtual Result<GenerationHandle> generate_async(
        SessionHandle session,
        const GenerationRequest& request,
        TokenCallback token_cb,
        CompletionCallback complete_cb
    ) noexcept = 0;

    virtual Status cancel(GenerationHandle handle) noexcept = 0;

    // Backend Switching
    virtual Status switch_backend(BackendType target) noexcept = 0;

    // Hardware Inspection & Telemetry
    virtual EngineMetrics get_metrics() const noexcept = 0;
    virtual HardwareProfile query_hardware() const noexcept = 0;
    virtual ProcessMemoryTelemetry get_process_memory() const noexcept = 0;
    virtual Result<ContextTelemetry> get_session_telemetry(SessionHandle session) const noexcept = 0;
    virtual Status set_session_threads(SessionHandle session, uint32_t n_threads) noexcept = 0;

    /**
     * @brief Diagnostics for the most recently completed generation on this session.
     *
     * Returns a default-initialized struct (benchmark_status == "idle") when no generation has
     * completed yet. Backend identity and offload evidence are filled from the backend's own
     * session description, never from configuration intent.
     */
    virtual GenerationDiagnostics get_generation_diagnostics(SessionHandle /*session*/) const noexcept {
        return GenerationDiagnostics{};
    }
};

/**
 * @brief Factory function creating an instance of the default MobileLLM Engine.
 * 
 * @return std::unique_ptr<IEngine> pointer to the created engine instance.
 */
std::unique_ptr<IEngine> create_engine() noexcept;

/**
 * @brief Process-wide hardware detection (device model, SoC, GPU adapter).
 *
 * Independent of any engine instance so it can be used by `mllm_detect_hardware()` before a model
 * is loaded. Returns real values on Android and an explicit "Host Machine" otherwise.
 */
HardwareProfile detect_hardware() noexcept;

} // namespace mobilellm
