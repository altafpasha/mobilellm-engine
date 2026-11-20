#pragma once

#include "mobilellm/types.h"
#include <string>
#include <vector>
#include <memory>
#include <cstdint>

namespace mobilellm {

using BackendModelHandle = uint64_t;
using BackendSessionHandle = uint64_t;

struct BackendConfig {
    uint32_t default_threads{4};
    bool enable_profiling{false};
};

struct SessionContextTelemetry {
    uint32_t context_capacity{0};
    uint32_t used_tokens{0};
    uint32_t remaining_capacity{0};
    bool is_available{false};
};

/**
 * @brief Abstract interface defining compute and inference operations across backends.
 */
class IBackend {
public:
    virtual ~IBackend() = default;

    virtual Status initialize(const BackendConfig& config) noexcept = 0;
    virtual Status shutdown() noexcept = 0;

    virtual Result<BackendModelHandle> load_model(const ModelLoadConfig& config) noexcept = 0;
    virtual Status unload_model(BackendModelHandle model) noexcept = 0;

    virtual Result<BackendSessionHandle> create_session(BackendModelHandle model, const SessionConfig& config) noexcept = 0;
    virtual Status destroy_session(BackendSessionHandle session) noexcept = 0;

    virtual Status evaluate_prompt(
        BackendSessionHandle session,
        const std::string& prompt,
        std::vector<int32_t>& out_tokens,
        PromptEvalInfo* out_info = nullptr
    ) noexcept = 0;

    virtual Result<int32_t> sample_next_token(
        BackendSessionHandle session,
        const SamplingParams& params,
        TokenTiming* out_timing = nullptr
    ) noexcept = 0;

    virtual std::string decode_token(BackendSessionHandle session, int32_t token_id) noexcept = 0;
    virtual bool is_eos_token(BackendSessionHandle session, int32_t token_id) const noexcept = 0;

    /**
     * @brief Describes the *actual* execution configuration of a live session.
     *
     * Backends that cannot report anything (e.g. the mock backend) return a default-initialized
     * struct, which callers must treat as "unknown" rather than as "CPU with zero layers".
     */
    virtual GenerationDiagnostics describe_session(BackendSessionHandle /*session*/) const noexcept {
        return GenerationDiagnostics{};
    }

    virtual SessionContextTelemetry get_session_telemetry(BackendSessionHandle /*session*/) const noexcept {
        return SessionContextTelemetry{};
    }

    virtual Status set_threads(BackendSessionHandle /*session*/, uint32_t /*n_threads*/) noexcept {
        return Status::ok();
    }

    virtual Status reset_session(BackendSessionHandle /*session*/) noexcept {
        return Status::ok();
    }

    virtual Result<std::vector<float>> get_last_logits(BackendSessionHandle /*session*/) const noexcept {
        return Status::error(StatusCode::NOT_IMPLEMENTED, "get_last_logits not implemented on this backend");
    }

    virtual BackendType type() const noexcept = 0;
    virtual std::string name() const noexcept = 0;

    /**
     * @brief Human-readable description of the GPU device this backend can use, or "" if none.
     * Used by hardware detection so the reported device is the real adapter, not a build constant.
     */
    virtual std::string gpu_device_description() const noexcept { return {}; }
};

} // namespace mobilellm

