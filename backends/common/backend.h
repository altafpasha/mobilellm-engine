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
        std::vector<int32_t>& out_tokens
    ) noexcept = 0;

    virtual Result<int32_t> sample_next_token(
        BackendSessionHandle session,
        const SamplingParams& params
    ) noexcept = 0;

    virtual std::string decode_token(BackendSessionHandle session, int32_t token_id) noexcept = 0;
    virtual bool is_eos_token(BackendSessionHandle session, int32_t token_id) const noexcept = 0;

    virtual BackendType type() const noexcept = 0;
    virtual std::string name() const noexcept = 0;
};

} // namespace mobilellm
