#pragma once

#include "mobilellm/types.h"
#include "backends/common/backend.h"
#include "mobilellm/scheduler/kv_cache_manager.h"
#include <cstdint>
#include <unordered_map>
#include <mutex>
#include <vector>

namespace mobilellm::scheduler {

enum class SessionState : uint32_t {
    IDLE = 0,
    ACTIVE = 1,
    PREEMPTED = 2,
    TRUNCATED = 3,
    CLOSED = 4
};

enum class ContextOverflowPolicy : uint32_t {
    REJECT_NEW = 0,       // Return CONTEXT_EXCEEDED
    SLIDING_WINDOW = 1    // Truncate oldest context tokens
};

struct SessionMetadata {
    SessionHandle handle{0};
    ModelHandle model_handle{0};
    BackendSessionHandle backend_handle{0};
    SessionState state{SessionState::IDLE};
    uint32_t current_tokens{0};
    uint32_t max_context_tokens{2048};
    uint64_t last_active_us{0};
    uint32_t turn_count{0};
    ContextOverflowPolicy overflow_policy{ContextOverflowPolicy::SLIDING_WINDOW};
};

/**
 * @brief Manages conversation session lifecycles, context limits, LRU tracking, and cleanup.
 */
class SessionManager {
public:
    explicit SessionManager(std::shared_ptr<KVCacheManager> kv_manager = nullptr);
    ~SessionManager() = default;

    Result<SessionHandle> create_session(
        ModelHandle model,
        const SessionConfig& config,
        ContextOverflowPolicy policy = ContextOverflowPolicy::SLIDING_WINDOW
    ) noexcept;

    Result<SessionHandle> create_session_with_handle(
        SessionHandle handle,
        ModelHandle model,
        BackendSessionHandle backend_handle,
        const SessionConfig& config,
        ContextOverflowPolicy policy = ContextOverflowPolicy::SLIDING_WINDOW
    ) noexcept;

    Status destroy_session(SessionHandle session) noexcept;

    Result<SessionMetadata> get_session_info(SessionHandle session) const noexcept;

    /**
     * @brief Prepares session context for new incoming tokens, applying truncation policy if necessary.
     */
    Status prepare_context_for_tokens(SessionHandle session, uint32_t incoming_tokens) noexcept;

    /**
     * @brief Records completion of an interaction turn.
     */
    Status record_turn_completion(SessionHandle session, uint32_t generated_tokens) noexcept;

    /**
     * @brief Finds and preempts/evicts the least recently used idle session to free memory.
     */
    Result<SessionHandle> evict_lru_session() noexcept;

    uint32_t active_session_count() const noexcept;
    uint32_t total_session_count() const noexcept;

    std::vector<SessionHandle> get_active_sessions() const noexcept;

private:
    mutable std::mutex mutex_;
    std::shared_ptr<KVCacheManager> kv_manager_;
    std::atomic<uint64_t> next_session_id_{1};
    std::unordered_map<SessionHandle, SessionMetadata> sessions_;

    static uint64_t get_current_time_us() noexcept;
};

} // namespace mobilellm::scheduler

