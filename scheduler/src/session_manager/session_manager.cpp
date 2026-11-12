#include "mobilellm/scheduler/session_manager.h"
#include <chrono>
#include <algorithm>

namespace mobilellm::scheduler {

uint64_t SessionManager::get_current_time_us() noexcept {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

SessionManager::SessionManager(std::shared_ptr<KVCacheManager> kv_manager)
    : kv_manager_(std::move(kv_manager)) {}

Result<SessionHandle> SessionManager::create_session(
    ModelHandle model,
    const SessionConfig& config,
    ContextOverflowPolicy policy
) noexcept {
    if (model == 0) {
        return Status::error(StatusCode::INVALID_ARGUMENT, "Invalid model handle for session creation");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    SessionHandle handle = next_session_id_++;

    SessionMetadata meta;
    meta.handle = handle;
    meta.model_handle = model;
    meta.backend_handle = handle;
    meta.state = SessionState::IDLE;
    meta.current_tokens = 0;
    meta.max_context_tokens = config.max_context_tokens > 0 ? config.max_context_tokens : 2048;
    meta.last_active_us = get_current_time_us();
    meta.turn_count = 0;
    meta.overflow_policy = policy;

    sessions_[handle] = meta;
    return handle;
}

Result<SessionHandle> SessionManager::create_session_with_handle(
    SessionHandle handle,
    ModelHandle model,
    BackendSessionHandle backend_handle,
    const SessionConfig& config,
    ContextOverflowPolicy policy
) noexcept {
    if (handle == 0) {
        return Status::error(StatusCode::INVALID_ARGUMENT, "Invalid session handle");
    }
    if (model == 0) {
        return Status::error(StatusCode::INVALID_ARGUMENT, "Invalid model handle for session creation");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    SessionMetadata meta;
    meta.handle = handle;
    meta.model_handle = model;
    meta.backend_handle = backend_handle > 0 ? backend_handle : handle;
    meta.state = SessionState::IDLE;
    meta.current_tokens = 0;
    meta.max_context_tokens = config.max_context_tokens > 0 ? config.max_context_tokens : 2048;
    meta.last_active_us = get_current_time_us();
    meta.turn_count = 0;
    meta.overflow_policy = policy;

    sessions_[handle] = meta;
    return handle;
}

Status SessionManager::destroy_session(SessionHandle session) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(session);
    if (it == sessions_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session handle not recognized");
    }

    if (kv_manager_) {
        kv_manager_->release_session_blocks(session);
    }

    sessions_.erase(it);
    return Status::ok();
}

Result<SessionMetadata> SessionManager::get_session_info(SessionHandle session) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(session);
    if (it == sessions_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session handle not found");
    }
    return it->second;
}

Status SessionManager::prepare_context_for_tokens(SessionHandle session, uint32_t incoming_tokens) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(session);
    if (it == sessions_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session handle not found");
    }

    SessionMetadata& meta = it->second;
    if (incoming_tokens > meta.max_context_tokens) {
        return Status::error(StatusCode::CONTEXT_EXCEEDED,
            "Prompt size of " + std::to_string(incoming_tokens) +
            " tokens exceeds maximum session context limit of " +
            std::to_string(meta.max_context_tokens) + " tokens");
    }

    if (meta.current_tokens + incoming_tokens > meta.max_context_tokens) {
        if (meta.overflow_policy == ContextOverflowPolicy::REJECT_NEW) {
            return Status::error(StatusCode::CONTEXT_EXCEEDED, "Context limit reached under REJECT_NEW policy");
        }

        // SLIDING_WINDOW: Truncate oldest context tokens
        uint32_t allowed_prior = meta.max_context_tokens - incoming_tokens;
        if (kv_manager_) {
            kv_manager_->truncate_session(session, allowed_prior);
        }
        meta.current_tokens = allowed_prior;
        meta.state = SessionState::TRUNCATED;
    }

    if (kv_manager_) {
        // Allocate blocks if not already mapped, or append tokens
        const auto* table = kv_manager_->get_session_table(session);
        if (!table) {
            auto b_res = kv_manager_->allocate_blocks_for_tokens(session, meta.current_tokens + incoming_tokens);
            if (!b_res.is_ok()) {
                return b_res.status();
            }
        } else {
            Status app_st = kv_manager_->append_tokens(session, incoming_tokens);
            if (!app_st.is_ok()) {
                return app_st;
            }
        }
    }

    meta.current_tokens += incoming_tokens;
    meta.state = SessionState::ACTIVE;
    meta.last_active_us = get_current_time_us();
    return Status::ok();
}

Status SessionManager::record_turn_completion(SessionHandle session, uint32_t generated_tokens) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = sessions_.find(session);
    if (it == sessions_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session handle not found");
    }

    SessionMetadata& meta = it->second;
    meta.current_tokens += generated_tokens;
    meta.turn_count++;
    meta.state = SessionState::IDLE;
    meta.last_active_us = get_current_time_us();
    return Status::ok();
}

Result<SessionHandle> SessionManager::evict_lru_session() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    SessionHandle lru_handle = 0;
    uint64_t oldest_time = UINT64_MAX;

    for (const auto& [handle, meta] : sessions_) {
        // Only evict idle or preempted sessions
        if (meta.state == SessionState::IDLE || meta.state == SessionState::PREEMPTED) {
            if (meta.last_active_us < oldest_time) {
                oldest_time = meta.last_active_us;
                lru_handle = handle;
            }
        }
    }

    if (lru_handle == 0) {
        return Status::error(StatusCode::OUT_OF_MEMORY, "No idle sessions available for eviction");
    }

    if (kv_manager_) {
        kv_manager_->release_session_blocks(lru_handle);
    }
    sessions_[lru_handle].state = SessionState::PREEMPTED;
    sessions_[lru_handle].current_tokens = 0;

    return lru_handle;
}

uint32_t SessionManager::active_session_count() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    uint32_t count = 0;
    for (const auto& [_, meta] : sessions_) {
        if (meta.state == SessionState::ACTIVE) count++;
    }
    return count;
}

uint32_t SessionManager::total_session_count() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<uint32_t>(sessions_.size());
}

std::vector<SessionHandle> SessionManager::get_active_sessions() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<SessionHandle> res;
    for (const auto& [handle, meta] : sessions_) {
        if (meta.state == SessionState::ACTIVE) {
            res.push_back(handle);
        }
    }
    return res;
}

} // namespace mobilellm::scheduler

