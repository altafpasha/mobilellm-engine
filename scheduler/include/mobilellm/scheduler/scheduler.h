#pragma once

#include "mobilellm/types.h"
#include "backends/common/backend.h"
#include "mobilellm/scheduler/memory_manager.h"
#include "mobilellm/scheduler/kv_cache_manager.h"
#include "mobilellm/scheduler/session_manager.h"
#include "mobilellm/scheduler/request_queue.h"
#include <memory>
#include <atomic>
#include <thread>
#include <vector>

namespace mobilellm::scheduler {

struct SchedulerConfig {
    uint32_t max_concurrent_requests{1}; // Mobile default: 1 on CPU to avoid core thrashing
    uint32_t max_batch_size{4};          // Supported when backend supports continuous batching
    uint32_t total_kv_blocks{512};       // 512 blocks * 16 tokens = 8,192 tokens capacity
    size_t bytes_per_kv_block{64 * 1024};// 64 KB per block
    uint64_t memory_safety_margin_bytes{DEFAULT_SAFETY_MARGIN_BYTES};
};

/**
 * @brief Continuous Batching Scheduler coordinating memory budgeting, paged KV-cache,
 * request queuing, and iteration-level batch execution under mobile constraints.
 */
class ContinuousBatchScheduler {
public:
    ContinuousBatchScheduler(
        std::shared_ptr<IBackend> backend,
        const SchedulerConfig& config = SchedulerConfig{}
    );
    ~ContinuousBatchScheduler();

    Status start() noexcept;
    Status stop() noexcept;

    /**
     * @brief Submits a generation request to the scheduler.
     */
    Result<GenerationHandle> submit_request(
        SessionHandle session,
        const GenerationRequest& request,
        RequestPriority priority,
        TokenCallback token_cb,
        CompletionCallback complete_cb
    ) noexcept;

    /**
     * @brief Cancels an active or queued generation request.
     */
    Status cancel_request(GenerationHandle handle) noexcept;

    /**
     * @brief Executes a single iteration step of continuous batching (useful for synchronous testing).
     * @return Number of active requests processed in this step.
     */
    uint32_t step_batch() noexcept;

    // Subsystem accessors
    std::shared_ptr<MemoryBudgetManager> memory_manager() const noexcept { return memory_mgr_; }
    std::shared_ptr<KVCacheManager> kv_cache_manager() const noexcept { return kv_mgr_; }
    std::shared_ptr<SessionManager> session_manager() const noexcept { return session_mgr_; }
    std::shared_ptr<RequestQueue> request_queue() const noexcept { return queue_; }

    uint32_t active_batch_size() const noexcept;
    bool is_running() const noexcept { return is_running_.load(); }

private:
    std::shared_ptr<IBackend> backend_;
    SchedulerConfig config_;

    std::shared_ptr<MemoryBudgetManager> memory_mgr_;
    std::shared_ptr<KVCacheManager> kv_mgr_;
    std::shared_ptr<SessionManager> session_mgr_;
    std::shared_ptr<RequestQueue> queue_;

    std::atomic<bool> is_running_{false};
    std::atomic<uint64_t> next_request_id_{1};

    std::thread worker_thread_;
    mutable std::mutex batch_mutex_;
    std::vector<std::shared_ptr<QueuedRequest>> active_batch_;

    void scheduler_loop();
    void retire_request_locked(size_t index, const Status& status);
};

} // namespace mobilellm::scheduler
