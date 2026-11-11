#pragma once

#include "mobilellm/types.h"
#include <cstdint>
#include <string>
#include <memory>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace mobilellm::scheduler {

enum class RequestPriority : uint32_t {
    INTERACTIVE = 0,
    NORMAL = 1,
    BACKGROUND = 2
};

struct QueuedRequest {
    GenerationHandle request_id{0};
    SessionHandle session_handle{0};
    std::string prompt;
    SamplingParams sampling;
    RequestPriority priority{RequestPriority::NORMAL};
    TokenCallback token_cb;
    CompletionCallback complete_cb;
    uint64_t enqueue_time_us{0};
    std::atomic<bool> is_cancelled{false};
    uint32_t generated_tokens{0};
    GenerationMetrics metrics;
};

/**
 * @brief Thread-safe priority request queue supporting fair scheduling and concurrency throttling.
 */
class RequestQueue {
public:
    explicit RequestQueue(size_t max_capacity = 256);
    ~RequestQueue() = default;

    /**
     * @brief Enqueues a generation request with priority insertion.
     */
    Status enqueue(std::shared_ptr<QueuedRequest> request) noexcept;

    /**
     * @brief Pops the next highest priority, fair-selected request.
     * Blocks if wait_for_data is true and queue is empty.
     */
    std::shared_ptr<QueuedRequest> dequeue(bool wait_for_data = false) noexcept;

    /**
     * @brief Cancels a pending or active request by handle.
     */
    Status cancel(GenerationHandle request_id) noexcept;

    /**
     * @brief Returns number of pending requests.
     */
    size_t size() const noexcept;

    bool empty() const noexcept;

    void clear() noexcept;

    /**
     * @brief Wakes any waiting dequeue threads.
     */
    void stop() noexcept;

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    size_t max_capacity_;
    bool stopped_{false};

    // Priority queues: 0 = INTERACTIVE, 1 = NORMAL, 2 = BACKGROUND
    std::deque<std::shared_ptr<QueuedRequest>> interactive_queue_;
    std::deque<std::shared_ptr<QueuedRequest>> normal_queue_;
    std::deque<std::shared_ptr<QueuedRequest>> background_queue_;

    // Fair scheduling deficit counter across sessions
    [[maybe_unused]] SessionHandle last_served_session_{0};
};

} // namespace mobilellm::scheduler
