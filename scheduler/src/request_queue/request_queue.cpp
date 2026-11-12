#include "mobilellm/scheduler/request_queue.h"

namespace mobilellm::scheduler {

RequestQueue::RequestQueue(size_t max_capacity) : max_capacity_(max_capacity) {}

Status RequestQueue::enqueue(std::shared_ptr<QueuedRequest> request) noexcept {
    if (!request) {
        return Status::error(StatusCode::INVALID_ARGUMENT, "Cannot enqueue null request");
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_) {
        return Status::error(StatusCode::INTERNAL_ERROR, "Queue is stopped");
    }

    size_t total_size = interactive_queue_.size() + normal_queue_.size() + background_queue_.size();
    if (total_size >= max_capacity_) {
        return Status::error(StatusCode::OUT_OF_MEMORY, "Request queue capacity exceeded");
    }

    switch (request->priority) {
        case RequestPriority::INTERACTIVE:
            interactive_queue_.push_back(request);
            break;
        case RequestPriority::NORMAL:
            normal_queue_.push_back(request);
            break;
        case RequestPriority::BACKGROUND:
            background_queue_.push_back(request);
            break;
    }

    cv_.notify_one();
    return Status::ok();
}

std::shared_ptr<QueuedRequest> RequestQueue::dequeue(bool wait_for_data) noexcept {
    return dequeue(nullptr, wait_for_data);
}

std::shared_ptr<QueuedRequest> RequestQueue::dequeue(
    const std::function<bool(SessionHandle)>& can_admit_session,
    bool wait_for_data
) noexcept {
    std::unique_lock<std::mutex> lock(mutex_);

    auto has_items = [this]() {
        return stopped_ || !interactive_queue_.empty() || !normal_queue_.empty() || !background_queue_.empty();
    };

    if (wait_for_data && !has_items()) {
        cv_.wait(lock, has_items);
    }

    if (stopped_ && !has_items()) {
        return nullptr;
    }

    std::vector<std::shared_ptr<QueuedRequest>> cancelled_items;

    // Helper to pull from a sub-queue while discarding cancelled items and respecting session predicate
    auto pull_from = [&](std::deque<std::shared_ptr<QueuedRequest>>& q) -> std::shared_ptr<QueuedRequest> {
        for (auto it = q.begin(); it != q.end(); ) {
            auto item = *it;
            if (!item) {
                it = q.erase(it);
                continue;
            }
            if (item->is_cancelled.load(std::memory_order_relaxed)) {
                cancelled_items.push_back(item);
                it = q.erase(it);
                continue;
            }
            if (!can_admit_session || can_admit_session(item->session_handle)) {
                it = q.erase(it);
                return item;
            }
            ++it;
        }
        return nullptr;
    };

    // 1. Interactive priority
    auto req = pull_from(interactive_queue_);
    if (!req) {
        // 2. Normal priority
        req = pull_from(normal_queue_);
    }
    if (!req) {
        // 3. Background priority
        req = pull_from(background_queue_);
    }

    // Release lock before invoking any application callbacks
    lock.unlock();

    for (auto& item : cancelled_items) {
        if (item->complete_cb) {
            item->complete_cb(Status::error(StatusCode::CANCELLED, "Request cancelled in queue"), item->metrics);
        }
    }

    return req;
}

Status RequestQueue::cancel(GenerationHandle request_id) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);

    auto check_queue = [request_id](std::deque<std::shared_ptr<QueuedRequest>>& q) -> bool {
        for (auto& item : q) {
            if (item && item->request_id == request_id) {
                item->is_cancelled.store(true);
                return true;
            }
        }
        return false;
    };

    if (check_queue(interactive_queue_) || check_queue(normal_queue_) || check_queue(background_queue_)) {
        return Status::ok();
    }

    return Status::error(StatusCode::INVALID_ARGUMENT, "Request ID not found in queue");
}

size_t RequestQueue::size() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return interactive_queue_.size() + normal_queue_.size() + background_queue_.size();
}

bool RequestQueue::empty() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return interactive_queue_.empty() && normal_queue_.empty() && background_queue_.empty();
}

void RequestQueue::clear() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    interactive_queue_.clear();
    normal_queue_.clear();
    background_queue_.clear();
}

void RequestQueue::stop() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
    cv_.notify_all();
}

} // namespace mobilellm::scheduler

