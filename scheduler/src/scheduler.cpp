#include "mobilellm/scheduler/scheduler.h"
#include <chrono>

namespace mobilellm::scheduler {

ContinuousBatchScheduler::ContinuousBatchScheduler(
    std::shared_ptr<IBackend> backend,
    const SchedulerConfig& config
) : backend_(std::move(backend)), config_(config) {
    memory_mgr_ = std::make_shared<MemoryBudgetManager>(config.memory_safety_margin_bytes);
    kv_mgr_ = std::make_shared<KVCacheManager>();
    kv_mgr_->initialize(config.total_kv_blocks, config.bytes_per_kv_block);
    session_mgr_ = std::make_shared<SessionManager>(kv_mgr_);
    queue_ = std::make_shared<RequestQueue>(256);

    // Register emergency low-memory cleanup
    memory_mgr_->register_trim_callback([this](TrimLevel level) {
        if (level == TrimLevel::RUNNING_CRITICAL || level == TrimLevel::COMPLETE) {
            session_mgr_->evict_lru_session();
        }
    });
}

ContinuousBatchScheduler::~ContinuousBatchScheduler() {
    stop();
}

Status ContinuousBatchScheduler::start() noexcept {
    if (is_running_.load()) {
        return Status::ok();
    }
    is_running_.store(true);
    worker_thread_ = std::thread(&ContinuousBatchScheduler::scheduler_loop, this);
    return Status::ok();
}

Status ContinuousBatchScheduler::stop() noexcept {
    if (!is_running_.load()) {
        return Status::ok();
    }
    is_running_.store(false);
    if (queue_) {
        queue_->stop();
    }
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    std::lock_guard<std::mutex> lock(batch_mutex_);
    for (size_t i = 0; i < active_batch_.size(); ++i) {
        retire_request_locked(i, Status::error(StatusCode::CANCELLED, "Scheduler stopped"));
    }
    active_batch_.clear();
    return Status::ok();
}

Result<GenerationHandle> ContinuousBatchScheduler::submit_request(
    SessionHandle session,
    const GenerationRequest& request,
    RequestPriority priority,
    TokenCallback token_cb,
    CompletionCallback complete_cb
) noexcept {
    auto sess_res = session_mgr_->get_session_info(session);
    if (!sess_res.is_ok()) {
        return sess_res.status();
    }

    if (request.prompt.empty()) {
        return Status::error(StatusCode::INVALID_ARGUMENT, "Prompt cannot be empty");
    }

    GenerationHandle req_id = next_request_id_++;
    auto q_req = std::make_shared<QueuedRequest>();
    q_req->request_id = req_id;
    q_req->session_handle = session;
    q_req->prompt = request.prompt;
    q_req->sampling = request.sampling;
    q_req->priority = priority;
    q_req->token_cb = std::move(token_cb);
    q_req->complete_cb = std::move(complete_cb);

    auto now = std::chrono::steady_clock::now().time_since_epoch();
    q_req->enqueue_time_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now).count());

    Status enq_st = queue_->enqueue(q_req);
    if (!enq_st.is_ok()) {
        return enq_st;
    }

    return req_id;
}

Status ContinuousBatchScheduler::cancel_request(GenerationHandle handle) noexcept {
    // 1. Check in queue
    queue_->cancel(handle);

    // 2. Check in active batch
    std::lock_guard<std::mutex> lock(batch_mutex_);
    for (auto& req : active_batch_) {
        if (req && req->request_id == handle) {
            req->is_cancelled.store(true);
            return Status::ok();
        }
    }
    return Status::ok();
}

uint32_t ContinuousBatchScheduler::step_batch() noexcept {
    std::lock_guard<std::mutex> lock(batch_mutex_);

    // 1. Admission phase: Admit waiting requests into active batch slots up to max_batch_size
    while (active_batch_.size() < config_.max_batch_size && !queue_->empty()) {
        auto new_req = queue_->dequeue(false);
        if (!new_req) break;

        // Context preparation
        uint32_t prompt_tokens_est = static_cast<uint32_t>(new_req->prompt.size() / 3 + 4);
        Status prep_st = session_mgr_->prepare_context_for_tokens(new_req->session_handle, prompt_tokens_est);
        if (!prep_st.is_ok()) {
            if (new_req->complete_cb) {
                new_req->complete_cb(prep_st, new_req->metrics);
            }
            continue;
        }

        // Prefill evaluation on backend
        std::vector<int32_t> out_tokens;
        Status eval_st = backend_->evaluate_prompt(new_req->session_handle, new_req->prompt, out_tokens);
        if (!eval_st.is_ok()) {
            if (new_req->complete_cb) {
                new_req->complete_cb(eval_st, new_req->metrics);
            }
            continue;
        }

        new_req->metrics.prompt_tokens = static_cast<uint32_t>(out_tokens.size());
        active_batch_.push_back(new_req);
    }

    if (active_batch_.empty()) {
        return 0;
    }

    // 2. Decode iteration phase: sample 1 token for each request in the active batch
    for (size_t i = 0; i < active_batch_.size();) {
        auto req = active_batch_[i];

        if (req->is_cancelled.load(std::memory_order_relaxed)) {
            retire_request_locked(i, Status::error(StatusCode::CANCELLED, "Request cancelled"));
            active_batch_.erase(active_batch_.begin() + i);
            continue;
        }

        auto sample_res = backend_->sample_next_token(req->session_handle, req->sampling);
        if (!sample_res.is_ok()) {
            retire_request_locked(i, sample_res.status());
            active_batch_.erase(active_batch_.begin() + i);
            continue;
        }

        int32_t tok_id = sample_res.value();
        if (backend_->is_eos_token(req->session_handle, tok_id)) {
            retire_request_locked(i, Status::ok());
            active_batch_.erase(active_batch_.begin() + i);
            continue;
        }

        std::string piece = backend_->decode_token(req->session_handle, tok_id);
        req->generated_tokens++;
        req->metrics.completion_tokens = req->generated_tokens;

        bool keep_going = true;
        if (req->token_cb) {
            TokenChunk chunk{piece, tok_id, false, 0.0f};
            keep_going = req->token_cb(chunk);
        }

        if (!keep_going) {
            retire_request_locked(i, Status::error(StatusCode::CANCELLED, "Callback requested cancellation"));
            active_batch_.erase(active_batch_.begin() + i);
            continue;
        }

        if (req->generated_tokens >= req->sampling.max_tokens) {
            retire_request_locked(i, Status::ok());
            active_batch_.erase(active_batch_.begin() + i);
            continue;
        }

        ++i;
    }

    return static_cast<uint32_t>(active_batch_.size());
}

void ContinuousBatchScheduler::retire_request_locked(size_t index, const Status& status) {
    if (index >= active_batch_.size()) return;
    auto req = active_batch_[index];
    if (!req) return;

    session_mgr_->record_turn_completion(req->session_handle, req->generated_tokens);

    if (req->complete_cb) {
        req->complete_cb(status, req->metrics);
    }
}

void ContinuousBatchScheduler::scheduler_loop() {
    while (is_running_.load(std::memory_order_relaxed)) {
        uint32_t active = step_batch();
        if (active == 0 && queue_->empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
}

uint32_t ContinuousBatchScheduler::active_batch_size() const noexcept {
    std::lock_guard<std::mutex> lock(batch_mutex_);
    return static_cast<uint32_t>(active_batch_.size());
}

} // namespace mobilellm::scheduler

