#include "mobilellm/scheduler/scheduler.h"
#include <chrono>

namespace mobilellm::scheduler {

ContinuousBatchScheduler::ContinuousBatchScheduler(
    std::shared_ptr<IBackend> backend,
    const SchedulerConfig& config
) : ContinuousBatchScheduler(std::move(backend), nullptr, config) {}

ContinuousBatchScheduler::ContinuousBatchScheduler(
    std::shared_ptr<IBackend> backend,
    std::shared_ptr<MemoryBudgetManager> memory_mgr,
    const SchedulerConfig& config
) : backend_(std::move(backend)), config_(config) {
    if (memory_mgr) {
        memory_mgr_ = std::move(memory_mgr);
    } else {
        memory_mgr_ = std::make_shared<MemoryBudgetManager>(config.memory_safety_margin_bytes);
    }
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

    std::vector<std::pair<CompletionCallback, GenerationMetrics>> to_complete;
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        for (auto& req : active_batch_) {
            if (req) {
                session_mgr_->record_turn_completion(req->session_handle, req->generated_tokens);
                if (req->complete_cb) {
                    to_complete.emplace_back(std::move(req->complete_cb), req->metrics);
                }
            }
        }
        active_batch_.clear();
    }

    for (auto& [cb, metrics] : to_complete) {
        cb(Status::error(StatusCode::CANCELLED, "Scheduler stopped"), metrics);
    }
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
    q_req->backend_session_handle = sess_res.value().backend_handle > 0 ? sess_res.value().backend_handle : session;
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
    struct CompletionDispatch {
        CompletionCallback cb;
        Status status;
        GenerationMetrics metrics;
    };
    std::vector<CompletionDispatch> to_complete;

    // 1. Admission phase: Admit waiting requests into active batch slots up to max_batch_size
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        auto can_admit_session = [this](SessionHandle s) {
            for (const auto& r : active_batch_) {
                if (r && r->session_handle == s) {
                    return false; // Session already in active batch
                }
            }
            return true;
        };

        while (active_batch_.size() < config_.max_batch_size && !queue_->empty()) {
            auto new_req = queue_->dequeue(can_admit_session, false);
            if (!new_req) break;

            if (new_req->is_cancelled.load(std::memory_order_relaxed)) {
                if (new_req->complete_cb) {
                    to_complete.push_back({std::move(new_req->complete_cb),
                                          Status::error(StatusCode::CANCELLED, "Request cancelled before execution"),
                                          new_req->metrics});
                }
                continue;
            }

            auto now = std::chrono::steady_clock::now().time_since_epoch();
            new_req->dequeue_time_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
            new_req->queue_wait_time_ms = (new_req->dequeue_time_us >= new_req->enqueue_time_us)
                ? static_cast<double>(new_req->dequeue_time_us - new_req->enqueue_time_us) / 1000.0 : 0.0;

            // Context preparation
            uint32_t prompt_tokens_est = static_cast<uint32_t>(new_req->prompt.size() / 3 + 4);
            Status prep_st = session_mgr_->prepare_context_for_tokens(new_req->session_handle, prompt_tokens_est);
            if (!prep_st.is_ok()) {
                if (new_req->complete_cb) {
                    to_complete.push_back({std::move(new_req->complete_cb), prep_st, new_req->metrics});
                }
                continue;
            }

            // Prefill evaluation on backend
            auto t_prefill_start = std::chrono::steady_clock::now();
            new_req->prefill_start_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t_prefill_start.time_since_epoch()).count());

            std::vector<int32_t> out_tokens;
            Status eval_st = backend_->evaluate_prompt(new_req->backend_session_handle, new_req->prompt, out_tokens);

            auto t_prefill_end = std::chrono::steady_clock::now();
            new_req->prefill_end_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(t_prefill_end.time_since_epoch()).count());
            new_req->prefill_time_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();

            if (!eval_st.is_ok()) {
                if (new_req->complete_cb) {
                    to_complete.push_back({std::move(new_req->complete_cb), eval_st, new_req->metrics});
                }
                continue;
            }

            new_req->metrics.prompt_tokens = static_cast<uint32_t>(out_tokens.size());
            new_req->decode_start_us = new_req->prefill_end_us;
            active_batch_.push_back(new_req);
        }
    }

    // Dispatch any admission failure completions outside batch_mutex_
    for (auto& item : to_complete) {
        if (item.cb) item.cb(item.status, item.metrics);
    }
    to_complete.clear();

    // 2. Decode iteration phase: sample 1 token for each request in the active batch
    std::vector<std::shared_ptr<QueuedRequest>> current_batch_snapshot;
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        current_batch_snapshot = active_batch_;
    }

    if (current_batch_snapshot.empty()) {
        return 0;
    }

    std::vector<GenerationHandle> requests_to_retire;
    std::unordered_map<GenerationHandle, Status> retirement_statuses;

    for (auto& req : current_batch_snapshot) {
        if (!req) continue;

        if (req->is_cancelled.load(std::memory_order_relaxed)) {
            requests_to_retire.push_back(req->request_id);
            retirement_statuses[req->request_id] = Status::error(StatusCode::CANCELLED, "Request cancelled");
            continue;
        }

        auto sample_res = backend_->sample_next_token(req->backend_session_handle, req->sampling);
        if (!sample_res.is_ok()) {
            requests_to_retire.push_back(req->request_id);
            retirement_statuses[req->request_id] = sample_res.status();
            continue;
        }

        int32_t tok_id = sample_res.value();
        if (backend_->is_eos_token(req->backend_session_handle, tok_id)) {
            requests_to_retire.push_back(req->request_id);
            retirement_statuses[req->request_id] = Status::ok();
            continue;
        }

        std::string piece = backend_->decode_token(req->backend_session_handle, tok_id);
        req->generated_tokens++;
        req->metrics.completion_tokens = req->generated_tokens;

        auto now_steady = std::chrono::steady_clock::now();
        if (!req->first_token_emitted) {
            req->first_token_emitted = true;
            uint64_t now_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now_steady.time_since_epoch()).count());
            req->metrics.time_to_first_token_ms = (now_us >= req->enqueue_time_us)
                ? static_cast<double>(now_us - req->enqueue_time_us) / 1000.0 : 0.0;
        }

        // Invoke token callback WITHOUT holding batch_mutex_
        bool keep_going = true;
        if (req->token_cb) {
            TokenChunk chunk{piece, tok_id, false, 0.0f};
            keep_going = req->token_cb(chunk);
        }

        if (!keep_going) {
            req->is_cancelled.store(true);
            requests_to_retire.push_back(req->request_id);
            retirement_statuses[req->request_id] = Status::error(StatusCode::CANCELLED, "Callback requested cancellation");
            continue;
        }

        if (req->generated_tokens >= req->sampling.max_tokens) {
            requests_to_retire.push_back(req->request_id);
            retirement_statuses[req->request_id] = Status::ok();
            continue;
        }
    }

    // 3. Retire completed requests under lock, then dispatch callbacks outside lock
    if (!requests_to_retire.empty()) {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        auto now_steady = std::chrono::steady_clock::now();
        uint64_t now_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now_steady.time_since_epoch()).count());

        for (auto req_id : requests_to_retire) {
            for (auto it = active_batch_.begin(); it != active_batch_.end(); ++it) {
                if ((*it)->request_id == req_id) {
                    auto req = *it;
                    active_batch_.erase(it);

                    req->decode_end_us = now_us;
                    double total_decode_sec = (req->decode_end_us >= req->decode_start_us)
                        ? static_cast<double>(req->decode_end_us - req->decode_start_us) / 1000000.0 : 0.0;
                    req->decode_time_ms = total_decode_sec * 1000.0;
                    if (total_decode_sec > 0.0 && req->generated_tokens > 0) {
                        req->metrics.decode_tokens_per_second = static_cast<double>(req->generated_tokens) / total_decode_sec;
                    }

                    session_mgr_->record_turn_completion(req->session_handle, req->generated_tokens);

                    if (req->complete_cb) {
                        to_complete.push_back({std::move(req->complete_cb), retirement_statuses[req_id], req->metrics});
                    }
                    break;
                }
            }
        }
    }

    // Dispatch all completion callbacks outside locks
    for (auto& item : to_complete) {
        if (item.cb) item.cb(item.status, item.metrics);
    }

    std::lock_guard<std::mutex> lock(batch_mutex_);
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
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }
}

uint32_t ContinuousBatchScheduler::active_batch_size() const noexcept {
    std::lock_guard<std::mutex> lock(batch_mutex_);
    return static_cast<uint32_t>(active_batch_.size());
}

} // namespace mobilellm::scheduler

