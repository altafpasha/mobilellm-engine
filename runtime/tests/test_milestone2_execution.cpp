#include <gtest/gtest.h>
#include "mobilellm/engine.h"
#include "runtime/src/engine/engine_impl.h"
#include "backends/mock/mock_backend.h"
#include <future>
#include <atomic>
#include <vector>
#include <string>
#include <chrono>

using namespace mobilellm;

class Milestone2ExecutionTest : public ::testing::Test {
protected:
    std::unique_ptr<EngineImpl> engine;
    MockBackend* raw_mock{nullptr};
    ModelHandle model_handle{0};

    void SetUp() override {
        engine = std::make_unique<EngineImpl>();
        auto mock = std::make_unique<MockBackend>();
        raw_mock = mock.get();
        engine->set_backend_override(std::move(mock));

        EngineConfig cfg;
        cfg.max_total_memory_budget_mb = 1024.0f;
        ASSERT_TRUE(engine->initialize(cfg).is_ok());

        ModelLoadConfig m_cfg;
        m_cfg.model_path = "/tmp/mock_model.gguf";
        m_cfg.memory_budget_mb = 256.0f;
        auto m_res = engine->load_model(m_cfg);
        ASSERT_TRUE(m_res.is_ok());
        model_handle = m_res.value();
    }

    void TearDown() override {
        if (engine) {
            engine->shutdown();
            engine.reset();
        }
    }
};

// 1. Single request through scheduler
TEST_F(Milestone2ExecutionTest, SingleRequestThroughScheduler) {
    SessionConfig s_cfg;
    auto s_res = engine->create_session(model_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle session = s_res.value();

    GenerationRequest req{"Hello MobileLLM", {0.7f, 0.9f, 40, 0.05f, 1.1f, 5, {}}};
    std::vector<TokenChunk> chunks;
    std::promise<std::pair<Status, GenerationMetrics>> comp_promise;
    auto comp_future = comp_promise.get_future();

    auto gen_res = engine->generate_async(
        session,
        req,
        [&](const TokenChunk& chunk) {
            chunks.push_back(chunk);
            return true;
        },
        [&](const Status& st, const GenerationMetrics& m) {
            comp_promise.set_value({st, m});
        }
    );
    ASSERT_TRUE(gen_res.is_ok());

    ASSERT_EQ(comp_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    auto [status, metrics] = comp_future.get();

    EXPECT_TRUE(status.is_ok());
    EXPECT_EQ(chunks.size(), 5);
    EXPECT_EQ(metrics.completion_tokens, 5);
    EXPECT_GT(metrics.prompt_tokens, 0);
    EXPECT_GT(metrics.time_to_first_token_ms, 0.0);
    EXPECT_GT(metrics.decode_tokens_per_second, 0.0);

    for (size_t i = 0; i < chunks.size(); ++i) {
        EXPECT_FALSE(chunks[i].text.empty());
    }
}

// 2. Multiple queued requests
TEST_F(Milestone2ExecutionTest, MultipleQueuedRequests) {
    SessionConfig s_cfg;
    auto s_res = engine->create_session(model_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle session = s_res.value();

    const int NUM_REQS = 4;
    std::vector<std::promise<Status>> promises(NUM_REQS);
    std::vector<std::atomic<uint32_t>> token_counts(NUM_REQS);

    for (int i = 0; i < NUM_REQS; ++i) {
        token_counts[i] = 0;
        GenerationRequest req{"Queued prompt " + std::to_string(i), {0.7f, 0.9f, 40, 0.05f, 1.1f, 4, {}}};
        auto r_res = engine->generate_async(
            session,
            req,
            [&, i](const TokenChunk&) {
                token_counts[i]++;
                return true;
            },
            [&, i](const Status& st, const GenerationMetrics&) {
                promises[i].set_value(st);
            }
        );
        ASSERT_TRUE(r_res.is_ok());
    }

    for (int i = 0; i < NUM_REQS; ++i) {
        auto fut = promises[i].get_future();
        ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        EXPECT_TRUE(fut.get().is_ok());
        EXPECT_EQ(token_counts[i].load(), 4);
    }
}

// 3. Streaming callback order
TEST_F(Milestone2ExecutionTest, StreamingCallbackOrder) {
    SessionConfig s_cfg;
    auto s_res = engine->create_session(model_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle session = s_res.value();

    GenerationRequest req{"Testing order", {0.7f, 0.9f, 40, 0.05f, 1.1f, 6, {}}};
    std::vector<int32_t> token_ids;
    std::promise<void> done;

    auto gen_res = engine->generate_async(
        session,
        req,
        [&](const TokenChunk& chunk) {
            token_ids.push_back(chunk.token_id);
            return true;
        },
        [&](const Status&, const GenerationMetrics&) {
            done.set_value();
        }
    );
    ASSERT_TRUE(gen_res.is_ok());
    ASSERT_EQ(done.get_future().wait_for(std::chrono::seconds(5)), std::future_status::ready);

    EXPECT_EQ(token_ids.size(), 6);
    // In MockBackend, tokens are deterministic: 'a' + (step % 26)
    for (size_t i = 0; i < token_ids.size(); ++i) {
        EXPECT_EQ(token_ids[i], 'a' + static_cast<int32_t>(i));
    }
}

// 4. Cancellation before execution
TEST_F(Milestone2ExecutionTest, CancellationBeforeExecution) {
    SessionConfig s_cfg;
    auto s_res = engine->create_session(model_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle session = s_res.value();

    std::promise<Status> comp_promise;
    std::atomic<uint32_t> tokens_received{0};

    GenerationRequest req{"Cancel early", {0.7f, 0.9f, 40, 0.05f, 1.1f, 10, {}}};
    auto gen_res = engine->generate_async(
        session,
        req,
        [&](const TokenChunk&) {
            tokens_received++;
            return true;
        },
        [&](const Status& st, const GenerationMetrics&) {
            comp_promise.set_value(st);
        }
    );
    ASSERT_TRUE(gen_res.is_ok());
    GenerationHandle handle = gen_res.value();

    // Cancel immediately before or right as execution starts
    EXPECT_TRUE(engine->cancel(handle).is_ok());

    auto fut = comp_promise.get_future();
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    Status status = fut.get();
    EXPECT_EQ(status.code, StatusCode::CANCELLED);
}

// 5. Cancellation during generation
TEST_F(Milestone2ExecutionTest, CancellationDuringGeneration) {
    SessionConfig s_cfg;
    auto s_res = engine->create_session(model_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle session = s_res.value();

    std::promise<Status> comp_promise;
    std::atomic<uint32_t> tokens_received{0};

    GenerationRequest req{"Cancel via callback", {0.7f, 0.9f, 40, 0.05f, 1.1f, 10, {}}};
    auto gen_res = engine->generate_async(
        session,
        req,
        [&](const TokenChunk&) {
            tokens_received++;
            if (tokens_received == 3) {
                return false; // Request abort via callback
            }
            return true;
        },
        [&](const Status& st, const GenerationMetrics&) {
            comp_promise.set_value(st);
        }
    );
    ASSERT_TRUE(gen_res.is_ok());

    auto fut = comp_promise.get_future();
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    Status status = fut.get();
    EXPECT_EQ(status.code, StatusCode::CANCELLED);
    EXPECT_EQ(tokens_received.load(), 3);
}

// 6. Backend error propagation
TEST_F(Milestone2ExecutionTest, BackendErrorPropagation) {
    SessionConfig s_cfg;
    auto s_res = engine->create_session(model_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle session = s_res.value();

    raw_mock->set_simulate_eval_failure(true);

    std::promise<Status> comp_promise;
    GenerationRequest req{"Should fail eval", {0.7f, 0.9f, 40, 0.05f, 1.1f, 5, {}}};
    auto gen_res = engine->generate_async(
        session,
        req,
        nullptr,
        [&](const Status& st, const GenerationMetrics&) {
            comp_promise.set_value(st);
        }
    );
    ASSERT_TRUE(gen_res.is_ok());

    auto fut = comp_promise.get_future();
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    Status status = fut.get();
    EXPECT_FALSE(status.is_ok());
    EXPECT_EQ(status.code, StatusCode::BACKEND_EXECUTION_ERROR);

    raw_mock->set_simulate_eval_failure(false);
}

// 7. Queue rejection
TEST_F(Milestone2ExecutionTest, QueueRejection) {
    // Test that RequestQueue rejects enqueues when capacity is exhausted
    scheduler::RequestQueue small_queue(2); // capacity of 2 requests

    auto req1 = std::make_shared<scheduler::QueuedRequest>();
    req1->request_id = 1;
    auto req2 = std::make_shared<scheduler::QueuedRequest>();
    req2->request_id = 2;
    auto req3 = std::make_shared<scheduler::QueuedRequest>();
    req3->request_id = 3;

    EXPECT_TRUE(small_queue.enqueue(req1).is_ok());
    EXPECT_TRUE(small_queue.enqueue(req2).is_ok());
    // Exceed capacity
    Status rej = small_queue.enqueue(req3);
    EXPECT_FALSE(rej.is_ok());
    EXPECT_EQ(rej.code, StatusCode::OUT_OF_MEMORY);
}

// 8. Two independent sessions generating concurrently
TEST_F(Milestone2ExecutionTest, TwoIndependentSessionsGeneratingConcurrently) {
    SessionConfig s_cfg;
    auto s1_res = engine->create_session(model_handle, s_cfg);
    auto s2_res = engine->create_session(model_handle, s_cfg);
    ASSERT_TRUE(s1_res.is_ok());
    ASSERT_TRUE(s2_res.is_ok());

    SessionHandle s1 = s1_res.value();
    SessionHandle s2 = s2_res.value();

    std::promise<Status> p1, p2;
    std::atomic<uint32_t> count1{0}, count2{0};

    GenerationRequest req1{"Prompt One", {0.7f, 0.9f, 40, 0.05f, 1.1f, 4, {}}};
    GenerationRequest req2{"Prompt Two", {0.7f, 0.9f, 40, 0.05f, 1.1f, 4, {}}};

    ASSERT_TRUE(engine->generate_async(
        s1, req1,
        [&](const TokenChunk&) { count1++; return true; },
        [&](const Status& st, const GenerationMetrics&) { p1.set_value(st); }
    ).is_ok());

    ASSERT_TRUE(engine->generate_async(
        s2, req2,
        [&](const TokenChunk&) { count2++; return true; },
        [&](const Status& st, const GenerationMetrics&) { p2.set_value(st); }
    ).is_ok());

    auto f1 = p1.get_future();
    auto f2 = p2.get_future();

    ASSERT_EQ(f1.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_EQ(f2.wait_for(std::chrono::seconds(5)), std::future_status::ready);

    EXPECT_TRUE(f1.get().is_ok());
    EXPECT_TRUE(f2.get().is_ok());
    EXPECT_EQ(count1.load(), 4);
    EXPECT_EQ(count2.load(), 4);
}

// 9. Model unload with active requests
TEST_F(Milestone2ExecutionTest, ModelUnloadWithActiveRequests) {
    SessionConfig s_cfg;
    auto s_res = engine->create_session(model_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle session = s_res.value();

    std::promise<Status> comp_promise;
    GenerationRequest req{"Long running prompt", {0.7f, 0.9f, 40, 0.05f, 1.1f, 50, {}}};

    auto gen_res = engine->generate_async(
        session,
        req,
        [](const TokenChunk&) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            return true;
        },
        [&](const Status& st, const GenerationMetrics&) {
            comp_promise.set_value(st);
        }
    );
    ASSERT_TRUE(gen_res.is_ok());

    // Unload model while request is active!
    Status un_st = engine->unload_model(model_handle);
    EXPECT_TRUE(un_st.is_ok());

    auto fut = comp_promise.get_future();
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    Status comp_st = fut.get();
    // Request must retire with either OK (if finished) or CANCELLED
    EXPECT_TRUE(comp_st.is_ok() || comp_st.code == StatusCode::CANCELLED);
}

// 10. Engine shutdown with queued and active requests
TEST_F(Milestone2ExecutionTest, EngineShutdownWithQueuedAndActiveRequests) {
    SessionConfig s_cfg;
    auto s_res = engine->create_session(model_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle session = s_res.value();

    for (int i = 0; i < 5; ++i) {
        GenerationRequest req{"Shutdown stress prompt " + std::to_string(i), {0.7f, 0.9f, 40, 0.05f, 1.1f, 30, {}}};
        engine->generate_async(
            session,
            req,
            [](const TokenChunk&) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                return true;
            },
            nullptr
        );
    }

    // Call shutdown immediately with active and queued requests
    Status shut_st = engine->shutdown();
    EXPECT_TRUE(shut_st.is_ok());
}

// 11. Callback reentrancy (calling engine API inside callback without deadlock)
TEST_F(Milestone2ExecutionTest, CallbackReentrancy) {
    SessionConfig s_cfg;
    auto s_res = engine->create_session(model_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle session = s_res.value();

    std::promise<void> done;
    std::atomic<bool> reentrancy_succeeded{false};

    GenerationRequest req{"Reentrancy test", {0.7f, 0.9f, 40, 0.05f, 1.1f, 3, {}}};
    auto gen_res = engine->generate_async(
        session,
        req,
        [&](const TokenChunk&) {
            // Re-entrant call inside token callback!
            EngineMetrics m = engine->get_metrics();
            EXPECT_GT(m.active_sessions, 0);
            return true;
        },
        [&](const Status& st, const GenerationMetrics&) {
            // Re-entrant call inside completion callback!
            EngineMetrics m = engine->get_metrics();
            EXPECT_GT(m.active_sessions, 0);
            if (st.is_ok()) {
                reentrancy_succeeded.store(true);
            }
            done.set_value();
        }
    );
    ASSERT_TRUE(gen_res.is_ok());

    ASSERT_EQ(done.get_future().wait_for(std::chrono::seconds(5)), std::future_status::ready);
    EXPECT_TRUE(reentrancy_succeeded.load());
}

// 12. Callback lifetime and cleanup
TEST_F(Milestone2ExecutionTest, CallbackLifetimeAndCleanup) {
    SessionConfig s_cfg;
    auto s_res = engine->create_session(model_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle session = s_res.value();

    auto lifetime_token = std::make_shared<int>(42);
    std::weak_ptr<int> weak_token = lifetime_token;
    std::promise<void> done;

    GenerationRequest req{"Lifetime check", {0.7f, 0.9f, 40, 0.05f, 1.1f, 2, {}}};
    auto gen_res = engine->generate_async(
        session,
        req,
        [lifetime_token](const TokenChunk&) {
            EXPECT_EQ(*lifetime_token, 42);
            return true;
        },
        [lifetime_token, &done](const Status&, const GenerationMetrics&) {
            EXPECT_EQ(*lifetime_token, 42);
            done.set_value();
        }
    );
    ASSERT_TRUE(gen_res.is_ok());

    // Release caller's strong ref
    lifetime_token.reset();
    EXPECT_FALSE(weak_token.expired()); // Still held by scheduler request

    ASSERT_EQ(done.get_future().wait_for(std::chrono::seconds(5)), std::future_status::ready);

    // Give scheduler thread a brief moment to finish step retirement
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    EXPECT_TRUE(weak_token.expired()); // Successfully released after retirement!
}

// 13. Repeated request submission and cancellation
TEST_F(Milestone2ExecutionTest, RepeatedRequestSubmissionAndCancellation) {
    SessionConfig s_cfg;
    auto s_res = engine->create_session(model_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle session = s_res.value();

    for (int cycle = 0; cycle < 20; ++cycle) {
        std::promise<void> cycle_done;
        GenerationRequest req{"Rapid cycle " + std::to_string(cycle), {0.7f, 0.9f, 40, 0.05f, 1.1f, 10, {}}};
        auto g_res = engine->generate_async(
            session,
            req,
            nullptr,
            [&cycle_done](const Status&, const GenerationMetrics&) {
                cycle_done.set_value();
            }
        );
        ASSERT_TRUE(g_res.is_ok());
        EXPECT_TRUE(engine->cancel(g_res.value()).is_ok());
        ASSERT_EQ(cycle_done.get_future().wait_for(std::chrono::seconds(2)), std::future_status::ready);
    }
}

// 14. No regression in existing memory reservation tests
TEST_F(Milestone2ExecutionTest, NoRegressionInMemoryReservationTests) {
    auto mem_mgr = engine->memory_manager();
    ASSERT_NE(mem_mgr, nullptr);

    auto status_before = mem_mgr->get_status();
    EXPECT_GT(status_before.current_usage_bytes, 0); // Model reserved

    SessionConfig s_cfg;
    auto s_res = engine->create_session(model_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle session = s_res.value();

    auto status_session = mem_mgr->get_status();
    EXPECT_GT(status_session.kv_cache_bytes, 0); // Session KV reserved

    // Perform generation
    std::promise<void> done;
    GenerationRequest req{"Memory test prompt", {0.7f, 0.9f, 40, 0.05f, 1.1f, 3, {}}};
    ASSERT_TRUE(engine->generate_async(
        session,
        req,
        nullptr,
        [&done](const Status& st, const GenerationMetrics&) {
            EXPECT_TRUE(st.is_ok());
            done.set_value();
        }
    ).is_ok());

    ASSERT_EQ(done.get_future().wait_for(std::chrono::seconds(5)), std::future_status::ready);

    // Destroy session releases session reservation
    EXPECT_TRUE(engine->destroy_session(session).is_ok());
    auto status_after_destroy = mem_mgr->get_status();
    EXPECT_EQ(status_after_destroy.kv_cache_bytes, 0);

    // Unload model releases model reservation
    EXPECT_TRUE(engine->unload_model(model_handle).is_ok());
    auto status_after_unload = mem_mgr->get_status();
    EXPECT_EQ(status_after_unload.current_usage_bytes, 0);
}

