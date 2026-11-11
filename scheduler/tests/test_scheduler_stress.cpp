#include <gtest/gtest.h>
#include "mobilellm/scheduler/scheduler.h"
#include "backends/mock/mock_backend.h"
#include <atomic>
#include <future>
#include <vector>

using namespace mobilellm::scheduler;
using namespace mobilellm;

TEST(SchedulerStressTest, MultiThreadedAsyncGeneration) {
    auto backend = std::make_shared<MockBackend>();
    ASSERT_TRUE(backend->initialize(BackendConfig{}).is_ok());

    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/mock.gguf";
    auto m_res = backend->load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok());

    SchedulerConfig cfg;
    cfg.max_batch_size = 4;
    cfg.total_kv_blocks = 256;
    ContinuousBatchScheduler scheduler(backend, cfg);
    ASSERT_TRUE(scheduler.start().is_ok());

    const int NUM_CONCURRENT_REQUESTS = 16;
    std::vector<std::promise<void>> promises(NUM_CONCURRENT_REQUESTS);
    std::atomic<int> completed_count{0};

    for (int i = 0; i < NUM_CONCURRENT_REQUESTS; ++i) {
        SessionConfig s_cfg;
        backend->create_session(m_res.value(), s_cfg);
        SessionHandle s = scheduler.session_manager()->create_session(m_res.value(), s_cfg).value();

        GenerationRequest req{"Stress prompt " + std::to_string(i), {0.7f, 0.9f, 40, 0.05f, 1.1f, 8, {}}};

        scheduler.submit_request(s, req, RequestPriority::NORMAL,
            nullptr,
            [&, i](const Status& st, const GenerationMetrics&) {
                if (st.is_ok()) {
                    completed_count++;
                }
                promises[i].set_value();
            }
        );
    }

    for (int i = 0; i < NUM_CONCURRENT_REQUESTS; ++i) {
        auto fut = promises[i].get_future();
        ASSERT_EQ(fut.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    }

    EXPECT_EQ(completed_count.load(), NUM_CONCURRENT_REQUESTS);
    scheduler.stop();

    // Verify all sessions can be cleanly closed
    auto sessions = scheduler.session_manager()->get_active_sessions();
    for (auto s : sessions) {
        scheduler.session_manager()->destroy_session(s);
    }
}

TEST(SchedulerStressTest, RapidSessionAllocationLeakCheck) {
    auto kv = std::make_shared<KVCacheManager>();
    const uint32_t TOTAL_BLOCKS = 100;
    ASSERT_TRUE(kv->initialize(TOTAL_BLOCKS, 1024).is_ok());

    SessionManager sm(kv);
    SessionConfig s_cfg;

    // Run 50 cycles of allocating and destroying sessions
    for (int cycle = 0; cycle < 50; ++cycle) {
        SessionHandle s = sm.create_session(1, s_cfg).value();
        EXPECT_TRUE(sm.prepare_context_for_tokens(s, 48).is_ok()); // 3 blocks
        EXPECT_TRUE(sm.record_turn_completion(s, 16).is_ok());     // +1 block
        EXPECT_TRUE(sm.destroy_session(s).is_ok());
    }

    // Verify ZERO block leakage: all blocks must be back in the free pool!
    EXPECT_EQ(kv->allocated_blocks_count(), 0);
    EXPECT_EQ(kv->free_blocks_count(), TOTAL_BLOCKS);
}
