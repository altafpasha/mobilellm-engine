#include <gtest/gtest.h>
#include "mobilellm/scheduler/scheduler.h"
#include "backends/mock/mock_backend.h"

using namespace mobilellm::scheduler;
using namespace mobilellm;

TEST(ContinuousBatchingTest, DynamicAdmissionAndRetirement) {
    auto backend = std::make_shared<MockBackend>();
    ASSERT_TRUE(backend->initialize(BackendConfig{}).is_ok());

    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/mock.gguf";
    auto m_res = backend->load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok());
    BackendModelHandle m_handle = m_res.value();

    SchedulerConfig cfg;
    cfg.max_batch_size = 2; // Batch capacity of 2 requests
    ContinuousBatchScheduler scheduler(backend, cfg);

    // Create 3 sessions
    SessionConfig s_cfg;
    auto s1_res = backend->create_session(m_handle, s_cfg);
    auto s2_res = backend->create_session(m_handle, s_cfg);
    auto s3_res = backend->create_session(m_handle, s_cfg);

    SessionHandle s1 = scheduler.session_manager()->create_session(m_handle, s_cfg).value();
    SessionHandle s2 = scheduler.session_manager()->create_session(m_handle, s_cfg).value();
    SessionHandle s3 = scheduler.session_manager()->create_session(m_handle, s_cfg).value();

    // 1. Submit Request 1 (5 tokens)
    GenerationRequest req1{"Prompt 1", {0.7f, 0.9f, 40, 0.05f, 1.1f, 5, {}}};
    uint32_t s1_tokens = 0;
    bool s1_completed = false;
    auto r1_res = scheduler.submit_request(s1, req1, RequestPriority::NORMAL,
        [&](const TokenChunk&) { s1_tokens++; return true; },
        [&](const Status& st, const GenerationMetrics&) { s1_completed = true; }
    );
    ASSERT_TRUE(r1_res.is_ok());

    // Step 1: Request 1 is admitted into batch and generates token 1
    uint32_t active = scheduler.step_batch();
    EXPECT_EQ(active, 1);
    EXPECT_EQ(s1_tokens, 1);

    // 2. Submit Request 2 (2 tokens) while Request 1 is in-flight!
    GenerationRequest req2{"Prompt 2", {0.7f, 0.9f, 40, 0.05f, 1.1f, 2, {}}};
    uint32_t s2_tokens = 0;
    bool s2_completed = false;
    auto r2_res = scheduler.submit_request(s2, req2, RequestPriority::NORMAL,
        [&](const TokenChunk&) { s2_tokens++; return true; },
        [&](const Status& st, const GenerationMetrics&) { s2_completed = true; }
    );
    ASSERT_TRUE(r2_res.is_ok());

    // Step 2: Request 2 is admitted into the batch. Both R1 and R2 decode!
    active = scheduler.step_batch();
    EXPECT_EQ(active, 2);
    EXPECT_EQ(s1_tokens, 2);
    EXPECT_EQ(s2_tokens, 1);

    // Step 3: Decode step. R2 hits max_tokens (2 tokens) and retires!
    active = scheduler.step_batch();
    EXPECT_EQ(s2_tokens, 2);
    EXPECT_TRUE(s2_completed); // R2 is completed and retired!
    EXPECT_EQ(s1_tokens, 3);

    // 3. Submit Request 3 (3 tokens) -> immediately fills the slot vacated by R2!
    GenerationRequest req3{"Prompt 3", {0.7f, 0.9f, 40, 0.05f, 1.1f, 3, {}}};
    uint32_t s3_tokens = 0;
    bool s3_completed = false;
    auto r3_res = scheduler.submit_request(s3, req3, RequestPriority::NORMAL,
        [&](const TokenChunk&) { s3_tokens++; return true; },
        [&](const Status& st, const GenerationMetrics&) { s3_completed = true; }
    );
    ASSERT_TRUE(r3_res.is_ok());

    // Step 4: R3 admitted alongside remaining R1
    active = scheduler.step_batch();
    EXPECT_EQ(s1_tokens, 4);
    EXPECT_EQ(s3_tokens, 1);

    // Step 5: R1 finishes (5 tokens)
    active = scheduler.step_batch();
    EXPECT_EQ(s1_tokens, 5);
    EXPECT_TRUE(s1_completed);
    EXPECT_EQ(s3_tokens, 2);

    // Step 6: R3 finishes
    active = scheduler.step_batch();
    EXPECT_EQ(s3_tokens, 3);
    EXPECT_TRUE(s3_completed);

    // Step 7: Batch is now empty
    active = scheduler.step_batch();
    EXPECT_EQ(active, 0);
}

