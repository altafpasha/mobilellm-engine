#include <gtest/gtest.h>
#include "mobilellm/scheduler/session_manager.h"
#include <thread>

using namespace mobilellm::scheduler;
using namespace mobilellm;

TEST(SessionManagerTest, LifecycleAndLimits) {
    auto kv = std::make_shared<KVCacheManager>();
    ASSERT_TRUE(kv->initialize(20, 1024).is_ok());

    SessionManager sm(kv);
    ModelHandle m_handle = 1;

    SessionConfig s_cfg;
    s_cfg.max_context_tokens = 64; // Max 64 tokens

    auto sess_res = sm.create_session(m_handle, s_cfg, ContextOverflowPolicy::REJECT_NEW);
    ASSERT_TRUE(sess_res.is_ok());
    SessionHandle s_handle = sess_res.value();

    EXPECT_EQ(sm.total_session_count(), 1);

    // Prepare 30 tokens
    EXPECT_TRUE(sm.prepare_context_for_tokens(s_handle, 30).is_ok());
    auto info = sm.get_session_info(s_handle);
    ASSERT_TRUE(info.is_ok());
    EXPECT_EQ(info.value().current_tokens, 30);
    EXPECT_EQ(info.value().state, SessionState::ACTIVE);

    // Record turn completion (generated 10 tokens)
    EXPECT_TRUE(sm.record_turn_completion(s_handle, 10).is_ok());
    info = sm.get_session_info(s_handle);
    EXPECT_EQ(info.value().current_tokens, 40);
    EXPECT_EQ(info.value().turn_count, 1);
    EXPECT_EQ(info.value().state, SessionState::IDLE);

    // Incoming 30 tokens -> 40 + 30 = 70 > 64 -> Under REJECT_NEW, must fail with CONTEXT_EXCEEDED
    Status reject_st = sm.prepare_context_for_tokens(s_handle, 30);
    EXPECT_FALSE(reject_st.is_ok());
    EXPECT_EQ(reject_st.code, StatusCode::CONTEXT_EXCEEDED);

    EXPECT_TRUE(sm.destroy_session(s_handle).is_ok());
    EXPECT_EQ(sm.total_session_count(), 0);
}

TEST(SessionManagerTest, SlidingWindowTruncation) {
    auto kv = std::make_shared<KVCacheManager>();
    ASSERT_TRUE(kv->initialize(20, 1024).is_ok());

    SessionManager sm(kv);
    SessionConfig s_cfg;
    s_cfg.max_context_tokens = 50;

    auto sess_res = sm.create_session(1, s_cfg, ContextOverflowPolicy::SLIDING_WINDOW);
    ASSERT_TRUE(sess_res.is_ok());
    SessionHandle s = sess_res.value();

    // Fill with 40 tokens
    EXPECT_TRUE(sm.prepare_context_for_tokens(s, 40).is_ok());
    EXPECT_TRUE(sm.record_turn_completion(s, 0).is_ok());

    // Add 25 new tokens -> 40 + 25 = 65 > 50 -> Under SLIDING_WINDOW:
    // Retains allowed_prior = 50 - 25 = 25 tokens, then adds 25 = total 50 tokens
    EXPECT_TRUE(sm.prepare_context_for_tokens(s, 25).is_ok());
    auto info = sm.get_session_info(s);
    ASSERT_TRUE(info.is_ok());
    EXPECT_EQ(info.value().current_tokens, 50);

    sm.destroy_session(s);
}

TEST(SessionManagerTest, LRUEviction) {
    auto kv = std::make_shared<KVCacheManager>();
    ASSERT_TRUE(kv->initialize(20, 1024).is_ok());

    SessionManager sm(kv);
    SessionConfig s_cfg;

    auto s1 = sm.create_session(1, s_cfg).value();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    [[maybe_unused]] auto s2 = sm.create_session(1, s_cfg).value();

    // Both are idle; s1 is older
    auto evict_res = sm.evict_lru_session();
    ASSERT_TRUE(evict_res.is_ok());
    EXPECT_EQ(evict_res.value(), s1); // s1 was evicted!

    auto info1 = sm.get_session_info(s1);
    EXPECT_EQ(info1.value().state, SessionState::PREEMPTED);
}
