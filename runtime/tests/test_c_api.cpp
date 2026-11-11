#include <gtest/gtest.h>
#include "mobilellm/c_api.h"
#include <cstring>
#include <future>
#include <chrono>

TEST(CApiTest, GetVersion) {
    char buf[32];
    mllm_status_t st = mllm_get_version(buf, sizeof(buf));
    EXPECT_EQ(st.code, 0);
    EXPECT_STREQ(buf, "0.1.0-dev");

    // Null buffer
    mllm_status_t err = mllm_get_version(nullptr, 32);
    EXPECT_NE(err.code, 0);
}

TEST(CApiTest, DetectHardware) {
    char buf[256];
    mllm_status_t st = mllm_detect_hardware(buf, sizeof(buf));
    EXPECT_EQ(st.code, 0);
    EXPECT_TRUE(strstr(buf, "\"arch\":\"ARM64\"") != nullptr);
}

TEST(CApiTest, FullLifecycle) {
    mllm_engine_t engine = 0;
    mllm_status_t st = mllm_engine_create(&engine);
    ASSERT_EQ(st.code, 0);
    ASSERT_NE(engine, 0);

    // Load model
    mllm_model_config_t m_cfg;
    memset(&m_cfg, 0, sizeof(m_cfg));
    m_cfg.model_path = "/tmp/mock_model.gguf";
    m_cfg.backend = 0; // auto
    m_cfg.context_length = 2048;

    mllm_model_t model = 0;
    st = mllm_model_load(engine, &m_cfg, &model);
    ASSERT_EQ(st.code, 0);
    ASSERT_NE(model, 0);

    // Create session
    mllm_session_config_t s_cfg;
    s_cfg.max_context_tokens = 2048;
    s_cfg.enable_prefix_caching = true;

    mllm_session_t session = 0;
    st = mllm_session_create(engine, model, &s_cfg, &session);
    ASSERT_EQ(st.code, 0);
    ASSERT_NE(session, 0);

    // Generate
    mllm_sampling_params_t sparams;
    memset(&sparams, 0, sizeof(sparams));
    sparams.temperature = 0.7f;
    sparams.max_tokens = 5;

    struct TestContext {
        std::promise<void> promise;
        int tokens_seen{0};
        int complete_code{0};
    } ctx;

    auto token_fn = [](const mllm_token_chunk_t*, void* udata) -> bool {
        auto* c = static_cast<TestContext*>(udata);
        c->tokens_seen++;
        return true;
    };

    auto complete_fn = [](mllm_status_t status, const mllm_metrics_t*, void* udata) {
        auto* c = static_cast<TestContext*>(udata);
        c->complete_code = status.code;
        c->promise.set_value();
    };

    mllm_gen_handle_t gen_handle = 0;
    st = mllm_generate(
        engine,
        session,
        "Hello C-API",
        &sparams,
        token_fn,
        complete_fn,
        &ctx,
        &gen_handle
    );
    ASSERT_EQ(st.code, 0);

    auto fut = ctx.promise.get_future();
    ASSERT_EQ(fut.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_EQ(ctx.complete_code, 0);
    EXPECT_EQ(ctx.tokens_seen, 5);

    // Destroy session, unload model, destroy engine
    EXPECT_EQ(mllm_session_destroy(engine, session).code, 0);
    EXPECT_EQ(mllm_model_unload(engine, model).code, 0);
    EXPECT_EQ(mllm_engine_destroy(engine).code, 0);
}

