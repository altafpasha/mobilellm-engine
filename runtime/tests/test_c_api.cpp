#include <gtest/gtest.h>
#include "mobilellm/c_api.h"
#include "backends/common/backend_registry.h"
#include "backends/mock/mock_backend.h"
#include <cstring>
#include <future>
#include <chrono>
#include <string>
#include <vector>

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
    // Ensure MockBackend is registered for Auto in this test
    mobilellm::BackendRegistry::instance().register_backend(mobilellm::BackendType::CPU_NEON, []() {
        return std::make_unique<mobilellm::MockBackend>();
    });

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

TEST(CApiTest, ErrorMessageLifetimeAfterTemporary) {
    char buf[128];
    std::string bad_path = "/path/to/definitely_nonexistent_model.gguf";
    mllm_status_t inspect_st = mllm_model_inspect(bad_path.c_str(), buf, sizeof(buf));
    EXPECT_NE(inspect_st.code, 0);
    ASSERT_NE(inspect_st.message, nullptr);
    EXPECT_GT(strlen(inspect_st.message), 0u);

    // Verify error message persists across function calls on this thread
    std::string captured_msg = inspect_st.message;
    EXPECT_NE(captured_msg.find("definitely_nonexistent_model.gguf"), std::string::npos);

    // Verify mllm_get_last_error returns the identical string
    const char* last_err = mllm_get_last_error();
    ASSERT_NE(last_err, nullptr);
    EXPECT_STREQ(last_err, captured_msg.c_str());

    // Also verify invalid engine handle error through c_error
    mllm_status_t invalid_handle_st = mllm_engine_destroy(999999);
    EXPECT_NE(invalid_handle_st.code, 0);
    ASSERT_NE(invalid_handle_st.message, nullptr);
    EXPECT_STREQ(invalid_handle_st.message, "Invalid engine handle");
    EXPECT_STREQ(mllm_get_last_error(), "Invalid engine handle");

    // Also verify null config pointer validation
    mllm_engine_t engine = 0;
    ASSERT_EQ(mllm_engine_create(&engine).code, 0);
    mllm_model_t mdl = 0;
    mllm_status_t null_cfg_st = mllm_model_load(engine, nullptr, &mdl);
    EXPECT_NE(null_cfg_st.code, 0);
    ASSERT_NE(null_cfg_st.message, nullptr);
    EXPECT_STREQ(null_cfg_st.message, "Config or out_model pointer is null");
    EXPECT_STREQ(mllm_get_last_error(), "Config or out_model pointer is null");
    EXPECT_EQ(mllm_engine_destroy(engine).code, 0);
}

TEST(CApiTest, RepeatedErrorRetrievalAcrossCalls) {
    for (int i = 0; i < 5; ++i) {
        std::string bad_path = "/tmp/bad_model_" + std::to_string(i) + ".gguf";
        char buf[128];
        mllm_status_t st = mllm_model_inspect(bad_path.c_str(), buf, sizeof(buf));
        EXPECT_NE(st.code, 0);
        ASSERT_NE(st.message, nullptr);
        EXPECT_NE(std::string(st.message).find("bad_model_" + std::to_string(i)), std::string::npos);
        EXPECT_STREQ(mllm_get_last_error(), st.message);
    }
}

TEST(CApiTest, ConcurrentErrorsOnMultipleThreads) {
    constexpr int NUM_THREADS = 8;
    std::vector<std::future<void>> futures;

    for (int i = 0; i < NUM_THREADS; ++i) {
        futures.push_back(std::async(std::launch::async, [i]() {
            std::string path = "/nonexistent/thread_" + std::to_string(i) + ".gguf";
            char buf[128];
            mllm_status_t st = mllm_model_inspect(path.c_str(), buf, sizeof(buf));
            EXPECT_NE(st.code, 0);
            ASSERT_NE(st.message, nullptr);

            std::string msg = st.message;
            EXPECT_NE(msg.find("thread_" + std::to_string(i)), std::string::npos);
            EXPECT_STREQ(mllm_get_last_error(), msg.c_str());
        }));
    }

    for (auto& f : futures) {
        ASSERT_EQ(f.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    }
}

