#include <gtest/gtest.h>
#include "backends/common/backend_registry.h"
#include "backends/mock/mock_backend.h"

using namespace mobilellm;

TEST(BackendTest, RegistryAddAndRetrieve) {
    BackendRegistry& reg = BackendRegistry::instance();
    reg.register_backend(BackendType::CPU_NEON, []() {
        return std::make_unique<MockBackend>();
    });

    EXPECT_TRUE(reg.has_backend(BackendType::CPU_NEON));
    auto backend = reg.create(BackendType::CPU_NEON);
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->type(), BackendType::CPU_NEON);
    EXPECT_EQ(backend->name(), "MockBackend");
}

TEST(BackendTest, MockBackendLifecycle) {
    MockBackend backend;
    Status st = backend.initialize(BackendConfig{});
    EXPECT_TRUE(st.is_ok());

    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/mock_model.gguf";
    auto load_res = backend.load_model(m_cfg);
    ASSERT_TRUE(load_res.is_ok());
    BackendModelHandle m_handle = load_res.value();

    SessionConfig s_cfg;
    auto sess_res = backend.create_session(m_handle, s_cfg);
    ASSERT_TRUE(sess_res.is_ok());
    BackendSessionHandle s_handle = sess_res.value();

    // Evaluate prompt
    std::vector<int32_t> tokens;
    Status eval_st = backend.evaluate_prompt(s_handle, "Hello world", tokens);
    EXPECT_TRUE(eval_st.is_ok());
    EXPECT_GT(tokens.size(), 0);

    // Sample token
    SamplingParams sparams;
    sparams.max_tokens = 5;
    auto tok_res = backend.sample_next_token(s_handle, sparams);
    ASSERT_TRUE(tok_res.is_ok());

    std::string piece = backend.decode_token(s_handle, tok_res.value());
    EXPECT_FALSE(piece.empty());

    // Destroy session and unload
    EXPECT_TRUE(backend.destroy_session(s_handle).is_ok());
    EXPECT_TRUE(backend.unload_model(m_handle).is_ok());

    // Unloading again should fail with MODEL_NOT_LOADED
    EXPECT_EQ(backend.unload_model(m_handle).code, StatusCode::MODEL_NOT_LOADED);
}

TEST(BackendTest, MockBackendSimulatedOOM) {
    MockBackend backend;
    backend.set_simulate_oom(true);

    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/huge_model.gguf";
    auto res = backend.load_model(m_cfg);
    EXPECT_FALSE(res.is_ok());
    EXPECT_EQ(res.status().code, StatusCode::OUT_OF_MEMORY);
}
