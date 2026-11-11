#include <gtest/gtest.h>
#include "runtime/src/engine/engine_impl.h"
#include "backends/mock/mock_backend.h"
#include <chrono>
#include <future>

using namespace mobilellm;

class EngineTest : public ::testing::Test {
protected:
    std::unique_ptr<EngineImpl> engine;
    ModelHandle model_handle{0};
    SessionHandle session_handle{0};

    void SetUp() override {
        engine = std::make_unique<EngineImpl>();
        auto mock = std::make_unique<MockBackend>();
        engine->set_backend_override(std::move(mock));

        EngineConfig cfg;
        ASSERT_TRUE(engine->initialize(cfg).is_ok());

        ModelLoadConfig m_cfg;
        m_cfg.model_path = "/tmp/test_model.gguf";
        auto m_res = engine->load_model(m_cfg);
        ASSERT_TRUE(m_res.is_ok());
        model_handle = m_res.value();

        SessionConfig s_cfg;
        auto s_res = engine->create_session(model_handle, s_cfg);
        ASSERT_TRUE(s_res.is_ok());
        session_handle = s_res.value();
    }

    void TearDown() override {
        if (engine) {
            engine->shutdown();
        }
    }
};

TEST_F(EngineTest, StreamingGenerationCompletes) {
    GenerationRequest req;
    req.prompt = "Tell me a story";
    req.sampling.max_tokens = 10;

    std::promise<void> done_promise;
    auto done_future = done_promise.get_future();

    std::vector<std::string> received_tokens;
    GenerationMetrics final_metrics;
    Status final_status = Status::ok();

    auto token_cb = [&](const TokenChunk& chunk) -> bool {
        received_tokens.push_back(chunk.text);
        return true;
    };

    auto complete_cb = [&](const Status& st, const GenerationMetrics& m) {
        final_status = st;
        final_metrics = m;
        done_promise.set_value();
    };

    auto gen_res = engine->generate_async(session_handle, req, token_cb, complete_cb);
    ASSERT_TRUE(gen_res.is_ok());

    ASSERT_EQ(done_future.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_TRUE(final_status.is_ok());
    EXPECT_EQ(final_metrics.completion_tokens, 10);
    EXPECT_EQ(received_tokens.size(), 10);
    EXPECT_GT(final_metrics.decode_tokens_per_second, 0.0);
}

TEST_F(EngineTest, CooperativeCancellationViaApi) {
    GenerationRequest req;
    req.prompt = "Write an essay";
    req.sampling.max_tokens = 100;

    std::promise<void> done_promise;
    auto done_future = done_promise.get_future();

    Status final_status = Status::ok();
    uint32_t tokens_received = 0;

    auto token_cb = [&](const TokenChunk&) -> bool {
        tokens_received++;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        return true;
    };

    auto complete_cb = [&](const Status& st, const GenerationMetrics&) {
        final_status = st;
        done_promise.set_value();
    };

    auto gen_res = engine->generate_async(session_handle, req, token_cb, complete_cb);
    ASSERT_TRUE(gen_res.is_ok());

    // Allow a few tokens then cancel
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    EXPECT_TRUE(engine->cancel(gen_res.value()).is_ok());

    ASSERT_EQ(done_future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_EQ(final_status.code, StatusCode::CANCELLED);
    EXPECT_LT(tokens_received, 100);
}

TEST_F(EngineTest, CancellationViaTokenCallback) {
    GenerationRequest req;
    req.prompt = "Count to 50";
    req.sampling.max_tokens = 50;

    std::promise<void> done_promise;
    auto done_future = done_promise.get_future();

    Status final_status = Status::ok();
    int count = 0;

    auto token_cb = [&](const TokenChunk&) -> bool {
        count++;
        if (count >= 5) {
            return false; // Request cancellation
        }
        return true;
    };

    auto complete_cb = [&](const Status& st, const GenerationMetrics&) {
        final_status = st;
        done_promise.set_value();
    };

    auto gen_res = engine->generate_async(session_handle, req, token_cb, complete_cb);
    ASSERT_TRUE(gen_res.is_ok());

    ASSERT_EQ(done_future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
    EXPECT_EQ(final_status.code, StatusCode::CANCELLED);
    EXPECT_EQ(count, 5);
}

TEST_F(EngineTest, ErrorOnEmptyPrompt) {
    GenerationRequest req;
    req.prompt = ""; // Invalid

    auto gen_res = engine->generate_async(session_handle, req, nullptr, nullptr);
    EXPECT_FALSE(gen_res.is_ok());
    EXPECT_EQ(gen_res.status().code, StatusCode::INVALID_ARGUMENT);
}

TEST_F(EngineTest, ErrorOnInvalidSession) {
    GenerationRequest req;
    req.prompt = "Test";

    auto gen_res = engine->generate_async(99999, req, nullptr, nullptr);
    EXPECT_FALSE(gen_res.is_ok());
    EXPECT_EQ(gen_res.status().code, StatusCode::SESSION_NOT_FOUND);
}

TEST_F(EngineTest, ConcurrentSessions) {
    SessionConfig s_cfg;
    auto s2_res = engine->create_session(model_handle, s_cfg);
    ASSERT_TRUE(s2_res.is_ok());
    SessionHandle session_2 = s2_res.value();

    GenerationRequest req1{"Session 1 prompt", {0.7f, 0.9f, 40, 0.05f, 1.1f, 15, {}}};
    GenerationRequest req2{"Session 2 prompt", {0.7f, 0.9f, 40, 0.05f, 1.1f, 15, {}}};

    std::promise<void> p1, p2;
    auto f1 = p1.get_future();
    auto f2 = p2.get_future();

    engine->generate_async(session_handle, req1, nullptr, [&](const Status&, const GenerationMetrics&) {
        p1.set_value();
    });

    engine->generate_async(session_2, req2, nullptr, [&](const Status&, const GenerationMetrics&) {
        p2.set_value();
    });

    EXPECT_EQ(f1.wait_for(std::chrono::seconds(3)), std::future_status::ready);
    EXPECT_EQ(f2.wait_for(std::chrono::seconds(3)), std::future_status::ready);

    EXPECT_TRUE(engine->destroy_session(session_2).is_ok());
}
