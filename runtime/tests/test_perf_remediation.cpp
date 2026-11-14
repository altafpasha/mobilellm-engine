#include <gtest/gtest.h>
#include "mobilellm/engine.h"
#include "mobilellm/c_api.h"
#include "backends/mock/mock_backend.h"
#include "backends/common/backend_registry.h"
#include "runtime/src/sampling/sampler.h"
#include <chrono>
#include <future>
#include <thread>
#include <vector>

using namespace mobilellm;

class PerfRemediationTest : public ::testing::Test {
protected:
    void SetUp() override {
        BackendRegistry& reg = BackendRegistry::instance();
        reg.register_backend(BackendType::CPU_NEON, []() {
            return std::make_unique<MockBackend>();
        });
        engine = create_engine();
        EngineConfig cfg;
        ASSERT_TRUE(engine->initialize(cfg).is_ok());

        ModelLoadConfig m_cfg;
        m_cfg.model_path = "/tmp/mock_model.gguf";
        auto res = engine->load_model(m_cfg);
        ASSERT_TRUE(res.is_ok());
        model_handle = res.value();
    }

    void TearDown() override {
        if (engine) {
            engine->shutdown();
        }
    }

    std::unique_ptr<IEngine> engine;
    ModelHandle model_handle{0};
};

// 1. Reusable logits buffer allocation & sizing
TEST_F(PerfRemediationTest, LogitsBufferAllocationAndSizing) {
    MockBackend backend;
    ASSERT_TRUE(backend.initialize(BackendConfig{}).is_ok());

    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/test_model.gguf";
    auto m_res = backend.load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok());

    SessionConfig s_cfg;
    auto s_res = backend.create_session(m_res.value(), s_cfg);
    ASSERT_TRUE(s_res.is_ok());

    // Prompt evaluation
    std::vector<int32_t> out_tokens;
    ASSERT_TRUE(backend.evaluate_prompt(s_res.value(), "Test prompt", out_tokens).is_ok());

    // Sample tokens repeatedly and verify buffer reuse without crashes
    SamplingParams sparams;
    sparams.max_tokens = 20;
    for (int i = 0; i < 20; ++i) {
        auto tok_res = backend.sample_next_token(s_res.value(), sparams);
        ASSERT_TRUE(tok_res.is_ok());
    }

    ASSERT_TRUE(backend.destroy_session(s_res.value()).is_ok());
    ASSERT_TRUE(backend.unload_model(m_res.value()).is_ok());
}

// 2. Buffer reuse across repeated token sampling preserves deterministic output
TEST_F(PerfRemediationTest, DeterministicSamplingEquivalence) {
    MockBackend b1, b2;
    ASSERT_TRUE(b1.initialize(BackendConfig{}).is_ok());
    ASSERT_TRUE(b2.initialize(BackendConfig{}).is_ok());

    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/test_model.gguf";
    auto m1 = b1.load_model(m_cfg).value();
    auto m2 = b2.load_model(m_cfg).value();

    SessionConfig s_cfg;
    auto s1 = b1.create_session(m1, s_cfg).value();
    auto s2 = b2.create_session(m2, s_cfg).value();

    std::vector<int32_t> out1, out2;
    b1.evaluate_prompt(s1, "Same input", out1);
    b2.evaluate_prompt(s2, "Same input", out2);

    SamplingParams sparams;
    sparams.max_tokens = 10;
    sparams.temperature = 0.0f; // greedy

    for (int i = 0; i < 10; ++i) {
        auto tok1 = b1.sample_next_token(s1, sparams).value();
        auto tok2 = b2.sample_next_token(s2, sparams).value();
        EXPECT_EQ(tok1, tok2) << "Token mismatch at step " << i;
    }

    b1.destroy_session(s1);
    b2.destroy_session(s2);
}

// 3. Independent buffer ownership across concurrent sessions
TEST_F(PerfRemediationTest, ConcurrentIndependentSessionBuffers) {
    MockBackend backend;
    ASSERT_TRUE(backend.initialize(BackendConfig{}).is_ok());

    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/test_model.gguf";
    auto m = backend.load_model(m_cfg).value();

    SessionConfig s_cfg;
    auto s1 = backend.create_session(m, s_cfg).value();
    auto s2 = backend.create_session(m, s_cfg).value();

    std::vector<int32_t> out1, out2;
    backend.evaluate_prompt(s1, "Session One", out1);
    backend.evaluate_prompt(s2, "Session Two Different", out2);

    std::atomic<bool> start_signal{false};
    std::vector<int32_t> tokens1, tokens2;

    auto worker = [&](BackendSessionHandle sess, std::vector<int32_t>& out_toks) {
        while (!start_signal.load()) {
            std::this_thread::yield();
        }
        SamplingParams sparams;
        sparams.max_tokens = 15;
        for (int i = 0; i < 15; ++i) {
            auto res = backend.sample_next_token(sess, sparams);
            if (res.is_ok()) {
                out_toks.push_back(res.value());
            }
        }
    };

    std::thread t1(worker, s1, std::ref(tokens1));
    std::thread t2(worker, s2, std::ref(tokens2));

    start_signal.store(true);
    t1.join();
    t2.join();

    EXPECT_EQ(tokens1.size(), 15);
    EXPECT_EQ(tokens2.size(), 15);

    backend.destroy_session(s1);
    backend.destroy_session(s2);
    backend.unload_model(m);
}

// 4. Allocation failure handling
TEST_F(PerfRemediationTest, AllocationFailureHandling) {
    MockBackend backend;
    backend.set_simulate_oom(true);

    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/test_model.gguf";
    auto m_res = backend.load_model(m_cfg);
    EXPECT_FALSE(m_res.is_ok());
    EXPECT_EQ(m_res.status().code, StatusCode::OUT_OF_MEMORY);
}

// 5. Decomposed TTFT and Timing Metrics (PERF-01)
TEST_F(PerfRemediationTest, DecomposedTTFTMetricsValidation) {
    SessionConfig s_cfg;
    auto s_res = engine->create_session(model_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle session = s_res.value();

    GenerationRequest req;
    req.prompt = "Benchmark prompt for TTFT";
    req.sampling.max_tokens = 8;
    req.sampling.temperature = 0.0f;

    std::promise<GenerationMetrics> metric_promise;
    auto metric_future = metric_promise.get_future();

    auto gen_res = engine->generate_async(
        session,
        req,
        nullptr,
        [&](const Status& st, const GenerationMetrics& m) {
            if (st.is_ok()) {
                metric_promise.set_value(m);
            }
        }
    );
    ASSERT_TRUE(gen_res.is_ok());

    ASSERT_EQ(metric_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    GenerationMetrics m = metric_future.get();

    EXPECT_EQ(m.completion_tokens, 8);
    EXPECT_GT(m.prompt_tokens, 0);

    // Verify all 5 decomposed timing metrics
    EXPECT_GE(m.queue_wait_time_ms, 0.0);
    EXPECT_GE(m.prefill_time_ms, 0.0);
    EXPECT_GE(m.first_token_sampling_time_ms, 0.0);
    EXPECT_GE(m.pure_inference_ttft_ms, 0.0);
    EXPECT_GE(m.end_to_end_ttft_ms, 0.0);

    // Mathematical invariant: pure_inference_ttft_ms == prefill_time_ms + first_token_sampling_time_ms
    EXPECT_NEAR(m.pure_inference_ttft_ms, m.prefill_time_ms + m.first_token_sampling_time_ms, 0.001);

    // Invariant: end_to_end_ttft_ms includes queue wait and delivery
    EXPECT_GE(m.end_to_end_ttft_ms, m.pure_inference_ttft_ms);

    // Invariant: time_to_first_token_ms equals pure_inference_ttft_ms
    EXPECT_EQ(m.time_to_first_token_ms, m.pure_inference_ttft_ms);
}

// 6. Timing under concurrent queued requests
TEST_F(PerfRemediationTest, TimingUnderQueuedRequests) {
    SessionConfig s_cfg;
    auto s_res = engine->create_session(model_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle session = s_res.value();

    GenerationRequest req1{"Prompt 1", {0.0f, 0.9f, 40, 0.05f, 1.0f, 10, {}}};
    GenerationRequest req2{"Prompt 2", {0.0f, 0.9f, 40, 0.05f, 1.0f, 5, {}}};

    std::promise<GenerationMetrics> p1, p2;
    auto f1 = p1.get_future();
    auto f2 = p2.get_future();

    engine->generate_async(session, req1, nullptr, [&](const Status&, const GenerationMetrics& m) {
        p1.set_value(m);
    });
    engine->generate_async(session, req2, nullptr, [&](const Status&, const GenerationMetrics& m) {
        p2.set_value(m);
    });

    ASSERT_EQ(f1.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    ASSERT_EQ(f2.wait_for(std::chrono::seconds(5)), std::future_status::ready);

    GenerationMetrics m1 = f1.get();
    GenerationMetrics m2 = f2.get();

    EXPECT_GE(m1.queue_wait_time_ms, 0.0);
    // Request 2 was serialized behind request 1, so request 2 experienced queuing
    EXPECT_GE(m2.end_to_end_ttft_ms, m2.pure_inference_ttft_ms);
}

// 7. C-API ABI Preservation
TEST_F(PerfRemediationTest, CApiAbiPreservation) {
    // Ensure mllm_metrics_t size and alignment match frozen C ABI specification
    EXPECT_EQ(sizeof(mllm_metrics_t), sizeof(uint32_t) * 2 + sizeof(double) * 7);

    // Test C-API generation metrics mapping
    mllm_engine_t eng;
    ASSERT_EQ(mllm_engine_create(&eng).code, 0);

    mllm_model_t mdl;
    mllm_model_config_t m_cfg = {"/tmp/mock_model.gguf", 1, 2048, 4, false, false, 0.0f};
    ASSERT_EQ(mllm_model_load(eng, &m_cfg, &mdl).code, 0);

    mllm_session_t sess;
    mllm_session_config_t s_cfg = {2048, false};
    ASSERT_EQ(mllm_session_create(eng, mdl, &s_cfg, &sess).code, 0);

    mllm_sampling_params_t sparams = {0.0f, 0.9f, 40, 0.05f, 1.0f, 5};

    std::promise<mllm_metrics_t> c_metric_promise;
    auto c_metric_future = c_metric_promise.get_future();

    auto comp_cb = [](mllm_status_t st, const mllm_metrics_t* metrics, void* user_data) {
        auto* prom = static_cast<std::promise<mllm_metrics_t>*>(user_data);
        if (metrics) {
            prom->set_value(*metrics);
        }
    };

    mllm_gen_handle_t gen_handle;
    ASSERT_EQ(mllm_generate(eng, sess, "Test C ABI", &sparams, nullptr, comp_cb, &c_metric_promise, &gen_handle).code, 0);

    ASSERT_EQ(c_metric_future.wait_for(std::chrono::seconds(5)), std::future_status::ready);
    mllm_metrics_t cm = c_metric_future.get();

    EXPECT_EQ(cm.completion_tokens, 5);
    EXPECT_GT(cm.prompt_tokens, 0);
    EXPECT_GT(cm.ttft_ms, 0.0);
    EXPECT_GT(cm.tps, 0.0);

    mllm_session_destroy(eng, sess);
    mllm_model_unload(eng, mdl);
    mllm_engine_destroy(eng);
}
