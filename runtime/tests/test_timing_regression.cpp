// Agent 10E — Automated timing regression tests.
//
// These tests lock down the timing contract that the Performance & Telemetry dashboard relies on:
//   1. Inference execution time must be distinguishable from end-to-end request latency.
//   2. Decode throughput must be derived from the decode window, and a *pure* decode rate must be
//      derivable that excludes streaming token-callback (JNI/UI delivery) overhead.
//   3. The decomposed TTFT metrics must remain mathematically consistent.
//
// They run against the deterministic MockBackend so they are fast and hardware-independent, yet
// they exercise the real ContinuousBatchScheduler timing code paths.

#include <gtest/gtest.h>
#include "mobilellm/engine.h"
#include "mobilellm/c_api.h"
#include "backends/mock/mock_backend.h"
#include "backends/common/backend_registry.h"
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <thread>

using namespace mobilellm;

namespace {

class TimingRegressionTest : public ::testing::Test {
protected:
    void SetUp() override {
        BackendRegistry::instance().register_backend(BackendType::CPU_NEON, []() {
            return std::make_unique<MockBackend>();
        });

        engine = create_engine();
        EngineConfig cfg;
        ASSERT_TRUE(engine->initialize(cfg).is_ok());

        ModelLoadConfig m_cfg;
        m_cfg.model_path = "/tmp/mock_timing_model.gguf";
        auto res = engine->load_model(m_cfg);
        ASSERT_TRUE(res.is_ok());
        model_handle = res.value();
    }

    void TearDown() override {
        if (engine) {
            engine->shutdown();
        }
    }

    // Runs one generation and returns the final metrics.
    GenerationMetrics run_generation(uint32_t max_tokens,
                                     TokenCallback token_cb = nullptr) {
        auto s_res = engine->create_session(model_handle, SessionConfig{});
        EXPECT_TRUE(s_res.is_ok());
        SessionHandle session = s_res.value();

        GenerationRequest req;
        req.prompt = "timing regression probe prompt";
        req.sampling.max_tokens = max_tokens;
        req.sampling.temperature = 0.0f;

        std::promise<GenerationMetrics> prom;
        auto fut = prom.get_future();
        auto gen = engine->generate_async(
            session, req, std::move(token_cb),
            [&](const Status& st, const GenerationMetrics& m) {
                if (st.is_ok()) {
                    prom.set_value(m);
                } else {
                    prom.set_value(GenerationMetrics{});
                }
            });
        EXPECT_TRUE(gen.is_ok());

        EXPECT_EQ(fut.wait_for(std::chrono::seconds(10)), std::future_status::ready);
        GenerationMetrics m = fut.get();
        engine->destroy_session(session);
        return m;
    }

    std::unique_ptr<IEngine> engine;
    ModelHandle model_handle{0};
};

} // namespace

// 1. Core timing invariants: total >= prefill + decode, and the decomposed TTFT identities hold.
TEST_F(TimingRegressionTest, TimingInvariantsAreConsistent) {
    GenerationMetrics m = run_generation(12);

    ASSERT_EQ(m.completion_tokens, 12u);
    ASSERT_GT(m.prompt_tokens, 0u);

    // Inference execution stages must be positive.
    EXPECT_GT(m.prefill_time_ms, 0.0);
    EXPECT_GT(m.decode_time_ms, 0.0);
    EXPECT_GT(m.total_duration_ms, 0.0);

    // End-to-end request latency is the superset: queue wait + prefill + decode.
    EXPECT_GE(m.total_duration_ms, m.prefill_time_ms + m.decode_time_ms - 1e-6);

    // Decode window can never exceed the whole request.
    EXPECT_LE(m.decode_time_ms, m.total_duration_ms + 1e-6);

    // Decomposed TTFT identities (PERF-01 contract).
    EXPECT_NEAR(m.pure_inference_ttft_ms,
                m.prefill_time_ms + m.first_token_sampling_time_ms, 1e-3);
    EXPECT_GE(m.end_to_end_ttft_ms, m.pure_inference_ttft_ms - 1e-6);

    // Legacy public field must remain the end-to-end TTFT.
    EXPECT_DOUBLE_EQ(m.time_to_first_token_ms, m.end_to_end_ttft_ms);
}

// 2. Throughput metrics must be derived from the timings they claim to describe.
TEST_F(TimingRegressionTest, ThroughputFormulasMatchDecomposedTimings) {
    GenerationMetrics m = run_generation(16);

    ASSERT_EQ(m.completion_tokens, 16u);

    // decode_tokens_per_second == completion_tokens / decode_time_seconds
    const double expected_decode = static_cast<double>(m.completion_tokens) / (m.decode_time_ms / 1000.0);
    EXPECT_NEAR(m.decode_tokens_per_second, expected_decode, expected_decode * 0.05 + 1e-3);

    // prefill_tokens_per_second == prompt_tokens / prefill_time_seconds
    const double expected_prefill = static_cast<double>(m.prompt_tokens) / (m.prefill_time_ms / 1000.0);
    EXPECT_NEAR(m.prefill_tokens_per_second, expected_prefill, expected_prefill * 0.05 + 1e-3);
}

// 3. The pure decode rate must isolate backend compute from streaming-callback overhead.
//    This is the regression guard for the defect where the dashboard reported end-to-end
//    decode latency (including per-token JNI/UI delivery) as raw inference throughput.
TEST_F(TimingRegressionTest, PureDecodeIsolatesStreamingCallbackOverhead) {
    // Simulate a slow streaming consumer (e.g. UI/rendering back-pressure) on the callback path.
    std::atomic<int> emitted{0};
    auto slow_cb = [&](const TokenChunk&) -> bool {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        emitted.fetch_add(1);
        return true;
    };

    GenerationMetrics m = run_generation(10, slow_cb);

    ASSERT_EQ(m.completion_tokens, 10u);
    EXPECT_EQ(emitted.load(), 10);

    // The end-to-end decode window includes the ~5ms/token callback delivery ...
    EXPECT_GT(m.decode_time_ms, 20.0);
    // ... while pure compute time must be far smaller.
    EXPECT_GT(m.decode_compute_time_ms, 0.0);
    EXPECT_LT(m.decode_compute_time_ms, m.decode_time_ms);

    // Therefore the pure decode rate must exceed the end-to-end decode rate.
    EXPECT_GT(m.pure_decode_tokens_per_second, m.decode_tokens_per_second);

    // And the pure decode rate must equal completion_tokens / compute_time.
    const double expected_pure = static_cast<double>(m.completion_tokens) / (m.decode_compute_time_ms / 1000.0);
    EXPECT_NEAR(m.pure_decode_tokens_per_second, expected_pure, expected_pure * 0.05 + 1e-3);
}

// 4. All emitted timing metrics must be finite and non-negative (guards divide-by-zero regressions
//    and the prefix-cache "zero prefill time" edge case).
TEST_F(TimingRegressionTest, AllTimingMetricsAreFiniteAndNonNegative) {
    GenerationMetrics m = run_generation(8);

    const double values[] = {
        m.time_to_first_token_ms, m.decode_tokens_per_second, m.prefill_tokens_per_second,
        m.prefill_time_ms, m.decode_time_ms, m.total_duration_ms, m.queue_wait_time_ms,
        m.first_token_sampling_time_ms, m.pure_inference_ttft_ms, m.end_to_end_ttft_ms,
        m.decode_compute_time_ms, m.pure_decode_tokens_per_second,
    };
    for (double v : values) {
        EXPECT_TRUE(std::isfinite(v)) << "non-finite timing metric";
        EXPECT_GE(v, 0.0) << "negative timing metric";
    }
}
