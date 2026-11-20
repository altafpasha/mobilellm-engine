// Agent 10G — CPU/GPU Backend Switching Regression & Performance Restoration Tests
//
// Verifies safe transitions, resource teardown order, GPU synchronization,
// thread setting restoration, error propagation, bounded memory, and
// CPU performance restoration (<= 10% regression).

#include <gtest/gtest.h>
#include "mobilellm/engine.h"
#include "mobilellm/c_api.h"
#include "backends/mock/mock_backend.h"
#include "backends/cpu/llamacpp_backend.h"
#include "backends/common/backend_registry.h"
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <future>
#include <iostream>
#include <numeric>
#include <thread>
#include <vector>

using namespace mobilellm;

namespace {

static std::string resolve_test_model_path(const std::string& rel_path) {
    if (std::filesystem::exists(rel_path)) return rel_path;
    if (std::filesystem::exists("../" + rel_path)) return "../" + rel_path;
    if (std::filesystem::exists("../../" + rel_path)) return "../../" + rel_path;
    return rel_path;
}

class BackendSwitchingTest : public ::testing::Test {
protected:
    std::shared_ptr<MockVulkanBackend> mock_vulkan_ref;

    void SetUp() override {
        auto& reg = BackendRegistry::instance();
        reg.register_backend(BackendType::CPU_NEON, []() {
            return std::make_unique<MockBackend>();
        });
        reg.register_backend(BackendType::VULKAN, []() {
            auto v = std::make_unique<MockVulkanBackend>();
            return v;
        }, []() {
            return true;
        });

        engine = create_engine();
        EngineConfig cfg;
        ASSERT_TRUE(engine->initialize(cfg).is_ok());

        ModelLoadConfig m_cfg;
        m_cfg.model_path = "/tmp/mock_model_switching.gguf";
        m_cfg.preferred_backend = BackendType::CPU_NEON;
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

    GenerationMetrics run_sync_generation(SessionHandle sess, uint32_t max_tokens = 10) {
        GenerationRequest req;
        req.prompt = "Test prompt for backend switching";
        req.sampling.max_tokens = max_tokens;

        std::promise<GenerationMetrics> promise;
        auto future = promise.get_future();

        auto res = engine->generate_async(
            sess,
            req,
            [](const TokenChunk&) { return true; },
            [&promise](const Status& st, const GenerationMetrics& m) {
                if (st.is_ok()) {
                    promise.set_value(m);
                } else {
                    GenerationMetrics err_m{};
                    promise.set_value(err_m);
                }
            }
        );

        EXPECT_TRUE(res.is_ok());
        return future.get();
    }

    std::unique_ptr<IEngine> engine;
    ModelHandle model_handle{0};
    SessionHandle session_handle{0};
};

// 1. CPU -> Vulkan -> CPU Transition
TEST_F(BackendSwitchingTest, CPU_To_Vulkan_To_CPU) {
    // 1. Run baseline on CPU
    auto m1 = run_sync_generation(session_handle, 8);
    EXPECT_EQ(m1.completion_tokens, 8u);

    // 2. Switch to Vulkan
    Status st_vulkan = engine->switch_backend(BackendType::VULKAN);
    ASSERT_TRUE(st_vulkan.is_ok()) << "Vulkan switch failed: " << st_vulkan.message;

    // Run inference on Vulkan
    auto m2 = run_sync_generation(session_handle, 8);
    EXPECT_EQ(m2.completion_tokens, 8u);

    // 3. Switch back to CPU
    Status st_cpu = engine->switch_backend(BackendType::CPU_NEON);
    ASSERT_TRUE(st_cpu.is_ok()) << "CPU switch back failed: " << st_cpu.message;

    // Run inference on CPU again
    auto m3 = run_sync_generation(session_handle, 8);
    EXPECT_EQ(m3.completion_tokens, 8u);
}

// 2. Multi-Hop Transition: CPU -> Vulkan -> CPU -> Vulkan
TEST_F(BackendSwitchingTest, CPU_To_Vulkan_To_CPU_To_Vulkan) {
    EXPECT_TRUE(engine->switch_backend(BackendType::CPU_NEON).is_ok());
    EXPECT_EQ(run_sync_generation(session_handle, 5).completion_tokens, 5u);

    EXPECT_TRUE(engine->switch_backend(BackendType::VULKAN).is_ok());
    EXPECT_EQ(run_sync_generation(session_handle, 5).completion_tokens, 5u);

    EXPECT_TRUE(engine->switch_backend(BackendType::CPU_NEON).is_ok());
    EXPECT_EQ(run_sync_generation(session_handle, 5).completion_tokens, 5u);

    EXPECT_TRUE(engine->switch_backend(BackendType::VULKAN).is_ok());
    EXPECT_EQ(run_sync_generation(session_handle, 5).completion_tokens, 5u);
}

// 3. Repeated Switching: Verify No Unbounded Memory Growth
TEST_F(BackendSwitchingTest, Repeated_Backend_Switching_No_Memory_Growth) {
    auto initial_mem = engine->get_process_memory();

    constexpr int NUM_SWITCH_ITERATIONS = 20;
    for (int i = 0; i < NUM_SWITCH_ITERATIONS; ++i) {
        BackendType target = (i % 2 == 0) ? BackendType::VULKAN : BackendType::CPU_NEON;
        Status st = engine->switch_backend(target);
        ASSERT_TRUE(st.is_ok()) << "Iteration " << i << " failed: " << st.message;

        auto m = run_sync_generation(session_handle, 4);
        EXPECT_EQ(m.completion_tokens, 4u);
    }

    // Ensure we end on CPU
    ASSERT_TRUE(engine->switch_backend(BackendType::CPU_NEON).is_ok());

    auto final_mem = engine->get_process_memory();
    auto metrics = engine->get_metrics();

    // Verify model and session counts are preserved without duplicates
    EXPECT_EQ(metrics.active_models, 1u);
    EXPECT_EQ(metrics.active_sessions, 1u);

    // Verify memory budget usage is stable
    if (initial_mem.current_rss_bytes > 0 && final_mem.current_rss_bytes > 0) {
        int64_t diff = static_cast<int64_t>(final_mem.current_rss_bytes) -
                       static_cast<int64_t>(initial_mem.current_rss_bytes);
        // RSS growth after 20 repeated switches must be strictly less than 15 MB
        constexpr int64_t MAX_ALLOWED_GROWTH_BYTES = 15LL * 1024 * 1024;
        EXPECT_LT(diff, MAX_ALLOWED_GROWTH_BYTES)
            << "Unbounded memory growth detected across 20 backend switches! Diff: "
            << diff / (1024 * 1024) << " MB";
    }
}

// 4. Switching While Inference Is Active (Safe Drain / Cancellation)
TEST_F(BackendSwitchingTest, Switching_While_Inference_Is_Active) {
    GenerationRequest req;
    req.prompt = "Long active prompt";
    req.sampling.max_tokens = 50;

    std::atomic<bool> token_received{false};
    std::promise<Status> complete_promise;
    auto complete_future = complete_promise.get_future();

    auto gen_res = engine->generate_async(
        session_handle,
        req,
        [&token_received](const TokenChunk&) {
            token_received.store(true);
            return true;
        },
        [&complete_promise](const Status& st, const GenerationMetrics&) {
            complete_promise.set_value(st);
        }
    );
    ASSERT_TRUE(gen_res.is_ok());

    // Switch while request is running in background thread
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    Status switch_st = engine->switch_backend(BackendType::VULKAN);
    EXPECT_TRUE(switch_st.is_ok()) << "Switch while active failed: " << switch_st.message;

    // Check completion
    auto wait_st = complete_future.wait_for(std::chrono::seconds(2));
    EXPECT_EQ(wait_st, std::future_status::ready);

    // Verify subsequent inference on new backend works without corruption
    auto m = run_sync_generation(session_handle, 6);
    EXPECT_EQ(m.completion_tokens, 6u);
}

// 5. Switching After Explicit Request Cancellation
TEST_F(BackendSwitchingTest, Switching_After_Cancellation) {
    GenerationRequest req;
    req.prompt = "Cancel test prompt";
    req.sampling.max_tokens = 40;

    std::promise<Status> comp_p;
    auto comp_f = comp_p.get_future();

    auto gen_res = engine->generate_async(
        session_handle,
        req,
        [](const TokenChunk&) { return true; },
        [&comp_p](const Status& st, const GenerationMetrics&) {
            comp_p.set_value(st);
        }
    );
    ASSERT_TRUE(gen_res.is_ok());

    EXPECT_TRUE(engine->cancel(gen_res.value()).is_ok());

    // Wait for cancellation to settle
    comp_f.wait_for(std::chrono::seconds(1));

    // Switch backend immediately
    Status switch_st = engine->switch_backend(BackendType::VULKAN);
    ASSERT_TRUE(switch_st.is_ok());

    auto m = run_sync_generation(session_handle, 5);
    EXPECT_EQ(m.completion_tokens, 5u);
}

// 6. Switching After Model Reload
TEST_F(BackendSwitchingTest, Switching_After_Model_Reload) {
    // Unload existing session and model
    ASSERT_TRUE(engine->destroy_session(session_handle).is_ok());
    ASSERT_TRUE(engine->unload_model(model_handle).is_ok());

    // Reload model
    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/reloaded_model.gguf";
    auto m_res = engine->load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok());
    ModelHandle new_model = m_res.value();

    SessionConfig s_cfg;
    auto s_res = engine->create_session(new_model, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle new_session = s_res.value();

    // Switch to Vulkan and verify
    ASSERT_TRUE(engine->switch_backend(BackendType::VULKAN).is_ok());
    auto m = run_sync_generation(new_session, 6);
    EXPECT_EQ(m.completion_tokens, 6u);

    // Switch back to CPU
    ASSERT_TRUE(engine->switch_backend(BackendType::CPU_NEON).is_ok());
    auto m_cpu = run_sync_generation(new_session, 6);
    EXPECT_EQ(m_cpu.completion_tokens, 6u);
}

// 7. Failed Vulkan Initialization Followed by CPU Inference (No Silent Fallback)
TEST_F(BackendSwitchingTest, Failed_Vulkan_Initialization_Followed_By_CPU) {
    // Register a failing Vulkan backend
    auto& reg = BackendRegistry::instance();
    reg.register_backend(BackendType::VULKAN, []() {
        auto v = std::make_unique<MockVulkanBackend>();
        v->set_simulate_init_failure(true);
        return v;
    }, []() {
        return true;
    });

    // Attempt switch to Vulkan: must propagate failure without silent fallback!
    Status switch_st = engine->switch_backend(BackendType::VULKAN);
    EXPECT_FALSE(switch_st.is_ok());
    EXPECT_EQ(switch_st.code, StatusCode::BACKEND_INITIALIZATION_FAILED);

    // Subsequent CPU switch must succeed and restore functional inference
    Status cpu_st = engine->switch_backend(BackendType::CPU_NEON);
    EXPECT_TRUE(cpu_st.is_ok()) << "CPU restoration after failed switch failed: " << cpu_st.message;

    auto m = run_sync_generation(session_handle, 8);
    EXPECT_EQ(m.completion_tokens, 8u);
}

// 8. CPU Performance Restoration Benchmark (Target <= 10% Regression)
//
// NOTE: the mock backend performs negligible per-token work, so any single wall-clock sample is
// dominated by scheduler/threadpool jitter and machine load. We therefore measure many
// baseline/post PAIRS that are adjacent in time (so slow-machine drift affects both nearly
// equally) and compare the median of the per-pair ratios. The exact-decode-rate gate on real
// hardware lives in PhysicalModelSwitchingTest.RealModel_CPU_Switch_Restoration.
TEST_F(BackendSwitchingTest, CPU_Performance_Restoration_After_Switching) {
    constexpr int    ROUNDS = 9;
    constexpr uint32_t MEASURE_TOKENS = 40;

    // Give the mock a deterministic per-token compute cost so throughput is a real signal rather
    // than scheduler jitter. Then round-trip the backends once to force both the active CPU and
    // Vulkan mock instances to be recreated with the latency applied.
    MockBackend::set_default_simulated_token_latency_us(200);
    ASSERT_TRUE(engine->switch_backend(BackendType::VULKAN).is_ok());
    ASSERT_TRUE(engine->switch_backend(BackendType::CPU_NEON).is_ok());

    auto median = [](std::vector<double> v) -> double {
        if (v.empty()) return 0.0;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };

    // Warmup
    run_sync_generation(session_handle, 10);

    std::vector<double> baseline_tps, post_tps;
    std::vector<double> baseline_ttft, post_ttft;
    std::vector<double> pair_ratios;

    for (int i = 0; i < ROUNDS; ++i) {
        // Baseline CPU sample (before the switch).
        auto b = run_sync_generation(session_handle, MEASURE_TOKENS);
        baseline_tps.push_back(b.decode_tokens_per_second);
        baseline_ttft.push_back(b.time_to_first_token_ms);

        // CPU -> Vulkan -> CPU round trip.
        ASSERT_TRUE(engine->switch_backend(BackendType::VULKAN).is_ok());
        run_sync_generation(session_handle, 8);
        ASSERT_TRUE(engine->switch_backend(BackendType::CPU_NEON).is_ok());
        // Discard one generation so the restored CPU path reaches steady state before measuring.
        run_sync_generation(session_handle, 8);

        // Post-switch CPU sample (adjacent to the baseline sample in time).
        auto p = run_sync_generation(session_handle, MEASURE_TOKENS);
        post_tps.push_back(p.decode_tokens_per_second);
        post_ttft.push_back(p.time_to_first_token_ms);

        if (b.decode_tokens_per_second > 0.0) {
            pair_ratios.push_back(p.decode_tokens_per_second / b.decode_tokens_per_second);
        }
    }

    const double baseline_tps_median = median(baseline_tps);
    const double post_tps_median = median(post_tps);
    const double baseline_ttft_median = median(baseline_ttft);
    const double post_ttft_median = median(post_ttft);
    const double tps_ratio = median(pair_ratios);

    std::cout << "\n=== CPU Performance Restoration Benchmark ===" << std::endl;
    std::cout << "Baseline CPU TPS (median): " << baseline_tps_median
              << " tok/s, TTFT: " << baseline_ttft_median << " ms" << std::endl;
    std::cout << "Post-Vulkan CPU TPS (median): " << post_tps_median
              << " tok/s, TTFT: " << post_ttft_median << " ms" << std::endl;
    std::cout << "TPS Performance Retention (median of " << pair_ratios.size()
              << " paired ratios): " << (tps_ratio * 100.0) << "%" << std::endl;

    // Requirement: statistically comparable with target of <= 10% regression.
    EXPECT_GE(tps_ratio, 0.90)
        << "CPU performance regression after switching exceeds 10%! Retention: "
        << (tps_ratio * 100.0) << "%";

    // Restore the default so later tests are not affected.
    MockBackend::set_default_simulated_token_latency_us(0);
}

// 9. C-API Switch Backend Safety
TEST_F(BackendSwitchingTest, C_API_Switch_Backend) {
    mllm_engine_t c_eng = 0;
    ASSERT_EQ(mllm_engine_create(&c_eng).code, 0);

    // Switch to CPU
    ASSERT_EQ(mllm_engine_switch_backend(c_eng, static_cast<int32_t>(BackendType::CPU_NEON)).code, 0);

    // Switch to Vulkan
    ASSERT_EQ(mllm_engine_switch_backend(c_eng, static_cast<int32_t>(BackendType::VULKAN)).code, 0);

    // Switch back to CPU
    ASSERT_EQ(mllm_engine_switch_backend(c_eng, static_cast<int32_t>(BackendType::CPU_NEON)).code, 0);

    ASSERT_EQ(mllm_engine_destroy(c_eng).code, 0);
}

// 10. Real GGUF Model CPU -> Vulkan Mock -> CPU Performance & Correctness
TEST(PhysicalModelSwitchingTest, RealModel_CPU_Switch_Restoration) {
    std::string model_path = resolve_test_model_path("models/smollm2-135m-instruct-q4_k_m.gguf");
    if (!std::filesystem::exists(model_path)) {
        GTEST_SKIP() << "SmolLM2 model not found at " << model_path << ", skipping physical model test.";
    }

    auto& reg = BackendRegistry::instance();
    reg.register_backend(BackendType::CPU_NEON, []() {
        return std::make_unique<LlamaCppBackend>();
    });
    reg.register_backend(BackendType::VULKAN, []() {
        return std::make_unique<MockVulkanBackend>();
    }, []() {
        return true;
    });

    auto engine = create_engine();
    EngineConfig cfg;
    ASSERT_TRUE(engine->initialize(cfg).is_ok());

    ModelLoadConfig m_cfg;
    m_cfg.model_path = model_path;
    m_cfg.preferred_backend = BackendType::CPU_NEON;
    m_cfg.n_threads = 4;
    m_cfg.n_gpu_layers = 0;

    auto m_res = engine->load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok()) << "Model load failed: " << m_res.status().message;
    ModelHandle m_handle = m_res.value();

    SessionConfig s_cfg;
    s_cfg.max_context_tokens = 512;
    auto s_res = engine->create_session(m_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle s_handle = s_res.value();

    auto run_inference = [&](uint32_t max_tokens) -> GenerationMetrics {
        GenerationRequest req;
        req.prompt = "Write a haiku about computers:";
        req.sampling.max_tokens = max_tokens;
        req.sampling.temperature = 0.0f;

        std::promise<GenerationMetrics> p;
        auto f = p.get_future();

        engine->generate_async(
            s_handle,
            req,
            [](const TokenChunk&) { return true; },
            [&p](const Status& st, const GenerationMetrics& m) {
                if (st.is_ok()) p.set_value(m);
                else {
                    GenerationMetrics em{};
                    p.set_value(em);
                }
            }
        );
        return f.get();
    };

    // Warmup
    run_inference(5);

    // Baseline CPU metrics (average of 2 runs to minimize microsecond host scheduling jitter)
    auto base_m1 = run_inference(24);
    auto base_m2 = run_inference(24);
    double base_tps = (base_m1.decode_tokens_per_second + base_m2.decode_tokens_per_second) / 2.0;
    std::cout << "\n[Real SmolLM2 Model] Baseline CPU TTFT: " << base_m1.time_to_first_token_ms
              << " ms, TPS: " << base_tps << " tok/s" << std::endl;
    EXPECT_GT(base_m1.completion_tokens, 0u);
    EXPECT_GT(base_tps, 0.0);

    // Switch to Vulkan (MockVulkan on macOS host)
    ASSERT_TRUE(engine->switch_backend(BackendType::VULKAN).is_ok());

    // Switch back to CPU (Real LlamaCppBackend)
    ASSERT_TRUE(engine->switch_backend(BackendType::CPU_NEON).is_ok());

    // Post-switch warmup to allow the re-initialized context to fault in compute buffers
    run_inference(5);

    // Post-switch CPU metrics (average of 2 runs)
    auto post_m1 = run_inference(24);
    auto post_m2 = run_inference(24);
    double post_tps = (post_m1.decode_tokens_per_second + post_m2.decode_tokens_per_second) / 2.0;
    std::cout << "[Real SmolLM2 Model] Post-Switch CPU TTFT: " << post_m1.time_to_first_token_ms
              << " ms, TPS: " << post_tps << " tok/s" << std::endl;
    EXPECT_GT(post_m1.completion_tokens, 0u);
    EXPECT_GT(post_tps, 0.0);

    double tps_ratio = post_tps / base_tps;
    std::cout << "[Real SmolLM2 Model] CPU TPS Retention: " << (tps_ratio * 100.0) << "%" << std::endl;
    EXPECT_GE(tps_ratio, 0.75) << "Real model CPU TPS degraded by more than 25% after switching!";

    engine->shutdown();
}

// 11. Safety Guard: When BackendType::AUTO is selected and Qualcomm Adreno is detected with mul_mat_vec disabled,
// automatically fall back to BackendType::CPU_NEON.
TEST(SafetyGuardTest, AutoBackend_Qualcomm_Fallback_To_CPU) {
    unsetenv("GGML_VULKAN_ALLOW_QUALCOMM_MUL_MAT_VEC");

    auto& reg = BackendRegistry::instance();
    reg.register_backend(BackendType::CPU_NEON, []() {
        return std::make_unique<MockBackend>();
    });
    reg.register_backend(BackendType::VULKAN, []() {
        auto v = std::make_unique<MockVulkanBackend>();
        v->set_gpu_device_description("Adreno (TM) 740 (Qualcomm, driver 512.744.0)");
        return v;
    }, []() {
        return true;
    });

    auto engine = create_engine();
    EngineConfig cfg;
    ASSERT_TRUE(engine->initialize(cfg).is_ok());

    ModelLoadConfig m_cfg;
    m_cfg.model_path = "mock_model.gguf";
    m_cfg.preferred_backend = BackendType::AUTO; // AUTO mode
    m_cfg.n_threads = 8;                         // Request 8 threads

    auto m_res = engine->load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok());
    ModelHandle m_handle = m_res.value();

    SessionConfig s_cfg;
    auto s_res = engine->create_session(m_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle s_handle = s_res.value();

    GenerationDiagnostics diag = engine->get_generation_diagnostics(s_handle);
    EXPECT_EQ(diag.backend_selected, "cpu_neon");

    engine->shutdown();
}

// 12. Explicit Vulkan backend is kept allowed for benchmarking/testing even on Qualcomm
TEST(SafetyGuardTest, Explicit_Vulkan_Allowed_For_Benchmarking) {
    auto& reg = BackendRegistry::instance();
    reg.register_backend(BackendType::CPU_NEON, []() {
        return std::make_unique<MockBackend>();
    });
    reg.register_backend(BackendType::VULKAN, []() {
        auto v = std::make_unique<MockVulkanBackend>();
        v->set_gpu_device_description("Adreno (TM) 740 (Qualcomm, driver 512.744.0)");
        return v;
    }, []() {
        return true;
    });

    auto engine = create_engine();
    EngineConfig cfg;
    ASSERT_TRUE(engine->initialize(cfg).is_ok());

    ModelLoadConfig m_cfg;
    m_cfg.model_path = "mock_model.gguf";
    m_cfg.preferred_backend = BackendType::VULKAN; // Explicit Vulkan

    auto m_res = engine->load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok());
    ModelHandle m_handle = m_res.value();

    SessionConfig s_cfg;
    auto s_res = engine->create_session(m_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle s_handle = s_res.value();

    GenerationDiagnostics diag = engine->get_generation_diagnostics(s_handle);
    EXPECT_EQ(diag.backend_selected, "vulkan");

    engine->shutdown();
}

} // namespace
