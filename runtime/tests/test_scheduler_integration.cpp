#include <gtest/gtest.h>
#include "runtime/src/engine/engine_impl.h"
#include "backends/mock/mock_backend.h"
#include <vector>
#include <thread>
#include <future>

using namespace mobilellm;

class SchedulerIntegrationTest : public ::testing::Test {
protected:
    std::unique_ptr<EngineImpl> engine;
    MockBackend* raw_mock{nullptr};

    void SetUp() override {
        engine = std::make_unique<EngineImpl>();
        auto mock = std::make_unique<MockBackend>();
        raw_mock = mock.get();
        engine->set_backend_override(std::move(mock));

        EngineConfig cfg;
        ASSERT_TRUE(engine->initialize(cfg).is_ok());
    }

    void TearDown() override {
        if (engine) {
            engine->shutdown();
        }
    }
};

TEST_F(SchedulerIntegrationTest, SchedulerInitializationAndShutdown) {
    EXPECT_NE(engine->memory_manager(), nullptr);
    EXPECT_NE(engine->scheduler(), nullptr);

    auto mem_status = engine->memory_manager()->get_status();
    EXPECT_GT(mem_status.total_budget_ceiling_bytes, 0);
    EXPECT_EQ(mem_status.current_usage_bytes, 0);

    // Shutdown releases scheduler and flushes state
    EXPECT_TRUE(engine->shutdown().is_ok());
    EXPECT_EQ(engine->scheduler(), nullptr);
}

TEST_F(SchedulerIntegrationTest, ModelMemoryReservation) {
    auto mem_mgr = engine->memory_manager();
    ASSERT_NE(mem_mgr, nullptr);

    uint64_t initial_usage = mem_mgr->get_status().current_usage_bytes;
    EXPECT_EQ(initial_usage, 0);

    // Load model with explicit 150 MB budget
    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/mock_model.gguf";
    m_cfg.memory_budget_mb = 150.0f;

    auto m_res = engine->load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok());
    ModelHandle m_handle = m_res.value();

    uint64_t expected_bytes = 150ULL * 1024ULL * 1024ULL;
    auto status = mem_mgr->get_status();
    EXPECT_EQ(status.model_weights_bytes, expected_bytes);
    EXPECT_EQ(status.current_usage_bytes, expected_bytes);

    // Model unload releases reservation
    EXPECT_TRUE(engine->unload_model(m_handle).is_ok());
    status = mem_mgr->get_status();
    EXPECT_EQ(status.model_weights_bytes, 0);
    EXPECT_EQ(status.current_usage_bytes, 0);
}

TEST_F(SchedulerIntegrationTest, SessionMemoryReservation) {
    auto mem_mgr = engine->memory_manager();
    ASSERT_NE(mem_mgr, nullptr);

    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/mock_model.gguf";
    m_cfg.memory_budget_mb = 100.0f;
    auto m_res = engine->load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok());
    ModelHandle m_handle = m_res.value();

    uint64_t model_bytes = mem_mgr->get_status().model_weights_bytes;
    EXPECT_GT(model_bytes, 0);

    // Create session with 2048 context tokens
    SessionConfig s_cfg;
    s_cfg.max_context_tokens = 2048;
    auto s_res = engine->create_session(m_handle, s_cfg);
    ASSERT_TRUE(s_res.is_ok());
    SessionHandle s_handle = s_res.value();

    auto status = mem_mgr->get_status();
    EXPECT_GT(status.kv_cache_bytes, 0);
    EXPECT_EQ(status.current_usage_bytes, model_bytes + status.kv_cache_bytes);

    // Engine metrics reflect actual allocated KV cache
    EngineMetrics metrics = engine->get_metrics();
    EXPECT_EQ(metrics.active_models, 1);
    EXPECT_EQ(metrics.active_sessions, 1);
    EXPECT_EQ(metrics.kv_cache_allocated_bytes, status.kv_cache_bytes);
    EXPECT_EQ(metrics.resident_memory_bytes, status.current_usage_bytes);

    // Destroy session releases only session KV memory
    EXPECT_TRUE(engine->destroy_session(s_handle).is_ok());
    status = mem_mgr->get_status();
    EXPECT_EQ(status.kv_cache_bytes, 0);
    EXPECT_EQ(status.model_weights_bytes, model_bytes);
    EXPECT_EQ(status.current_usage_bytes, model_bytes);
}

TEST_F(SchedulerIntegrationTest, BudgetExhaustionReturnsOOM) {
    auto mem_mgr = engine->memory_manager();
    ASSERT_NE(mem_mgr, nullptr);

    // Constrain budget to a ceiling of 50 MB
    uint64_t small_ceiling = 50ULL * 1024ULL * 1024ULL;
    mem_mgr->recalculate_budget(small_ceiling);

    // Attempt to load a model requiring 100 MB -> must fail with OUT_OF_MEMORY
    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/too_large.gguf";
    m_cfg.memory_budget_mb = 100.0f;

    auto m_res = engine->load_model(m_cfg);
    ASSERT_FALSE(m_res.is_ok());
    EXPECT_EQ(m_res.status().code, StatusCode::OUT_OF_MEMORY);
    EXPECT_NE(m_res.status().message.find("exceeds available memory budget ceiling"), std::string::npos);

    // Memory usage must remain 0
    EXPECT_EQ(mem_mgr->get_status().current_usage_bytes, 0);
}

TEST_F(SchedulerIntegrationTest, SessionBudgetExhaustionReturnsOOM) {
    auto mem_mgr = engine->memory_manager();
    ASSERT_NE(mem_mgr, nullptr);

    // Set budget ceiling to 110 MB
    uint64_t tight_ceiling = 110ULL * 1024ULL * 1024ULL;
    mem_mgr->recalculate_budget(tight_ceiling);

    // Load model requiring 100 MB -> leaves only 10 MB headroom
    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/mock_model.gguf";
    m_cfg.memory_budget_mb = 100.0f;
    auto m_res = engine->load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok());

    // Attempt to create session requiring ~47 MB KV cache (2048 tokens * ~23 KB)
    SessionConfig s_cfg;
    s_cfg.max_context_tokens = 2048;
    auto s_res = engine->create_session(m_res.value(), s_cfg);
    ASSERT_FALSE(s_res.is_ok());
    EXPECT_EQ(s_res.status().code, StatusCode::OUT_OF_MEMORY);
    EXPECT_NE(s_res.status().message.find("exceeds available memory budget ceiling"), std::string::npos);

    // Memory usage must still be strictly 100 MB (no leaked KV reservation)
    EXPECT_EQ(mem_mgr->get_status().kv_cache_bytes, 0);
    EXPECT_EQ(mem_mgr->get_status().model_weights_bytes, 100ULL * 1024ULL * 1024ULL);
}

TEST_F(SchedulerIntegrationTest, ReservationRollbackOnModelLoadFailure) {
    auto mem_mgr = engine->memory_manager();
    ASSERT_NE(mem_mgr, nullptr);

    // Configure backend to fail model loading
    raw_mock->set_simulate_load_failure(true);

    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/failing_model.gguf";
    m_cfg.memory_budget_mb = 80.0f;

    auto m_res = engine->load_model(m_cfg);
    ASSERT_FALSE(m_res.is_ok());

    // Reservation must be rolled back completely
    EXPECT_EQ(mem_mgr->get_status().model_weights_bytes, 0);
    EXPECT_EQ(mem_mgr->get_status().current_usage_bytes, 0);
}

TEST_F(SchedulerIntegrationTest, ReservationRollbackOnSessionCreateFailure) {
    auto mem_mgr = engine->memory_manager();
    ASSERT_NE(mem_mgr, nullptr);

    // Load model successfully
    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/model.gguf";
    m_cfg.memory_budget_mb = 50.0f;
    auto m_res = engine->load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok());

    // Configure backend to fail session creation (e.g. simulated backend OOM)
    raw_mock->set_simulate_oom(true);

    SessionConfig s_cfg;
    s_cfg.max_context_tokens = 1024;
    auto s_res = engine->create_session(m_res.value(), s_cfg);
    ASSERT_FALSE(s_res.is_ok());

    // KV reservation must be rolled back, leaving only model reservation
    EXPECT_EQ(mem_mgr->get_status().kv_cache_bytes, 0);
    EXPECT_EQ(mem_mgr->get_status().model_weights_bytes, 50ULL * 1024ULL * 1024ULL);
}

TEST_F(SchedulerIntegrationTest, ModelUnloadReleasesAssociatedSessionReservations) {
    auto mem_mgr = engine->memory_manager();
    ASSERT_NE(mem_mgr, nullptr);

    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/model.gguf";
    m_cfg.memory_budget_mb = 50.0f;
    auto m_res = engine->load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok());
    ModelHandle m_handle = m_res.value();

    SessionConfig s_cfg;
    s_cfg.max_context_tokens = 1024;
    auto s1 = engine->create_session(m_handle, s_cfg);
    auto s2 = engine->create_session(m_handle, s_cfg);
    ASSERT_TRUE(s1.is_ok());
    ASSERT_TRUE(s2.is_ok());

    EXPECT_GT(mem_mgr->get_status().kv_cache_bytes, 0);
    EXPECT_GT(mem_mgr->get_status().model_weights_bytes, 0);

    // Unloading model must cascade: destroy sessions and release ALL reservations
    EXPECT_TRUE(engine->unload_model(m_handle).is_ok());
    EXPECT_EQ(mem_mgr->get_status().kv_cache_bytes, 0);
    EXPECT_EQ(mem_mgr->get_status().model_weights_bytes, 0);
    EXPECT_EQ(mem_mgr->get_status().current_usage_bytes, 0);
}

TEST_F(SchedulerIntegrationTest, MultipleIndependentSessionsAccounting) {
    auto mem_mgr = engine->memory_manager();
    ASSERT_NE(mem_mgr, nullptr);

    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/model.gguf";
    m_cfg.memory_budget_mb = 50.0f;
    auto m_res = engine->load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok());
    ModelHandle m_handle = m_res.value();

    SessionConfig s_cfg1;
    s_cfg1.max_context_tokens = 1024;
    auto s1_res = engine->create_session(m_handle, s_cfg1);
    ASSERT_TRUE(s1_res.is_ok());
    uint64_t kv_s1 = mem_mgr->get_status().kv_cache_bytes;

    SessionConfig s_cfg2;
    s_cfg2.max_context_tokens = 2048;
    auto s2_res = engine->create_session(m_handle, s_cfg2);
    ASSERT_TRUE(s2_res.is_ok());
    uint64_t kv_total = mem_mgr->get_status().kv_cache_bytes;

    EXPECT_GT(kv_total, kv_s1);
    uint64_t kv_s2 = kv_total - kv_s1;

    // Destroy session 1 -> kv_cache_bytes should drop to kv_s2
    EXPECT_TRUE(engine->destroy_session(s1_res.value()).is_ok());
    EXPECT_EQ(mem_mgr->get_status().kv_cache_bytes, kv_s2);

    // Destroy session 2 -> kv_cache_bytes should drop to 0
    EXPECT_TRUE(engine->destroy_session(s2_res.value()).is_ok());
    EXPECT_EQ(mem_mgr->get_status().kv_cache_bytes, 0);
}

TEST_F(SchedulerIntegrationTest, ConcurrentMemoryReservation) {
    auto mem_mgr = engine->memory_manager();
    ASSERT_NE(mem_mgr, nullptr);

    ModelLoadConfig m_cfg;
    m_cfg.model_path = "/tmp/concurrent_model.gguf";
    m_cfg.memory_budget_mb = 50.0f;
    auto m_res = engine->load_model(m_cfg);
    ASSERT_TRUE(m_res.is_ok());
    ModelHandle m_handle = m_res.value();

    constexpr int NUM_THREADS = 8;
    constexpr int SESSIONS_PER_THREAD = 4;
    std::vector<std::thread> threads;

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([this, m_handle]() {
            for (int s = 0; s < SESSIONS_PER_THREAD; ++s) {
                SessionConfig s_cfg;
                s_cfg.max_context_tokens = 512;
                auto s_res = engine->create_session(m_handle, s_cfg);
                if (s_res.is_ok()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    engine->destroy_session(s_res.value());
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    // After all threads finish, all session reservations must be zero
    EXPECT_EQ(mem_mgr->get_status().kv_cache_bytes, 0);
    EXPECT_EQ(mem_mgr->get_status().model_weights_bytes, 50ULL * 1024ULL * 1024ULL);
}

