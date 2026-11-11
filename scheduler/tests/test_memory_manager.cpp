#include <gtest/gtest.h>
#include "mobilellm/scheduler/memory_manager.h"

using namespace mobilellm::scheduler;
using namespace mobilellm;

TEST(MemoryManagerTest, QuerySystemMemory) {
    MemoryInfo info = MemoryBudgetManager::query_system_memory();
    EXPECT_GT(info.total_ram_bytes, 0);
    EXPECT_GT(info.available_ram_bytes, 0);
}

TEST(MemoryManagerTest, BudgetCeilingCalculation) {
    MemoryBudgetManager mgr(512ULL * 1024ULL * 1024ULL); // 512 MB safety margin
    // Simulate 4 GB total, 2 GB available RAM
    uint64_t total = 4ULL * 1024ULL * 1024ULL * 1024ULL;
    uint64_t avail = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    mgr.set_simulated_memory(total, avail);

    // Usable = 2048 MB - 512 MB = 1536 MB
    // Budget ceiling = 1536 MB * 0.70 = 1075.2 MB (~1,127,428,915 bytes)
    auto status = mgr.get_status();
    EXPECT_EQ(status.total_system_ram_bytes, total);
    EXPECT_GT(status.total_budget_ceiling_bytes, 1000ULL * 1024ULL * 1024ULL);
    EXPECT_LT(status.total_budget_ceiling_bytes, 1100ULL * 1024ULL * 1024ULL);
}

TEST(MemoryManagerTest, AllocationAndOOMRejection) {
    MemoryBudgetManager mgr;
    mgr.set_simulated_memory(4ULL * 1024ULL * 1024ULL * 1024ULL, 2ULL * 1024ULL * 1024ULL * 1024ULL);

    uint64_t small_alloc = 200ULL * 1024ULL * 1024ULL; // 200 MB
    EXPECT_TRUE(mgr.can_allocate(small_alloc, AllocationType::MODEL_WEIGHTS));
    EXPECT_TRUE(mgr.allocate(small_alloc, AllocationType::MODEL_WEIGHTS).is_ok());

    auto status = mgr.get_status();
    EXPECT_EQ(status.model_weights_bytes, small_alloc);
    EXPECT_EQ(status.current_usage_bytes, small_alloc);

    // Attempt massive allocation exceeding budget ceiling (e.g. 1.5 GB)
    uint64_t huge_alloc = 1500ULL * 1024ULL * 1024ULL;
    EXPECT_FALSE(mgr.can_allocate(huge_alloc, AllocationType::KV_CACHE));
    Status oom_st = mgr.allocate(huge_alloc, AllocationType::KV_CACHE);
    EXPECT_FALSE(oom_st.is_ok());
    EXPECT_EQ(oom_st.code, StatusCode::OUT_OF_MEMORY);

    // Deallocate
    EXPECT_TRUE(mgr.deallocate(small_alloc, AllocationType::MODEL_WEIGHTS).is_ok());
    EXPECT_EQ(mgr.get_status().current_usage_bytes, 0);
}

TEST(MemoryManagerTest, HandleTrimMemoryCallback) {
    MemoryBudgetManager mgr;
    bool callback_called = false;
    TrimLevel received_level = TrimLevel::RUNNING_MODERATE;

    mgr.register_trim_callback([&](TrimLevel lvl) {
        callback_called = true;
        received_level = lvl;
    });

    EXPECT_TRUE(mgr.handle_trim_memory(TrimLevel::RUNNING_CRITICAL).is_ok());
    EXPECT_TRUE(callback_called);
    EXPECT_EQ(received_level, TrimLevel::RUNNING_CRITICAL);
}
