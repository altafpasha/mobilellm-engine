#pragma once

#include "mobilellm/types.h"
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace mobilellm::scheduler {

constexpr uint64_t DEFAULT_SAFETY_MARGIN_BYTES = 512ULL * 1024ULL * 1024ULL; // 512 MB
constexpr float MAX_AVAILABLE_RAM_RATIO = 0.70f; // Max 70% of available RAM

enum class AllocationType : uint32_t {
    MODEL_WEIGHTS = 0,
    KV_CACHE = 1,
    SCRATCH_WORKSPACE = 2
};

enum class TrimLevel : uint32_t {
    RUNNING_MODERATE = 0,
    RUNNING_CRITICAL = 1,
    COMPLETE = 2
};

struct MemoryInfo {
    uint64_t total_ram_bytes{0};
    uint64_t available_ram_bytes{0};
    uint64_t free_ram_bytes{0};
};

struct MemoryBudgetStatus {
    uint64_t total_system_ram_bytes{0};
    uint64_t total_budget_ceiling_bytes{0};
    uint64_t current_usage_bytes{0};
    uint64_t model_weights_bytes{0};
    uint64_t kv_cache_bytes{0};
    uint64_t scratch_bytes{0};
    uint64_t safety_margin_bytes{DEFAULT_SAFETY_MARGIN_BYTES};
    uint64_t remaining_budget_bytes{0};
};

using TrimCallback = std::function<void(TrimLevel)>;

/**
 * @brief Dynamic memory budget manager enforcing strict mobile RAM constraints and OOM prevention.
 */
class MemoryBudgetManager {
public:
    explicit MemoryBudgetManager(uint64_t safety_margin = DEFAULT_SAFETY_MARGIN_BYTES);

    /**
     * @brief Queries physical system RAM from OS (/proc/meminfo on Linux/Android, sysctl on macOS).
     */
    static MemoryInfo query_system_memory() noexcept;

    /**
     * @brief Recalculates the active budget ceiling based on current system conditions.
     */
    Status recalculate_budget(uint64_t custom_ceiling_bytes = 0) noexcept;

    /**
     * @brief Checks if a requested memory allocation is safe within budget limits.
     */
    bool can_allocate(uint64_t bytes, AllocationType type) const noexcept;

    /**
     * @brief Records an allocation. Returns OUT_OF_MEMORY if budget ceiling is exceeded.
     */
    Status allocate(uint64_t bytes, AllocationType type) noexcept;

    /**
     * @brief Releases previously accounted memory.
     */
    Status deallocate(uint64_t bytes, AllocationType type) noexcept;

    /**
     * @brief Handles OS low-memory callbacks (Android onTrimMemory).
     */
    Status handle_trim_memory(TrimLevel level) noexcept;

    /**
     * @brief Registers listener for emergency eviction.
     */
    void register_trim_callback(TrimCallback cb);

    /**
     * @brief Returns current accounting statistics.
     */
    MemoryBudgetStatus get_status() const noexcept;

    /**
     * @brief For testing: override system RAM query with simulated values.
     */
    void set_simulated_memory(uint64_t total_ram, uint64_t available_ram);

private:
    mutable std::mutex mutex_;
    uint64_t safety_margin_{DEFAULT_SAFETY_MARGIN_BYTES};
    uint64_t budget_ceiling_{0};

    uint64_t model_allocated_{0};
    uint64_t kv_allocated_{0};
    uint64_t scratch_allocated_{0};

    bool simulated_{false};
    uint64_t sim_total_ram_{0};
    uint64_t sim_available_ram_{0};

    std::vector<TrimCallback> trim_callbacks_;
};

} // namespace mobilellm::scheduler
