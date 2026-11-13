#include "mobilellm/scheduler/memory_manager.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <sys/resource.h>

#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <mach/mach_types.h>
#include <mach/mach_init.h>
#include <mach/mach_host.h>
#endif

namespace mobilellm::scheduler {

MemoryBudgetManager::MemoryBudgetManager(uint64_t safety_margin)
    : safety_margin_(safety_margin) {
    recalculate_budget(0);
}

void MemoryBudgetManager::set_simulated_memory(uint64_t total_ram, uint64_t available_ram) {
    std::lock_guard<std::mutex> lock(mutex_);
    simulated_ = true;
    sim_total_ram_ = total_ram;
    sim_available_ram_ = available_ram;

    uint64_t usable = (sim_available_ram_ > safety_margin_) ? (sim_available_ram_ - safety_margin_) : 0;
    budget_ceiling_ = static_cast<uint64_t>(usable * MAX_AVAILABLE_RAM_RATIO);
}

MemoryInfo MemoryBudgetManager::query_system_memory() noexcept {
    MemoryInfo info;

#if defined(__APPLE__)
    // macOS host RAM detection
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    uint64_t memsize = 0;
    size_t len = sizeof(memsize);
    if (sysctl(mib, 2, &memsize, &len, nullptr, 0) == 0) {
        info.total_ram_bytes = memsize;
    }

    vm_size_t page_size;
    mach_port_t mach_port = mach_host_self();
    vm_statistics64_data_t vm_stats;
    mach_msg_type_number_t count = sizeof(vm_stats) / sizeof(natural_t);
    if (host_page_size(mach_port, &page_size) == KERN_SUCCESS &&
        host_statistics64(mach_port, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm_stats), &count) == KERN_SUCCESS) {
        uint64_t free_bytes = (static_cast<uint64_t>(vm_stats.free_count) + 
                               static_cast<uint64_t>(vm_stats.inactive_count)) * page_size;
        info.free_ram_bytes = free_bytes;
        info.available_ram_bytes = free_bytes;
    } else {
        info.available_ram_bytes = info.total_ram_bytes / 2;
    }
#else
    // Linux / Android /proc/meminfo query
    std::ifstream file("/proc/meminfo");
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string key;
            uint64_t value_kb = 0;
            std::string unit;
            if (iss >> key >> value_kb >> unit) {
                if (key == "MemTotal:") {
                    info.total_ram_bytes = value_kb * 1024ULL;
                } else if (key == "MemAvailable:") {
                    info.available_ram_bytes = value_kb * 1024ULL;
                } else if (key == "MemFree:") {
                    info.free_ram_bytes = value_kb * 1024ULL;
                }
            }
        }
    }

    if (info.available_ram_bytes == 0 && info.total_ram_bytes > 0) {
        info.available_ram_bytes = info.total_ram_bytes / 2;
    }
#endif

    if (info.total_ram_bytes == 0) {
        // Fallback default: 4 GB
        info.total_ram_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
        info.available_ram_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    }

    return info;
}

ProcessMemoryStats MemoryBudgetManager::query_process_memory() noexcept {
    ProcessMemoryStats stats;

#if defined(__APPLE__)
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        stats.current_rss_bytes = info.resident_size;
        stats.vmsize_bytes = info.virtual_size;
    }
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        // ru_maxrss is in bytes on macOS
        stats.peak_rss_bytes = static_cast<uint64_t>(usage.ru_maxrss);
    }
#else
    std::ifstream file("/proc/self/status");
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string key;
            uint64_t value_kb = 0;
            std::string unit;
            if (iss >> key >> value_kb >> unit) {
                if (key == "VmRSS:") {
                    stats.current_rss_bytes = value_kb * 1024ULL;
                } else if (key == "VmHWM:") {
                    stats.peak_rss_bytes = value_kb * 1024ULL;
                } else if (key == "VmSize:") {
                    stats.vmsize_bytes = value_kb * 1024ULL;
                }
            }
        }
    }
    if (stats.peak_rss_bytes == 0) {
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            // ru_maxrss is in kilobytes on Linux/Android
            stats.peak_rss_bytes = static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL;
        }
    }
#endif

    if (stats.peak_rss_bytes < stats.current_rss_bytes) {
        stats.peak_rss_bytes = stats.current_rss_bytes;
    }

    return stats;
}

Status MemoryBudgetManager::recalculate_budget(uint64_t custom_ceiling_bytes) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);

    uint64_t avail = 0;
    MemoryInfo sys_info;
    if (simulated_) {
        avail = sim_available_ram_;
    } else {
        sys_info = query_system_memory();
        avail = sys_info.available_ram_bytes;
    }

    uint64_t effective_avail = (avail > safety_margin_) ? (avail - safety_margin_) : 0;
    uint64_t calculated_ceiling = static_cast<uint64_t>(effective_avail * MAX_AVAILABLE_RAM_RATIO);

    if (!simulated_) {
        // On real hardware (Linux/Android), allow up to 85% of available RAM if it provides more headroom
        // while preserving at least a 256MB safety cushion.
        uint64_t min_safety_bytes = static_cast<uint64_t>(256ULL * 1024ULL * 1024ULL);
        uint64_t mobile_ceiling = static_cast<uint64_t>(avail * 0.85f);
        if (mobile_ceiling > calculated_ceiling && avail > min_safety_bytes) {
            uint64_t max_safe = avail - min_safety_bytes;
            calculated_ceiling = (mobile_ceiling < max_safe) ? mobile_ceiling : max_safe;
        }

        // On mobile devices, cached background processes are evicted by Android lmkd on demand.
        // If total_ram_bytes is known, allow up to 75% of total RAM minus OS reserve (1.5 GB).
        if (sys_info.total_ram_bytes > 0) {
            uint64_t os_reserve = 1536ULL * 1024ULL * 1024ULL; // 1.5 GB for OS
            if (sys_info.total_ram_bytes > os_reserve) {
                uint64_t ram_based_ceiling = static_cast<uint64_t>((sys_info.total_ram_bytes - os_reserve) * 0.75f);
                if (ram_based_ceiling > calculated_ceiling) {
                    calculated_ceiling = ram_based_ceiling;
                }
            }
        }
    }

    if (custom_ceiling_bytes > 0) {
        budget_ceiling_ = custom_ceiling_bytes;
    } else {
        budget_ceiling_ = calculated_ceiling;
    }

    return Status::ok();
}

bool MemoryBudgetManager::can_allocate(uint64_t bytes, AllocationType /*type*/) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t current = model_allocated_ + kv_allocated_ + scratch_allocated_;
    return (current + bytes) <= budget_ceiling_;
}

Status MemoryBudgetManager::allocate(uint64_t bytes, AllocationType type) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    uint64_t current = model_allocated_ + kv_allocated_ + scratch_allocated_;
    if (current + bytes > budget_ceiling_) {
        return Status::error(StatusCode::OUT_OF_MEMORY,
            "Allocation of " + std::to_string(bytes / (1024 * 1024)) +
            " MB exceeds mobile budget ceiling of " +
            std::to_string(budget_ceiling_ / (1024 * 1024)) + " MB");
    }

    switch (type) {
        case AllocationType::MODEL_WEIGHTS:
            model_allocated_ += bytes;
            break;
        case AllocationType::KV_CACHE:
            kv_allocated_ += bytes;
            break;
        case AllocationType::SCRATCH_WORKSPACE:
            scratch_allocated_ += bytes;
            break;
    }

    return Status::ok();
}

Status MemoryBudgetManager::deallocate(uint64_t bytes, AllocationType type) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    switch (type) {
        case AllocationType::MODEL_WEIGHTS:
            model_allocated_ = (model_allocated_ >= bytes) ? (model_allocated_ - bytes) : 0;
            break;
        case AllocationType::KV_CACHE:
            kv_allocated_ = (kv_allocated_ >= bytes) ? (kv_allocated_ - bytes) : 0;
            break;
        case AllocationType::SCRATCH_WORKSPACE:
            scratch_allocated_ = (scratch_allocated_ >= bytes) ? (scratch_allocated_ - bytes) : 0;
            break;
    }
    return Status::ok();
}

Status MemoryBudgetManager::handle_trim_memory(TrimLevel level) noexcept {
    std::vector<TrimCallback> callbacks;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        callbacks = trim_callbacks_;
    }

    for (const auto& cb : callbacks) {
        if (cb) {
            cb(level);
        }
    }

    // Refresh budget after OS trim signal
    recalculate_budget(0);
    return Status::ok();
}

void MemoryBudgetManager::register_trim_callback(TrimCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    trim_callbacks_.push_back(std::move(cb));
}

MemoryBudgetStatus MemoryBudgetManager::get_status() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    MemoryBudgetStatus s;
    if (simulated_) {
        s.total_system_ram_bytes = sim_total_ram_;
    } else {
        s.total_system_ram_bytes = query_system_memory().total_ram_bytes;
    }

    s.total_budget_ceiling_bytes = budget_ceiling_;
    s.model_weights_bytes = model_allocated_;
    s.kv_cache_bytes = kv_allocated_;
    s.scratch_bytes = scratch_allocated_;
    s.current_usage_bytes = model_allocated_ + kv_allocated_ + scratch_allocated_;
    s.safety_margin_bytes = safety_margin_;
    s.remaining_budget_bytes = (budget_ceiling_ >= s.current_usage_bytes)
        ? (budget_ceiling_ - s.current_usage_bytes) : 0;

    return s;
}

} // namespace mobilellm::scheduler

