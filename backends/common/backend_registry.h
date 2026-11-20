#pragma once

#include "backends/common/backend.h"
#include <unordered_map>
#include <functional>
#include <memory>
#include <mutex>

namespace mobilellm {

using BackendFactory = std::function<std::unique_ptr<IBackend>()>;
using BackendProbe = std::function<bool()>;

class BackendRegistry {
public:
    static BackendRegistry& instance() {
        static BackendRegistry s_instance;
        return s_instance;
    }

    void register_backend(BackendType type, BackendFactory factory, BackendProbe probe = nullptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        factories_[type] = std::move(factory);
        if (probe) {
            probes_[type] = std::move(probe);
        } else {
            probes_.erase(type);
        }
    }

    std::unique_ptr<IBackend> create(BackendType type) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (type == BackendType::AUTO) {
            // Priority order: CPU_NEON (primary mobile standard) -> VULKAN (if hardware supported) -> any available
            if (is_available_locked(BackendType::CPU_NEON)) {
                return factories_[BackendType::CPU_NEON]();
            }
            if (is_available_locked(BackendType::VULKAN)) {
                return factories_[BackendType::VULKAN]();
            }
            if (!factories_.empty()) {
                return factories_.begin()->second();
            }
            return nullptr;
        }

        auto it = factories_.find(type);
        if (it != factories_.end()) {
            if (!is_available_locked(type)) {
                return nullptr;
            }
            return it->second();
        }
        return nullptr;
    }

    bool has_backend(BackendType type) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return factories_.count(type) > 0;
    }

    bool is_available(BackendType type) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return is_available_locked(type);
    }

private:
    BackendRegistry() = default;

    bool is_available_locked(BackendType type) const {
        auto it = factories_.find(type);
        if (it == factories_.end()) return false;
        auto pit = probes_.find(type);
        if (pit != probes_.end() && pit->second) {
            return pit->second();
        }
        return true;
    }

    mutable std::mutex mutex_;
    std::unordered_map<BackendType, BackendFactory> factories_;
    std::unordered_map<BackendType, BackendProbe> probes_;
};

} // namespace mobilellm

