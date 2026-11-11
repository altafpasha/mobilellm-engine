#pragma once

#include "backends/common/backend.h"
#include <unordered_map>
#include <functional>
#include <memory>
#include <mutex>

namespace mobilellm {

using BackendFactory = std::function<std::unique_ptr<IBackend>()>;

class BackendRegistry {
public:
    static BackendRegistry& instance() {
        static BackendRegistry s_instance;
        return s_instance;
    }

    void register_backend(BackendType type, BackendFactory factory) {
        std::lock_guard<std::mutex> lock(mutex_);
        factories_[type] = std::move(factory);
    }

    std::unique_ptr<IBackend> create(BackendType type) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (type == BackendType::AUTO) {
            // Priority order: VULKAN -> CPU_NEON -> any available
            if (factories_.count(BackendType::VULKAN)) {
                return factories_[BackendType::VULKAN]();
            }
            if (factories_.count(BackendType::CPU_NEON)) {
                return factories_[BackendType::CPU_NEON]();
            }
            if (!factories_.empty()) {
                return factories_.begin()->second();
            }
            return nullptr;
        }

        auto it = factories_.find(type);
        if (it != factories_.end()) {
            return it->second();
        }
        return nullptr;
    }

    bool has_backend(BackendType type) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return factories_.count(type) > 0;
    }

private:
    BackendRegistry() = default;
    mutable std::mutex mutex_;
    std::unordered_map<BackendType, BackendFactory> factories_;
};

} // namespace mobilellm
