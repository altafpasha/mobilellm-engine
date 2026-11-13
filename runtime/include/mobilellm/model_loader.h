#pragma once

#include "mobilellm/types.h"
#include <string>
#include <cstdint>
#include <vector>

namespace mobilellm {

constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF" in little-endian
constexpr uint32_t GGUF_VERSION_2 = 2;
constexpr uint32_t GGUF_VERSION_3 = 3;

struct GGUFHeader {
    uint32_t magic{0};
    uint32_t version{0};
    uint64_t tensor_count{0};
    uint64_t metadata_kv_count{0};
};

struct ModelMetadata {
    std::string filepath;
    uint32_t gguf_version{0};
    std::string architecture;
    std::string description;
    uint32_t context_length{2048};
    uint32_t embedding_length{0};
    uint32_t block_count{0};
    uint32_t head_count{0};
    uint32_t head_count_kv{0};
    uint64_t tensor_count{0};
    uint64_t metadata_kv_count{0};
    uint64_t file_size_bytes{0};
    std::string quantization;
};

/**
 * @brief Utility for safely parsing and validating GGUF model files before allocation.
 */
class ModelLoader {
public:
    /**
     * @brief Inspects and extracts metadata from a GGUF model file without loading all weights.
     */
    static Result<ModelMetadata> inspect_file(const std::string& filepath) noexcept;

    /**
     * @brief Validates GGUF magic, supported version, and structural integrity.
     */
    static Status validate_file(const std::string& filepath) noexcept;

    /**
     * @brief Memory maps a file into the process virtual address space.
     */
    static Result<void*> mmap_file(const std::string& filepath, size_t& out_size) noexcept;

    /**
     * @brief Unmaps a previously memory-mapped file.
     */
    static Status munmap_file(void* addr, size_t size) noexcept;
};

} // namespace mobilellm

