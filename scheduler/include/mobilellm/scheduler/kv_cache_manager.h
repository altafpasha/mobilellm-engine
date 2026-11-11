#pragma once

#include "mobilellm/types.h"
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <memory>

namespace mobilellm::scheduler {

constexpr uint32_t TOKENS_PER_BLOCK = 16;
using BlockId = uint32_t;
constexpr BlockId INVALID_BLOCK_ID = 0xFFFFFFFF;

/**
 * @brief Discrete physical block holding key and value representations for TOKENS_PER_BLOCK tokens.
 */
struct KVBlock {
    BlockId id{INVALID_BLOCK_ID};
    uint32_t ref_count{0};
    uint64_t prefix_hash{0};
    uint32_t valid_tokens{0};
    std::vector<uint8_t> buffer; // Raw tensor storage in native memory (Zero SQLite)
};

/**
 * @brief Block mapping table mapping a session's logical tokens to physical KV blocks.
 */
struct SessionBlockTable {
    SessionHandle session_handle{0};
    std::vector<BlockId> block_ids;
    uint32_t total_tokens{0};
};

/**
 * @brief Paged KV-Cache Manager allocating discrete 16-token virtual blocks,
 * eliminating heap fragmentation and supporting prefix sharing.
 */
class KVCacheManager {
public:
    KVCacheManager();
    ~KVCacheManager() = default;

    /**
     * @brief Initializes the physical block pool arena.
     * @param total_blocks Total number of discrete physical blocks available.
     * @param bytes_per_block Memory byte size required for 16 tokens across all model layers.
     */
    Status initialize(uint32_t total_blocks, size_t bytes_per_block) noexcept;

    /**
     * @brief Allocates sufficient blocks for a session to store the requested number of tokens.
     */
    Result<std::vector<BlockId>> allocate_blocks_for_tokens(
        SessionHandle session,
        uint32_t num_tokens
    ) noexcept;

    /**
     * @brief Appends token capacity to an existing session, allocating additional blocks as needed.
     */
    Status append_tokens(SessionHandle session, uint32_t new_token_count) noexcept;

    /**
     * @brief Releases all KV blocks held by a session back to the free block pool.
     */
    Status release_session_blocks(SessionHandle session) noexcept;

    /**
     * @brief Truncates a session's context, releasing trailing blocks.
     */
    Status truncate_session(SessionHandle session, uint32_t keep_tokens) noexcept;

    /**
     * @brief Retrieves or allocates a shared prefix block with the given prefix hash.
     */
    Result<BlockId> acquire_shared_prefix_block(
        SessionHandle session,
        uint64_t prefix_hash,
        const uint8_t* initial_data,
        size_t data_size
    ) noexcept;

    // Telemetry & Inspection
    uint32_t total_blocks() const noexcept;
    uint32_t free_blocks_count() const noexcept;
    uint32_t allocated_blocks_count() const noexcept;
    size_t bytes_per_block() const noexcept;
    size_t total_memory_bytes() const noexcept;
    size_t active_memory_bytes() const noexcept;

    const SessionBlockTable* get_session_table(SessionHandle session) const noexcept;

private:
    mutable std::mutex mutex_;
    uint32_t total_blocks_{0};
    size_t bytes_per_block_{0};

    std::vector<KVBlock> block_pool_;
    std::vector<BlockId> free_block_indices_;

    std::unordered_map<SessionHandle, SessionBlockTable> session_tables_;
    std::unordered_map<uint64_t, BlockId> prefix_hash_to_block_;

    BlockId pop_free_block_locked() noexcept;
    void push_free_block_locked(BlockId id) noexcept;
};

} // namespace mobilellm::scheduler

