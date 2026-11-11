#include "mobilellm/scheduler/kv_cache_manager.h"
#include <cstring>

namespace mobilellm::scheduler {

KVCacheManager::KVCacheManager() = default;

Status KVCacheManager::initialize(uint32_t total_blocks, size_t bytes_per_block) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (total_blocks == 0 || bytes_per_block == 0) {
        return Status::error(StatusCode::INVALID_ARGUMENT, "total_blocks and bytes_per_block must be positive");
    }

    total_blocks_ = total_blocks;
    bytes_per_block_ = bytes_per_block;

    block_pool_.clear();
    block_pool_.resize(total_blocks);
    free_block_indices_.clear();
    free_block_indices_.reserve(total_blocks);

    for (uint32_t i = 0; i < total_blocks; ++i) {
        block_pool_[i].id = i;
        block_pool_[i].ref_count = 0;
        block_pool_[i].prefix_hash = 0;
        block_pool_[i].valid_tokens = 0;
        block_pool_[i].buffer.resize(bytes_per_block, 0);
        free_block_indices_.push_back(i);
    }

    session_tables_.clear();
    prefix_hash_to_block_.clear();
    return Status::ok();
}

BlockId KVCacheManager::pop_free_block_locked() noexcept {
    if (free_block_indices_.empty()) {
        return INVALID_BLOCK_ID;
    }
    BlockId id = free_block_indices_.back();
    free_block_indices_.pop_back();
    return id;
}

void KVCacheManager::push_free_block_locked(BlockId id) noexcept {
    if (id < block_pool_.size()) {
        block_pool_[id].ref_count = 0;
        block_pool_[id].prefix_hash = 0;
        block_pool_[id].valid_tokens = 0;
        free_block_indices_.push_back(id);
    }
}

Result<std::vector<BlockId>> KVCacheManager::allocate_blocks_for_tokens(
    SessionHandle session,
    uint32_t num_tokens
) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (session == 0) {
        return Status::error(StatusCode::INVALID_ARGUMENT, "Invalid session handle");
    }

    uint32_t needed_blocks = (num_tokens + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
    if (needed_blocks == 0) {
        needed_blocks = 1;
    }

    if (free_block_indices_.size() < needed_blocks) {
        return Status::error(StatusCode::OUT_OF_MEMORY,
            "KV-cache block exhaustion: requested " + std::to_string(needed_blocks) +
            " blocks, but only " + std::to_string(free_block_indices_.size()) + " available");
    }

    std::vector<BlockId> allocated;
    allocated.reserve(needed_blocks);
    for (uint32_t i = 0; i < needed_blocks; ++i) {
        BlockId b_id = pop_free_block_locked();
        block_pool_[b_id].ref_count = 1;
        block_pool_[b_id].valid_tokens = 0;
        allocated.push_back(b_id);
    }

    SessionBlockTable table;
    table.session_handle = session;
    table.block_ids = allocated;
    table.total_tokens = num_tokens;

    session_tables_[session] = std::move(table);
    return allocated;
}

Status KVCacheManager::append_tokens(SessionHandle session, uint32_t new_token_count) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = session_tables_.find(session);
    if (it == session_tables_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session KV block table not found");
    }

    SessionBlockTable& table = it->second;
    uint32_t total_after = table.total_tokens + new_token_count;
    uint32_t needed_total_blocks = (total_after + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;

    if (needed_total_blocks > table.block_ids.size()) {
        uint32_t additional = needed_total_blocks - static_cast<uint32_t>(table.block_ids.size());
        if (free_block_indices_.size() < additional) {
            return Status::error(StatusCode::OUT_OF_MEMORY, "KV-cache full: cannot allocate additional blocks");
        }

        for (uint32_t i = 0; i < additional; ++i) {
            BlockId b_id = pop_free_block_locked();
            block_pool_[b_id].ref_count = 1;
            table.block_ids.push_back(b_id);
        }
    }

    table.total_tokens = total_after;
    return Status::ok();
}

Status KVCacheManager::release_session_blocks(SessionHandle session) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = session_tables_.find(session);
    if (it == session_tables_.end()) {
        return Status::ok(); // Already released
    }

    for (BlockId b_id : it->second.block_ids) {
        if (b_id < block_pool_.size()) {
            if (block_pool_[b_id].ref_count > 0) {
                block_pool_[b_id].ref_count--;
            }
            if (block_pool_[b_id].ref_count == 0) {
                if (block_pool_[b_id].prefix_hash != 0) {
                    prefix_hash_to_block_.erase(block_pool_[b_id].prefix_hash);
                }
                push_free_block_locked(b_id);
            }
        }
    }

    session_tables_.erase(it);
    return Status::ok();
}

Status KVCacheManager::truncate_session(SessionHandle session, uint32_t keep_tokens) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = session_tables_.find(session);
    if (it == session_tables_.end()) {
        return Status::error(StatusCode::SESSION_NOT_FOUND, "Session not found for truncation");
    }

    SessionBlockTable& table = it->second;
    if (keep_tokens >= table.total_tokens) {
        return Status::ok(); // Nothing to truncate
    }

    uint32_t blocks_to_keep = (keep_tokens + TOKENS_PER_BLOCK - 1) / TOKENS_PER_BLOCK;
    while (table.block_ids.size() > blocks_to_keep) {
        BlockId b_id = table.block_ids.back();
        table.block_ids.pop_back();

        if (b_id < block_pool_.size()) {
            if (block_pool_[b_id].ref_count > 0) {
                block_pool_[b_id].ref_count--;
            }
            if (block_pool_[b_id].ref_count == 0) {
                if (block_pool_[b_id].prefix_hash != 0) {
                    prefix_hash_to_block_.erase(block_pool_[b_id].prefix_hash);
                }
                push_free_block_locked(b_id);
            }
        }
    }

    table.total_tokens = keep_tokens;
    return Status::ok();
}

Result<BlockId> KVCacheManager::acquire_shared_prefix_block(
    SessionHandle session,
    uint64_t prefix_hash,
    const uint8_t* initial_data,
    size_t data_size
) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (prefix_hash == 0) {
        return Status::error(StatusCode::INVALID_ARGUMENT, "Prefix hash cannot be zero");
    }

    auto p_it = prefix_hash_to_block_.find(prefix_hash);
    if (p_it != prefix_hash_to_block_.end()) {
        BlockId existing_id = p_it->second;
        block_pool_[existing_id].ref_count++;
        session_tables_[session].block_ids.push_back(existing_id);
        session_tables_[session].total_tokens += TOKENS_PER_BLOCK;
        return existing_id;
    }

    // Allocate new block for prefix
    BlockId new_id = pop_free_block_locked();
    if (new_id == INVALID_BLOCK_ID) {
        return Status::error(StatusCode::OUT_OF_MEMORY, "No free KV blocks for prefix cache");
    }

    KVBlock& blk = block_pool_[new_id];
    blk.ref_count = 1;
    blk.prefix_hash = prefix_hash;
    blk.valid_tokens = TOKENS_PER_BLOCK;

    if (initial_data && data_size > 0) {
        size_t copy_bytes = std::min(data_size, blk.buffer.size());
        std::memcpy(blk.buffer.data(), initial_data, copy_bytes);
    }

    prefix_hash_to_block_[prefix_hash] = new_id;
    session_tables_[session].block_ids.push_back(new_id);
    session_tables_[session].total_tokens += TOKENS_PER_BLOCK;
    return new_id;
}

uint32_t KVCacheManager::total_blocks() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_blocks_;
}

uint32_t KVCacheManager::free_blocks_count() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<uint32_t>(free_block_indices_.size());
}

uint32_t KVCacheManager::allocated_blocks_count() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return total_blocks_ - static_cast<uint32_t>(free_block_indices_.size());
}

size_t KVCacheManager::bytes_per_block() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return bytes_per_block_;
}

size_t KVCacheManager::total_memory_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<size_t>(total_blocks_) * bytes_per_block_;
}

size_t KVCacheManager::active_memory_bytes() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<size_t>(total_blocks_ - free_block_indices_.size()) * bytes_per_block_;
}

const SessionBlockTable* KVCacheManager::get_session_table(SessionHandle session) const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = session_tables_.find(session);
    return (it != session_tables_.end()) ? &it->second : nullptr;
}

} // namespace mobilellm::scheduler

