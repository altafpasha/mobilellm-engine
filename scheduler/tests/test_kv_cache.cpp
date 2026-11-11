#include <gtest/gtest.h>
#include "mobilellm/scheduler/kv_cache_manager.h"

using namespace mobilellm::scheduler;
using namespace mobilellm;

TEST(KVCacheTest, BlockAllocationAndRecycling) {
    KVCacheManager kv;
    // 10 blocks, 1024 bytes per block
    ASSERT_TRUE(kv.initialize(10, 1024).is_ok());
    EXPECT_EQ(kv.total_blocks(), 10);
    EXPECT_EQ(kv.free_blocks_count(), 10);
    EXPECT_EQ(kv.allocated_blocks_count(), 0);

    SessionHandle session_1 = 101;
    // Request 32 tokens -> exactly 2 blocks (16 tokens/block)
    auto alloc_res = kv.allocate_blocks_for_tokens(session_1, 32);
    ASSERT_TRUE(alloc_res.is_ok());
    EXPECT_EQ(alloc_res.value().size(), 2);
    EXPECT_EQ(kv.free_blocks_count(), 8);
    EXPECT_EQ(kv.allocated_blocks_count(), 2);

    // Append 17 tokens -> total 49 tokens -> requires 4 blocks (+2 new blocks)
    EXPECT_TRUE(kv.append_tokens(session_1, 17).is_ok());
    EXPECT_EQ(kv.free_blocks_count(), 6);
    EXPECT_EQ(kv.allocated_blocks_count(), 4);

    // Release session -> all 4 blocks must return to free pool
    EXPECT_TRUE(kv.release_session_blocks(session_1).is_ok());
    EXPECT_EQ(kv.free_blocks_count(), 10);
    EXPECT_EQ(kv.allocated_blocks_count(), 0);
}

TEST(KVCacheTest, ContextTruncationReclaimsBlocks) {
    KVCacheManager kv;
    ASSERT_TRUE(kv.initialize(10, 1024).is_ok());

    SessionHandle session = 202;
    // Allocate 64 tokens = 4 blocks
    auto res = kv.allocate_blocks_for_tokens(session, 64);
    ASSERT_TRUE(res.is_ok());
    EXPECT_EQ(kv.free_blocks_count(), 6);

    // Truncate to 16 tokens = 1 block
    EXPECT_TRUE(kv.truncate_session(session, 16).is_ok());
    EXPECT_EQ(kv.free_blocks_count(), 9);
    EXPECT_EQ(kv.allocated_blocks_count(), 1);

    kv.release_session_blocks(session);
    EXPECT_EQ(kv.free_blocks_count(), 10);
}

TEST(KVCacheTest, PrefixBlockSharing) {
    KVCacheManager kv;
    ASSERT_TRUE(kv.initialize(10, 1024).is_ok());

    SessionHandle session_1 = 1;
    SessionHandle session_2 = 2;

    uint64_t sys_prompt_hash = 0xABCDEF1234567890ULL;
    uint8_t dummy_data[32] = {0x42};

    // Session 1 acquires shared prefix block
    auto b1_res = kv.acquire_shared_prefix_block(session_1, sys_prompt_hash, dummy_data, sizeof(dummy_data));
    ASSERT_TRUE(b1_res.is_ok());
    BlockId b1_id = b1_res.value();
    EXPECT_EQ(kv.allocated_blocks_count(), 1);

    // Session 2 acquires the same prefix block
    auto b2_res = kv.acquire_shared_prefix_block(session_2, sys_prompt_hash, dummy_data, sizeof(dummy_data));
    ASSERT_TRUE(b2_res.is_ok());
    BlockId b2_id = b2_res.value();
    EXPECT_EQ(b1_id, b2_id); // Reused identical physical block!
    EXPECT_EQ(kv.allocated_blocks_count(), 1); // No new block allocated!

    // Session 1 closes -> block is NOT freed because session 2 still holds a ref
    EXPECT_TRUE(kv.release_session_blocks(session_1).is_ok());
    EXPECT_EQ(kv.allocated_blocks_count(), 1);

    // Session 2 closes -> block is now freed
    EXPECT_TRUE(kv.release_session_blocks(session_2).is_ok());
    EXPECT_EQ(kv.allocated_blocks_count(), 0);
    EXPECT_EQ(kv.free_blocks_count(), 10);
}

TEST(KVCacheTest, BlockExhaustionReturnsOOM) {
    KVCacheManager kv;
    ASSERT_TRUE(kv.initialize(2, 1024).is_ok()); // Only 2 blocks = 32 tokens max

    SessionHandle session = 1;
    auto res = kv.allocate_blocks_for_tokens(session, 48); // Needs 3 blocks
    EXPECT_FALSE(res.is_ok());
    EXPECT_EQ(res.status().code, StatusCode::OUT_OF_MEMORY);
}

