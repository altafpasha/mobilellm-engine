#include <gtest/gtest.h>
#include "mobilellm/scheduler/request_queue.h"

using namespace mobilellm::scheduler;
using namespace mobilellm;

TEST(RequestQueueTest, PriorityOrdering) {
    RequestQueue queue(10);

    auto req_bg = std::make_shared<QueuedRequest>();
    req_bg->request_id = 1;
    req_bg->priority = RequestPriority::BACKGROUND;

    auto req_norm = std::make_shared<QueuedRequest>();
    req_norm->request_id = 2;
    req_norm->priority = RequestPriority::NORMAL;

    auto req_int = std::make_shared<QueuedRequest>();
    req_int->request_id = 3;
    req_int->priority = RequestPriority::INTERACTIVE;

    // Enqueue background, then normal, then interactive
    EXPECT_TRUE(queue.enqueue(req_bg).is_ok());
    EXPECT_TRUE(queue.enqueue(req_norm).is_ok());
    EXPECT_TRUE(queue.enqueue(req_int).is_ok());
    EXPECT_EQ(queue.size(), 3);

    // Dequeue must yield INTERACTIVE first, then NORMAL, then BACKGROUND
    auto first = queue.dequeue(false);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->request_id, 3);

    auto second = queue.dequeue(false);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(second->request_id, 2);

    auto third = queue.dequeue(false);
    ASSERT_NE(third, nullptr);
    EXPECT_EQ(third->request_id, 1);

    EXPECT_TRUE(queue.empty());
}

TEST(RequestQueueTest, CancellationInQueue) {
    RequestQueue queue(10);

    auto req = std::make_shared<QueuedRequest>();
    req->request_id = 42;
    req->priority = RequestPriority::NORMAL;

    EXPECT_TRUE(queue.enqueue(req).is_ok());
    EXPECT_TRUE(queue.cancel(42).is_ok());

    // Dequeue should skip the cancelled request
    auto popped = queue.dequeue(false);
    EXPECT_EQ(popped, nullptr);
}

TEST(RequestQueueTest, CapacityExhaustion) {
    RequestQueue queue(2); // Capacity = 2

    auto r1 = std::make_shared<QueuedRequest>();
    auto r2 = std::make_shared<QueuedRequest>();
    auto r3 = std::make_shared<QueuedRequest>();

    EXPECT_TRUE(queue.enqueue(r1).is_ok());
    EXPECT_TRUE(queue.enqueue(r2).is_ok());
    Status err = queue.enqueue(r3);
    EXPECT_FALSE(err.is_ok());
    EXPECT_EQ(err.code, StatusCode::OUT_OF_MEMORY);
}

