#include <gtest/gtest.h>

#include "mcfcg/graph/d_ary_heap.h"

TEST(DAryHeap, BasicPushPop) {
    mcfcg::d_ary_heap<4, int64_t> heap(10);

    heap.push(3, 30);
    heap.push(1, 10);
    heap.push(5, 50);
    heap.push(2, 20);

    EXPECT_EQ(heap.size(), 4u);
    EXPECT_EQ(heap.top_vertex(), 1u);
    EXPECT_EQ(heap.top_priority(), 10);

    heap.pop();
    EXPECT_EQ(heap.top_vertex(), 2u);
    EXPECT_EQ(heap.top_priority(), 20);

    heap.pop();
    EXPECT_EQ(heap.top_vertex(), 3u);

    heap.pop();
    EXPECT_EQ(heap.top_vertex(), 5u);

    heap.pop();
    EXPECT_TRUE(heap.empty());
}

TEST(DAryHeap, Contains) {
    mcfcg::d_ary_heap<4, int64_t> heap(10);
    EXPECT_FALSE(heap.contains(3));
    heap.push(3, 30);
    EXPECT_TRUE(heap.contains(3));
    heap.pop();
    EXPECT_FALSE(heap.contains(3));
}

TEST(DAryHeap, Promote) {
    mcfcg::d_ary_heap<4, int64_t> heap(10);
    heap.push(1, 100);
    heap.push(2, 200);
    heap.push(3, 300);

    EXPECT_EQ(heap.top_vertex(), 1u);

    heap.promote(3, 50);
    EXPECT_EQ(heap.top_vertex(), 3u);
    EXPECT_EQ(heap.top_priority(), 50);
}

TEST(DAryHeap, PushOrPromote) {
    mcfcg::d_ary_heap<4, int64_t> heap(10);

    // First time: push
    heap.push_or_promote(5, 50);
    EXPECT_TRUE(heap.contains(5));
    EXPECT_EQ(heap.priority(5), 50);

    // Promote
    heap.push_or_promote(5, 25);
    EXPECT_EQ(heap.priority(5), 25);

    // No change (not better)
    heap.push_or_promote(5, 30);
    EXPECT_EQ(heap.priority(5), 25);
}

TEST(DAryHeap, Clear) {
    mcfcg::d_ary_heap<4, int64_t> heap(10);
    heap.push(1, 10);
    heap.push(2, 20);
    heap.clear();
    EXPECT_TRUE(heap.empty());
    EXPECT_FALSE(heap.contains(1));
    EXPECT_FALSE(heap.contains(2));

    // Can re-add after clear
    heap.push(1, 5);
    EXPECT_EQ(heap.top_vertex(), 1u);
}

TEST(DAryHeap, BinaryHeap) {
    mcfcg::d_ary_heap<2, int64_t> heap(10);
    heap.push(0, 5);
    heap.push(1, 3);
    heap.push(2, 7);
    heap.push(3, 1);

    EXPECT_EQ(heap.top_vertex(), 3u);
    heap.pop();
    EXPECT_EQ(heap.top_vertex(), 1u);
}
