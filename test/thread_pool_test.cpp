#include "mcfcg/util/thread_pool.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <gtest/gtest.h>
#include <thread>
#include <vector>

using mcfcg::thread_pool;

TEST(ThreadPool, ExecutesAllIndicesExactlyOnce) {
    thread_pool pool(4);
    constexpr uint32_t n = 10000;
    std::vector<std::atomic<int>> visited(n);
    for (auto& v : visited) {
        v.store(0);
    }

    pool.parallel_for(n, [&](uint32_t idx, uint32_t /*tid*/) {
        visited[idx].fetch_add(1, std::memory_order_relaxed);
    });

    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(visited[i].load(), 1) << "index " << i;
    }
}

TEST(ThreadPool, ThreadIdInRange) {
    thread_pool pool(4);
    constexpr uint32_t n = 1000;
    std::atomic<uint32_t> max_tid{0};

    pool.parallel_for(n, [&](uint32_t /*idx*/, uint32_t tid) {
        uint32_t prev = max_tid.load();
        while (tid > prev && !max_tid.compare_exchange_weak(prev, tid)) {
        }
    });

    EXPECT_LT(max_tid.load(), pool.num_threads());
}

TEST(ThreadPool, RepeatedCallsDoNotRace) {
    thread_pool pool(4);
    for (int round = 0; round < 100; ++round) {
        const uint32_t n = 50 + (round % 13);
        std::vector<std::atomic<int>> visited(n);
        for (auto& v : visited) {
            v.store(0);
        }

        pool.parallel_for(n, [&](uint32_t idx, uint32_t /*tid*/) {
            visited[idx].fetch_add(1, std::memory_order_relaxed);
        });

        for (uint32_t i = 0; i < n; ++i) {
            ASSERT_EQ(visited[i].load(), 1) << "round " << round << " index " << i;
        }
    }
}

TEST(ThreadPool, NZeroIsNoop) {
    thread_pool pool(4);
    std::atomic<int> calls{0};
    pool.parallel_for(0, [&](uint32_t, uint32_t) { calls.fetch_add(1); });
    EXPECT_EQ(calls.load(), 0);
}

TEST(ThreadPool, NOneRunsOnce) {
    thread_pool pool(4);
    std::atomic<int> calls{0};
    pool.parallel_for(1, [&](uint32_t idx, uint32_t /*tid*/) {
        EXPECT_EQ(idx, 0u);
        calls.fetch_add(1);
    });
    EXPECT_EQ(calls.load(), 1);
}

TEST(ThreadPool, NLessThanNumThreads) {
    thread_pool pool(8);
    constexpr uint32_t n = 3;
    std::vector<std::atomic<int>> visited(n);
    for (auto& v : visited) {
        v.store(0);
    }

    pool.parallel_for(n, [&](uint32_t idx, uint32_t /*tid*/) { visited[idx].fetch_add(1); });

    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(visited[i].load(), 1);
    }
}

TEST(ThreadPool, SingleThreadedPool) {
    thread_pool pool(1);
    constexpr uint32_t n = 100;
    std::vector<int> visited(n, 0);
    pool.parallel_for(n, [&](uint32_t idx, uint32_t tid) {
        EXPECT_EQ(tid, 0u);
        visited[idx]++;
    });
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_EQ(visited[i], 1);
    }
}

// With static chunking, one slow task on a chunk would force every thread
// to wait for that chunk to drain. With dynamic dispatch, fast threads
// pick up the remaining work while one thread handles the slow task,
// so wall time should be ~max(task) instead of ~sum/threads.
TEST(ThreadPool, DynamicDispatchBalancesUnevenWork) {
    constexpr uint32_t num_threads = 4;
    thread_pool pool(num_threads);

    // 1 slow task (50 ms) + many fast tasks (0). With static chunking,
    // the thread that owns the slow task is alone for ~50 ms.
    // With dynamic dispatch, fast threads keep grabbing fast tasks
    // and the wall time is still ~50 ms.
    constexpr uint32_t n = 200;
    constexpr auto slow_task_ms = std::chrono::milliseconds(50);

    auto start = std::chrono::steady_clock::now();
    pool.parallel_for(n, [&](uint32_t idx, uint32_t /*tid*/) {
        if (idx == 0) {
            std::this_thread::sleep_for(slow_task_ms);
        }
    });
    auto elapsed = std::chrono::steady_clock::now() - start;

    // Generous upper bound: the slow task plus scheduling slack.
    auto upper = slow_task_ms + std::chrono::milliseconds(50);
    EXPECT_LT(elapsed, upper)
        << "wall time " << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
        << " ms exceeded " << upper.count() << " ms";
}
