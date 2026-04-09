#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace mcfcg {

// Lightweight fork-join thread pool for parallel_for workloads.
// Workers are created once and reused across calls. The calling
// thread participates as thread_id 0 to avoid wasting a core.
class thread_pool {
public:
    explicit thread_pool(uint32_t num_threads) : _num_threads(std::max(num_threads, uint32_t{1})) {
        _workers.reserve(_num_threads - 1);
        for (uint32_t ti = 1; ti < _num_threads; ++ti) {
            _workers.emplace_back([this, ti] { worker_loop(ti); });
        }
    }

    ~thread_pool() {
        {
            std::lock_guard lock(_mutex);
            _stop = true;
            ++_generation;
        }
        _work_cv.notify_all();
        for (auto& worker : _workers)
            worker.join();
    }

    thread_pool(const thread_pool&) = delete;
    thread_pool& operator=(const thread_pool&) = delete;

    // Execute task(task_index, thread_id) for task_index in [0, n).
    // Blocks until all tasks complete.
    template <typename F>
    void parallel_for(uint32_t n, F&& task) {
        if (n == 0)
            return;

        if (_num_threads <= 1 || n == 1) {
            for (uint32_t idx = 0; idx < n; ++idx)
                task(idx, uint32_t{0});
            return;
        }

        // Publish work inside the lock so the mutex release establishes
        // happens-before with worker acquire (guarantees visibility).
        {
            std::lock_guard lock(_mutex);
            _task_fn = [&task](uint32_t idx, uint32_t tid) { task(idx, tid); };
            _task_count = n;
            _done_count = 0;
            ++_generation;
        }
        _work_cv.notify_all();

        // Calling thread does its share as thread 0
        run_range(0, n);

        // Wait for all workers to finish
        {
            std::unique_lock lock(_mutex);
            _done_cv.wait(lock, [this] { return _done_count >= _num_threads - 1; });
        }

        _task_fn = nullptr;
    }

    uint32_t num_threads() const noexcept { return _num_threads; }

private:
    void worker_loop(uint32_t thread_id) {
        uint64_t seen_gen = 0;
        while (true) {
            {
                std::unique_lock lock(_mutex);
                _work_cv.wait(lock, [&] { return _generation > seen_gen || _stop; });
                if (_stop)
                    return;
                seen_gen = _generation;
            }

            run_range(thread_id, _task_count);

            {
                std::lock_guard lock(_mutex);
                ++_done_count;
            }
            _done_cv.notify_one();
        }
    }

    void run_range(uint32_t thread_id, uint32_t n) {
        uint32_t active = std::min(_num_threads, n);
        if (thread_id >= active)
            return;

        uint32_t chunk = n / active;
        uint32_t remainder = n % active;

        uint32_t begin = thread_id * chunk + std::min(thread_id, remainder);
        uint32_t end = begin + chunk + (thread_id < remainder ? 1 : 0);

        for (uint32_t idx = begin; idx < end; ++idx)
            _task_fn(idx, thread_id);
    }

    uint32_t _num_threads;
    std::vector<std::thread> _workers;

    // Synchronization
    std::mutex _mutex;
    std::condition_variable _work_cv;
    std::condition_variable _done_cv;
    uint64_t _generation = 0;
    uint32_t _done_count = 0;
    bool _stop = false;

    // Current task
    std::function<void(uint32_t, uint32_t)> _task_fn;
    uint32_t _task_count = 0;
};

inline std::unique_ptr<thread_pool> make_thread_pool(uint32_t num_threads) {
    if (num_threads == 0)
        num_threads = std::max(1u, std::thread::hardware_concurrency());
    if (num_threads <= 1)
        return nullptr;
    return std::make_unique<thread_pool>(num_threads);
}

}  // namespace mcfcg
