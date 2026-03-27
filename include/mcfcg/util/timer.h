#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace mcfcg {

enum class TimerCat : uint8_t { LP = 0, Pricing = 1, Separation = 2, Total = 3 };

class Timer {
    static constexpr size_t kNumCats = 4;
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    std::array<double, kNumCats> _elapsed{};
    std::array<TimePoint, kNumCats> _start{};

public:
    void start(TimerCat cat) { _start[static_cast<size_t>(cat)] = Clock::now(); }

    void stop(TimerCat cat) {
        auto idx = static_cast<size_t>(cat);
        _elapsed[idx] += std::chrono::duration<double>(Clock::now() - _start[idx]).count();
    }

    [[nodiscard]] double elapsed(TimerCat cat) const { return _elapsed[static_cast<size_t>(cat)]; }

    void reset() { _elapsed.fill(0.0); }
};

}  // namespace mcfcg
