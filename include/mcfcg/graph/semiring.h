#pragma once

#include <limits>

namespace mcfcg {

template <typename T>
struct shortest_path_semiring {
    using value_type = T;
    static constexpr T ZERO = T{0};
    static constexpr T INFTY = std::numeric_limits<T>::max();

    // Saturating addition to avoid overflow with scaled int64_t costs.
    static constexpr T plus(T a, T b) noexcept {
        if (a > 0 && b > INFTY - a) {
            return INFTY;
        }
        return a + b;
    }
    static constexpr bool less(T a, T b) noexcept { return a < b; }
};

}  // namespace mcfcg
