#pragma once

// Solver-wide numeric sentinels. Centralized so we avoid duplicate ad-hoc
// definitions of "infinity" in headers (LP bounds, logger, master, etc.).

#include <cstdint>
#include <limits>

namespace mcfcg {

// Double-precision infinity sentinel for LP bounds, objective values, and
// uninitialized lower-bound placeholders in iteration logs.
inline constexpr double INF = std::numeric_limits<double>::infinity();

// 32-bit "unlimited" sentinel for configuration limits (column aging,
// row inactivity, etc.). Chosen as UINT32_MAX/2 instead of UINT32_MAX so a
// naive `++age` on a counter comparing against this limit cannot overflow
// within any realistic CG run (2.1e9 iterations).
inline constexpr std::uint32_t INF_U32 = std::numeric_limits<std::uint32_t>::max() / 2;

// Minimum problem sizes below which thread_pool dispatch overhead
// dominates the per-element work, so the parallel paths in master/pricer
// fall back to a serial loop.  Tuned for typical instance sizes.
inline constexpr std::uint32_t PAR_ARC_THRESHOLD = 4096;
inline constexpr std::uint32_t PAR_COL_THRESHOLD = 256;
inline constexpr std::uint32_t PAR_SOURCE_THRESHOLD = 64;

}  // namespace mcfcg
