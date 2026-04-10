#pragma once

// Test fixtures write small instance files in the build cwd.  Under
// `ctest -j$(nproc)` each gtest test runs in its own `mcfcg_tests` process,
// and several of those processes share the same cwd.  Without a per-process
// prefix, two parallel tests using the same fixture can race on the file:
// one's TearDown deletes a file the other is still reading.
//
// `unique_test_path(base)` returns a path that includes the current process
// id, so concurrent test processes never collide.

#include <string>
#include <string_view>
#include <unistd.h>

namespace mcfcg::test {

inline std::string unique_test_path(std::string_view base) {
    return std::to_string(::getpid()) + "_" + std::string(base);
}

}  // namespace mcfcg::test
