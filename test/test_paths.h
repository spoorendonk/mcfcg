#pragma once

// Test fixtures write small instance files in the build cwd.  Under
// `ctest -j$(nproc)` each gtest test runs in its own `mcfcg_tests` process,
// and several of those processes share the same cwd.  Without a per-process
// prefix, two parallel tests using the same fixture can race on the file:
// one's TearDown deletes a file the other is still reading.
//
// `unique_test_path(base)` returns a path that includes the current process
// id, so concurrent test processes never collide.

#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
#include <unistd.h>

namespace mcfcg::test {

inline std::string unique_test_path(std::string_view base) {
    return std::to_string(::getpid()) + "_" + std::string(base);
}

// CommaLab plain-numeric writer for test fixtures.  Header:
//   <num_vertices>\n<num_arcs>\n<num_commodities>\n
// followed by arc_lines (each `src dst cost cap`) and commodity_lines
// (each `origin destination demand`), 1-indexed.  Shared across test
// files to avoid duplicate local helpers.
inline void writeInstance(const std::string& path, uint32_t vertices, uint32_t arcs,
                          uint32_t commodities, const std::string& arc_lines,
                          const std::string& commodity_lines) {
    std::ofstream out(path);
    out << vertices << '\n' << arcs << '\n' << commodities << '\n';
    out << arc_lines << commodity_lines;
}

}  // namespace mcfcg::test
