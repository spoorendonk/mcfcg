#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mcfcg/instance.h"

namespace mcfcg {

// Backend-neutral compact source-based LP (paper section 2.2). One variable
// f^s_e per (source s, edge e): the total flow on edge e originating at s.
//
//   min  sum_s sum_e c_e f^s_e
//   s.t. flow conservation per (s, vertex):  = D_s at s, -d at sinks, 0 else
//        sum_s f^s_e <= u_e  for every capacitated edge
//        f^s_e >= 0
//
// |S|*|E| columns, |S|*|V| equality rows + one row per capacitated edge.
// Stored column-major (CSC): col_start has num_cols+1 entries, the i-th
// column's nonzeros are row_index[col_start[i] .. col_start[i+1]) with the
// matching value[]. Unbounded entries use +/-mcfcg::INF sentinels.
struct SourceLP {
    uint32_t num_cols = 0;
    uint32_t num_rows = 0;
    std::vector<double> col_cost;
    std::vector<double> col_lower;
    std::vector<double> col_upper;
    std::vector<double> row_lower;
    std::vector<double> row_upper;
    std::vector<uint32_t> col_start;  // size num_cols + 1, col_start.back() == nnz
    std::vector<uint32_t> row_index;  // size nnz
    std::vector<double> value;        // size nnz
};

// Dimensions of the compact source LP, from one pass over the arcs and without
// materializing it, so it also answers for instances too large to build.
//
// nnz is exact, not the 3*|S|*|E| bound build_source_lp guards on: a column
// holds +1 at the tail and -1 at the head unless both land on the same row (a
// self-loop, which cancels), plus a capacity entry only when the arc is
// capacitated. Uncapacitated arcs (CommaLab's negative-capacity sentinel, read
// as INF) have no capacity row; on intermodal instances they can be half the
// arcs.
struct SourceLPSize {
    uint64_t cols = 0;  // |S| * |E|
    uint64_t rows = 0;  // |S| * |V| + capacitated arcs
    uint64_t nnz = 0;   // |S| * (2*(|E| - self-loops) + capacitated arcs)
    uint32_t capacitated_arcs = 0;
    uint32_t self_loop_arcs = 0;
};

// Size the compact source LP without building it. Never throws: it has no
// 32-bit index limit to respect, so unlike build_source_lp it also answers for
// instances the exporter refuses.
SourceLPSize source_lp_size(const Instance& inst);

// Build the compact source LP for an instance.
SourceLP build_source_lp(const Instance& inst);

// Write the compact source LP as a free-format MPS file via a streaming,
// backend-neutral writer (zlib only — no HiGHS/COPT/cuOpt). The path extension
// drives compression: gzip when it ends in .gz, plain text otherwise. Returns
// false on any open/write/close failure.
bool write_source_mps(const Instance& inst, const std::string& path);

}  // namespace mcfcg
