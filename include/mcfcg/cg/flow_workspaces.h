#pragma once

#include "mcfcg/graph/static_digraph.h"
#include "mcfcg/graph/static_map.h"

#include <cstdint>
#include <vector>

namespace mcfcg {

// Per-thread scratch space reused across CG iterations for the two
// parallel passes in MasterBase:
//
//   * compute_arc_flow: each thread accumulates its chunk of columns'
//     demand-weighted arc flow into a dedicated arc-indexed map (one
//     per thread).  The per-map layout is identical so the merge-sum
//     that follows is straight indexed arithmetic.
//
//   * find_violated_arcs: each thread appends arcs whose merged flow
//     exceeds capacity into a per-thread vector, then the main thread
//     concatenates and sorts (sort gives deterministic ordering across
//     runs regardless of pool scheduling).
//
// Both vectors are sized [num_threads] at master init and empty when
// the master has no pool — the serial paths in compute_arc_flow /
// find_violated_arcs use stack locals and do not touch these.
//
// The reason this lives outside MasterBase: keeping the workspaces in
// one place makes it explicit that flow accumulation and violation
// scanning share a lifetime and the same per-thread indexing scheme.
// The logic itself stays on MasterBase because it is CRTP-templated on
// the Derived's for_each_arc_coeff hook.
struct FlowWorkspaces {
    std::vector<static_map<uint32_t, double>> flow;
    std::vector<std::vector<uint32_t>> violated_arcs;

    // Resize both vectors to num_threads.  `flow` entries are
    // arc-indexed zero-initialised maps.  Callers fill() before each
    // parallel pass to reset to zero.  No-op when num_threads == 0.
    void init(const static_digraph& graph, uint32_t num_threads) {
        flow.clear();
        violated_arcs.clear();
        if (num_threads == 0) {
            return;
        }
        flow.reserve(num_threads);
        for (uint32_t thread_i = 0; thread_i < num_threads; ++thread_i) {
            flow.push_back(graph.create_arc_map<double>(0.0));
        }
        violated_arcs.resize(num_threads);
    }
};

}  // namespace mcfcg
