#pragma once

#include <cstdint>
#include <vector>

namespace mcfcg {

struct Column {
    double cost;
    // Reduced cost of this column. INVARIANT: every column emitted by the
    // pricer has this set in PathPricer::process_source before push_back.
    // Stored as a member (rather than a parallel array or recomputed during
    // sort) so the per-iter column-cap selection in cg_loop::solve_cg can
    // partial_sort by reduced cost without an O(arcs) recomputation per
    // comparison.  The 0.0 default is only a safety net — if a non-pricer
    // code path ever creates Columns, update the partial_sort comparator.
    double reduced_cost = 0.0;
    uint32_t commodity;
    std::vector<uint32_t> arcs;  // path arcs from source to sink
};

}  // namespace mcfcg
