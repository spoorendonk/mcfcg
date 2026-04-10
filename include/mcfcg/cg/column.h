#pragma once

#include <cstdint>
#include <vector>

namespace mcfcg {

struct Column {
    double cost;
    // Reduced cost assigned by the pricer.  Only consulted when the
    // per-iter column cap truncates the batch — see cg_loop::solve_cg.
    double reduced_cost = 0.0;
    uint32_t commodity;
    std::vector<uint32_t> arcs;  // path arcs from source to sink
};

}  // namespace mcfcg
