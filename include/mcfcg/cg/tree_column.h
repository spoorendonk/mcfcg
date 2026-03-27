#pragma once

#include <cstdint>
#include <vector>

namespace mcfcg {

struct TreeColumn {
    uint32_t source_idx;  // index into Instance::sources
    double cost;          // total tree cost = sum_k d_k * dist[t_k]

    // Per-arc flow: f_bar_e = sum_{k: s_k=s, e on path s->t_k} d_k
    // Stored sparsely as (arc_id, flow) pairs
    struct ArcFlow {
        uint32_t arc;
        double flow;
    };
    std::vector<ArcFlow> arc_flows;
};

}  // namespace mcfcg
