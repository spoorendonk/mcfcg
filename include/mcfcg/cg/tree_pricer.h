#pragma once

#include "mcfcg/cg/pricer_base.h"
#include "mcfcg/cg/tree_column.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace mcfcg {

class TreePricer : public PricerBase<TreePricer, TreeColumn> {
    friend class PricerBase<TreePricer, TreeColumn>;

    void process_source(uint32_t s_idx, const Source& src, const std::vector<double>& pi_s,
                        const static_map<uint32_t, double>& mu, auto& dijk,
                        std::vector<TreeColumn>& new_columns, uint32_t thread_id) {
        TreeColumn col;
        col.source_idx = s_idx;
        col.cost = 0.0;
        double tree_rc = -pi_s[s_idx];

        if (_track_arcs) {
            _source_arcs[s_idx].clear();
        }

        // Reuse the per-thread scratch map: clear() retains the bucket
        // storage, so the second process_source call on this thread
        // skips the initial hash-table allocation.  Iteration order
        // drifts with bucket-count history across calls — that drift
        // shows up as small LP-backend dual noise in downstream RC
        // recomputation, and is absorbed by EXISTING_COL_RC_TOL.
        auto& arc_flow_map = _thread_arc_flow[thread_id];
        arc_flow_map.clear();

        for (uint32_t k : src.commodity_indices) {
            uint32_t sink = _inst->commodities[k].sink;
            // A* exhausts its heap when no path to sink exists (disconnected
            // source→sink).  Skip the unreachable commodity and keep
            // building a partial tree over the remaining reachable sinks.
            // The partial tree still contributes xi=1 to its source's
            // convexity row, so the LP picks it as a valid candidate;
            // unmet demand for the unreachable sink is absorbed by
            // CommodityRows-mode demand slacks or causes LP infeasibility
            // in EdgeRows mode (no demand slacks exist).  Preprocess
            // disconnected instances via mcfcg_clean before solving.
            if (!dijk.visited(sink))
                continue;
            double d = _inst->commodities[k].demand;

            double path_orig_cost = 0.0;
            double path_rc = 0.0;
            uint32_t v = sink;
            while (dijk.has_pred(v)) {
                uint32_t a = dijk.pred_arc(v);
                path_orig_cost += _inst->cost[a];
                path_rc += _inst->cost[a] - mu[a];
                arc_flow_map[a] += d;
                v = _inst->graph.arc_source(a);
            }
            tree_rc += d * path_rc;
            col.cost += d * path_orig_cost;
        }

        if (_track_arcs) {
            _source_arcs[s_idx].reserve(arc_flow_map.size());
            for (auto& [arc, flow] : arc_flow_map) {
                _source_arcs[s_idx].push_back(arc);
            }
        }

        if (tree_rc >= _neg_rc_tol) {
            _source_postponed[s_idx] = 1;
            return;
        }

        _source_postponed[s_idx] = 0;
        col.reduced_cost = tree_rc;

        for (auto& [arc, flow] : arc_flow_map) {
            col.arc_flows.push_back({arc, flow});
        }

        new_columns.push_back(std::move(col));
    }
};

}  // namespace mcfcg
