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

        // Fresh scratch map per call — a per-thread reused map keeps the
        // bucket array allocated but its iteration order drifts with
        // bucket-count history, which perturbs downstream LP numerics
        // enough to trip the tight EXISTING_COL_RC_TOL invariant (the
        // test's acceptance bound must match the pricer's NEG_RC_TOL to
        // avoid a duplicate-column window).  Per-call allocation is
        // cheap compared to the Dijkstra that precedes it.
        std::unordered_map<uint32_t, double> arc_flow_map;

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
            uint32_t path_arcs = 0;
            uint32_t v = sink;
            while (dijk.has_pred(v)) {
                uint32_t a = dijk.pred_arc(v);
                path_orig_cost += _inst->cost[a];
                path_rc += _inst->cost[a] - mu[a];
                arc_flow_map[a] += d;
                v = _inst->graph.arc_source(a);
                ++path_arcs;
            }
            tree_rc += d * path_rc;
            col.cost += d * path_orig_cost;
            // Tree column's rc is the demand-weighted sum of its per-
            // commodity path rcs, so the rounding-error budget is
            // demand-weighted too: d * L / SCALE.
            _thread_rc_error_bound[thread_id] += d * static_cast<double>(path_arcs) / SCALE;
        }

        if (_track_arcs) {
            _source_arcs[s_idx].reserve(arc_flow_map.size());
            for (auto& [arc, flow] : arc_flow_map) {
                _source_arcs[s_idx].push_back(arc);
            }
        }

        // Lagrangian LB accumulator: this source's best tree RC,
        // regardless of whether the col gets emitted.  Zero for
        // non-attractive sources.
        if (tree_rc < 0.0) {
            _thread_min_rc_sum[thread_id] += tree_rc;
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
