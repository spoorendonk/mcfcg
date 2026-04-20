#pragma once

#include "mcfcg/cg/pricer_base.h"
#include "mcfcg/cg/tree_column.h"

#include <cassert>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace mcfcg {

class TreePricer : public PricerBase<TreePricer, TreeColumn> {
    friend class PricerBase<TreePricer, TreeColumn>;

    void process_source(uint32_t s_idx, const Source& src, const std::vector<double>& pi_s,
                        const static_map<uint32_t, double>& mu, auto& dijk,
                        std::vector<TreeColumn>& new_columns) {
        TreeColumn col;
        col.source_idx = s_idx;
        col.cost = 0.0;
        double tree_rc = -pi_s[s_idx];

        if (_track_arcs) {
            _source_arcs[s_idx].clear();
        }

        std::unordered_map<uint32_t, double> arc_flow_map;

        for (uint32_t k : src.commodity_indices) {
            uint32_t sink = _inst->commodities[k].sink;
            // A* runs until every target sink is settled; an unvisited
            // sink would mean the instance graph is disconnected between
            // source and sink, which is a caller-level error.
            assert(dijk.visited(sink));
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
