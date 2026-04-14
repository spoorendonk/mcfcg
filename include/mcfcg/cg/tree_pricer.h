#pragma once

#include "mcfcg/cg/pricer_base.h"
#include "mcfcg/cg/tree_column.h"

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace mcfcg {

class TreePricer : public PricerBase<TreePricer, TreeColumn> {
    friend class PricerBase<TreePricer, TreeColumn>;

    void process_source(uint32_t s_idx, const Source& src, const std::vector<double>& pi_s,
                        const static_map<uint32_t, double>& mu, auto& dijk,
                        std::vector<TreeColumn>& new_columns) {
        bool all_reachable = true;
        TreeColumn col;
        col.source_idx = s_idx;
        col.cost = 0.0;
        double tree_rc = -pi_s[s_idx];

        std::unordered_map<uint32_t, double> arc_flow_map;

        for (uint32_t k : src.commodity_indices) {
            uint32_t sink = _inst->commodities[k].sink;
            if (!dijk.visited(sink)) {
                all_reachable = false;
                break;
            }
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

        // Record arcs used by this source for capacity filtering
        if (_track_arcs) {
            _source_arcs[s_idx].clear();
            _source_arcs[s_idx].reserve(arc_flow_map.size());
            for (auto& [arc, flow] : arc_flow_map) {
                _source_arcs[s_idx].push_back(arc);
            }
        }

        if (!all_reachable || tree_rc >= _neg_rc_tol) {
            _source_postponed[s_idx] = true;
            return;
        }

        _source_postponed[s_idx] = false;
        col.reduced_cost = tree_rc;

        for (auto& [arc, flow] : arc_flow_map) {
            col.arc_flows.push_back({arc, flow});
        }

        new_columns.push_back(std::move(col));
    }

    void price_source_dijkstra(uint32_t s_idx, const Source& src, uint32_t source_v,
                               const std::vector<double>& pi_s,
                               const static_map<uint32_t, double>& mu,
                               std::vector<TreeColumn>& new_columns, uint32_t thread_id) {
        constexpr auto MAX_BOUND = shortest_path_semiring<int64_t>::infty / 2;

        // Shrinking-budget bound: budget starts at pi_s * SCALE.
        // As targets are settled, subtract d_k * dist_k from budget.
        // Bound = budget / min_remaining_demand.
        struct target_info {
            uint32_t commodity;
            double demand;
        };
        std::unordered_map<uint32_t, std::vector<target_info>> remaining;
        double min_demand = std::numeric_limits<double>::max();
        for (uint32_t k : src.commodity_indices) {
            uint32_t sink = _inst->commodities[k].sink;
            double d = _inst->commodities[k].demand;
            remaining[sink].push_back({k, d});
            min_demand = std::min(min_demand, d);
        }

        double budget = pi_s[s_idx] * SCALE;
        uint32_t num_remaining = static_cast<uint32_t>(src.commodity_indices.size());

        auto compute_bound = [&]() -> int64_t {
            if (num_remaining == 0 || budget <= 0.0)
                return int64_t{0};
            double raw = budget / min_demand;
            return raw >= static_cast<double>(MAX_BOUND) ? MAX_BOUND
                                                         : static_cast<int64_t>(std::ceil(raw));
        };

        int64_t dual_bound = compute_bound();

        auto& ws = _workspaces[thread_id];
        ws.reset();
        dijkstra<dijkstra_store_paths> dijk(_inst->graph, _rc, ws);
        dijk.add_source(source_v);

        while (!dijk.finished() && num_remaining > 0) {
            auto [u, u_dist] = dijk.current();
            if (u_dist > dual_bound)
                break;
            dijk.advance();

            auto rit = remaining.find(u);
            if (rit != remaining.end()) {
                for (auto& ti : rit->second) {
                    budget -= ti.demand * static_cast<double>(u_dist);
                    --num_remaining;
                }
                remaining.erase(rit);
                if (num_remaining > 0) {
                    min_demand = std::numeric_limits<double>::max();
                    for (auto& [_, infos] : remaining) {
                        for (auto& ti : infos) {
                            min_demand = std::min(min_demand, ti.demand);
                        }
                    }
                }
                dual_bound = compute_bound();
            }
        }

        process_source(s_idx, src, pi_s, mu, dijk, new_columns);
    }
};

}  // namespace mcfcg
