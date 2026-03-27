#pragma once

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mcfcg/cg/tree_column.h"
#include "mcfcg/graph/dijkstra.h"
#include "mcfcg/graph/semiring.h"
#include "mcfcg/instance.h"

namespace mcfcg {

class TreePricer {
   public:
    static constexpr double SCALE = 1e9;
    static constexpr double NEG_RC_TOL = -1e-6;

   private:
    const Instance * _inst = nullptr;
    std::vector<bool> _source_postponed;

   public:
    TreePricer() = default;

    void init(const Instance & inst) {
        _inst = &inst;
        _source_postponed.assign(inst.sources.size(), false);
    }

    // Price all sources. Returns tree columns with negative reduced cost.
    // pi_s: source convexity duals
    // mu: capacity duals (per arc)
    std::vector<TreeColumn> price(
        const std::vector<double> & pi_s,
        const std::unordered_map<uint32_t, double> & mu,
        bool final_round = false) {
        std::unordered_set<uint32_t> no_forbidden;
        return price(pi_s, mu, no_forbidden, final_round);
    }

    std::vector<TreeColumn> price(
        const std::vector<double> & pi_s,
        const std::unordered_map<uint32_t, double> & mu,
        const std::unordered_set<uint32_t> & forbidden_arcs, bool final_round) {
        auto rc = _inst->graph.create_arc_map<int64_t>();
        constexpr int64_t BIG = shortest_path_semiring<int64_t>::infty / 2;
        for (auto a : _inst->graph.arcs()) {
            if (forbidden_arcs.count(a) > 0) {
                rc[a] = BIG;
                continue;
            }
            double cost_a = _inst->cost[a];
            double mu_a = 0.0;
            auto it = mu.find(a);
            if (it != mu.end())
                mu_a = it->second;
            double val = cost_a - mu_a;
            rc[a] = (val <= 0.0)
                        ? int64_t{0}
                        : static_cast<int64_t>(std::round(val * SCALE));
        }

        std::vector<TreeColumn> new_columns;

        for (uint32_t s_idx = 0; s_idx < _inst->sources.size(); ++s_idx) {
            if (!final_round && _source_postponed[s_idx])
                continue;

            const auto & src = _inst->sources[s_idx];
            uint32_t source_v = src.vertex;

            // Run Dijkstra from source to completion (full SPT)
            dijkstra<dijkstra_store_paths> dijk(_inst->graph, rc);
            dijk.add_source(source_v);
            dijk.run();

            // Build tree column and compute true reduced cost.
            // Dijkstra used clamped costs; true RC uses original c_e + mu_e.
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

                // Trace path, accumulate arc flow and costs
                double path_orig_cost = 0.0;
                double path_rc = 0.0;
                uint32_t v = sink;
                while (dijk.has_pred(v)) {
                    uint32_t a = dijk.pred_arc(v);
                    double mu_a = 0.0;
                    auto mit = mu.find(a);
                    if (mit != mu.end())
                        mu_a = mit->second;
                    path_orig_cost += _inst->cost[a];
                    path_rc += _inst->cost[a] - mu_a;
                    arc_flow_map[a] += d;
                    v = _inst->graph.arc_source(a);
                }
                tree_rc += d * path_rc;
                col.cost += d * path_orig_cost;
            }

            if (!all_reachable || tree_rc >= NEG_RC_TOL) {
                _source_postponed[s_idx] = true;
                continue;
            }

            _source_postponed[s_idx] = false;

            for (auto & [arc, flow] : arc_flow_map) {
                col.arc_flows.push_back({arc, flow});
            }

            new_columns.push_back(std::move(col));
        }

        return new_columns;
    }

    void reset_postponed() {
        std::fill(_source_postponed.begin(), _source_postponed.end(), false);
    }
};

}  // namespace mcfcg
