#pragma once

#include "mcfcg/cg/pricer.h"
#include "mcfcg/cg/tree_column.h"
#include "mcfcg/graph/dijkstra.h"
#include "mcfcg/graph/semiring.h"
#include "mcfcg/instance.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mcfcg {

class TreePricer {
public:
    static constexpr double SCALE = 1e9;
    static constexpr double NEG_RC_TOL = -1e-6;

private:
    const Instance* _inst = nullptr;
    std::vector<bool> _source_postponed;
    std::vector<std::vector<uint32_t>> _source_arcs;  // arcs used per source in last pricing
    PricingMode _mode = PricingMode::AStar;
    static_map<uint32_t, int64_t> _lower_bounds;

public:
    TreePricer() = default;

    void init(const Instance& inst, PricingMode mode = PricingMode::AStar) {
        _inst = &inst;
        _source_postponed.assign(inst.sources.size(), false);
        _source_arcs.resize(inst.sources.size());
        _mode = mode;
        if (_mode == PricingMode::AStar) {
            _lower_bounds = compute_lower_bounds_to_targets(inst, SCALE);
        }
    }

    // Price all sources. Returns tree columns with negative reduced cost.
    // pi_s: source convexity duals
    // mu: capacity duals (per arc)
    std::vector<TreeColumn> price(const std::vector<double>& pi_s,
                                  const std::unordered_map<uint32_t, double>& mu,
                                  bool final_round = false) {
        std::unordered_set<uint32_t> no_forbidden;
        return price(pi_s, mu, no_forbidden, final_round);
    }

    std::vector<TreeColumn> price(const std::vector<double>& pi_s,
                                  const std::unordered_map<uint32_t, double>& mu,
                                  const std::unordered_set<uint32_t>& forbidden_arcs,
                                  bool final_round) {
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
            rc[a] = (val <= 0.0) ? int64_t{0} : static_cast<int64_t>(std::round(val * SCALE));
        }

        std::vector<TreeColumn> new_columns;

        for (uint32_t s_idx = 0; s_idx < _inst->sources.size(); ++s_idx) {
            if (!final_round && _source_postponed[s_idx])
                continue;

            const auto& src = _inst->sources[s_idx];
            uint32_t source_v = src.vertex;

            if (_mode == PricingMode::AStar) {
                price_source_astar(s_idx, src, source_v, rc, pi_s, mu, new_columns);
            } else {
                price_source_dijkstra(s_idx, src, source_v, rc, pi_s, mu, new_columns);
            }
        }

        return new_columns;
    }

    // After new capacity constraints are added, mark sources for re-pricing
    // based on whether their last shortest-path tree used any newly constrained
    // arc.  Affected sources are un-postponed (their reduced costs changed);
    // unaffected sources are postponed (their reduced costs are unchanged).
    void filter_for_new_caps(const std::vector<uint32_t>& new_cap_arcs) {
        std::unordered_set<uint32_t> cap_set(new_cap_arcs.begin(), new_cap_arcs.end());
        for (uint32_t s = 0; s < _source_postponed.size(); ++s) {
            bool affected = std::any_of(_source_arcs[s].begin(), _source_arcs[s].end(),
                                        [&](uint32_t a) { return cap_set.contains(a); });
            _source_postponed[s] = !affected;
        }
    }

    void reset_postponed() { std::fill(_source_postponed.begin(), _source_postponed.end(), false); }

private:
    void build_tree_column(uint32_t s_idx, const Source& src, const std::vector<double>& pi_s,
                           const std::unordered_map<uint32_t, double>& mu, auto& dijk,
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

        // Record arcs used by this source for capacity filtering
        _source_arcs[s_idx].clear();
        _source_arcs[s_idx].reserve(arc_flow_map.size());
        for (auto& [arc, flow] : arc_flow_map) {
            _source_arcs[s_idx].push_back(arc);
        }

        if (!all_reachable || tree_rc >= NEG_RC_TOL) {
            _source_postponed[s_idx] = true;
            return;
        }

        _source_postponed[s_idx] = false;

        for (auto& [arc, flow] : arc_flow_map) {
            col.arc_flows.push_back({arc, flow});
        }

        new_columns.push_back(std::move(col));
    }

    void price_source_dijkstra(uint32_t s_idx, const Source& src, uint32_t source_v,
                               const static_map<uint32_t, int64_t>& rc,
                               const std::vector<double>& pi_s,
                               const std::unordered_map<uint32_t, double>& mu,
                               std::vector<TreeColumn>& new_columns) {
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

        dijkstra<dijkstra_store_paths> dijk(_inst->graph, rc);
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

        build_tree_column(s_idx, src, pi_s, mu, dijk, new_columns);
    }

    void price_source_astar(uint32_t s_idx, const Source& src, uint32_t source_v,
                            const static_map<uint32_t, int64_t>& rc,
                            const std::vector<double>& pi_s,
                            const std::unordered_map<uint32_t, double>& mu,
                            std::vector<TreeColumn>& new_columns) {
        // Build target set: all unique sinks.
        auto is_target = _inst->graph.create_vertex_map<bool>(false);
        uint32_t num_targets = 0;
        for (uint32_t k : src.commodity_indices) {
            uint32_t sink = _inst->commodities[k].sink;
            if (!is_target[sink]) {
                is_target[sink] = true;
                ++num_targets;
            }
        }

        // Use precomputed lower bounds (original costs, computed once at init).
        astar_dijkstra<dijkstra_store_paths> dijk(_inst->graph, rc, _lower_bounds);
        dijk.add_source(source_v);
        dijk.run_until_targets(is_target, num_targets);

        build_tree_column(s_idx, src, pi_s, mu, dijk, new_columns);
    }
};

}  // namespace mcfcg
