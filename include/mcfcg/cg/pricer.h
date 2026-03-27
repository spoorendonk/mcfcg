#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mcfcg/cg/column.h"
#include "mcfcg/graph/dijkstra.h"
#include "mcfcg/instance.h"

namespace mcfcg {

class PathPricer {
   public:
    using vertex_t = uint32_t;

    static constexpr double SCALE = 1e9;
    static constexpr double NEG_RC_TOL = -1e-6;

   private:
    const Instance * _inst = nullptr;
    std::vector<bool> _source_postponed;

   public:
    PathPricer() = default;

    void init(const Instance & inst) {
        _inst = &inst;
        _source_postponed.assign(inst.sources.size(), false);
    }

    std::vector<Column> price(const std::vector<double> & pi,
                              const std::unordered_map<uint32_t, double> & mu,
                              bool final_round = false) {
        std::unordered_set<uint32_t> no_forbidden;
        return price(pi, mu, no_forbidden, final_round);
    }

    std::vector<Column> price(
        const std::vector<double> & pi,
        const std::unordered_map<uint32_t, double> & mu,
        const std::unordered_set<uint32_t> & forbidden_arcs, bool final_round) {
        // Compute clamped reduced costs for Dijkstra.
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

        std::vector<Column> new_columns;

        for (uint32_t s_idx = 0; s_idx < _inst->sources.size(); ++s_idx) {
            if (!final_round && _source_postponed[s_idx])
                continue;

            const auto & src = _inst->sources[s_idx];
            vertex_t source_v = src.vertex;

            dijkstra<dijkstra_store_paths> dijk(_inst->graph, rc);
            dijk.add_source(source_v);
            dijk.run();

            bool found_any = false;

            for (uint32_t k : src.commodity_indices) {
                vertex_t sink = _inst->commodities[k].sink;
                if (!dijk.visited(sink))
                    continue;

                // Extract path and compute true reduced cost
                Column col;
                col.cost = 0.0;
                col.commodity = k;
                double true_rc = -pi[k];
                vertex_t v = sink;
                while (dijk.has_pred(v)) {
                    uint32_t a = dijk.pred_arc(v);
                    col.arcs.push_back(a);
                    col.cost += _inst->cost[a];
                    double mu_a = 0.0;
                    auto mit = mu.find(a);
                    if (mit != mu.end())
                        mu_a = mit->second;
                    true_rc += _inst->cost[a] - mu_a;
                    v = _inst->graph.arc_source(a);
                }

                if (true_rc >= NEG_RC_TOL)
                    continue;

                found_any = true;
                std::reverse(col.arcs.begin(), col.arcs.end());
                new_columns.push_back(std::move(col));
            }

            _source_postponed[s_idx] = !found_any;
        }

        return new_columns;
    }

    void reset_postponed() {
        std::fill(_source_postponed.begin(), _source_postponed.end(), false);
    }
};

}  // namespace mcfcg
