#pragma once

#include "mcfcg/cg/column.h"
#include "mcfcg/cg/pricer_base.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace mcfcg {

class PathPricer : public PricerBase<PathPricer, Column> {
    friend class PricerBase<PathPricer, Column>;

    void process_source(uint32_t s_idx, const Source& src, const std::vector<double>& pi,
                        const static_map<uint32_t, double>& mu, auto& dijk,
                        std::vector<Column>& new_columns) {
        bool found_any = false;
        if (_track_arcs)
            _source_arcs[s_idx].clear();

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
                if (_track_arcs)
                    _source_arcs[s_idx].push_back(a);
                col.cost += _inst->cost[a];
                true_rc += _inst->cost[a] - mu[a];
                v = _inst->graph.arc_source(a);
            }

            if (true_rc >= _neg_rc_tol)
                continue;

            col.reduced_cost = true_rc;
            found_any = true;
            std::reverse(col.arcs.begin(), col.arcs.end());
            new_columns.push_back(std::move(col));
        }

        _source_postponed[s_idx] = !found_any;
    }

    void price_source_dijkstra(uint32_t s_idx, const Source& src, vertex_t source_v,
                               const std::vector<double>& pi,
                               const static_map<uint32_t, double>& mu,
                               std::vector<Column>& new_columns, uint32_t thread_id) {
        constexpr auto MAX_BOUND = shortest_path_semiring<int64_t>::infty / 2;

        // Per-target cutoffs: cutoff[sink] = max pi[k]*SCALE over
        // commodities k with that sink. Dijkstra stops when min key
        // exceeds max_cutoff.
        std::unordered_map<vertex_t, int64_t> cutoff;
        int64_t max_cutoff = 0;
        for (uint32_t k : src.commodity_indices) {
            vertex_t sink = _inst->commodities[k].sink;
            double raw = pi[k] * SCALE;
            auto scaled = raw >= static_cast<double>(MAX_BOUND)
                              ? MAX_BOUND
                              : static_cast<int64_t>(std::ceil(raw));
            auto it = cutoff.find(sink);
            if (it == cutoff.end()) {
                cutoff[sink] = scaled;
            } else {
                it->second = std::max(it->second, scaled);
            }
            max_cutoff = std::max(max_cutoff, scaled);
        }

        uint32_t remaining = static_cast<uint32_t>(cutoff.size());

        auto& ws = _workspaces[thread_id];
        ws.reset();
        dijkstra<dijkstra_store_paths> dijk(_inst->graph, _rc, ws);
        dijk.add_source(source_v);

        while (!dijk.finished() && remaining > 0) {
            auto [u, u_dist] = dijk.current();
            if (u_dist > max_cutoff)
                break;
            dijk.advance();

            auto cit = cutoff.find(u);
            if (cit != cutoff.end()) {
                --remaining;
                cutoff.erase(cit);
                if (remaining > 0) {
                    max_cutoff = 0;
                    for (auto& [_, c] : cutoff) {
                        max_cutoff = std::max(max_cutoff, c);
                    }
                }
            }
        }

        process_source(s_idx, src, pi, mu, dijk, new_columns);
    }
};

}  // namespace mcfcg
