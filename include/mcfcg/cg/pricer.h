#pragma once

#include "mcfcg/cg/column.h"
#include "mcfcg/cg/pricer_base.h"

#include <cassert>
#include <cstdint>
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
            // A* runs until every target sink is settled; an unvisited
            // sink would mean the instance graph is disconnected between
            // source and sink, which is a caller-level error.
            assert(dijk.visited(sink));

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

        _source_postponed[s_idx] = found_any ? 0 : 1;
    }
};

}  // namespace mcfcg
