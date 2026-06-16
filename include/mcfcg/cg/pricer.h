#pragma once

#include "mcfcg/cg/column.h"
#include "mcfcg/cg/pricer_base.h"

#include <cstdint>
#include <vector>

namespace mcfcg {

class PathPricer : public PricerBase<PathPricer, Column> {
    friend class PricerBase<PathPricer, Column>;

    void process_source(uint32_t s_idx, const Source& src, const std::vector<double>& pi,
                        const static_map<uint32_t, double>& mu, auto& dijk,
                        std::vector<Column>& new_columns, uint32_t /*thread_id*/) {
        bool found_any = false;
        if (_track_arcs)
            _source_arcs[s_idx].clear();

        // Per-source LB accumulators.  All commodities rooted at this
        // source are processed sequentially in this call, so local
        // accumulation is race-free.  Written to the pricer's
        // per-source slot at the end for deterministic final sum.
        double source_rc_error = 0.0;
        double source_lagr_sum = 0.0;

        for (uint32_t k : src.commodity_indices) {
            vertex_t sink = _inst->commodities[k].sink;
            // A* exhausts its heap when no path to sink exists (disconnected
            // source→sink).  Skip that commodity and keep pricing the others.
            // In CommodityRows slack mode the master's demand-row slack
            // absorbs the unmet demand; in EdgeRows mode there is no demand
            // slack so a disconnected commodity will surface as LP
            // infeasibility on the first solve — the CG loop exits with
            // optimal=false.  Callers with potentially disconnected
            // commodities should preprocess the instance (e.g. via
            // mcfcg_clean) before handing it to the solver.
            if (!dijk.visited(sink))
                continue;

            // Extract path and compute true reduced cost
            Column col;
            col.cost = 0.0;
            col.commodity = k;
            double true_rc = -pi[k];
            // π-free reduced-cost path sum sp_k(c−μ), accumulated separately
            // from true_rc so the Lagrangian LB never forms (sp_k − π_k) and
            // re-adds π_k — catastrophic cancellation when a basic slack pins
            // π_k at the bumped slack cost (~1e7).
            double path_rc_sum = 0.0;
            vertex_t v = sink;
            while (dijk.has_pred(v)) {
                uint32_t a = dijk.pred_arc(v);
                col.arcs.push_back(a);
                if (_track_arcs)
                    _source_arcs[s_idx].push_back(a);
                col.cost += _inst->cost[a];
                true_rc += _inst->cost[a] - mu[a];
                path_rc_sum += _inst->cost[a] - mu[a];
                v = _inst->graph.arc_source(a);
            }

            // π-free Lagrangian LB term: d_k · sp_k(c−μ), the demand-weighted
            // reduced-cost shortest-path sum, UNCLAMPED and WITHOUT the −π_k
            // seed (the structural dual cancels Σπ_k d_k in cg_loop's L(μ)).
            // The rounding-error budget is demand-weighted to match its units;
            // LP_FEAS_TOL per arc bounds both integer-scale rounding and the
            // val<=0 clamp in compute_rc (|val| is bounded by LP_FEAS_TOL at
            // numerical noise, the only regime where the clamp fires under the
            // correct mu<=0 sign convention).
            double demand = _inst->commodities[k].demand;
            source_lagr_sum += demand * path_rc_sum;
            source_rc_error += demand * static_cast<double>(col.arcs.size()) * LP_FEAS_TOL;

            if (true_rc >= _neg_rc_tol)
                continue;

            col.reduced_cost = true_rc;
            found_any = true;
            std::reverse(col.arcs.begin(), col.arcs.end());
            new_columns.push_back(std::move(col));
        }

        _source_postponed[s_idx] = found_any ? 0 : 1;
        _source_rc_error[s_idx] = source_rc_error;
        _source_lagr_sum[s_idx] = source_lagr_sum;
    }
};

}  // namespace mcfcg
