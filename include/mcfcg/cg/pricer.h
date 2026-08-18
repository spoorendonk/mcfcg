#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "mcfcg/cg/column.h"
#include "mcfcg/cg/pricer_base.h"

namespace mcfcg {

class PathPricer : public PricerBase<PathPricer, Column> {
    friend class PricerBase<PathPricer, Column>;

    // Dual pricing cutoff for the path formulation.  A commodity k is
    // attractive only if its reduced-cost distance beats its demand dual, so
    // once the A* frontier passes max{π_k : k not yet settled} no unsettled
    // commodity of this source can price out.  The bound tightens as sinks
    // settle, which is what makes it worth more than the static max_k π_k.
    //
    // Entries are sorted by π descending and `head` walks past sinks the
    // search has already settled — asking `dijk` which are settled avoids a
    // separate settled-set.  Duplicate sinks (two commodities of one source
    // sharing a sink) need no merging: the max over unsettled entries is the
    // max over unsettled sinks either way.
    class Cutoff {
        std::vector<CutoffEntry>* _entries;
        size_t _head = 0;
        int64_t _bound;

    public:
        explicit Cutoff(std::vector<CutoffEntry>& entries) : _entries(&entries) {
            _bound = entries.empty() ? -MAX_BOUND : entries.front().key;
        }

        int64_t bound() const noexcept { return _bound; }

        void on_settle(const auto& dijk, uint32_t /*sink*/, int64_t /*g_dist*/) noexcept {
            auto& entries = *_entries;
            while (_head < entries.size() && dijk.visited(entries[_head].sink)) ++_head;
            // Only reachable once every target is settled, when the driver
            // loop is about to stop anyway; cut rather than run unbounded.
            _bound = _head < entries.size() ? entries[_head].key : -MAX_BOUND;
        }
    };

    Cutoff make_cutoff(uint32_t /*s_idx*/, const Source& src, const std::vector<double>& pi,
                       std::vector<CutoffEntry>& scratch) const {
        scratch.clear();
        scratch.reserve(src.commodity_indices.size());
        // Key = SCALE·π_k plus an allowance that makes the cut provably no more
        // aggressive than _neg_rc_tol.  A* compares integer-scaled distances,
        // which overstate SCALE·(true cost) by at most
        // _round_slack_per_demand·SCALE, so cutting on
        //     frontier > SCALE·(π_k + _round_slack_per_demand + _neg_rc_tol)
        // gives true_rc_k ≥ _neg_rc_tol.  Folding it in before scale_dual keeps
        // the warm start's +inf saturating to MAX_BOUND.  The _neg_rc_tol term
        // is negative and dominates at any realistic V, so this is in practice
        // a slightly tighter bound than the naive SCALE·π_k, not a looser one.
        double allowance = _round_slack_per_demand + _neg_rc_tol;
        for (uint32_t k : src.commodity_indices)
            scratch.push_back({_inst->commodities[k].sink, scale_dual(pi[k] + allowance), 0.0});
        std::sort(scratch.begin(), scratch.end(),
                  [](const CutoffEntry& lhs, const CutoffEntry& rhs) { return lhs.key > rhs.key; });
        return Cutoff(scratch);
    }

    void process_source(uint32_t s_idx, const Source& src, const std::vector<double>& pi,
                        const static_map<uint32_t, double>& mu, auto& dijk,
                        std::vector<Column>& new_columns, uint32_t /*thread_id*/,
                        std::optional<int64_t> cutoff_f) {
        bool found_any = false;
        const bool record_arcs = should_record_arcs(cutoff_f);
        if (record_arcs) _source_arcs[s_idx].clear();

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
            //
            // The dual cutoff leaves sinks unsettled too, and those are not
            // unreachable — the frontier only proved their reduced cost is
            // non-negative.  Salvage the Lagrangian term the truncated search
            // never computed from the frontier value it stopped at, so an
            // aggressive cutoff does not hollow out the lower bound.
            if (!dijk.visited(sink)) {
                source_lagr_sum += salvage_lagr_term(cutoff_f, _inst->commodities[k].demand);
                continue;
            }

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
                if (record_arcs) _source_arcs[s_idx].push_back(a);
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

            if (true_rc >= _neg_rc_tol) continue;

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
