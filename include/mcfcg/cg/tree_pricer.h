#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "mcfcg/cg/pricer_base.h"
#include "mcfcg/cg/tree_column.h"

namespace mcfcg {

class TreePricer : public PricerBase<TreePricer, TreeColumn> {
    friend class PricerBase<TreePricer, TreeColumn>;

    // Bounded pricing for the tree formulation.  The tree's reduced cost
    // is −π_s + Σ_k d_k·sp_k, so it stays non-negative once the settled sinks
    // have consumed the convexity dual: with `budget` = π_s·SCALE − Σ_settled
    // d_k·g_k and `rem_demand` = Σ_unsettled d_k, every unsettled sink is at
    // distance ≥ the frontier, so a frontier above budget/rem_demand proves
    // the whole tree prices out.  Dividing by the *sum* of remaining demands
    // (manuscript §3.3) is sharper than the min-demand form the removed
    // Dijkstra-mode code used.
    class PricingBound {
        std::vector<BoundEntry>* _entries;  // sorted by sink ascending
        double _budget;
        double _rem_demand;
        int64_t _bound;

        // The division is the reason `bound()` is cached rather than computed
        // on demand: the driver queries it once per settled vertex, but it only
        // changes when a target settles.
        void recompute() noexcept {
            // Budget consumed by the sinks settled so far: every remaining term
            // is non-negative, so no tree here can price out.  Negated
            // comparison so a NaN budget (never expected, but cheap to survive)
            // cuts rather than runs unbounded.
            if (!(_budget > 0.0)) {
                _bound = -1;  // frontier >= 0 > -1
                return;
            }
            // Budget left but no remaining demand: the unsettled sinks all
            // belong to zero-demand commodities (CommaLab keeps those verbatim;
            // only the TNTP reader drops demand <= 0).  They add 0 to the tree
            // reduced cost, so the budget is NOT proven consumed — cutting here
            // would suppress a strictly improving tree on every iteration,
            // including the final_round retry, and the CG loop would report
            // that as optimal.
            if (!(_rem_demand > 0.0)) {
                _bound = MAX_BOUND;  // price_source_astar never cuts at/above it
                return;
            }
            double raw = _budget / _rem_demand;
            // Floor: the driver breaks on frontier > bound, and for an integer
            // frontier that is exactly frontier·rem_demand > budget.
            _bound = (raw < static_cast<double>(MAX_BOUND)) ? static_cast<int64_t>(raw) : MAX_BOUND;
        }

    public:
        PricingBound(std::vector<BoundEntry>& entries, double budget, double rem_demand)
            : _entries(&entries), _budget(budget), _rem_demand(rem_demand) {
            recompute();
        }

        [[nodiscard]] int64_t bound() const noexcept { return _bound; }

        void on_settle(const auto& /*dijk*/, uint32_t sink, int64_t g_dist) noexcept {
            auto& entries = *_entries;
            auto pos = std::lower_bound(
                entries.begin(), entries.end(), sink,
                [](const BoundEntry& entry, uint32_t key) { return entry.sink < key; });
            // Every commodity sharing this sink is settled by this one pop, so
            // the whole equal-sink run leaves the budget at once.
            for (; pos != entries.end() && pos->sink == sink; ++pos) {
                _budget -= pos->demand * static_cast<double>(g_dist);
                _rem_demand -= pos->demand;
            }
            recompute();
        }
    };

    PricingBound make_bound(uint32_t s_idx, const Source& src, const std::vector<double>& pi_s,
                            std::vector<BoundEntry>& scratch) const {
        scratch.clear();
        scratch.reserve(src.commodity_indices.size());
        double rem_demand = 0.0;
        for (uint32_t k : src.commodity_indices) {
            double demand = _inst->commodities[k].demand;
            scratch.push_back({_inst->commodities[k].sink, 0, demand});
            rem_demand += demand;
        }
        std::ranges::sort(scratch, [](const BoundEntry& lhs, const BoundEntry& rhs) {
            return lhs.sink < rhs.sink;
        });
        // Budget = SCALE·π_s, plus an allowance that makes the cut provably no
        // more aggressive than _neg_rc_tol.  The search compares integer-scaled
        // distances, which overstate SCALE·(true cost) by at most
        // _round_slack_per_demand·SCALE per unit of demand, so cutting on
        //     frontier·rem_demand > SCALE·π_s − Σ_settled d·g + allowance
        // with allowance = SCALE·(D·_round_slack_per_demand + _neg_rc_tol)
        // gives true_tree_rc ≥ _neg_rc_tol, where D is the source's total
        // demand.  Without it the demand weighting scales the rounding gap by D
        // — on a source with D ≈ 1e5 that is ~5e-3, five times NEG_RC_TOL, so
        // the bound could suppress a column the pricer would otherwise accept.
        // The _neg_rc_tol term is negative and usually dominates, making the
        // bound slightly *tighter* than the naive one rather than looser.
        //
        // Computed in raw double rather than through scale_dual so the warm
        // start's +inf survives: scale_dual saturates it at MAX_BOUND, which
        // recompute() then divides by rem_demand back into the range of real
        // frontier values, letting the seeding pass cut and leave a source with
        // no column at all (fatal in EdgeRows slack mode, whose feasibility
        // rests on warm start seeding every structural row).  NaN still lands
        // in the !(_budget > 0) branch and cuts, as documented there.
        double budget = (pi_s[s_idx] + rem_demand * _round_slack_per_demand + _neg_rc_tol) * SCALE;
        return {scratch, budget, rem_demand};
    }

    void process_source(uint32_t s_idx, const Source& src, const std::vector<double>& pi_s,
                        const static_map<uint32_t, double>& mu, auto& dijk,
                        std::vector<TreeColumn>& new_columns, uint32_t /*thread_id*/,
                        std::optional<int64_t> bound_f) {
        TreeColumn col;
        col.source_idx = s_idx;
        col.cost = 0.0;
        double tree_rc = -pi_s[s_idx];
        double source_rc_error = 0.0;
        // π-free Lagrangian path sum Σ_k d_k·sp_k(c−μ) for this source,
        // accumulated WITHOUT the −π_s seed so the convexity dual cancels
        // in cg_loop's L(μ) (tree convexity RHS=1 ⇒ the per-source weight
        // collapses).
        double source_lagr_sum = 0.0;

        const bool record_arcs = should_record_arcs(bound_f);
        if (record_arcs) {
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
            // source→sink).  Skip the unreachable commodity and keep building a
            // partial tree over the remaining reachable sinks, which still
            // contributes xi=1 to its source's convexity row and so is a valid
            // LP candidate.  Note this is genuinely lossy for the tree
            // formulation: the master's structural rows count trees, not
            // demand, so nothing notices that this column serves fewer
            // commodities than a complete tree would, and the LP will happily
            // prefer it.  The objective is then an underestimate — acceptable
            // only because a disconnected commodity makes the MCF infeasible
            // anyway.  Preprocess disconnected instances via mcfcg_clean before
            // solving, as the class docs ask.
            //
            // Bounded pricing also leaves sinks unsettled, but those are
            // merely proven unattractive, not unreachable — salvage their
            // Lagrangian term from the frontier the search stopped at.  That
            // only feeds the lower bound: a cut source emits no column at all
            // (see below), precisely so the bound never manufactures a partial
            // tree.
            if (!dijk.visited(sink)) {
                source_lagr_sum += salvage_lagr_term(bound_f, _inst->commodities[k].demand);
                continue;
            }
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
            source_lagr_sum += d * path_rc;
            col.cost += d * path_orig_cost;
            // Tree column's rc is the demand-weighted sum of its per-
            // commodity path rcs, so the rounding-error budget is
            // demand-weighted too.  LP_FEAS_TOL per arc bounds both
            // integer-scale rounding and the val<=0 clamp in
            // compute_rc (see pricer.h for the rationale).
            source_rc_error += d * static_cast<double>(path_arcs) * LP_FEAS_TOL;
        }

        if (record_arcs) {
            _source_arcs[s_idx].reserve(arc_flow_map.size());
            for (auto& [arc, flow] : arc_flow_map) {
                _source_arcs[s_idx].push_back(arc);
            }
        }

        // Per-source LB slots (deterministic final summation).  source_lagr_sum
        // is the π-free Σ_k d_k·sp_k(c−μ) for this source; source_rc_error its
        // rounding budget.
        _source_rc_error[s_idx] = source_rc_error;
        _source_lagr_sum[s_idx] = source_lagr_sum;

        // A cut search settled only part of the source's sinks, so tree_rc
        // above is missing those commodities' (non-negative) contributions and
        // reads more negative than the true tree reduced cost.  The bound
        // already proved the true value is non-negative, so the source is
        // simply postponed.  Emitting the partial tree instead would be a
        // correctness bug, not just a weak column: nothing in the tree master
        // constrains a column to serve every commodity of its source — the
        // convexity row counts trees, not demand — so under-served demand
        // would go unnoticed and drag the objective below the true optimum.
        if (bound_f.has_value()) {
            _source_postponed[s_idx] = 1;
            return;
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
