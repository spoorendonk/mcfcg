#pragma once

#include "mcfcg/cg/master_base.h"
#include "mcfcg/cg/path_cg.h"
#include "mcfcg/cg/pricer_base.h"
#include "mcfcg/util/limits.h"
#include "mcfcg/util/thread_pool.h"
#include "mcfcg/util/timer.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace mcfcg {

// Slack-cost bump factor shared with the test helpers in
// test/cg_test_util.h.  Any slack still basic with positive primal has
// its cost multiplied by this factor once per CG iteration; the LP
// then pivots the slack out on the next solve once the slack cost
// exceeds the reduced cost of whatever column serves the row.  Replaces
// the legacy fixed BIG_M = 1e8; MasterBase::bump_active_slacks clamps
// each slack cost to an absolute ceiling inside the method.
inline constexpr double SLACK_BUMP_FACTOR = 10.0;

// Generic CG loop parameterized on Master, Pricer, and a dual-extraction callable.
// GetDuals: (const Master&) -> std::vector<double>
template <typename Master, typename Pricer, typename GetDuals>
CGResult solve_cg(const Instance& inst, const CGParams& params, GetDuals get_pricing_duals,
                  uint32_t num_entities) {
    auto pool = make_thread_pool(params.num_threads);

    // Strategy bundle: resolve effective params from params.strategy.
    // `PricerHeavy` caps columns per iter, disables column aging, enables the
    // source pricing filter, and defers pricing in iterations that added
    // lazy capacity rows.
    const bool pricer_heavy = (params.strategy == CGStrategy::PricerHeavy);
    const uint32_t effective_col_limit = pricer_heavy ? num_entities : params.max_cols_per_iter;
    const uint32_t effective_col_age_limit = pricer_heavy ? INF_U32 : params.col_age_limit;
    const bool effective_pricing_filter = pricer_heavy || params.pricing_filter;

    Master master;
    master.init(inst, params.solver_factory ? params.solver_factory() : nullptr, pool.get(),
                params.warm_start);

    Pricer pricer;
    pricer.init(inst, pool.get(), params.pricing_batch_size, params.neg_rc_tol);
    pricer.set_track_arcs(effective_pricing_filter);

    Timer timer;
    CGLogger logger(params.verbosity);
    logger.print_header();

    CGResult result{};
    result.optimal = false;
    bool solved = false;
    // Monotonically non-increasing upper bound.  Set only on iterations
    // whose LP primal is MCF-feasible (no slack basic, no fresh capacity
    // violation).  Never reset to INF once established — a later iter
    // can only tighten it.
    double best_ub = INF;
    // Last successful LP obj — informative fallback for result.objective
    // if the loop exits with no MCF-feasible iter (best_ub stays INF).
    // NOT a valid MCF bound (carries slack penalty / infeasible flow),
    // but keeps result.objective a real number for log / CSV parsers.
    double last_lp_obj = 0.0;
    // Monotonically non-decreasing Lagrangian/Farley lower bound.
    // LB_iter = LP_obj + pricer.min_rc_sum() when pricer.priced_all()
    // (every source was visited in that iter's pricing).  Adding
    // capacity rows or having slacks basic does NOT invalidate the
    // bound — both can only relax the LP, so LP_obj is still an
    // under-estimate of the constrained master that the Farley
    // correction lifts toward OPT.
    double best_lb = -INF;

    auto populate_timing = [&] {
        result.time_lp = timer.elapsed(TimerCat::LP);
        result.time_pricing = timer.elapsed(TimerCat::Pricing);
        result.time_separation = timer.elapsed(TimerCat::Separation);
        result.time_total = timer.elapsed(TimerCat::Total);
    };

    auto set_optimal = [&](double obj, uint32_t iter) {
        result.objective = obj;
        result.iterations = iter + 1;
        result.total_columns = master.num_columns();
        result.optimal = true;
        populate_timing();
        double gap_tol = RELATIVE_FEAS_TOL * std::max(1.0, std::abs(obj));
        logger.print_summary(result.iterations, obj, true, best_lb, gap_tol, result.time_lp,
                             result.time_pricing, result.time_separation, result.time_total);
    };

    timer.start(TimerCat::Total);

    if (params.warm_start) {
        // One-shot initialization: price every source against +inf duals to
        // seed the master with at least one column per source.  +inf
        // disables the dijkstra cutoff (clamped to MAX_BOUND in the
        // pricer) so we explore the full reachable graph.  Replaces the
        // legacy Master::BIG_M coupling — the warm-start cutoff has
        // nothing to do with the slack cost.  This pass intentionally
        // bypasses effective_col_limit (the per-iter cap only applies
        // inside the main loop below).
        timer.start(TimerCat::Pricing);
        std::vector<double> big_duals(num_entities, std::numeric_limits<double>::infinity());
        auto empty_mu = inst.graph.create_arc_map<double>(0.0);
        auto init_cols = pricer.price(big_duals, empty_mu, true);
        if (!init_cols.empty()) {
            master.add_columns(std::move(init_cols));
        }
        pricer.reset_postponed();
        timer.stop(TimerCat::Pricing);
    }

    for (uint32_t iter = 0; iter < params.max_iterations; ++iter) {
        Timer iter_timer;
        iter_timer.start(TimerCat::Total);

        // Log the iteration and record the iter number on
        // result.iterations.  All four exit points share this printout
        // (PricerHeavy cuts-only continue, pricing-exhausted optimal,
        // pricing-exhausted non-optimal continue, end-of-iter with new
        // columns) — only added / purged / num_purged_cuts differ.
        // UB is the running minimum over all MCF-feasible iterations
        // (no slack basic, no new cap row added this iter).  Once set,
        // it stays set — a later infeasible iter can't push it back to
        // inf.  LP_obj is always the LP's own objective so convergence
        // is visible regardless of feasibility.
        auto finish_iter = [&](double obj, uint32_t num_new_caps, uint32_t num_active_slacks,
                               uint32_t added, uint32_t purged, uint32_t num_purged_cuts) {
            iter_timer.stop(TimerCat::Total);
            logger.print_iteration(
                iter + 1, best_ub, best_lb, obj, master.num_lp_cols(), master.num_lp_rows(),
                num_active_slacks, added, purged, num_new_caps, num_purged_cuts,
                iter_timer.elapsed(TimerCat::LP), iter_timer.elapsed(TimerCat::Pricing),
                iter_timer.elapsed(TimerCat::Separation), iter_timer.elapsed(TimerCat::Total));
            result.iterations = iter + 1;
        };

        // --- LP solve (exactly one per iter) ---
        timer.start(TimerCat::LP);
        iter_timer.start(TimerCat::LP);
        auto status = master.solve();
        iter_timer.stop(TimerCat::LP);
        timer.stop(TimerCat::LP);

        if (status != LPStatus::Optimal)
            break;
        solved = true;

        double obj = master.get_obj();
        last_lp_obj = obj;

        // --- All LP reads here, BEFORE any mutation.  Some backends
        // (COPT barrier) drop the ability to return duals once the LP
        // has been mutated (add_rows / set_col_cost / delete_*), even
        // before the next solve.  Capture everything needed, then
        // mutate.
        auto primals = master.get_primals();
        auto pi = get_pricing_duals(master);
        const auto& mu = master.get_capacity_duals();
        master.update_capacity_row_activity(iter);
        master.update_column_ages(primals);
        uint32_t num_active_slacks = master.count_active_slacks(primals);

        // --- Separation (first mutation — add_rows for violated caps) ---
        timer.start(TimerCat::Separation);
        iter_timer.start(TimerCat::Separation);
        auto new_cap_arcs = master.add_violated_capacity_constraints(primals, iter);
        iter_timer.stop(TimerCat::Separation);
        timer.stop(TimerCat::Separation);

        uint32_t num_new_caps = static_cast<uint32_t>(new_cap_arcs.size());

        // Tighten the running UB only when the LP primal is
        // MCF-feasible: no slack basic AND separation found no new
        // violations.  Otherwise obj carries a feasibility penalty
        // and/or reflects flow that exceeds capacity.
        if (num_active_slacks == 0 && num_new_caps == 0) {
            best_ub = std::min(best_ub, obj);
        }

        if (effective_pricing_filter && num_new_caps > 0) {
            pricer.filter_for_new_caps(new_cap_arcs);
        }

        // PricerHeavy: when cuts were added this iter, defer pricing.
        // Just commit the cuts and let the next iter's LP solve digest
        // them with fresh duals.  No bump, no purge — nothing got
        // priced that could have aged out.
        if (pricer_heavy && num_new_caps > 0) {
            finish_iter(obj, num_new_caps, num_active_slacks, 0, 0, 0);
            continue;
        }

        // --- Pricing (duals captured above; stale wrt any cap rows
        // separation just added — the next iter picks them up).
        timer.start(TimerCat::Pricing);
        iter_timer.start(TimerCat::Pricing);

        auto new_cols = pricer.price(pi, mu, false, effective_col_limit);
        if (new_cols.empty()) {
            new_cols = pricer.price(pi, mu, true, effective_col_limit);
        }
        if (!new_cols.empty()) {
            pricer.reset_postponed();
        }

        // Lagrangian/Farley LB.  Valid only when the LP is MCF-
        // feasible (same gate as UB: no slack basic, no fresh capacity
        // violation) AND the pricer visited every source (no
        // postponement skipped anyone, no max_cols early break).
        // A slack-basic LP has duals polluted by the slack penalty
        // and a cut-pending LP has mu missing entries; either gives
        // a nonsensical Lagrangian reconstruction (observed LB orders
        // of magnitude above OPT on EdgeRows path).
        // The rounding-error budget accounts for Dijkstra minimizing
        // scaled-integer edge weights rather than true reduced cost.
        if (pricer.priced_all() && num_active_slacks == 0 && num_new_caps == 0) {
            // Use the LP dual obj (Σ pi·b + Σ cap*mu) rather than the
            // primal obj so the Lagrangian reconstruction is exact at
            // LP optimum even when the backend's primal and dual
            // differ at solver tolerance (barrier-without-crossover).
            double dual_obj = master.compute_dual_obj(pi, mu);
            double lb_iter = dual_obj + pricer.min_rc_sum() - pricer.lb_error_bound();
            best_lb = std::max(best_lb, lb_iter);
        }

        // Early termination on UB-LB relative gap.  best_ub is a valid
        // MCF UB (LP obj on an MCF-feasible iter); best_lb is a valid
        // MCF LB (Lagrangian - scale margin).  When the relative gap
        // drops below the design feasibility tolerance, the current UB
        // is within tolerance of OPT and there is no point iterating.
        // Report the cols the pricer *found* (not added, since we're
        // bailing out) so the log line explains why the LB tightened
        // enough to close the gap.
        if (best_ub < INF) {
            double gap = best_ub - best_lb;
            double gap_tol = RELATIVE_FEAS_TOL * std::max(1.0, std::abs(best_ub));
            // Require gap >= 0 as well: a transient LB > UB would
            // otherwise trip the check trivially.  In practice the
            // MCF-feasibility gate on LB prevents this, but bounding
            // below zero is cheap defense against LP backend numerical
            // surprises.
            if (gap >= 0.0 && gap < gap_tol) {
                iter_timer.stop(TimerCat::Pricing);
                timer.stop(TimerCat::Pricing);
                timer.stop(TimerCat::Total);
                finish_iter(obj, num_new_caps, num_active_slacks,
                            static_cast<uint32_t>(new_cols.size()), 0, 0);
                set_optimal(best_ub, iter);
                return result;
            }
        }

        // Cap columns at the per-iter limit. Keep the best-reduced-cost
        // columns rather than the first-found ones so the master LP makes
        // maximal progress per iter.
        if (new_cols.size() > effective_col_limit) {
            std::partial_sort(
                new_cols.begin(), new_cols.begin() + effective_col_limit, new_cols.end(),
                [](const auto& a, const auto& b) { return a.reduced_cost < b.reduced_cost; });
            new_cols.resize(effective_col_limit);
        }

        iter_timer.stop(TimerCat::Pricing);
        timer.stop(TimerCat::Pricing);

        // Pricing exhausted: optimal iff separation also found nothing
        // and no slack is basic.  Otherwise the next iter's LP solve
        // (with new caps and/or bumped slack costs) will make progress.
        if (new_cols.empty()) {
            if (num_new_caps == 0 && num_active_slacks == 0) {
                timer.stop(TimerCat::Total);
                finish_iter(obj, num_new_caps, 0, 0, 0, 0);
                set_optimal(obj, iter);
                return result;
            }
            if (num_active_slacks > 0) {
                (void)master.bump_active_slacks(primals, SLACK_BUMP_FACTOR);
            }
            pricer.reset_postponed();
            finish_iter(obj, num_new_caps, num_active_slacks, 0, 0, 0);
            continue;
        }

        // --- Mutations: bump + purge + add_columns ---
        (void)master.bump_active_slacks(primals, SLACK_BUMP_FACTOR);
        uint32_t purged = master.purge_aged_columns(effective_col_age_limit);
        uint32_t num_purged =
            master.purge_nonbinding_capacity_rows(iter, params.row_inactivity_threshold);

        uint32_t added = master.add_columns(std::move(new_cols));

        result.total_columns = master.num_columns();
        finish_iter(obj, num_new_caps, num_active_slacks, added, purged, num_purged);
    }

    timer.stop(TimerCat::Total);

    // Report the best UB captured inside the loop.  If the loop exited
    // (e.g. max_iterations) with no MCF-feasible iteration ever seen,
    // best_ub stays INF — fall back to the last LP obj so
    // result.objective is a real number for log / CSV parsers.  Callers
    // should consult result.optimal to know whether the objective is a
    // certified UB or just an informative LP value.
    if (solved) {
        result.objective = best_ub < INF ? best_ub : last_lp_obj;
    }
    populate_timing();
    double gap_tol = RELATIVE_FEAS_TOL * std::max(1.0, std::abs(result.objective));
    logger.print_summary(result.iterations, result.objective, result.optimal, best_lb, gap_tol,
                         result.time_lp, result.time_pricing, result.time_separation,
                         result.time_total);
    return result;
}

}  // namespace mcfcg
