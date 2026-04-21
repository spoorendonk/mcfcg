#pragma once

#include "mcfcg/cg/master_base.h"
#include "mcfcg/cg/path_cg.h"
#include "mcfcg/cg/pricer_base.h"
#include "mcfcg/util/limits.h"
#include "mcfcg/util/thread_pool.h"
#include "mcfcg/util/timer.h"

#include <algorithm>
#include <cstdio>
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
    double last_obj = 0.0;  // last successful LP obj; survives a hit on max_iterations

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
        logger.print_summary(result.iterations, obj, true, result.time_lp, result.time_pricing,
                             result.time_separation, result.time_total);
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
        // result.iterations.  All exit points in this loop share this
        // printout — only added / purged / num_purged_cuts differ.
        auto finish_iter = [&](double obj, uint32_t num_new_caps, uint32_t added, uint32_t purged,
                               uint32_t num_purged_cuts) {
            iter_timer.stop(TimerCat::Total);
            logger.print_iteration(
                iter + 1, obj, -INF, obj, master.num_lp_cols(), master.num_lp_rows(), added, purged,
                num_new_caps, num_purged_cuts, iter_timer.elapsed(TimerCat::LP),
                iter_timer.elapsed(TimerCat::Pricing), iter_timer.elapsed(TimerCat::Separation),
                iter_timer.elapsed(TimerCat::Total));
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
        last_obj = obj;

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

        // --- Separation (first mutation — add_rows for violated caps) ---
        timer.start(TimerCat::Separation);
        iter_timer.start(TimerCat::Separation);
        auto new_cap_arcs = master.add_violated_capacity_constraints(primals, iter);
        iter_timer.stop(TimerCat::Separation);
        timer.stop(TimerCat::Separation);

        uint32_t num_new_caps = static_cast<uint32_t>(new_cap_arcs.size());

        if (effective_pricing_filter && num_new_caps > 0) {
            pricer.filter_for_new_caps(new_cap_arcs);
        }

        // PricerHeavy: when cuts were added this iter, defer pricing.
        // Just commit the cuts and let the next iter's LP solve digest
        // them with fresh duals.  No bump, no purge — nothing got
        // priced that could have aged out.
        if (pricer_heavy && num_new_caps > 0) {
            finish_iter(obj, num_new_caps, 0, 0, 0);
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
            if (num_new_caps == 0 && !master.has_active_slacks(primals)) {
                timer.stop(TimerCat::Total);
                finish_iter(obj, 0, 0, 0, 0);
                set_optimal(obj, iter);
                return result;
            }
            if (master.has_active_slacks(primals)) {
                (void)master.bump_active_slacks(primals, SLACK_BUMP_FACTOR);
            }
            pricer.reset_postponed();
            finish_iter(obj, num_new_caps, 0, 0, 0);
            continue;
        }

        // --- Mutations: bump + purge + add_columns ---
        (void)master.bump_active_slacks(primals, SLACK_BUMP_FACTOR);
        uint32_t purged = master.purge_aged_columns(effective_col_age_limit);
        uint32_t num_purged =
            master.purge_nonbinding_capacity_rows(iter, params.row_inactivity_threshold);

        uint32_t added = master.add_columns(std::move(new_cols));

        result.total_columns = master.num_columns();
        finish_iter(obj, num_new_caps, added, purged, num_purged);
    }

    timer.stop(TimerCat::Total);

    // Use the last successful obj captured inside the loop, not
    // master.get_obj() — by the time we reach here, the LP state has
    // been mutated by add_columns/purge/bump without a re-solve, so
    // get_obj() returns 0 / stale on HiGHS and COPT.
    if (solved)
        result.objective = last_obj;
    populate_timing();
    logger.print_summary(result.iterations, result.objective, result.optimal, result.time_lp,
                         result.time_pricing, result.time_separation, result.time_total);
    return result;
}

}  // namespace mcfcg
