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
    // `PricerLight` caps columns per iter, disables column aging, enables the
    // source pricing filter, and defers pricing in iterations that added
    // lazy capacity rows.
    const bool pricer_light = (params.strategy == CGStrategy::PricerLight);
    const uint32_t effective_col_limit = pricer_light ? num_entities : params.max_cols_per_iter;
    const uint32_t effective_col_age_limit = pricer_light ? INF_U32 : params.col_age_limit;
    const bool effective_pricing_filter = pricer_light || params.pricing_filter;

    Master master;
    master.init(inst, params.solver_factory ? params.solver_factory() : nullptr, pool.get(),
                params.warm_start);
    if (params.verbosity >= Verbosity::Iteration) {
        std::fprintf(stderr, "CG slack mode: %s\n",
                     master.slack_mode() == SlackMode::EdgeRows ? "EdgeRows" : "CommodityRows");
    }

    Pricer pricer;
    pricer.init(inst, PricingMode::AStar, pool.get(), params.pricing_batch_size, params.neg_rc_tol);
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

        // --- LP solve ---
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
        auto primals = master.get_primals();

        // --- Separation ---
        timer.start(TimerCat::Separation);
        iter_timer.start(TimerCat::Separation);
        auto new_cap_arcs = master.add_violated_capacity_constraints(primals, iter);
        iter_timer.stop(TimerCat::Separation);
        timer.stop(TimerCat::Separation);

        uint32_t num_new_caps = static_cast<uint32_t>(new_cap_arcs.size());

        // If cuts were added, re-solve to get correct duals before pricing
        if (!new_cap_arcs.empty()) {
            if (effective_pricing_filter) {
                pricer.filter_for_new_caps(new_cap_arcs);
            }

            timer.start(TimerCat::LP);
            iter_timer.start(TimerCat::LP);
            status = master.solve();
            iter_timer.stop(TimerCat::LP);
            timer.stop(TimerCat::LP);

            if (status != LPStatus::Optimal)
                break;
            obj = master.get_obj();
            primals = master.get_primals();

            // PricerLight protocol — cuts before cols.  This is a mandatory
            // part of the strategy bundle (see CGStrategy::PricerLight in
            // path_cg.h): when new lazy capacity rows were just added, defer
            // pricing entirely so the master reaches a cut-stable state.  The
            // next iteration re-separates with fresh duals; once no more rows
            // are violated, that iteration finally prices.  All post-pricing
            // housekeeping (column aging, purge, add_columns) is also skipped
            // for this deferred iteration since there is nothing to add.
            if (pricer_light) {
                iter_timer.stop(TimerCat::Total);
                logger.print_iteration(
                    iter + 1, obj, -INF, obj, master.num_lp_cols(), master.num_lp_rows(), 0, 0,
                    num_new_caps, 0, iter_timer.elapsed(TimerCat::LP),
                    iter_timer.elapsed(TimerCat::Pricing), iter_timer.elapsed(TimerCat::Separation),
                    iter_timer.elapsed(TimerCat::Total));
                result.iterations = iter + 1;
                continue;
            }
        }

        // --- Pricing (duals are from the latest solve, not affected by purges) ---
        timer.start(TimerCat::Pricing);
        iter_timer.start(TimerCat::Pricing);
        auto pi = get_pricing_duals(master);
        const auto& mu = master.get_capacity_duals();

        auto new_cols = pricer.price(pi, mu, false, effective_col_limit);

        if (new_cols.empty()) {
            new_cols = pricer.price(pi, mu, true, effective_col_limit);
            if (new_cols.empty()) {
                // Pricing exhausted.  Optimal iff no slack is still
                // basic.  Otherwise bump and let the next iteration's
                // solve pivot the slacks out — we do not terminate on
                // saturated bumps here; the LP backend's numerical
                // ceiling is enforced inside bump_active_slacks, and
                // the outer max_iterations guard is the only escape
                // hatch for pathologically slack-dominated instances.
                if (master.has_active_slacks(primals)) {
                    (void)master.bump_active_slacks(primals, SLACK_BUMP_FACTOR);
                    pricer.reset_postponed();
                    iter_timer.stop(TimerCat::Pricing);
                    timer.stop(TimerCat::Pricing);
                    iter_timer.stop(TimerCat::Total);
                    logger.print_iteration(iter + 1, obj, -INF, obj, master.num_lp_cols(),
                                           master.num_lp_rows(), 0, 0, num_new_caps, 0,
                                           iter_timer.elapsed(TimerCat::LP),
                                           iter_timer.elapsed(TimerCat::Pricing),
                                           iter_timer.elapsed(TimerCat::Separation),
                                           iter_timer.elapsed(TimerCat::Total));
                    result.iterations = iter + 1;
                    continue;
                }

                iter_timer.stop(TimerCat::Pricing);
                timer.stop(TimerCat::Pricing);
                timer.stop(TimerCat::Total);

                iter_timer.stop(TimerCat::Total);
                logger.print_iteration(
                    iter + 1, obj, -INF, obj, master.num_lp_cols(), master.num_lp_rows(), 0, 0,
                    num_new_caps, 0, iter_timer.elapsed(TimerCat::LP),
                    iter_timer.elapsed(TimerCat::Pricing), iter_timer.elapsed(TimerCat::Separation),
                    iter_timer.elapsed(TimerCat::Total));

                set_optimal(obj, iter);
                return result;
            }
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

        // --- Purge (after pricing consumed duals, before add_columns so
        // primals and LP column indices stay consistent) ---
        // Activity updates AND the slack-cost bump must happen BEFORE
        // any delete_*, because some backends (COPT, HiGHS) invalidate
        // the cached primal/dual solution on delete and subsequent
        // get_primals / get_duals / get_reduced_costs calls return
        // stale or empty vectors.
        master.update_capacity_row_activity(iter);
        master.update_column_ages(primals);
        // Bump any slacks that were basic in this iteration's LP.
        // Return value intentionally ignored here: at end-of-iter we
        // just want to grow the cost for next iter; whether the cap
        // was hit doesn't change the loop structure.
        (void)master.bump_active_slacks(primals, SLACK_BUMP_FACTOR);
        uint32_t purged = master.purge_aged_columns(effective_col_age_limit);
        uint32_t num_purged =
            master.purge_nonbinding_capacity_rows(iter, params.row_inactivity_threshold);

        uint32_t added = master.add_columns(std::move(new_cols));

        result.iterations = iter + 1;
        result.total_columns = master.num_columns();

        iter_timer.stop(TimerCat::Total);
        logger.print_iteration(
            iter + 1, obj, -INF, obj, master.num_lp_cols(), master.num_lp_rows(), added, purged,
            num_new_caps, num_purged, iter_timer.elapsed(TimerCat::LP),
            iter_timer.elapsed(TimerCat::Pricing), iter_timer.elapsed(TimerCat::Separation),
            iter_timer.elapsed(TimerCat::Total));
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
