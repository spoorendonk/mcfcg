#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <limits>
#include <vector>

#include "mcfcg/cg/master_base.h"
#include "mcfcg/cg/path_cg.h"
#include "mcfcg/cg/pricer_base.h"
#include "mcfcg/util/limits.h"
#include "mcfcg/util/thread_pool.h"
#include "mcfcg/util/timer.h"

namespace mcfcg {

// Slack-cost bump factor shared with the test helpers in
// test/cg_test_util.h.  Any slack still basic with positive primal has
// its cost multiplied by this factor once per CG iteration; the LP
// then pivots the slack out on the next solve once the slack cost
// exceeds the reduced cost of whatever column serves the row.
// MasterBase::bump_active_slacks clamps each slack cost to the
// per-instance absolute ceiling set in MasterBase::init.
inline constexpr double SLACK_BUMP_FACTOR = 10.0;

// Generic CG loop parameterized on Master, Pricer, and a dual-extraction callable.
// GetDuals: (const Master&) -> std::vector<double>
template <typename Master, typename Pricer, typename GetDuals>
CGResult solve_cg(const Instance& inst, const CGParams& params, GetDuals get_pricing_duals,
                  uint32_t num_entities) {
    auto pool = make_thread_pool(params.num_threads);

    // Resolve the CGStrategy bundle into effective_* locals; see the
    // CGStrategy enum doc in path_cg.h for the per-preset contents.
    const bool pricer_heavy = (params.strategy == CGStrategy::PricerHeavy);
    const uint32_t effective_col_limit = pricer_heavy ? num_entities : params.max_cols_per_iter;
    const uint32_t effective_col_age_limit = pricer_heavy ? INF_U32 : params.col_age_limit;
    const bool effective_pricing_filter = pricer_heavy || params.pricing_filter;
    const uint32_t pool_threads = pool ? pool->num_threads() : 1U;
    const uint32_t n_sources = static_cast<uint32_t>(inst.sources.size());
    const uint32_t effective_batch_size = compute_partial_pricing_batch_size(
        params.pricing_batch_size, pricer_heavy, pool_threads, n_sources);

    Master master;
    master.init(inst, params.solver_factory ? params.solver_factory() : nullptr, pool.get(),
                params.warm_start);

    Pricer pricer;
    pricer.init(inst, pool.get(), effective_batch_size, params.neg_rc_tol, params.pricing_cutoff);
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
    // Monotonically non-decreasing π-free capacity-relaxation Lagrangian LB.
    // LB_iter = cap_dual_term + pricer.lagrangian_path_sum() −
    // pricer.lb_error_bound(), taken when pricer.priced_all() (every source
    // visited this iter).  Valid for any μ≤0 by weak Lagrangian duality —
    // independent of slack/feasibility state — so slacks basic or fresh
    // capacity rows do NOT invalidate it.  See the LB block below.
    double best_lb = -INF;
    // Stall-recovery state.  When pricing is exhausted (no improving column)
    // but slacks are still basic, the next LP solve is requested as a *certify*
    // solve (HiGHS runs crossover; other barriers no-op) to rule out a basic
    // slack that is merely an artifact of a non-vertex interior-point solution.
    // tried_certify latches the attempt so we crossover at most once per stable
    // column set — it is reset whenever new columns are added (a fresh column
    // set is a new situation worth re-certifying), bounding the extra solves.
    bool certify_next = false;
    bool tried_certify = false;

    // Accumulate the cutoff fire rate across every pricing sweep, including the
    // warm start and the final_round retries, so the reported rate covers all
    // the pricing work the run actually did.
    auto tally_pricing = [&] {
        result.cutoff_sources += pricer.last_cutoff_count();
        result.priced_sources += pricer.last_priced_count();
    };

    auto populate_timing = [&] {
        result.time_lp = timer.elapsed(TimerCat::LP);
        result.time_pricing = timer.elapsed(TimerCat::Pricing);
        result.time_separation = timer.elapsed(TimerCat::Separation);
        result.time_total = timer.elapsed(TimerCat::Total);
    };

    auto set_optimal = [&](double obj, uint32_t iter) {
        result.objective = obj;
        result.lower_bound = best_lb;
        result.iterations = iter + 1;
        result.total_columns = master.num_columns();
        result.optimal = true;
        populate_timing();
        double gap_tol = RELATIVE_FEAS_TOL * std::max(1.0, std::abs(obj));
        logger.print_summary(result.iterations, obj, true, best_lb, gap_tol, result.time_lp,
                             result.time_pricing, result.time_separation, result.time_total);
    };

    timer.start(TimerCat::Total);
    // Wall-clock anchor for the optional time budget (params.time_limit_seconds).
    // The Timer only exposes accumulated stopped time, so the live check needs
    // its own monotonic start point.
    const auto wall_start = std::chrono::steady_clock::now();

    if (params.warm_start) {
        // One-shot initialization: price every source against +inf duals to
        // seed the master with at least one column per source.  Every column
        // prices out against +inf, so the pass explores the full reachable
        // graph — and PricerBase::scale_dual saturates +inf at MAX_BOUND,
        // leaving the optional dual pricing cutoff inert here.  Replaces the
        // legacy Master::BIG_M coupling — the seeding pass has
        // nothing to do with the slack cost.  This pass intentionally
        // bypasses effective_col_limit (the per-iter cap only applies
        // inside the main loop below).
        timer.start(TimerCat::Pricing);
        std::vector<double> big_duals(num_entities, std::numeric_limits<double>::infinity());
        auto empty_mu = inst.graph.create_arc_map<double>(0.0);
        auto init_cols = pricer.price(big_duals, empty_mu, true);
        tally_pricing();
        if (!init_cols.empty()) {
            master.add_columns(std::move(init_cols));
        }
        pricer.reset_postponed();
        timer.stop(TimerCat::Pricing);
    }

    for (uint32_t iter = 0; iter < params.max_iterations; ++iter) {
        // Wall-clock budget check at the iteration boundary.  Breaking here
        // (rather than mid-iteration) leaves the master/pricer in a consistent
        // state; the shared exit path below stops the Total timer and emits the
        // best UB/LB as a non-optimal result.
        if (params.time_limit_seconds > 0.0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count() >
                params.time_limit_seconds) {
            break;
        }

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
                               uint32_t added, bool added_not_committed, uint32_t purged,
                               uint32_t num_purged_cuts) {
            iter_timer.stop(TimerCat::Total);
            logger.print_iteration(
                iter + 1, best_ub, best_lb, obj, master.num_lp_cols(), master.num_lp_rows(),
                num_active_slacks, added, added_not_committed, purged, num_new_caps,
                num_purged_cuts, iter_timer.elapsed(TimerCat::LP),
                iter_timer.elapsed(TimerCat::Pricing), iter_timer.elapsed(TimerCat::Separation),
                iter_timer.elapsed(TimerCat::Total));
            result.iterations = iter + 1;
        };

        // --- LP solve (exactly one per iter) ---
        timer.start(TimerCat::LP);
        iter_timer.start(TimerCat::LP);
        bool did_certify = certify_next;
        auto status = master.solve(certify_next);
        certify_next = false;
        // An interior-point solve (HiGHS HiPO, crossover off) can spuriously
        // report infeasible on a master that is feasible at a vertex — e.g.
        // after separation adds capacity rows in EdgeRows mode. Retry once with
        // a certified (crossover) solve before giving up, so a non-vertex
        // numerical artifact does not abort an otherwise-feasible CG run. Only
        // when certify actually runs crossover on this backend — otherwise the
        // retry would just repeat the identical solve (cuOpt has no crossover).
        if (status != LPStatus::Optimal && !did_certify && master.certify_runs_crossover()) {
            status = master.solve(true);
            did_certify = true;
        }
        iter_timer.stop(TimerCat::LP);
        timer.stop(TimerCat::LP);

        if (status != LPStatus::Optimal) break;
        solved = true;

        double obj = master.get_obj();

        // --- All LP reads here, BEFORE any mutation.  Some backends
        // (COPT barrier) drop the ability to return duals once the LP
        // has been mutated (add_rows / set_col_cost / delete_*), even
        // before the next solve.  Capture everything needed, then
        // mutate.
        auto primals = master.get_primals();
        auto pi = get_pricing_duals(master);
        const auto& mu = master.get_capacity_duals();
        // Snapshot the capacity-dual term Σ_a cap_a·μ_a NOW, against the row
        // set the LP was solved with — before separation mutates
        // _cap_row_to_arc below.  This is the master-side half of the π-free
        // Lagrangian LB; capturing it pre-separation is what lets the LB fire
        // even on iters that add cuts (the new rows carry μ=0, contributing 0,
        // which is exactly a valid relaxation of the not-yet-dualized rows).
        double cap_dual_term = master.compute_capacity_dual_term(mu);
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
            finish_iter(obj, num_new_caps, num_active_slacks, 0, false, 0, 0);
            continue;
        }

        // --- Pricing (duals captured above; stale wrt any cap rows
        // separation just added — the next iter picks them up).
        timer.start(TimerCat::Pricing);
        iter_timer.start(TimerCat::Pricing);

        auto new_cols = pricer.price(pi, mu, false, effective_col_limit);
        tally_pricing();
        if (new_cols.empty()) {
            new_cols = pricer.price(pi, mu, true, effective_col_limit);
            tally_pricing();
        }
        if (!new_cols.empty()) {
            // Keep _last_source_idx so partial pricing under PricerHeavy
            // resumes from its parked cursor next iter; reset_postponed()
            // would wipe the cursor and defeat partial pricing.
            pricer.clear_postponed();
        }

        // π-free capacity-relaxation Lagrangian LB:
        //   L(μ) = Σ_a cap_a·μ_a + Σ_k d_k·sp_k(c−μ) − rounding_margin
        // valid for ANY μ≤0 by weak Lagrangian duality — independent of π
        // and of slack/feasibility state.  The pricer accumulates sp_k(c−μ)
        // WITHOUT subtracting π_k (lagrangian_path_sum), so the structural
        // duals cancel analytically rather than numerically: this is what
        // makes the bound valid while a slack is basic (which would otherwise
        // pin π_k at the bumped slack cost and inflate the old clamped
        // reconstruction "orders of magnitude above OPT").  cap_dual_term was
        // snapshotted pre-separation against the solved row set.  Gated only
        // on priced_all() — the sum must cover every commodity; the slack and
        // new-cap gates are no longer needed.
        // Precondition: arc costs are non-negative, so c−μ ≥ 0 (μ≤0) and every
        // sp_k(c−μ) ≥ 0 — this is what makes dropping an unreachable
        // commodity's term only LOWER the bound (still ≤ OPT) rather than
        // risk raising it.  If negative arc costs are ever introduced,
        // revisit this and the val<=0 clamp in the pricer's compute_rc.
        if (pricer.priced_all()) {
            double lb_iter = cap_dual_term + pricer.lagrangian_path_sum() - pricer.lb_error_bound();
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
                // Flag the "+col" count as uncommitted (prefixed '*')
                // since we're returning without calling add_columns.
                finish_iter(obj, num_new_caps, num_active_slacks,
                            static_cast<uint32_t>(new_cols.size()), true, 0, 0);
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
                finish_iter(obj, num_new_caps, 0, 0, false, 0, 0);
                // Report the incumbent, not this iter's LP objective — same as
                // the gap-test exit above.  The UB-tightening guard upstream is
                // exactly this branch's condition, so best_ub = min(best_ub, obj)
                // has already run for this iter and best_ub <= obj is invariant.
                // A barrier that lands above an earlier solve on a strictly
                // larger column set (observed with cuOpt/MOSEK) would otherwise
                // make CG report a value worse than a solution it already had.
                assert(best_ub <= obj && "UB-update guard must cover this branch");
                set_optimal(best_ub, iter);
                return result;
            }
            // Pricing is exhausted but we are NOT optimal — slacks are still
            // basic and/or separation just added cuts.  On a pure interior-point
            // backend (HiGHS HiPO) this can be a non-vertex artifact: the
            // central duals fail to expose an improving column, or basic slacks
            // never collapse to 0.  Ask for one certified (vertex) solve next
            // iter — crossover rounds the interior point to a vertex, yielding
            // discriminating duals (so pricing can resume) and exact slacks (so
            // a slack-free UB can be recorded).  Latched so we crossover at most
            // once per stable column set; reset on add_columns below. Skipped on
            // backends where certify is a no-op (no crossover, e.g. cuOpt).
            if (!tried_certify && master.certify_runs_crossover()) {
                tried_certify = true;
                certify_next = true;
            }
            if (num_active_slacks > 0) {
                (void)master.bump_active_slacks(primals, SLACK_BUMP_FACTOR);
            }
            pricer.reset_postponed();
            finish_iter(obj, num_new_caps, num_active_slacks, 0, false, 0, 0);
            continue;
        }

        // --- Mutations: bump + purge + add_columns ---
        (void)master.bump_active_slacks(primals, SLACK_BUMP_FACTOR);
        uint32_t purged = master.purge_aged_columns(effective_col_age_limit);
        uint32_t num_purged =
            master.purge_nonbinding_capacity_rows(iter, params.row_inactivity_threshold);

        uint32_t added = master.add_columns(std::move(new_cols));
        // Column set changed: a future stall is a new situation, so re-arm the
        // certify attempt latched in the pricing-exhausted branch above.
        tried_certify = false;

        result.total_columns = master.num_columns();
        finish_iter(obj, num_new_caps, num_active_slacks, added, false, purged, num_purged);
    }

    timer.stop(TimerCat::Total);

    // Report the best UB captured inside the loop.  best_ub is a certified
    // MCF-feasible UB (set only on a slack-free iteration).  If the loop exited
    // non-optimally (max_iterations / time limit / LP solve failure) with no
    // MCF-feasible iteration ever seen, best_ub stays INF.  In that case the
    // last LP objective is a feasibility-penalty value (slacks basic), NOT a
    // routing cost — it can be orders of magnitude wrong (#35) — so do not
    // report it.  Fall back to the valid Lagrangian lower bound instead, which
    // is the only trustworthy quantity here.  result.optimal stays false so
    // callers know the objective is not a certified optimum.
    if (solved) {
        result.objective = best_ub < INF ? best_ub : best_lb;
        result.lower_bound = best_lb;
    }
    populate_timing();
    double gap_tol = RELATIVE_FEAS_TOL * std::max(1.0, std::abs(result.objective));
    // Report the true bounds: best_ub (INF when no slack-free incumbent was
    // ever found) and best_lb.  result.objective falls back to best_lb for the
    // CSV (#35), but the human summary must not present that LB as a zero-gap
    // UB — passing best_ub here makes the no-incumbent case print UB=inf.
    logger.print_summary(result.iterations, best_ub, result.optimal, best_lb, gap_tol,
                         result.time_lp, result.time_pricing, result.time_separation,
                         result.time_total);
    return result;
}

}  // namespace mcfcg
