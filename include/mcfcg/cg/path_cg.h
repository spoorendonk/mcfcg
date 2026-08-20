#pragma once

#include <algorithm>
#include <cstdint>
#include <functional>

#include "mcfcg/instance.h"
#include "mcfcg/lp/lp_solver.h"
#include "mcfcg/util/limits.h"
#include "mcfcg/util/logger.h"
#include "mcfcg/util/tolerances.h"

namespace mcfcg {

struct CGResult {
    // -INF = no objective established (e.g. the first LP solve failed before any
    // bound was recorded). Matches `lower_bound`'s "no info" sentinel; combine
    // with `optimal` to know whether it is a certified optimum.
    double objective = -INF;
    // Best Lagrangian/Farley lower bound seen during solve.  -INF means
    // LB tracking never fired (no MCF-feasible iter where pricer
    // visited every source).  Combine with `objective` to report a
    // gap.  A PricerHeavy regression that broke priced_all would leave
    // this at -INF even on convergent runs.
    double lower_bound = -INF;
    // CG-loop master iterations.  Counts every pass through the loop body,
    // including iterations where pricing was deferred under
    // CGStrategy::PricerHeavy (iterations that only added lazy capacity
    // rows; the next iteration's single solve picks them up).  Under
    // CGStrategy::PricerLight every counted iteration also priced.
    uint32_t iterations = 0;
    uint32_t total_columns = 0;
    bool optimal = false;
    // Bounded-pricing instrumentation, summed over every pricing sweep of the
    // run: how many source prices the bound stopped short, out of how many it
    // ran.  Both stay 0 / N when CGParams::bounded_pricing is off.  A null
    // speed-up is only interpretable next to the fire rate, so the CLI reports
    // it rather than leaving "did it ever trigger?" unanswered.
    uint64_t bounded_sources = 0;
    uint64_t priced_sources = 0;
    double time_lp = 0;
    double time_pricing = 0;
    double time_separation = 0;
    double time_total = 0;
};

// Batch size for the pricer's source-level dispatcher.  Partial pricing
// only engages when the instance has more sources than fit in one
// thread-pool batch; below that threshold the col-cap early break has
// no mid-sweep to park in, and we return 0 (single big batch) — the
// simple default.  For larger instances under PricerHeavy, ~4 batches
// per sweep (n_sources/4), floored at pool_threads to keep every batch
// able to saturate the pool.  An explicit caller setting
// (explicit_batch_size > 0) always wins.
inline uint32_t compute_partial_pricing_batch_size(uint32_t explicit_batch_size, bool pricer_heavy,
                                                   uint32_t pool_threads,
                                                   uint32_t n_sources) noexcept {
    if (explicit_batch_size > 0) {
        return explicit_batch_size;
    }
    if (!pricer_heavy || n_sources <= pool_threads) {
        return 0U;
    }
    return std::max(pool_threads, n_sources / 4U);
}

using SolverFactory = std::function<std::unique_ptr<LPSolver>()>;

// High-level CG strategy preset.  Bundles several lower-level knobs that
// together express a stance on how expensive the pricer is relative to the
// master LP.  Lower-level fields in CGParams remain as overrides, except
// where the bundle is documented to supersede them.
//
// CLI spellings (see src/main.cpp --strategy flag): "pricer-light",
// "pricer-heavy".
enum class CGStrategy : uint8_t {
    // Default: the pricer is cheap relative to LP solves, so push lots of
    // cols and rows at the master each iteration and keep master iterations
    // to a minimum.  Large col cap, column aging on, cuts and columns added
    // in the same iteration.
    PricerLight,
    // Pricer is expensive; throttle master iterations to as few pricing
    // sweeps as possible.  Bundle (rationale in CLAUDE.md):
    //  * cap columns per iter at num_entities
    //  * disable column aging (overrides CGParams::col_age_limit)
    //  * force the source pricing filter on
    //  * defer pricing on cut-adding iterations
    //  * partial pricing via compute_partial_pricing_batch_size (below)
    //    — engages only when n_sources > pool_threads; overridden when
    //    CGParams::pricing_batch_size > 0
    PricerHeavy,
};

struct CGParams {
    uint32_t max_iterations = 10000;
    uint32_t max_cols_per_iter = 50000;
    bool warm_start = true;
    Verbosity verbosity = Verbosity::Silent;
    // Strategy preset; see CGStrategy enum above for the bundled behaviors.
    CGStrategy strategy = CGStrategy::PricerLight;
    bool pricing_filter = false;
    // Bounded single-source pricing (manuscript §3.3): stop each source's A*
    // as soon as the frontier proves no negative-reduced-cost column remains
    // for it, instead of running until every sink of the source is settled.
    //
    // Off by default.  The full argument, with the logs behind every number, is
    // results/ablation/README.md; read it before changing this default.  In
    // short, from the two committed rounds:
    //
    //  * It always saves pricing time — t_PR fell in every measured group of
    //    both rounds — and pricing share is the ceiling on turning that into
    //    wall clock.  On intermodal (tree, PricerHeavy) pricing is 80−85% of the
    //    clock and the bound is worth −3.6%/−3.7% under copt-gpu/copt-cpu, where
    //    essentially the whole saving is pricing and the LP is flat.
    //  * Everywhere else pricing is 1−4% of the clock, so nothing reaches it:
    //    grid tree loses 22% of its pricing time and the wall clock moves
    //    +1.95%, LP noise being larger than the entire pricing term.
    //  * On a backend whose LP is a large share (HiGHS, ~42% on intermodal) the
    //    trajectory shift the bound causes is larger than the pricing saving and
    //    decides the sign in either direction; that total is not reproducible
    //    between sessions even when each session measures both arms.
    //
    // Two mechanism notes that are properties of the code rather than of a
    // measurement.  The bound fires on most searches only because of the
    // _neg_rc_tol allowance folded into both bounds — at a master optimum every
    // structural row is served by a basic column of reduced cost 0, so the A*
    // frontier reaches the sink at almost exactly the dual and an exact-zero
    // test would almost never trigger.  And the two formulations differ
    // sharply: the tree bound (residual budget over the sum of remaining
    // demands) tightens on every settle, while the path bound (max π over
    // unsettled sinks) is hostage to the single most expensive remaining
    // commodity, which usually settles last — so a path search runs nearly to
    // completion regardless.  Measured on the grid/planar cells whose pricing
    // time is large enough to read: tree −24.5% per price, path +1.1 to +3.4%.
    //
    // Caveat if you do enable it: on an instance with unreachable sinks the
    // bound can stop a search while an unreachable target is still pending
    // whenever some *other* sink keeps the A* heuristic finite, which
    // suppresses the partial tree column the unbounded path would emit.
    // Preprocess with mcfcg_clean, as the pricer already asks.
    bool bounded_pricing = false;
    uint32_t num_threads = 0;         // 0 = auto-detect via hardware_concurrency
    uint32_t pricing_batch_size = 0;  // 0 = all sources in one batch
    double neg_rc_tol = NEG_RC_TOL;   // see tolerances.h
    uint32_t row_inactivity_threshold =
        5;  // remove capacity rows inactive for this many iterations
    // Purge inactive columns after this many iters (0=off).  Ignored under
    // CGStrategy::PricerHeavy, which disables column aging entirely.
    uint32_t col_age_limit = 5;
    // Wall-clock budget for the CG loop in seconds (0 = no limit).  When
    // exceeded the loop breaks gracefully at the next iteration boundary,
    // reporting the best UB/LB found so far (result.optimal stays false) so
    // the CLI/benchmark always emits a result instead of being killed.
    double time_limit_seconds = 0.0;
    SolverFactory solver_factory;  // Custom LP solver; uses HiGHS if null
};

CGResult solve_path_cg(const Instance& inst, const CGParams& params = {});

}  // namespace mcfcg
