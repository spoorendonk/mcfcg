#pragma once

#include "mcfcg/instance.h"
#include "mcfcg/lp/lp_solver.h"
#include "mcfcg/util/limits.h"
#include "mcfcg/util/logger.h"
#include "mcfcg/util/tolerances.h"

#include <algorithm>
#include <cstdint>
#include <functional>

namespace mcfcg {

struct CGResult {
    double objective;
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
    uint32_t iterations;
    uint32_t total_columns;
    bool optimal;
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
    uint32_t num_threads = 0;         // 0 = auto-detect via hardware_concurrency
    uint32_t pricing_batch_size = 0;  // 0 = all sources in one batch
    double neg_rc_tol = NEG_RC_TOL;   // see tolerances.h
    uint32_t row_inactivity_threshold =
        5;  // remove capacity rows inactive for this many iterations
    // Purge inactive columns after this many iters (0=off).  Ignored under
    // CGStrategy::PricerHeavy, which disables column aging entirely.
    uint32_t col_age_limit = 5;
    SolverFactory solver_factory;  // Custom LP solver; uses HiGHS if null
};

CGResult solve_path_cg(const Instance& inst, const CGParams& params = {});

}  // namespace mcfcg
