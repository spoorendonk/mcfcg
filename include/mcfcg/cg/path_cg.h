#pragma once

#include "mcfcg/instance.h"
#include "mcfcg/lp/lp_solver.h"
#include "mcfcg/util/logger.h"
#include "mcfcg/util/tolerances.h"

#include <cstdint>
#include <functional>

namespace mcfcg {

struct CGResult {
    double objective;
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
    // The pricer is expensive relative to LP solves, so run as few pricing
    // problems per master iteration as possible.  Shift work toward the
    // master:
    //  * cap columns per iter at num_entities (one per source/commodity)
    //  * disable column aging (overrides CGParams::col_age_limit)
    //  * force the source pricing filter on
    //  * defer pricing in iterations that added lazy capacity rows — let the
    //    master re-solve and reach a cut-stable state before pricing again
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
    uint32_t num_threads = 1;         // 0 = auto-detect via hardware_concurrency
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
