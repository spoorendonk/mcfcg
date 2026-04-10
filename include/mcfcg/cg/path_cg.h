#pragma once

#include "mcfcg/instance.h"
#include "mcfcg/lp/lp_solver.h"
#include "mcfcg/util/logger.h"

#include <cstdint>
#include <functional>

namespace mcfcg {

struct CGResult {
    double objective;
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
// together express a stance on how much work to push onto the pricer per CG
// iteration.  Lower-level fields in CGParams remain as overrides, except
// where the bundle is documented to supersede them.
//
// CLI spellings (see src/main.cpp --strategy flag): "pricer-heavy",
// "pricer-light".
enum class CGStrategy : uint8_t {
    // Default: keep the pricer busy.  Large col cap, column aging on, cuts
    // and columns added in the same iteration.  Good when LP solves are
    // expensive relative to the pricer.
    PricerHeavy,
    // Throttle the pricer.  Shift work toward the master:
    //  * cap columns per iter at num_entities (one per source/commodity)
    //  * disable column aging (overrides CGParams::col_age_limit)
    //  * force the source pricing filter on
    //  * defer pricing in iterations that added lazy capacity rows — let the
    //    master re-solve and reach a cut-stable state before pricing again
    // Good when the pricer is expensive relative to LP solves.
    PricerLight,
};

struct CGParams {
    uint32_t max_iterations = 10000;
    uint32_t max_cols_per_iter = 50000;
    bool warm_start = true;
    Verbosity verbosity = Verbosity::Silent;
    CGStrategy strategy = CGStrategy::PricerHeavy;
    bool pricing_filter = false;
    uint32_t num_threads = 1;         // 0 = auto-detect via hardware_concurrency
    uint32_t pricing_batch_size = 0;  // 0 = all sources in one batch
    double neg_rc_tol = -1e-6;        // reduced cost tolerance for column acceptance
    uint32_t row_inactivity_threshold =
        5;  // remove capacity rows inactive for this many iterations
    // Purge inactive columns after this many iters (0=off).  Ignored under
    // CGStrategy::PricerLight, which disables column aging entirely.
    uint32_t col_age_limit = 5;
    SolverFactory solver_factory;  // Custom LP solver; uses HiGHS if null
};

CGResult solve_path_cg(const Instance& inst, const CGParams& params = {});

}  // namespace mcfcg
