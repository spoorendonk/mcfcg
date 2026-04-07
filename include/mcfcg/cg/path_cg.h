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

struct CGParams {
    uint32_t max_iterations = 10000;
    uint32_t max_cols_per_iter = 1000;
    bool warm_start = true;
    Verbosity verbosity = Verbosity::Silent;
    // When true, cap columns per iteration at the number of sources (one per
    // source) so the master LP is re-solved more frequently with fresh duals.
    bool prefer_master = false;
    bool pricing_filter = false;
    SolverFactory solver_factory;  // Custom LP solver; uses HiGHS if null
};

CGResult solve_path_cg(const Instance& inst, const CGParams& params = {});

}  // namespace mcfcg
