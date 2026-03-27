#pragma once

#include <cstdint>

#include "mcfcg/instance.h"

namespace mcfcg {

struct CGResult {
    double objective;
    uint32_t iterations;
    uint32_t total_columns;
    bool optimal;
};

struct CGParams {
    uint32_t max_iterations = 10000;
    uint32_t max_cols_per_iter = 1000;
};

CGResult solve_path_cg(const Instance & inst, const CGParams & params = {});

}  // namespace mcfcg
