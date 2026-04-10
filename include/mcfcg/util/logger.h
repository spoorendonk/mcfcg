#pragma once

#include "mcfcg/util/limits.h"

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace mcfcg {

enum class Verbosity : uint8_t { Silent, Summary, Iteration, Debug };

class CGLogger {
    Verbosity _verbosity;

public:
    explicit CGLogger(Verbosity verbosity) : _verbosity(verbosity) {}

    void print_header() const {
        if (_verbosity < Verbosity::Iteration) {
            return;
        }
        std::fprintf(stderr, "%5s %10s %10s %10s %6s %6s %6s %6s %6s %6s %7s %7s %7s %7s\n", "It",
                     "UB", "LB", "LP_obj", "#col", "#row", "+col", "-col", "+cut", "-cut", "t_LP",
                     "t_PR", "t_SP", "t_Tot");
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    void print_iteration(uint32_t iter, double upper_bound, double lower_bound, double lp_obj,
                         uint32_t num_col, uint32_t num_row, uint32_t added_col,
                         uint32_t removed_col, uint32_t added_cut, uint32_t removed_cut,
                         double t_lp, double t_pr, double t_sp, double t_tot) const {
        // NOLINTEND(bugprone-easily-swappable-parameters)
        if (_verbosity < Verbosity::Iteration) {
            return;
        }

        char ub_buf[16];
        if (std::isinf(upper_bound)) {
            std::snprintf(ub_buf, sizeof(ub_buf), "inf");
        } else {
            std::snprintf(ub_buf, sizeof(ub_buf), "%.2e", upper_bound);
        }

        char lb_buf[16];
        if (lower_bound == -INF) {
            std::snprintf(lb_buf, sizeof(lb_buf), "-inf");
        } else {
            std::snprintf(lb_buf, sizeof(lb_buf), "%.2e", lower_bound);
        }

        char obj_buf[16];
        std::snprintf(obj_buf, sizeof(obj_buf), "%.2e", lp_obj);

        std::fprintf(stderr, "%5u %10s %10s %10s %6u %6u %6u %6u %6u %6u %7.3f %7.3f %7.3f %7.3f\n",
                     iter, ub_buf, lb_buf, obj_buf, num_col, num_row, added_col, removed_col,
                     added_cut, removed_cut, t_lp, t_pr, t_sp, t_tot);
    }

    void print_summary(uint32_t iters, double obj, bool optimal, double t_lp, double t_pr,
                       double t_sp, double t_tot) const {
        if (_verbosity < Verbosity::Summary) {
            return;
        }
        std::fprintf(stderr,
                     "CG %s after %u iterations. obj=%.6f  "
                     "t_LP=%.3f  t_PR=%.3f  t_SP=%.3f  t_Tot=%.3f\n",
                     optimal ? "optimal" : "stopped", iters, obj, t_lp, t_pr, t_sp, t_tot);
    }

    [[nodiscard]] Verbosity verbosity() const { return _verbosity; }
};

}  // namespace mcfcg
