#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "mcfcg/util/limits.h"

namespace mcfcg {

enum class Verbosity : uint8_t { Silent, Summary, Iteration, Debug };

class CGLogger {
    Verbosity _verbosity;
    // Running sum of per-iteration total time, printed as the t_acc column so a
    // long run's wall-clock progress is visible without summing the log by hand.
    double _t_acc = 0.0;

public:
    explicit CGLogger(Verbosity verbosity) : _verbosity(verbosity) {}

    void print_header() const {
        if (_verbosity < Verbosity::Iteration) {
            return;
        }
        std::fprintf(stderr, "%5s %12s %12s %12s %6s %6s %5s %6s %6s %6s %6s %7s %7s %7s %7s %9s\n",
                     "It", "UB", "LB", "LP_obj", "#col", "#row", "#slk", "+col", "-col", "+cut",
                     "-cut", "t_LP", "t_PR", "t_SP", "t_Tot", "t_acc");
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    void print_iteration(uint32_t iter, double upper_bound, double lower_bound, double lp_obj,
                         uint32_t num_col, uint32_t num_row, uint32_t num_active_slacks,
                         uint32_t added_col, bool added_not_committed, uint32_t removed_col,
                         uint32_t added_cut, uint32_t removed_cut, double t_lp, double t_pr,
                         double t_sp, double t_tot) {
        // NOLINTEND(bugprone-easily-swappable-parameters)
        _t_acc += t_tot;
        if (_verbosity < Verbosity::Iteration) {
            return;
        }

        char ub_buf[16];
        if (std::isinf(upper_bound)) {
            std::snprintf(ub_buf, sizeof(ub_buf), "inf");
        } else {
            std::snprintf(ub_buf, sizeof(ub_buf), "%.4e", upper_bound);
        }

        char lb_buf[16];
        if (lower_bound == -INF) {
            std::snprintf(lb_buf, sizeof(lb_buf), "-inf");
        } else {
            std::snprintf(lb_buf, sizeof(lb_buf), "%.4e", lower_bound);
        }

        char obj_buf[16];
        std::snprintf(obj_buf, sizeof(obj_buf), "%.4e", lp_obj);

        // Prefix "+col" with '*' when the pricer produced columns that
        // were NOT actually added to the master (gap-based early
        // termination prints the pricer's output for diagnostic value).
        char added_buf[16];
        if (added_not_committed) {
            std::snprintf(added_buf, sizeof(added_buf), "*%u", added_col);
        } else {
            std::snprintf(added_buf, sizeof(added_buf), "%u", added_col);
        }
        std::fprintf(
            stderr,
            "%5u %12s %12s %12s %6u %6u %5u %6s %6u %6u %6u %7.3f %7.3f %7.3f %7.3f %9.3f\n", iter,
            ub_buf, lb_buf, obj_buf, num_col, num_row, num_active_slacks, added_buf, removed_col,
            added_cut, removed_cut, t_lp, t_pr, t_sp, t_tot, _t_acc);
    }

    // `upper_bound` is the certified MCF-feasible upper bound; it is +INF when
    // the run exited without ever finding a slack-free feasible incumbent (e.g.
    // an LP backend that stalls or spuriously reports infeasible on a large
    // master).  `lower_bound` is the Lagrangian lower bound, -INF until the
    // pricer first sweeps every source.  Format each side independently and
    // only show a numeric gap when BOTH bounds are finite — otherwise print
    // UB/LB=inf and gap=inf rather than letting an INF fallback masquerade as a
    // zero-gap optimum.
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    void print_summary(uint32_t iters, double upper_bound, bool optimal, double lower_bound,
                       double gap_tol, double t_lp, double t_pr, double t_sp, double t_tot) const {
        // NOLINTEND(bugprone-easily-swappable-parameters)
        if (_verbosity < Verbosity::Summary) {
            return;
        }
        const bool ub_inf = std::isinf(upper_bound);
        const bool lb_inf = (lower_bound == -INF) || std::isinf(lower_bound);

        char ub_buf[24];
        std::snprintf(ub_buf, sizeof(ub_buf), ub_inf ? "inf" : "%.6f", upper_bound);
        char lb_buf[24];
        std::snprintf(lb_buf, sizeof(lb_buf), lb_inf ? "-inf" : "%.6f", lower_bound);
        char gap_buf[24];
        if (ub_inf || lb_inf) {
            std::snprintf(gap_buf, sizeof(gap_buf), "inf");
        } else {
            std::snprintf(gap_buf, sizeof(gap_buf), "%.3e", upper_bound - lower_bound);
        }

        std::fprintf(stderr,
                     "CG %s after %u iterations. UB=%s LB=%s gap=%s tol=%.3e  "
                     "t_LP=%.3f  t_PR=%.3f  t_SP=%.3f  t_Tot=%.3f\n",
                     optimal ? "optimal" : "stopped", iters, ub_buf, lb_buf, gap_buf, gap_tol, t_lp,
                     t_pr, t_sp, t_tot);
    }
};

}  // namespace mcfcg
