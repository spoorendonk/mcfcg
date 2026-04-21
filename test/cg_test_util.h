#pragma once

#include "mcfcg/cg/cg_loop.h"
#include "mcfcg/cg/column.h"
#include "mcfcg/cg/master.h"
#include "mcfcg/cg/pricer.h"
#include "mcfcg/cg/tree_column.h"
#include "mcfcg/cg/tree_master.h"
#include "mcfcg/cg/tree_pricer.h"
#include "mcfcg/graph/static_map.h"
#include "mcfcg/instance.h"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace mcfcg::test {

// Test RC tolerances derived from the central RELATIVE_FEAS_TOL.
// NEW_COL_RC_TOL: new columns must have genuinely negative RC, not
// just dual noise (which is ~10 × LP_FEAS_TOL for a 10-arc path).
// EXISTING_COL_RC_TOL: existing columns may drift slightly negative
// from demand-weighted FP accumulation in the tree formulation.  The
// 5× multiplier accommodates the LP-backend dual noise that shows up
// when CSC row ordering changes (e.g. per-thread scratch reuse in the
// tree pricer's arc_flow_map), which is not a correctness concern but
// can drift a few existing RCs past a tight 1× threshold.
constexpr double NEW_COL_RC_TOL = COL_ACTIVE_EPS;
constexpr double EXISTING_COL_RC_TOL = 5 * COL_ACTIVE_EPS;

inline double recompute_path_rc(const Column& col, const std::vector<double>& pi,
                                const static_map<uint32_t, double>& mu) {
    double rc = col.cost - pi[col.commodity];
    for (uint32_t arc : col.arcs) {
        rc -= mu[arc];
    }
    return rc;
}

inline double recompute_tree_rc(const TreeColumn& col, const std::vector<double>& pi_s,
                                const static_map<uint32_t, double>& mu) {
    double rc = col.cost - pi_s[col.source_idx];
    for (const auto& af : col.arc_flows) {
        rc -= af.flow * mu[af.arc];
    }
    return rc;
}

// Run path CG manually, validating RC at each iteration.
// Checks: new columns have negative RC, existing columns have non-negative RC.
// If check_duplicates is true, also asserts no duplicate columns.
inline void solve_and_validate_path_rc(const Instance& inst, double ref_obj,
                                       double tol = RELATIVE_FEAS_TOL * 10,
                                       bool check_duplicates = false) {
    PathMaster master;
    master.init(inst);
    PathPricer pricer;
    pricer.init(inst);

    std::vector<double> big_pi(inst.commodities.size(), std::numeric_limits<double>::infinity());
    auto empty_mu = inst.graph.create_arc_map<double>(0.0);
    auto init_cols = pricer.price(big_pi, empty_mu, true);
    if (!init_cols.empty()) {
        master.add_columns(std::move(init_cols));
    }
    pricer.reset_postponed();

    bool optimal = false;
    double obj = 0;
    for (uint32_t iter = 0; iter < 10000; ++iter) {
        auto status = master.solve();
        ASSERT_EQ(status, LPStatus::Optimal) << "LP not optimal at iter " << iter;

        obj = master.get_obj();
        auto primals = master.get_primals();
        if (!master.add_violated_capacity_constraints(primals).empty()) {
            continue;
        }

        auto pi = master.get_structural_duals();
        const auto& mu = master.get_capacity_duals();

        for (const auto& col : master.columns()) {
            double rc = recompute_path_rc(col, pi, mu);
            EXPECT_GE(rc, -EXISTING_COL_RC_TOL)
                << "Existing column k=" << col.commodity << " RC=" << rc;
        }

        auto new_cols = pricer.price(pi, mu, false);
        if (new_cols.empty()) {
            new_cols = pricer.price(pi, mu, true);
            if (new_cols.empty()) {
                // Pricing exhausted.  Only optimal if no slack is still
                // carrying demand — otherwise bump and try again so the
                // LP can pivot the slacks out in the next iteration.
                if (!master.has_active_slacks(primals)) {
                    optimal = true;
                    break;
                }
                (void)master.bump_active_slacks(primals, SLACK_BUMP_FACTOR);
                pricer.reset_postponed();
                continue;
            }
            pricer.reset_postponed();
        }

        for (const auto& col : new_cols) {
            double rc = recompute_path_rc(col, pi, mu);
            EXPECT_LT(rc, NEW_COL_RC_TOL) << "New column k=" << col.commodity << " RC=" << rc;
        }

        if (check_duplicates) {
            for (const auto& col : new_cols) {
                for (const auto& existing : master.columns()) {
                    if (existing.commodity == col.commodity && existing.arcs == col.arcs) {
                        ADD_FAILURE() << "Duplicate column for commodity " << col.commodity
                                      << " at iteration " << iter;
                    }
                }
            }
        }

        if (new_cols.size() > 1000) {
            new_cols.resize(1000);
        }
        // Bump before add_columns, same reason as cg_loop.h: the primals
        // captured above reflect the solved LP; any purge would
        // invalidate get_primals().
        (void)master.bump_active_slacks(primals, SLACK_BUMP_FACTOR);
        master.add_columns(std::move(new_cols));
    }
    EXPECT_TRUE(optimal);
    EXPECT_GE(obj, ref_obj * (1.0 - tol));
    EXPECT_LE(obj, ref_obj * (1.0 + tol));
}

// Run tree CG manually, validating RC at each iteration.
inline void solve_and_validate_tree_rc(const Instance& inst, double ref_obj,
                                       double tol = RELATIVE_FEAS_TOL * 10,
                                       bool check_duplicates = false) {
    TreeMaster master;
    master.init(inst);
    TreePricer pricer;
    pricer.init(inst);

    std::vector<double> big_pi(inst.sources.size(), std::numeric_limits<double>::infinity());
    auto empty_mu = inst.graph.create_arc_map<double>(0.0);
    auto init_cols = pricer.price(big_pi, empty_mu, true);
    if (!init_cols.empty()) {
        master.add_columns(std::move(init_cols));
    }
    pricer.reset_postponed();

    bool optimal = false;
    double obj = 0;
    for (uint32_t iter = 0; iter < 10000; ++iter) {
        auto status = master.solve();
        ASSERT_EQ(status, LPStatus::Optimal) << "LP not optimal at iter " << iter;

        obj = master.get_obj();
        auto primals = master.get_primals();
        if (!master.add_violated_capacity_constraints(primals).empty()) {
            continue;
        }

        auto pi_s = master.get_structural_duals();
        const auto& mu = master.get_capacity_duals();

        for (const auto& col : master.columns()) {
            double rc = recompute_tree_rc(col, pi_s, mu);
            EXPECT_GE(rc, -EXISTING_COL_RC_TOL)
                << "Existing tree col s=" << col.source_idx << " RC=" << rc;
        }

        auto new_cols = pricer.price(pi_s, mu, false);
        if (new_cols.empty()) {
            new_cols = pricer.price(pi_s, mu, true);
            if (new_cols.empty()) {
                if (!master.has_active_slacks(primals)) {
                    optimal = true;
                    break;
                }
                (void)master.bump_active_slacks(primals, SLACK_BUMP_FACTOR);
                pricer.reset_postponed();
                continue;
            }
            pricer.reset_postponed();
        }

        for (const auto& col : new_cols) {
            double rc = recompute_tree_rc(col, pi_s, mu);
            EXPECT_LT(rc, NEW_COL_RC_TOL) << "New tree col s=" << col.source_idx << " RC=" << rc;
        }

        if (check_duplicates) {
            for (const auto& col : new_cols) {
                for (const auto& existing : master.columns()) {
                    if (existing.source_idx != col.source_idx) {
                        continue;
                    }
                    if (existing.arc_flows.size() != col.arc_flows.size()) {
                        continue;
                    }
                    bool match = true;
                    for (const auto& af : col.arc_flows) {
                        bool found = false;
                        for (const auto& eaf : existing.arc_flows) {
                            if (eaf.arc == af.arc && std::abs(eaf.flow - af.flow) < 1e-10) {
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        ADD_FAILURE() << "Duplicate tree column for source " << col.source_idx
                                      << " at iteration " << iter;
                    }
                }
            }
        }

        if (new_cols.size() > 1000) {
            new_cols.resize(1000);
        }
        (void)master.bump_active_slacks(primals, SLACK_BUMP_FACTOR);
        master.add_columns(std::move(new_cols));
    }
    EXPECT_TRUE(optimal);
    EXPECT_GE(obj, ref_obj * (1.0 - tol));
    EXPECT_LE(obj, ref_obj * (1.0 + tol));
}

}  // namespace mcfcg::test
