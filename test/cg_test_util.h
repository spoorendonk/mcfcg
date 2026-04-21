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
// from demand-weighted FP accumulation.  Kept identical to NEG_RC_TOL
// (the pricer's new-column acceptance threshold) so there is no
// window in which the test tolerates drift that the pricer would
// re-accept as a new column — which would invite duplicate columns.
constexpr double NEW_COL_RC_TOL = COL_ACTIVE_EPS;
constexpr double EXISTING_COL_RC_TOL = COL_ACTIVE_EPS;

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
// Mirrors the production single-solve-per-iter flow in cg_loop.h:
// capture primals/duals once, then separate and price with those same
// duals (stale wrt any cap rows separation just added — the next iter
// picks them up).  Existing cols must have RC ≥ -EXISTING_COL_RC_TOL
// at the captured duals (LP optimality of the previous solve).  New
// cols must have RC < NEW_COL_RC_TOL at the captured duals (pricer's
// own decision, so this is the self-consistency check).
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
        // Capture primals/duals BEFORE any mutation (add_rows /
        // set_col_cost / delete_*).  Some backends invalidate cached
        // solution state on mutation — see cg_loop.h.
        auto primals = master.get_primals();
        auto pi = master.get_structural_duals();
        const auto& mu = master.get_capacity_duals();

        auto new_cap_arcs = master.add_violated_capacity_constraints(primals);
        uint32_t num_new_caps = static_cast<uint32_t>(new_cap_arcs.size());

        // Existing-column RC invariant holds only against the current
        // LP's row set.  When separation found new violations this iter,
        // the LP just solved did NOT contain those rows — any col whose
        // flow traverses a newly-violated arc can legitimately have
        // negative RC against the pre-sep duals (the missing cap dual
        // would have pushed RC up).  Gate the invariant on cut-stable
        // iters to avoid false positives.  The new-column RC check
        // below still runs every iter — the pricer must be internally
        // consistent with its own duals regardless.
        if (num_new_caps == 0) {
            for (const auto& col : master.columns()) {
                double rc = recompute_path_rc(col, pi, mu);
                EXPECT_GE(rc, -EXISTING_COL_RC_TOL)
                    << "Existing column k=" << col.commodity << " RC=" << rc;
            }
        }

        auto new_cols = pricer.price(pi, mu, false);
        if (new_cols.empty()) {
            new_cols = pricer.price(pi, mu, true);
        }
        if (!new_cols.empty()) {
            pricer.reset_postponed();
        }

        if (new_cols.empty()) {
            const bool slacks_active = master.count_active_slacks(primals) > 0;
            if (num_new_caps == 0 && !slacks_active) {
                optimal = true;
                break;
            }
            if (slacks_active) {
                (void)master.bump_active_slacks(primals, SLACK_BUMP_FACTOR);
            }
            pricer.reset_postponed();
            continue;
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
        (void)master.bump_active_slacks(primals, SLACK_BUMP_FACTOR);
        master.add_columns(std::move(new_cols));
    }
    EXPECT_TRUE(optimal);
    EXPECT_GE(obj, ref_obj * (1.0 - tol));
    EXPECT_LE(obj, ref_obj * (1.0 + tol));
}

// Run tree CG manually, validating RC at each iteration.
// Mirrors the production single-solve-per-iter flow in cg_loop.h; see
// solve_and_validate_path_rc for the full rationale.
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
        auto pi_s = master.get_structural_duals();
        const auto& mu = master.get_capacity_duals();

        auto new_cap_arcs = master.add_violated_capacity_constraints(primals);
        uint32_t num_new_caps = static_cast<uint32_t>(new_cap_arcs.size());

        // See solve_and_validate_path_rc for the gating rationale.
        if (num_new_caps == 0) {
            for (const auto& col : master.columns()) {
                double rc = recompute_tree_rc(col, pi_s, mu);
                EXPECT_GE(rc, -EXISTING_COL_RC_TOL)
                    << "Existing tree col s=" << col.source_idx << " RC=" << rc;
            }
        }

        auto new_cols = pricer.price(pi_s, mu, false);
        if (new_cols.empty()) {
            new_cols = pricer.price(pi_s, mu, true);
        }
        if (!new_cols.empty()) {
            pricer.reset_postponed();
        }

        if (new_cols.empty()) {
            const bool slacks_active = master.count_active_slacks(primals) > 0;
            if (num_new_caps == 0 && !slacks_active) {
                optimal = true;
                break;
            }
            if (slacks_active) {
                (void)master.bump_active_slacks(primals, SLACK_BUMP_FACTOR);
            }
            pricer.reset_postponed();
            continue;
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
