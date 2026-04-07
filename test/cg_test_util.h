#pragma once

#include "mcfcg/cg/column.h"
#include "mcfcg/cg/master.h"
#include "mcfcg/cg/pricer.h"
#include "mcfcg/cg/tree_column.h"
#include "mcfcg/cg/tree_master.h"
#include "mcfcg/cg/tree_pricer.h"
#include "mcfcg/instance.h"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

namespace mcfcg::test {

// Tolerance for new columns: must have strictly negative RC
constexpr double NEW_COL_RC_TOL = 1e-6;

// Tolerance for existing columns at optimality: allow small numerical noise.
// Tree formulation accumulates demand-weighted floating-point error, so existing
// columns can show slightly negative recomputed RC (observed up to ~3.5e-4 on
// planar100 tree). This grows with instance size.
constexpr double EXISTING_COL_RC_TOL = 1e-3;

inline double recompute_path_rc(const Column& col, const std::vector<double>& pi,
                                const std::unordered_map<uint32_t, double>& mu) {
    double rc = col.cost - pi[col.commodity];
    for (uint32_t arc : col.arcs) {
        auto it = mu.find(arc);
        if (it != mu.end()) {
            rc -= it->second;
        }
    }
    return rc;
}

inline double recompute_tree_rc(const TreeColumn& col, const std::vector<double>& pi_s,
                                const std::unordered_map<uint32_t, double>& mu) {
    double rc = col.cost - pi_s[col.source_idx];
    for (const auto& af : col.arc_flows) {
        auto it = mu.find(af.arc);
        if (it != mu.end()) {
            rc -= af.flow * it->second;
        }
    }
    return rc;
}

// Run path CG manually, validating RC at each iteration.
// Checks: new columns have negative RC, existing columns have non-negative RC.
// If check_duplicates is true, also asserts no duplicate columns.
inline void solve_and_validate_path_rc(const Instance& inst, double ref_obj, double tol = 0.0001,
                                       bool check_duplicates = false) {
    PathMaster master;
    master.init(inst);
    PathPricer pricer;
    pricer.init(inst);

    std::vector<double> big_pi(inst.commodities.size(), PathMaster::BIG_M);
    std::unordered_map<uint32_t, double> empty_mu;
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

        auto pi = master.get_demand_duals();
        auto mu = master.get_capacity_duals();

        for (const auto& col : master.columns()) {
            double rc = recompute_path_rc(col, pi, mu);
            EXPECT_GE(rc, -EXISTING_COL_RC_TOL)
                << "Existing column k=" << col.commodity << " RC=" << rc;
        }

        auto new_cols = pricer.price(pi, mu, false);
        if (new_cols.empty()) {
            new_cols = pricer.price(pi, mu, true);
            if (new_cols.empty()) {
                optimal = true;
                break;
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
        master.add_columns(std::move(new_cols));
    }
    EXPECT_TRUE(optimal);
    EXPECT_GE(obj, ref_obj * (1.0 - tol));
    EXPECT_LE(obj, ref_obj * (1.0 + tol));
}

// Run tree CG manually, validating RC at each iteration.
inline void solve_and_validate_tree_rc(const Instance& inst, double ref_obj, double tol = 0.0001,
                                       bool check_duplicates = false) {
    TreeMaster master;
    master.init(inst);
    TreePricer pricer;
    pricer.init(inst);

    std::vector<double> big_pi(inst.sources.size(), TreeMaster::BIG_M);
    std::unordered_map<uint32_t, double> empty_mu;
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

        auto pi_s = master.get_source_duals();
        auto mu = master.get_capacity_duals();

        for (const auto& col : master.columns()) {
            double rc = recompute_tree_rc(col, pi_s, mu);
            EXPECT_GE(rc, -EXISTING_COL_RC_TOL)
                << "Existing tree col s=" << col.source_idx << " RC=" << rc;
        }

        auto new_cols = pricer.price(pi_s, mu, false);
        if (new_cols.empty()) {
            new_cols = pricer.price(pi_s, mu, true);
            if (new_cols.empty()) {
                optimal = true;
                break;
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
        master.add_columns(std::move(new_cols));
    }
    EXPECT_TRUE(optimal);
    EXPECT_GE(obj, ref_obj * (1.0 - tol));
    EXPECT_LE(obj, ref_obj * (1.0 + tol));
}

}  // namespace mcfcg::test
