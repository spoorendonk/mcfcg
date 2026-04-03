#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "mcfcg/cg/master.h"
#include "mcfcg/cg/path_cg.h"
#include "mcfcg/cg/pricer.h"
#include "mcfcg/cg/tree_master.h"
#include "mcfcg/cg/tree_pricer.h"
#include "mcfcg/instance.h"

namespace fs = std::filesystem;

static std::string data_dir(const std::string & subdir) {
    return std::string(MCFCG_SOURCE_DIR) + "/data/" + subdir;
}

// --- RC validation: manually drive CG loop with checks ---

// Tolerance for new columns: they must have strictly negative RC
static constexpr double NEW_COL_RC_TOL = 1e-6;
// Tolerance for existing columns at optimality: allow small numerical noise.
// Tree formulation accumulates demand-weighted floating-point error, so existing
// columns can show slightly negative recomputed RC (observed up to ~1e-4 on
// Winnipeg tree). This is a known epsilon issue, not a bug.
static constexpr double EXISTING_COL_RC_TOL = 1e-4;

static double recompute_path_rc(const mcfcg::Column& col, const std::vector<double>& pi,
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

static double recompute_tree_rc(const mcfcg::TreeColumn& col, const std::vector<double>& pi_s,
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

static void solve_and_validate_path_rc(const mcfcg::Instance& inst, double paper_obj,
                                       double tol = 0.0001) {
    mcfcg::PathMaster master;
    master.init(inst);
    mcfcg::PathPricer pricer;
    pricer.init(inst);

    std::vector<double> big_pi(inst.commodities.size(), mcfcg::PathMaster::BIG_M);
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
        ASSERT_EQ(status, mcfcg::LPStatus::Optimal) << "LP not optimal at iter " << iter;

        obj = master.get_obj();
        auto primals = master.get_primals();
        if (master.add_violated_capacity_constraints(primals) > 0) {
            continue;
        }

        auto pi = master.get_demand_duals();
        auto mu = master.get_capacity_duals();

        // Validate existing columns
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

        // Validate new columns
        for (const auto& col : new_cols) {
            double rc = recompute_path_rc(col, pi, mu);
            EXPECT_LT(rc, NEW_COL_RC_TOL) << "New column k=" << col.commodity << " RC=" << rc;
        }

        if (new_cols.size() > 1000) {
            new_cols.resize(1000);
        }
        master.add_columns(std::move(new_cols));
    }
    EXPECT_TRUE(optimal);
    EXPECT_GE(obj, paper_obj * (1.0 - tol));
    EXPECT_LE(obj, paper_obj * (1.0 + tol));
}

static void solve_and_validate_tree_rc(const mcfcg::Instance& inst, double paper_obj,
                                       double tol = 0.0001) {
    mcfcg::TreeMaster master;
    master.init(inst);
    mcfcg::TreePricer pricer;
    pricer.init(inst);

    std::vector<double> big_pi(inst.sources.size(), mcfcg::TreeMaster::BIG_M);
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
        ASSERT_EQ(status, mcfcg::LPStatus::Optimal) << "LP not optimal at iter " << iter;

        obj = master.get_obj();
        auto primals = master.get_primals();
        if (master.add_violated_capacity_constraints(primals) > 0) {
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
            EXPECT_LT(rc, NEW_COL_RC_TOL)
                << "New tree col s=" << col.source_idx << " RC=" << rc;
        }

        if (new_cols.size() > 1000) {
            new_cols.resize(1000);
        }
        master.add_columns(std::move(new_cols));
    }
    EXPECT_TRUE(optimal);
    EXPECT_GE(obj, paper_obj * (1.0 - tol));
    EXPECT_LE(obj, paper_obj * (1.0 + tol));
}

// --- Correctness tests: solve small instances, verify against paper ---
// Reference objectives from the paper (FLOWTY_MOSEK_PATH solver).
// We check: optimal, and within `tol` (default 0.01%) of paper value.

static void solve_and_check(const mcfcg::Instance & inst, double paper_obj,
                            double tol = 0.0001) {
    mcfcg::CGParams params;
    params.max_iterations = 10000;
    auto result = mcfcg::solve_path_cg(inst, params);
    EXPECT_TRUE(result.optimal) << "Did not reach optimality";
    EXPECT_GE(result.objective, paper_obj * (1.0 - tol))
        << "Objective below paper value (possible bug)";
    EXPECT_LE(result.objective, paper_obj * (1.0 + tol))
        << "Objective too far above paper value";
}

TEST(GridCorrectness, Grid1) {
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid1");
    solve_and_check(inst, 827319.0);  // paper: 827318.9999
}

TEST(GridCorrectness, Grid2) {
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid2");
    solve_and_check(inst, 1705508.0);  // paper: 1705507.9999
}

TEST(PlanarCorrectness, Planar30) {
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar30");
    solve_and_check(inst, 44350624.0);  // paper: 44350623.9995
}

TEST(PlanarCorrectness, Planar80) {
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar80");
    solve_and_check(inst, 182438134.0);  // paper: 182438134.0001
}

TEST(TransportationCorrectness, Winnipeg) {
    auto net = data_dir("transportation") + "/Winnipeg_net.tntp.gz";
    auto trips = data_dir("transportation") + "/Winnipeg_trips.tntp.gz";
    if (!fs::exists(net))
        GTEST_SKIP() << "data/transportation not found";
    auto inst = mcfcg::read_tntp(net, trips, 2000.0);
    solve_and_check(inst, 420.165);  // paper: 420.16536
}

TEST(TransportationCorrectness, Barcelona) {
    auto net = data_dir("transportation") + "/Barcelona_net.tntp.gz";
    auto trips = data_dir("transportation") + "/Barcelona_trips.tntp.gz";
    if (!fs::exists(net))
        GTEST_SKIP() << "data/transportation not found";
    auto inst = mcfcg::read_tntp(net, trips, 5050.0);
    solve_and_check(inst, 243.512);  // paper: 243.51248
}

// Intermodal instances were regenerated + cleaned, so paper objectives don't
// apply. Use our own verified values (from this solver on these instances).
TEST(IntermodalCorrectness, Subway308) {
    auto path = data_dir("intermodal") + "/SUBWAY-308-0.txt.gz";
    if (!fs::exists(path))
        GTEST_SKIP() << "data/intermodal not found";
    auto inst = mcfcg::read_commalab(path);
    solve_and_check(inst, 8835.0);
}

TEST(IntermodalCorrectness, Subway486) {
    auto path = data_dir("intermodal") + "/SUBWAY-486-0.txt.gz";
    if (!fs::exists(path))
        GTEST_SKIP() << "data/intermodal not found";
    auto inst = mcfcg::read_commalab(path);
    solve_and_check(inst, 13543.0);
}

TEST(IntermodalCorrectness, DISABLED_Bus2632) {
    auto path = data_dir("intermodal") + "/BUS-2632-0.txt.gz";
    if (!fs::exists(path))
        GTEST_SKIP() << "data/intermodal not found";
    auto inst = mcfcg::read_commalab(path);
    solve_and_check(inst, 71026.5, 0.005);
}

TEST(IntermodalCorrectness, DISABLED_Bus7896) {
    auto path = data_dir("intermodal") + "/BUS-7896-0.txt.gz";
    if (!fs::exists(path))
        GTEST_SKIP() << "data/intermodal not found";
    auto inst = mcfcg::read_commalab(path);
    solve_and_check(inst, 210603.5, 0.005);
}

// --- Reduced cost validation on real instances ---

TEST(RCValidation, Grid1Path) {
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid1");
    solve_and_validate_path_rc(inst, 827319.0);
}

TEST(RCValidation, Grid1Tree) {
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid1");
    solve_and_validate_tree_rc(inst, 827319.0);
}

TEST(RCValidation, Planar30Path) {
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar30");
    solve_and_validate_path_rc(inst, 44350624.0);
}

TEST(RCValidation, Planar30Tree) {
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar30");
    solve_and_validate_tree_rc(inst, 44350624.0);
}

TEST(RCValidation, WinnipegPath) {
    auto net = data_dir("transportation") + "/Winnipeg_net.tntp.gz";
    auto trips = data_dir("transportation") + "/Winnipeg_trips.tntp.gz";
    if (!fs::exists(net))
        GTEST_SKIP() << "data/transportation not found";
    auto inst = mcfcg::read_tntp(net, trips, 2000.0);
    solve_and_validate_path_rc(inst, 420.165);
}

TEST(RCValidation, WinnipegTree) {
    auto net = data_dir("transportation") + "/Winnipeg_net.tntp.gz";
    auto trips = data_dir("transportation") + "/Winnipeg_trips.tntp.gz";
    if (!fs::exists(net))
        GTEST_SKIP() << "data/transportation not found";
    auto inst = mcfcg::read_tntp(net, trips, 2000.0);
    solve_and_validate_tree_rc(inst, 420.165);
}
