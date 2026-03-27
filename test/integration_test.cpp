#include <gtest/gtest.h>

#include <filesystem>
#include <string>

#include "mcfcg/cg/path_cg.h"
#include "mcfcg/instance.h"

namespace fs = std::filesystem;

static std::string data_dir(const std::string & subdir) {
    return std::string(MCFCG_SOURCE_DIR) + "/data/" + subdir;
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
