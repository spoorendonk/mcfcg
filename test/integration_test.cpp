#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>

#include "cg_test_util.h"
#include "mcfcg/cg/path_cg.h"
#include "mcfcg/instance.h"

namespace fs = std::filesystem;

static std::string data_dir(const std::string& subdir) {
    return std::string(MCFCG_SOURCE_DIR) + "/data/" + subdir;
}

// Load optimal.csv from a data directory. Returns instance->optimal map.
static std::unordered_map<std::string, double> load_optimal(const std::string& dir) {
    std::unordered_map<std::string, double> result;
    auto path = dir + "/optimal.csv";
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open " + path);
    }
    std::string line;
    std::getline(file, line);  // skip header
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        auto comma = line.find(',');
        auto name = line.substr(0, comma);
        auto val = std::stod(line.substr(comma + 1));
        result[name] = val;
    }
    return result;
}

using mcfcg::test::solve_and_validate_path_rc;
using mcfcg::test::solve_and_validate_tree_rc;

// --- Correctness tests: solve instances, verify against optimal.csv ---

static void solve_and_check(const mcfcg::Instance& inst, double ref_obj, double tol = 0.0001) {
    mcfcg::CGParams params;
    params.max_iterations = 10000;
    auto result = mcfcg::solve_path_cg(inst, params);
    EXPECT_TRUE(result.optimal) << "Did not reach optimality";
    EXPECT_GE(result.objective, ref_obj * (1.0 - tol)) << "Objective below reference";
    EXPECT_LE(result.objective, ref_obj * (1.0 + tol)) << "Objective above reference";
}

TEST(GridCorrectness, Grid1) {
    auto opt = load_optimal(data_dir("commalab/grid"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid1");
    solve_and_check(inst, opt.at("grid1"));
}

TEST(GridCorrectness, Grid2) {
    auto opt = load_optimal(data_dir("commalab/grid"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid2");
    solve_and_check(inst, opt.at("grid2"));
}

TEST(PlanarCorrectness, Planar30) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar30");
    solve_and_check(inst, opt.at("planar30"));
}

TEST(PlanarCorrectness, Planar80) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar80");
    solve_and_check(inst, opt.at("planar80"));
}

TEST(TransportationCorrectness, Winnipeg) {
    auto net = data_dir("transportation") + "/Winnipeg_net.tntp.gz";
    auto trips = data_dir("transportation") + "/Winnipeg_trips.tntp.gz";
    if (!fs::exists(net))
        GTEST_SKIP() << "data/transportation not found";
    auto opt = load_optimal(data_dir("transportation"));
    auto inst = mcfcg::read_tntp(net, trips, 2000.0);
    solve_and_check(inst, opt.at("Winnipeg"));
}

TEST(TransportationCorrectness, Barcelona) {
    auto net = data_dir("transportation") + "/Barcelona_net.tntp.gz";
    auto trips = data_dir("transportation") + "/Barcelona_trips.tntp.gz";
    if (!fs::exists(net))
        GTEST_SKIP() << "data/transportation not found";
    auto opt = load_optimal(data_dir("transportation"));
    auto inst = mcfcg::read_tntp(net, trips, 5050.0);
    solve_and_check(inst, opt.at("Barcelona"));
}

TEST(IntermodalCorrectness, Subway308) {
    auto path = data_dir("intermodal") + "/SUBWAY-308-0.txt.gz";
    if (!fs::exists(path))
        GTEST_SKIP() << "data/intermodal not found";
    auto opt = load_optimal(data_dir("intermodal"));
    auto inst = mcfcg::read_commalab(path);
    solve_and_check(inst, opt.at("SUBWAY-308-0"));
}

TEST(IntermodalCorrectness, Subway486) {
    auto path = data_dir("intermodal") + "/SUBWAY-486-0.txt.gz";
    if (!fs::exists(path))
        GTEST_SKIP() << "data/intermodal not found";
    auto opt = load_optimal(data_dir("intermodal"));
    auto inst = mcfcg::read_commalab(path);
    solve_and_check(inst, opt.at("SUBWAY-486-0"));
}

TEST(IntermodalCorrectness, DISABLED_Bus2632) {
    auto path = data_dir("intermodal") + "/BUS-2632-0.txt.gz";
    if (!fs::exists(path))
        GTEST_SKIP() << "data/intermodal not found";
    auto opt = load_optimal(data_dir("intermodal"));
    auto inst = mcfcg::read_commalab(path);
    solve_and_check(inst, opt.at("BUS-2632-0"));
}

TEST(IntermodalCorrectness, DISABLED_Bus7896) {
    auto path = data_dir("intermodal") + "/BUS-7896-0.txt.gz";
    if (!fs::exists(path))
        GTEST_SKIP() << "data/intermodal not found";
    auto opt = load_optimal(data_dir("intermodal"));
    auto inst = mcfcg::read_commalab(path);
    solve_and_check(inst, opt.at("BUS-7896-0"));
}

// --- Reduced cost validation on real instances ---

TEST(RCValidation, Grid1Path) {
    auto opt = load_optimal(data_dir("commalab/grid"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid1");
    solve_and_validate_path_rc(inst, opt.at("grid1"));
}

TEST(RCValidation, Grid1Tree) {
    auto opt = load_optimal(data_dir("commalab/grid"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid1");
    solve_and_validate_tree_rc(inst, opt.at("grid1"));
}

TEST(RCValidation, Planar30Path) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar30");
    solve_and_validate_path_rc(inst, opt.at("planar30"));
}

TEST(RCValidation, Planar30Tree) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar30");
    solve_and_validate_tree_rc(inst, opt.at("planar30"));
}

TEST(RCValidation, Planar80Path) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar80");
    solve_and_validate_path_rc(inst, opt.at("planar80"));
}

TEST(RCValidation, Planar80Tree) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar80");
    solve_and_validate_tree_rc(inst, opt.at("planar80"));
}

TEST(RCValidation, Planar100Path) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar100");
    solve_and_validate_path_rc(inst, opt.at("planar100"));
}

TEST(RCValidation, Planar100Tree) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar100");
    solve_and_validate_tree_rc(inst, opt.at("planar100"));
}

TEST(RCValidation, WinnipegPath) {
    auto net = data_dir("transportation") + "/Winnipeg_net.tntp.gz";
    auto trips = data_dir("transportation") + "/Winnipeg_trips.tntp.gz";
    if (!fs::exists(net))
        GTEST_SKIP() << "data/transportation not found";
    auto opt = load_optimal(data_dir("transportation"));
    auto inst = mcfcg::read_tntp(net, trips, 2000.0);
    solve_and_validate_path_rc(inst, opt.at("Winnipeg"));
}

TEST(RCValidation, WinnipegTree) {
    auto net = data_dir("transportation") + "/Winnipeg_net.tntp.gz";
    auto trips = data_dir("transportation") + "/Winnipeg_trips.tntp.gz";
    if (!fs::exists(net))
        GTEST_SKIP() << "data/transportation not found";
    auto opt = load_optimal(data_dir("transportation"));
    auto inst = mcfcg::read_tntp(net, trips, 2000.0);
    solve_and_validate_tree_rc(inst, opt.at("Winnipeg"));
}

// --- cuOpt GPU solver tests ---

#ifdef MCFCG_USE_CUOPT

#include "mcfcg/lp/lp_solver.h"

static void solve_and_check_cuopt(const mcfcg::Instance& inst, double ref_obj, double tol = 0.0001) {
    mcfcg::CGParams params;
    params.max_iterations = 10000;
    params.solver_factory = []() { return mcfcg::create_cuopt_solver(); };
    auto result = mcfcg::solve_path_cg(inst, params);
    EXPECT_TRUE(result.optimal) << "Did not reach optimality with cuOpt solver";
    EXPECT_GE(result.objective, ref_obj * (1.0 - tol)) << "Objective below reference";
    EXPECT_LE(result.objective, ref_obj * (1.0 + tol)) << "Objective above reference";
}

TEST(CuOptCorrectness, Grid1) {
    auto opt = load_optimal(data_dir("commalab/grid"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid1");
    solve_and_check_cuopt(inst, opt.at("grid1"));
}

TEST(CuOptCorrectness, Grid2) {
    auto opt = load_optimal(data_dir("commalab/grid"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid2");
    solve_and_check_cuopt(inst, opt.at("grid2"));
}

TEST(CuOptCorrectness, Planar30) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar30");
    solve_and_check_cuopt(inst, opt.at("planar30"));
}

#endif  // MCFCG_USE_CUOPT
