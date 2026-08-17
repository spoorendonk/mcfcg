#include "cg_test_util.h"
#include "mcfcg/cg/master.h"
#include "mcfcg/cg/master_base.h"
#include "mcfcg/cg/path_cg.h"
#include "mcfcg/cg/pricer.h"
#include "mcfcg/cg/tree_cg.h"
#include "mcfcg/cg/tree_master.h"
#include "mcfcg/cg/tree_pricer.h"
#include "mcfcg/graph/static_digraph_builder.h"
#include "mcfcg/instance.h"
#include "mcfcg/lp/lp_solver.h"
#include "mcfcg/source/source_lp.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <unordered_map>

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

static void solve_and_check(const mcfcg::Instance& inst, double ref_obj,
                            double tol = mcfcg::RELATIVE_FEAS_TOL * 10) {
    mcfcg::CGParams params;
    params.max_iterations = 10000;
    auto result = mcfcg::solve_path_cg(inst, params);
    EXPECT_TRUE(result.optimal) << "Did not reach optimality";
    EXPECT_GE(result.objective, ref_obj * (1.0 - tol)) << "Objective below reference";
    EXPECT_LE(result.objective, ref_obj * (1.0 + tol)) << "Objective above reference";
    // The π-free Lagrangian LB is provably ≤ OPT; assert it never exceeds the
    // reference (any violation is a bug in the bound).
    if (std::isfinite(result.lower_bound)) {
        EXPECT_LE(result.lower_bound, ref_obj * (1.0 + tol)) << "LB exceeds reference optimum";
    }
}

// Path formulation stalls on large intermodal graphs (slacks stay
// basic under HiGHS, LP stops improving around iter 400 with UB=inf),
// so these tests use the tree formulation.  PricerHeavy and
// PricerLight both work with tree — we default to PricerHeavy since
// it exercises the col-cap + partial-pricing + filter bundle that the
// production intermodal runs use.
//
// Tolerance is tighter than solve_and_check's: observed LP-solver
// noise on SUBWAY intermodal is < 1e-5 relative, so 2 * RELATIVE_FEAS_TOL
// (2e-4) still catches any reintroduction of the 0.25% translator bug
// without flaking on float jitter.
static void solve_intermodal_and_check(const mcfcg::Instance& inst, double ref_obj,
                                       double tol = mcfcg::RELATIVE_FEAS_TOL * 2) {
    mcfcg::CGParams params;
    params.max_iterations = 10000;
    params.strategy = mcfcg::CGStrategy::PricerHeavy;
    auto result = mcfcg::solve_tree_cg(inst, params);
    EXPECT_TRUE(result.optimal) << "Did not reach optimality";
    EXPECT_GE(result.objective, ref_obj * (1.0 - tol)) << "Objective below reference";
    EXPECT_LE(result.objective, ref_obj * (1.0 + tol)) << "Objective above reference";
    if (std::isfinite(result.lower_bound)) {
        EXPECT_LE(result.lower_bound, ref_obj * (1.0 + tol)) << "LB exceeds reference optimum";
    }
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

// Winnipeg is the one shipped instance that triggers EdgeRows on path
// (2836 arcs < 4345 commodities).  Lock that in so a future TNTP
// reader change that flips the selector shows up as a test failure
// rather than silently demoting Winnipeg back to CommodityRows and
// leaving EdgeRows untested end-to-end.
TEST(TransportationCorrectness, WinnipegPathPicksEdgeRows) {
    auto net = data_dir("transportation") + "/Winnipeg_net.tntp.gz";
    auto trips = data_dir("transportation") + "/Winnipeg_trips.tntp.gz";
    if (!fs::exists(net))
        GTEST_SKIP() << "data/transportation not found";
    auto inst = mcfcg::read_tntp(net, trips, 2000.0);
    mcfcg::PathMaster master;
    master.init(inst);
    EXPECT_EQ(master.slack_mode(), mcfcg::SlackMode::EdgeRows);

    mcfcg::TreeMaster tree_master;
    tree_master.init(inst);
    EXPECT_EQ(tree_master.slack_mode(), mcfcg::SlackMode::CommodityRows);
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
    solve_intermodal_and_check(inst, opt.at("SUBWAY-308-0"));
}

TEST(IntermodalCorrectness, Subway486) {
    auto path = data_dir("intermodal") + "/SUBWAY-486-0.txt.gz";
    if (!fs::exists(path))
        GTEST_SKIP() << "data/intermodal not found";
    auto opt = load_optimal(data_dir("intermodal"));
    auto inst = mcfcg::read_commalab(path);
    solve_intermodal_and_check(inst, opt.at("SUBWAY-486-0"));
}

TEST(IntermodalCorrectness, Bus2632) {
    auto path = data_dir("intermodal") + "/BUS-2632-0.txt.gz";
    if (!fs::exists(path))
        GTEST_SKIP() << "data/intermodal not found";
    auto opt = load_optimal(data_dir("intermodal"));
    auto inst = mcfcg::read_commalab(path);
    solve_intermodal_and_check(inst, opt.at("BUS-2632-0"));
}

TEST(IntermodalCorrectness, Bus7896) {
    auto path = data_dir("intermodal") + "/BUS-7896-0.txt.gz";
    if (!fs::exists(path))
        GTEST_SKIP() << "data/intermodal not found";
    auto opt = load_optimal(data_dir("intermodal"));
    auto inst = mcfcg::read_commalab(path);
    solve_intermodal_and_check(inst, opt.at("BUS-7896-0"));
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

// --- Threaded execution: parallel paths must reach the same objective ---
//
// The default num_threads=1 sends a nullptr pool to the master and pricer,
// so all the parallel branches in master_base.h / pricer_base.h are
// dead-code in the rest of the integration suite.  These tests force a
// real pool by setting num_threads>1 and check that the solver still
// converges to the reference objective.  They also catch any FP
// non-determinism that flips cuts at the +1e-6 capacity threshold.

template <typename SolveFn>
static void solve_threaded(const mcfcg::Instance& inst, double ref_obj, SolveFn solve_fn,
                           uint32_t num_threads, double tol = mcfcg::RELATIVE_FEAS_TOL * 10) {
    mcfcg::CGParams params;
    params.max_iterations = 10000;
    params.num_threads = num_threads;
    auto result = solve_fn(inst, params);
    EXPECT_TRUE(result.optimal) << "Did not reach optimality with " << num_threads << " threads";
    EXPECT_GE(result.objective, ref_obj * (1.0 - tol));
    EXPECT_LE(result.objective, ref_obj * (1.0 + tol));
}

TEST(ThreadedExecution, Planar80Path) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar80");
    solve_threaded(inst, opt.at("planar80"), mcfcg::solve_path_cg, 4);
}

TEST(ThreadedExecution, Planar80Tree) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar80");
    solve_threaded(inst, opt.at("planar80"), mcfcg::solve_tree_cg, 4);
}

TEST(ThreadedExecution, Grid2Path) {
    auto opt = load_optimal(data_dir("commalab/grid"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid2");
    solve_threaded(inst, opt.at("grid2"), mcfcg::solve_path_cg, 4);
}

TEST(ThreadedExecution, Grid2Tree) {
    auto opt = load_optimal(data_dir("commalab/grid"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid2");
    solve_threaded(inst, opt.at("grid2"), mcfcg::solve_tree_cg, 4);
}

// Winnipeg has ~80k arcs which clears PAR_ARC_THRESHOLD (4096), so
// this is the only test that exercises the arc-scale parallel branches
// in compute_rc, find_violated_arcs, and the compute_arc_flow merge.
TEST(ThreadedExecution, WinnipegPath) {
    auto net = data_dir("transportation") + "/Winnipeg_net.tntp.gz";
    auto trips = data_dir("transportation") + "/Winnipeg_trips.tntp.gz";
    if (!fs::exists(net))
        GTEST_SKIP() << "data/transportation not found";
    auto opt = load_optimal(data_dir("transportation"));
    auto inst = mcfcg::read_tntp(net, trips, 2000.0);
    solve_threaded(inst, opt.at("Winnipeg"), mcfcg::solve_path_cg, 4);
}

// --- Feature tests: strategy bundle, pricing_filter, unreachable-sink handling ---

// Unreachable source→sink must not crash or corrupt the pricer.  The path
// pricer skips the affected commodity and emits columns for the reachable
// ones; the tree pricer builds a partial tree over the reachable sinks.
// This regression test locks in the behavior contract from commit c20b757
// by driving the pricers directly (the full CG loop is harder to observe
// because CommodityRows slacks eventually saturate at the cost ceiling and
// the loop runs to max_iterations on a truly disconnected instance).
namespace unreachable_test {
// 4 vertices.  Vertex 3 is isolated — no incident arcs.
// Source 0 has two commodities: 0→2 (reachable) and 0→3 (unreachable).
static mcfcg::Instance build_disconnected() {
    mcfcg::static_digraph_builder<double, double> builder(4);
    builder.add_arc(0, 1, 1.0, 10.0);  // 0→1
    builder.add_arc(1, 2, 2.0, 10.0);  // 1→2
    builder.add_arc(0, 2, 5.0, 10.0);  // 0→2
    auto [graph, cost_map, cap_map] = builder.build();

    std::vector<mcfcg::Commodity> commodities = {
        {0, 2, 1.0},  // reachable
        {0, 3, 1.0},  // unreachable — sink 3 has no in-arcs
    };
    auto sources = mcfcg::group_by_source(commodities);
    return mcfcg::Instance{std::move(graph), std::move(cost_map), std::move(cap_map),
                           std::move(commodities), std::move(sources)};
}
}  // namespace unreachable_test

TEST(FeatureTests, PathPricerSkipsUnreachableSink) {
    auto inst = unreachable_test::build_disconnected();

    mcfcg::PathPricer pricer;
    pricer.init(inst);

    // Price with zero duals: reachable commodity has RC < 0 (cost > 0 −
    // 0 dual = cost, but -pi[k] term with pi=0 leaves true_rc = cost);
    // to actually trigger negative-RC column emission we prime the
    // commodity's pi to a large-enough value.
    std::vector<double> pi(inst.commodities.size(), 100.0);
    auto mu = inst.graph.create_arc_map<double>(0.0);

    auto cols = pricer.price(pi, mu);
    // One column for the reachable commodity; none for the unreachable one.
    ASSERT_EQ(cols.size(), 1u);
    EXPECT_EQ(cols[0].commodity, 0u);  // reachable commodity index
    EXPECT_FALSE(cols[0].arcs.empty());
}

TEST(FeatureTests, TreePricerEmitsPartialTreeOnUnreachableSink) {
    auto inst = unreachable_test::build_disconnected();

    mcfcg::TreePricer pricer;
    pricer.init(inst);

    std::vector<double> pi_s(inst.sources.size(), 100.0);
    auto mu = inst.graph.create_arc_map<double>(0.0);

    auto cols = pricer.price(pi_s, mu);
    // Exactly one partial-tree column for source 0, covering only the
    // reachable sink (vertex 2).  The unreachable sink (3) contributes
    // no arc flow.
    ASSERT_EQ(cols.size(), 1u);
    EXPECT_EQ(cols[0].source_idx, 0u);
    EXPECT_FALSE(cols[0].arc_flows.empty());
    // All arc flows must be on arcs reachable from source 0 — none of
    // them terminate at vertex 3 (which has no in-arcs anyway).
    for (const auto& af : cols[0].arc_flows) {
        EXPECT_NE(inst.graph.arc_target(af.arc), 3u);
    }
}

// Verify PricerHeavy strategy produces same optimal objective.
TEST(FeatureTests, PricerHeavyPath) {
    auto opt = load_optimal(data_dir("commalab/grid"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid1");
    mcfcg::CGParams params;
    params.strategy = mcfcg::CGStrategy::PricerHeavy;
    auto result = mcfcg::solve_path_cg(inst, params);
    EXPECT_TRUE(result.optimal);
    solve_and_check(inst, opt.at("grid1"));
}

TEST(FeatureTests, PricerHeavyTree) {
    auto opt = load_optimal(data_dir("commalab/grid"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid1");
    mcfcg::CGParams params;
    params.strategy = mcfcg::CGStrategy::PricerHeavy;
    auto result = mcfcg::solve_tree_cg(inst, params);
    EXPECT_TRUE(result.optimal);
    EXPECT_GE(result.objective, opt.at("grid1") * (1.0 - 0.0001));
    EXPECT_LE(result.objective, opt.at("grid1") * (1.0 + 0.0001));
}

// Partial pricing regression guard.  If the col-cap early break fires
// inside pricer.price() (batch_size < n_sources and enough negative-RC
// cols found), the round-robin cursor must park mid-sweep so the NEXT
// price() call resumes from there.  Two failure modes this catches:
// (1) a misscaled batch_size that never triggers early break, and
// (2) reset_postponed() on the success path wiping the cursor to 0.
TEST(FeatureTests, PartialPricingParksCursor) {
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar30");
    ASSERT_GT(inst.sources.size(), 4U) << "need enough sources for partial pricing";

    mcfcg::PathPricer pricer;
    // batch_size=2 forces a col-cap check every 2 sources; max_cols=1
    // fires the early break as soon as the first batch produces a col.
    pricer.init(inst, /*pool=*/nullptr, /*batch_size=*/2, mcfcg::NEG_RC_TOL);

    // Pi = +INF makes every commodity's shortest path have very
    // negative reduced cost, so every source emits a column.
    std::vector<double> pi(inst.commodities.size(), std::numeric_limits<double>::infinity());
    auto mu = inst.graph.create_arc_map<double>(0.0);

    auto cols1 = pricer.price(pi, mu, /*final_round=*/false, /*max_cols=*/1);
    EXPECT_FALSE(cols1.empty()) << "pricer must produce at least one col with +INF duals";
    const uint32_t cursor1 = pricer.last_source_idx();
    // A mid-sweep break lands at 0 < cursor < n_sources; the modulo wrap
    // (finishing the sweep) lands back at 0 and is caught by EXPECT_GT.
    EXPECT_GT(cursor1, 0U) << "cursor did not advance from 0 — early break did not park mid-sweep";
    EXPECT_FALSE(pricer.priced_all()) << "priced_all should be false after early break";

    auto cols2 = pricer.price(pi, mu, /*final_round=*/false, /*max_cols=*/1);
    EXPECT_FALSE(cols2.empty());
    EXPECT_NE(pricer.last_source_idx(), cursor1)
        << "cursor did not advance on the second price() call";
    const uint32_t cursor2 = pricer.last_source_idx();

    // clear_postponed must preserve the cursor (this is the fix — the
    // main CG loop calls it on the success path).  reset_postponed
    // wipes the cursor and is reserved for warm-start / pricing-exhausted.
    pricer.clear_postponed();
    EXPECT_EQ(pricer.last_source_idx(), cursor2)
        << "clear_postponed() wiped the cursor — partial pricing would be inert";

    pricer.reset_postponed();
    EXPECT_EQ(pricer.last_source_idx(), 0U) << "reset_postponed() must rewind the cursor to 0";
}

// Partial-pricing batch-size formula (compute_partial_pricing_batch_size).
// Kept as a pure function so it's testable without an instance or pool.
TEST(FeatureTests, PartialPricingBatchSizeFormula) {
    using mcfcg::compute_partial_pricing_batch_size;

    // Explicit caller setting always wins.
    EXPECT_EQ(compute_partial_pricing_batch_size(50U, true, 32U, 1000U), 50U);
    EXPECT_EQ(compute_partial_pricing_batch_size(50U, false, 32U, 1000U), 50U);
    EXPECT_EQ(compute_partial_pricing_batch_size(1U, true, 32U, 4U), 1U);

    // PricerLight: always 0 (one big batch).
    EXPECT_EQ(compute_partial_pricing_batch_size(0U, false, 32U, 1000U), 0U);
    EXPECT_EQ(compute_partial_pricing_batch_size(0U, false, 1U, 10U), 0U);

    // PricerHeavy + small instance (n_sources <= pool_threads): 0.
    // Partial pricing can't fire so we don't pretend it does.
    EXPECT_EQ(compute_partial_pricing_batch_size(0U, true, 32U, 10U), 0U);
    EXPECT_EQ(compute_partial_pricing_batch_size(0U, true, 32U, 32U), 0U);
    EXPECT_EQ(compute_partial_pricing_batch_size(0U, true, 8U, 0U), 0U);

    // PricerHeavy + larger instance: max(pool_threads, n_sources/4).
    // pool_threads floor case (n_sources/4 < threads):
    EXPECT_EQ(compute_partial_pricing_batch_size(0U, true, 32U, 100U), 32U);
    // n_sources/4 dominates:
    EXPECT_EQ(compute_partial_pricing_batch_size(0U, true, 32U, 1000U), 250U);
    // Single-threaded (pool=nullptr → pool_threads=1): n_sources/4 rules.
    EXPECT_EQ(compute_partial_pricing_batch_size(0U, true, 1U, 100U), 25U);

    // Boundary: n_sources == 4 * pool_threads — both sides equal.
    EXPECT_EQ(compute_partial_pricing_batch_size(0U, true, 32U, 128U), 32U);
}

// LB-tracking invariant: a max_cols break that fires exactly on sweep
// completion must leave priced_all=true, otherwise Lagrangian/Farley LB
// tracking silently stops firing in the precise iterations where tree
// PricerHeavy hits its col cap (num_entities = n_sources, one col per
// source, cap triggered at batch end).  Guards the !early_break drop.
TEST(FeatureTests, PricerPricedAllSurvivesSweepCompletingBreak) {
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid1");
    const auto n_sources = static_cast<uint32_t>(inst.sources.size());

    mcfcg::TreePricer pricer;
    pricer.init(inst, /*pool=*/nullptr, /*batch_size=*/1, mcfcg::NEG_RC_TOL);

    // Large-but-finite dual per source so every tree has negative RC and
    // emits exactly one column.  +INF would also work but produces
    // degenerate rc_error bounds in the pricer.
    std::vector<double> pi_s(n_sources, 1e6);
    auto mu = inst.graph.create_arc_map<double>(0.0);

    // max_cols == n_sources: with batch=1 and one col per source, the
    // break fires on the last batch, after priced_count reached n_sources.
    auto cols = pricer.price(pi_s, mu, /*final_round=*/false, /*max_cols=*/n_sources);
    EXPECT_EQ(cols.size(), n_sources) << "tree pricer should emit one column per source";
    EXPECT_TRUE(pricer.priced_all())
        << "priced_all must remain true when priced_count == n_sources, even if the "
           "col-cap break fired on the final batch";
}

// End-to-end LB tracking under PricerHeavy.  planar80 has 80 sources;
// num_threads=4 forces partial pricing to engage (n_sources/4 = 20 > 4),
// independent of host hw_concurrency (an 80+-thread box would otherwise
// route through the small-instance single-batch branch and hide a
// partial-pricing regression).  Without the priced_all fix, LB tracking
// would be disabled in every iteration that hit the col cap, leaving
// result.lower_bound at -INF on convergent runs.
TEST(FeatureTests, PricerHeavyLagrangianBound) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar80");
    mcfcg::CGParams params;
    params.strategy = mcfcg::CGStrategy::PricerHeavy;
    params.num_threads = 4;
    auto result = mcfcg::solve_path_cg(inst, params);
    EXPECT_TRUE(result.optimal);
    EXPECT_LT(result.iterations, params.max_iterations);
    EXPECT_GT(result.lower_bound, -mcfcg::INF) << "LB tracking never fired under PricerHeavy";
    EXPECT_LE(result.lower_bound, result.objective + 1e-6) << "LB cannot exceed UB";
    double ref = opt.at("planar80");
    EXPECT_GE(result.objective, ref * (1.0 - 1e-4));
    EXPECT_LE(result.objective, ref * (1.0 + 1e-4));
}

// Solve planar150 under both formulations and check the reported
// objective is within RELATIVE_FEAS_TOL of the reference.  planar150
// is small enough to run fast but big enough that LB tracking and
// gap-based early termination both get exercised; reducing the LB
// (e.g. forgetting demand weighting or dropping the dual obj) will
// surface here as either a wrong objective or a non-optimal exit.
TEST(FeatureTests, LagrangianBoundPath) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar150");
    mcfcg::CGParams params;
    auto result = mcfcg::solve_path_cg(inst, params);
    EXPECT_TRUE(result.optimal);
    double ref = opt.at("planar150");
    double rel = std::abs(result.objective - ref) / std::max(1.0, std::abs(ref));
    EXPECT_LT(rel, mcfcg::RELATIVE_FEAS_TOL) << "obj=" << result.objective << " ref=" << ref;
}

TEST(FeatureTests, LagrangianBoundTree) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar150");
    mcfcg::CGParams params;
    auto result = mcfcg::solve_tree_cg(inst, params);
    EXPECT_TRUE(result.optimal);
    double ref = opt.at("planar150");
    double rel = std::abs(result.objective - ref) / std::max(1.0, std::abs(ref));
    EXPECT_LT(rel, mcfcg::RELATIVE_FEAS_TOL) << "obj=" << result.objective << " ref=" << ref;
}

// --- Dual pricing cutoff (CGParams::pricing_cutoff, gh #41) ---

namespace cutoff_test {
// 5 vertices.  Sink 3 is isolated (unreachable from the source) and vertex 4 is
// a *dead end*: reachable from the source but with no path to any sink, so
// compute_lower_bounds_to_targets gives it h = UNREACHED = MAX_BOUND.  That
// combination is what puts a saturated key on the A* frontier while a target is
// still pending — unreachable_test::build_disconnected cannot, because its
// heap empties first.
static mcfcg::Instance build_deadend() {
    mcfcg::static_digraph_builder<double, double> builder(5);
    builder.add_arc(0, 1, 1.0, 10.0);  // 0→1
    builder.add_arc(1, 2, 2.0, 10.0);  // 1→2  (2 is the reachable sink)
    builder.add_arc(0, 2, 5.0, 10.0);  // 0→2
    builder.add_arc(0, 4, 1.0, 10.0);  // 0→4  (4 has no out-arcs: dead end)
    auto [graph, cost_map, cap_map] = builder.build();

    std::vector<mcfcg::Commodity> commodities = {
        {0, 2, 1.0},  // reachable
        {0, 3, 1.0},  // unreachable — sink 3 has no incident arcs
    };
    auto sources = mcfcg::group_by_source(commodities);
    return mcfcg::Instance{std::move(graph), std::move(cost_map), std::move(cap_map),
                           std::move(commodities), std::move(sources)};
}

// 4-vertex chain 0→1→2→3.  The near sinks carry all the demand and the far sink
// none, so the demand-weighted budget is fully "spent" while a target is still
// unsettled.
//
// The demands are deliberately fractional and deliberately three.  With a single
// unit demand, Σd minus the settled demands lands on exactly +0.0, the division
// in TreePricer::Cutoff::recompute yields +inf, and the MAX_BOUND clamp then
// reproduces precisely what the `_rem_demand > 0` guard does — so the test
// passes with that guard deleted and pins nothing.  Fractional demands (the norm
// on TNTP) leave a small NEGATIVE residue instead, the division yields -inf, and
// its int64_t cast is UB (INT64_MIN in practice), so every frontier cuts and the
// improving column is lost.  That is the bug the guard exists for.
static mcfcg::Instance build_zero_demand() {
    mcfcg::static_digraph_builder<double, double> builder(4);
    builder.add_arc(0, 1, 1.0, 10.0);
    builder.add_arc(1, 2, 1.0, 10.0);
    builder.add_arc(2, 3, 1.0, 10.0);
    auto [graph, cost_map, cap_map] = builder.build();

    std::vector<mcfcg::Commodity> commodities = {
        {0, 1, 8.446},  // near sinks carry all the demand; the fractional values
        {0, 2, 7.582},  // make the residue land just *below* zero, not at +0.0
        {0, 3, 0.0},    // far sink, zero demand (CommaLab keeps these verbatim)
    };
    auto sources = mcfcg::group_by_source(commodities);
    return mcfcg::Instance{std::move(graph), std::move(cost_map), std::move(cap_map),
                           std::move(commodities), std::move(sources)};
}

// Two independent source→sink pairs plus a spare arc 0→5 that lies on no
// shortest path to any sink, so a capacity row on it touches neither source's
// recorded arc set.  Vertex 5 is nobody's sink, so it is a dead end.
static mcfcg::Instance build_two_sources_and_spare_arc() {
    mcfcg::static_digraph_builder<double, double> builder(6);
    builder.add_arc(0, 1, 1.0, 10.0);
    builder.add_arc(1, 2, 1.0, 10.0);
    builder.add_arc(3, 4, 1.0, 10.0);
    builder.add_arc(0, 5, 1.0, 10.0);  // the spare
    auto [graph, cost_map, cap_map] = builder.build();

    std::vector<mcfcg::Commodity> commodities = {
        {0, 2, 1.0},
        {3, 4, 1.0},
    };
    auto sources = mcfcg::group_by_source(commodities);
    return mcfcg::Instance{std::move(graph), std::move(cost_map), std::move(cap_map),
                           std::move(commodities), std::move(sources)};
}

// Match key for a column: each commodity (path) / source (tree) yields at most
// one column per sweep, so the key identifies a column within one price() call.
inline uint64_t column_key(const mcfcg::Column& col) {
    return col.commodity;
}
inline uint64_t column_key(const mcfcg::TreeColumn& col) {
    return col.source_idx;
}

// Bit-equality, not EXPECT_NEAR.  A cutoff run settles its sinks in the same
// order as a full run — run_until_targets is the same settle_next loop as the
// cutoff driver, minus the frontier checks — so the two extract the same path
// and accumulate the same floats in the same order.  Anything but bit-equality
// means the cutoff changed the answer.  Returns "" when the columns agree,
// otherwise a description of the first field that differs.
inline std::string column_diff(const mcfcg::Column& base, const mcfcg::Column& cut) {
    std::ostringstream os;
    os.precision(17);
    if (base.cost != cut.cost) {
        os << " cost " << base.cost << " vs " << cut.cost;
    }
    if (base.reduced_cost != cut.reduced_cost) {
        os << " rc " << base.reduced_cost << " vs " << cut.reduced_cost;
    }
    if (base.arcs != cut.arcs) {
        os << " arcs differ (" << base.arcs.size() << " vs " << cut.arcs.size() << ")";
    }
    return os.str();
}

inline std::string column_diff(const mcfcg::TreeColumn& base, const mcfcg::TreeColumn& cut) {
    std::ostringstream os;
    os.precision(17);
    if (base.cost != cut.cost) {
        os << " cost " << base.cost << " vs " << cut.cost;
    }
    if (base.reduced_cost != cut.reduced_cost) {
        os << " rc " << base.reduced_cost << " vs " << cut.reduced_cost;
    }
    // arc_flows is dumped from an unordered_map, so its order is not part of
    // the contract even though both pricers build the map identically.
    auto by_arc = [](std::vector<mcfcg::TreeColumn::ArcFlow> flows) {
        std::sort(flows.begin(), flows.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.arc < rhs.arc; });
        return flows;
    };
    auto base_flows = by_arc(base.arc_flows);
    auto cut_flows = by_arc(cut.arc_flows);
    if (base_flows.size() != cut_flows.size()) {
        os << " arc_flows " << base_flows.size() << " vs " << cut_flows.size();
        return os.str();
    }
    for (size_t i = 0; i < base_flows.size(); ++i) {
        if (base_flows[i].arc != cut_flows[i].arc || base_flows[i].flow != cut_flows[i].flow) {
            os << " arc_flow[" << i << "] " << base_flows[i].arc << ":" << base_flows[i].flow
               << " vs " << cut_flows[i].arc << ":" << cut_flows[i].flow;
            break;
        }
    }
    return os.str();
}
}  // namespace cutoff_test

// A frontier saturated at MAX_BOUND is not a dual proof — it means the search
// ran out of vertices that can reach any sink.  Treating it as a cutoff made
// salvage_lagr_term contribute ~MAX_BOUND/SCALE per unsettled sink (≈4.6e9),
// which latches into the monotone best_lb and is reported as the objective via
// the #35 fallback; it also made the tree pricer suppress the partial column
// the non-cutoff path emits for a disconnected source.  Both are silent wrong
// answers on an instance class the pricer documents as degraded-but-working.
TEST(FeatureTests, PricingCutoffIgnoresDeadEndFrontier) {
    auto inst = cutoff_test::build_deadend();
    mcfcg::TreePricer base_pricer;
    mcfcg::TreePricer cut_pricer;
    base_pricer.init(inst, nullptr, 0, mcfcg::NEG_RC_TOL, /*dual_cutoff=*/false);
    cut_pricer.init(inst, nullptr, 0, mcfcg::NEG_RC_TOL, /*dual_cutoff=*/true);

    std::vector<double> pi_s(inst.sources.size(), 50.0);
    auto mu = inst.graph.create_arc_map<double>(0.0);
    auto base_cols = base_pricer.price(pi_s, mu, true);
    auto cut_cols = cut_pricer.price(pi_s, mu, true);

    EXPECT_EQ(cut_pricer.last_cutoff_count(), 0U)
        << "a dead-end frontier must not be recorded as a cutoff";
    ASSERT_EQ(base_cols.size(), cut_cols.size()) << "partial tree suppressed by the cutoff";
    EXPECT_NEAR(base_pricer.lagrangian_path_sum(), cut_pricer.lagrangian_path_sum(), 1e-9)
        << "unreachable sink salvaged a saturated pseudo-distance into the LB";
}

// A zero-demand commodity drives the tree budget's remaining demand to 0 while
// the budget itself is untouched.  Cutting there proves nothing — the unsettled
// sinks contribute 0 to the tree reduced cost — and suppressed a strictly
// improving column on every iteration including the final_round retry, which
// the CG loop then reports as optimality.
TEST(FeatureTests, PricingCutoffKeepsColumnWithZeroDemandCommodity) {
    auto inst = cutoff_test::build_zero_demand();
    mcfcg::TreePricer base_pricer;
    mcfcg::TreePricer cut_pricer;
    base_pricer.init(inst, nullptr, 0, mcfcg::NEG_RC_TOL, /*dual_cutoff=*/false);
    cut_pricer.init(inst, nullptr, 0, mcfcg::NEG_RC_TOL, /*dual_cutoff=*/true);

    std::vector<double> pi_s(inst.sources.size(), 1e6);
    auto mu = inst.graph.create_arc_map<double>(0.0);
    auto base_cols = base_pricer.price(pi_s, mu, true);
    auto cut_cols = cut_pricer.price(pi_s, mu, true);

    ASSERT_EQ(base_cols.size(), 1U) << "baseline must find this obviously attractive tree";
    ASSERT_EQ(cut_cols.size(), base_cols.size()) << "cutoff dropped an improving column";
    EXPECT_NEAR(base_cols[0].reduced_cost, cut_cols[0].reduced_cost, 1e-9);
}

// A cut source keeps the arc set from its last complete price, so the source
// pricing filter answers "was any arc I route over just capacitated?" from an
// older routing.  Pinned rather than fixed: this is why switching the cutoff on
// moves the CG trajectory (see the Shadow tests for the invariant that DOES
// hold), and treating a cut source as affected instead costs +31% wall clock on
// intermodal — see should_record_arcs.
TEST(FeatureTests, PricingCutoffFilterUsesStaleArcsForCutSources) {
    auto inst = cutoff_test::build_two_sources_and_spare_arc();
    ASSERT_EQ(inst.sources.size(), 2U);
    // The spare arc lies on no source's routing; the first arc out of vertex 0
    // toward vertex 1 lies on source 0's.
    uint32_t spare = UINT32_MAX;
    uint32_t routed = UINT32_MAX;
    for (uint32_t arc : inst.graph.out_arcs(0)) {
        if (inst.graph.arc_target(arc) == 5) {
            spare = arc;
        } else {
            routed = arc;
        }
    }
    ASSERT_NE(spare, UINT32_MAX);
    ASSERT_NE(routed, UINT32_MAX);

    mcfcg::TreePricer pricer;
    pricer.init(inst, nullptr, 0, mcfcg::NEG_RC_TOL, /*dual_cutoff=*/true);
    pricer.set_track_arcs(true);
    auto mu = inst.graph.create_arc_map<double>(0.0);

    // Seed: +inf duals leave the cutoff inert, so both sources record a
    // complete arc set — neither of which contains the spare arc.
    std::vector<double> seed(inst.sources.size(), std::numeric_limits<double>::infinity());
    (void)pricer.price(seed, mu, true);
    ASSERT_EQ(pricer.last_cutoff_count(), 0U);

    // Source 0's convexity dual of 0 spends its budget before anything is
    // settled, so it is cut and never refreshes its arcs.  Source 1's budget is
    // large enough to run to completion.
    std::vector<double> pi_s = {0.0, 1e6};
    (void)pricer.price(pi_s, mu, true);
    ASSERT_EQ(pricer.last_cutoff_count(), 1U) << "test setup did not cut exactly one source";

    // Capacitating an arc on nobody's routing postpones everyone, the cut
    // source included: its stale set is consulted exactly like a fresh one.
    pricer.filter_for_new_caps({spare});
    (void)pricer.price(pi_s, mu, false);
    EXPECT_EQ(pricer.last_priced_count(), 0U) << "a stale arc set still postpones";

    // And the stale set is the *seed* routing, not an empty or partial one:
    // capacitating an arc it contains brings the cut source back.
    pricer.filter_for_new_caps({routed});
    (void)pricer.price(pi_s, mu, false);
    EXPECT_EQ(pricer.last_priced_count(), 1U) << "the retained arc set is the complete one";
}

// The cutoff is a proof that a source has no negative-reduced-cost column, so
// switching it on must not change which columns the pricer emits.  Drives a
// manual tree CG loop and prices every iteration twice — once with the cutoff,
// once without — off the same duals, asserting the two column sets agree.  A
// cutoff bound that is too aggressive (e.g. dividing the tree budget by the
// wrong demand aggregate) shows up here as a dropped column, which end-to-end
// objective checks would only catch if it happened to change the optimum.
TEST(FeatureTests, PricingCutoffEmitsSameColumns) {
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar30");

    mcfcg::TreeMaster master;
    master.init(inst);
    mcfcg::TreePricer base_pricer;
    mcfcg::TreePricer cut_pricer;
    base_pricer.init(inst, nullptr, 0, mcfcg::NEG_RC_TOL, /*dual_cutoff=*/false);
    cut_pricer.init(inst, nullptr, 0, mcfcg::NEG_RC_TOL, /*dual_cutoff=*/true);

    std::vector<double> big_pi(inst.sources.size(), std::numeric_limits<double>::infinity());
    auto empty_mu = inst.graph.create_arc_map<double>(0.0);
    auto init_cols = base_pricer.price(big_pi, empty_mu, true);
    // The warm start's +inf duals must leave the cutoff inert, otherwise the
    // seeding pass would explore less than the full reachable graph.
    (void)cut_pricer.price(big_pi, empty_mu, true);
    EXPECT_EQ(cut_pricer.last_cutoff_count(), 0U) << "+inf duals must disable the cutoff";
    ASSERT_FALSE(init_cols.empty());
    master.add_columns(std::move(init_cols));

    uint64_t total_cut = 0;
    for (uint32_t iter = 0; iter < 25; ++iter) {
        ASSERT_EQ(master.solve(), mcfcg::LPStatus::Optimal) << "LP not optimal at iter " << iter;
        auto primals = master.get_primals();
        auto pi_s = master.get_structural_duals();
        const auto& mu = master.get_capacity_duals();
        (void)master.add_violated_capacity_constraints(primals);

        // final_round on both so postponement can never make the two pricers
        // visit different sources — the comparison is about the cutoff alone.
        auto base_cols = base_pricer.price(pi_s, mu, true);
        auto cut_cols = cut_pricer.price(pi_s, mu, true);
        total_cut += cut_pricer.last_cutoff_count();

        ASSERT_EQ(base_cols.size(), cut_cols.size()) << "column count differs at iter " << iter;
        for (size_t i = 0; i < base_cols.size(); ++i) {
            ASSERT_EQ(base_cols[i].source_idx, cut_cols[i].source_idx) << "at iter " << iter;
            EXPECT_EQ(cutoff_test::column_diff(base_cols[i], cut_cols[i]), "")
                << "at iter " << iter;
        }

        if (base_cols.empty()) {
            break;
        }
        (void)master.bump_active_slacks(primals, mcfcg::SLACK_BUMP_FACTOR);
        master.add_columns(std::move(base_cols));
    }
    // A pass where the cutoff never fired would satisfy every assertion above
    // vacuously.
    EXPECT_GT(total_cut, 0U) << "cutoff never fired; the comparison proved nothing";
}

// The ablation in results/ablation/ concluded "ship it off", but nothing pinned
// that: every other PricingCutoff* test sets the flag explicitly, so a flipped
// default would leave the whole suite green while silently changing what every
// committed benchmark cell in results/cg_benchmark.csv means — and that CSV has
// no extra_args column to record it (PROVENANCE.txt section 5.1).
TEST(FeatureTests, PricingCutoffIsOffByDefault) {
    EXPECT_FALSE(mcfcg::CGParams{}.pricing_cutoff);
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid1");
    auto result = mcfcg::solve_tree_cg(inst, mcfcg::CGParams{});
    EXPECT_TRUE(result.optimal);
    EXPECT_EQ(result.cutoff_sources, 0U) << "a default run fired the pricing cutoff";
}

// End-to-end: the cutoff must not move the optimum or cost optimality, on both
// formulations and on the family it is actually meant for (intermodal, tree +
// PricerHeavy — where a cut source must suppress its whole tree column rather
// than emit a partial one).
TEST(FeatureTests, PricingCutoffPathMatchesReference) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar150");
    mcfcg::CGParams params;
    params.pricing_cutoff = true;
    auto result = mcfcg::solve_path_cg(inst, params);
    EXPECT_TRUE(result.optimal);
    double ref = opt.at("planar150");
    double rel = std::abs(result.objective - ref) / std::max(1.0, std::abs(ref));
    EXPECT_LT(rel, mcfcg::RELATIVE_FEAS_TOL) << "obj=" << result.objective << " ref=" << ref;
    EXPECT_LE(result.lower_bound, ref * (1.0 + mcfcg::RELATIVE_FEAS_TOL))
        << "salvaged LB exceeds OPT";
    // Without this the test is vacuous: an inert cutoff trivially reproduces
    // the reference.  Measured fire rate on planar150 path is ~19%.
    EXPECT_GT(result.cutoff_sources, 0U) << "cutoff never fired";
}

TEST(FeatureTests, PricingCutoffTreeMatchesReference) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar150");
    mcfcg::CGParams params;
    params.pricing_cutoff = true;
    auto result = mcfcg::solve_tree_cg(inst, params);
    EXPECT_TRUE(result.optimal);
    double ref = opt.at("planar150");
    double rel = std::abs(result.objective - ref) / std::max(1.0, std::abs(ref));
    EXPECT_LT(rel, mcfcg::RELATIVE_FEAS_TOL) << "obj=" << result.objective << " ref=" << ref;
    EXPECT_LE(result.lower_bound, ref * (1.0 + mcfcg::RELATIVE_FEAS_TOL))
        << "salvaged LB exceeds OPT";
    // Vacuity guard, as above.  Measured fire rate on planar150 tree is ~12%.
    EXPECT_GT(result.cutoff_sources, 0U) << "cutoff never fired";
}

TEST(FeatureTests, PricingCutoffIntermodalTree) {
    auto path = data_dir("intermodal") + "/BUS-2632-0.txt.gz";
    if (!fs::exists(path)) {
        GTEST_SKIP() << "data/intermodal not found";
    }
    auto opt = load_optimal(data_dir("intermodal"));
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.strategy = mcfcg::CGStrategy::PricerHeavy;
    params.pricing_cutoff = true;
    auto result = mcfcg::solve_tree_cg(inst, params);
    double ref = opt.at("BUS-2632-0");
    double tol = mcfcg::RELATIVE_FEAS_TOL * 2;
    EXPECT_TRUE(result.optimal) << "Did not reach optimality";
    EXPECT_GE(result.objective, ref * (1.0 - tol)) << "Objective below reference";
    EXPECT_LE(result.objective, ref * (1.0 + tol)) << "Objective above reference";
    EXPECT_GT(result.cutoff_sources, 0U) << "cutoff never fired on intermodal";
}

// --- Shadow pricing: the cutoff must emit identical columns on the dual
// trajectory a real solve visits, not just on a hand-rolled one ---

namespace cutoff_test {

// Drives the real CG loop with the cutoff OFF while shadowing every dual vector
// the loop produces with a second, cutoff-ON pricer, and checks the two emit
// the same columns down to the arc lists.  solve_cg is a template on Pricer and
// touches only the methods forwarded below, so this needs no production change.
//
// The shadow always prices final_round=true with no column cap, so its sweep
// covers every source; whatever the baseline emits — possibly truncated by
// max_cols or thinned by postponement — is a subset of that same sweep.  That
// is what lets the two be compared without synchronizing postponement: a cut
// source does not refresh its _source_arcs (see should_record_arcs), so
// filter_for_new_caps would otherwise postpone different sources in the two
// pricers and the comparison would be between different source sets.
//
// Results are static because solve_cg constructs the pricer itself and the test
// never gets a handle on it.  GoogleTest runs cases sequentially; reset()
// clears between them.
template <typename Inner>
class ShadowPricer {
public:
    static inline std::string failure;
    static inline uint64_t compared = 0;
    static inline uint64_t fired = 0;

    static void reset() {
        failure.clear();
        compared = 0;
        fired = 0;
    }

    void init(const mcfcg::Instance& inst, mcfcg::thread_pool* pool = nullptr,
              uint32_t batch_size = 0, double neg_rc_tol = mcfcg::NEG_RC_TOL,
              bool dual_cutoff = false) {
        // This harness owns BOTH arms, so the caller's flag is deliberately not
        // forwarded — it would otherwise be possible to run the shadow with the
        // cutoff off on both sides, comparing a pricer against itself and
        // passing vacuously.
        EXPECT_FALSE(dual_cutoff) << "ShadowPricer drives both arms; leave the flag off";
        _base.init(inst, pool, batch_size, neg_rc_tol, /*dual_cutoff=*/false);
        _cut.init(inst, pool, batch_size, neg_rc_tol, /*dual_cutoff=*/true);
    }

    void set_track_arcs(bool enabled) {
        _base.set_track_arcs(enabled);
        _cut.set_track_arcs(enabled);
    }

    auto price(const std::vector<double>& duals, const mcfcg::static_map<uint32_t, double>& mu,
               bool final_round = false, uint32_t max_cols = 0) {
        auto base_cols = _base.price(duals, mu, final_round, max_cols);
        auto cut_cols = _cut.price(duals, mu, /*final_round=*/true, /*max_cols=*/0);
        fired += _cut.last_cutoff_count();
        check(base_cols, cut_cols);
        return base_cols;
    }

    // Control flow follows the baseline; the reported cutoff stats follow the
    // shadow, which is the pricer those numbers describe.
    bool priced_all() const noexcept { return _base.priced_all(); }
    double lagrangian_path_sum() const noexcept { return _base.lagrangian_path_sum(); }
    double lb_error_bound() const noexcept { return _base.lb_error_bound(); }
    uint64_t last_cutoff_count() const noexcept { return _cut.last_cutoff_count(); }
    uint64_t last_priced_count() const noexcept { return _cut.last_priced_count(); }

    void filter_for_new_caps(const std::vector<uint32_t>& new_cap_arcs) {
        _base.filter_for_new_caps(new_cap_arcs);
        _cut.filter_for_new_caps(new_cap_arcs);
    }
    void clear_postponed() {
        _base.clear_postponed();
        _cut.clear_postponed();
    }
    void reset_postponed() {
        _base.reset_postponed();
        _cut.reset_postponed();
    }

private:
    // Every baseline column must appear in the shadow sweep, unchanged.  Stops
    // at the first mismatch: the loop runs for tens of iterations and a real
    // divergence would otherwise bury the first (and only diagnostic) one.
    template <typename ColumnT>
    static void check(const std::vector<ColumnT>& base_cols, const std::vector<ColumnT>& cut_cols) {
        if (!failure.empty()) {
            return;
        }
        std::unordered_map<uint64_t, const ColumnT*> by_key;
        by_key.reserve(cut_cols.size());
        for (const auto& col : cut_cols) {
            by_key.emplace(column_key(col), &col);
        }
        for (const auto& col : base_cols) {
            auto found = by_key.find(column_key(col));
            if (found == by_key.end()) {
                failure = "cutoff dropped the column for key " + std::to_string(column_key(col));
                return;
            }
            auto diff = column_diff(col, *found->second);
            if (!diff.empty()) {
                failure = "cutoff altered the column for key " + std::to_string(column_key(col)) +
                          ":" + diff;
                return;
            }
            ++compared;
        }
    }

    Inner _base;
    Inner _cut;
};

template <typename Shadow>
static void expect_shadow_agreed(const mcfcg::CGResult& result) {
    EXPECT_TRUE(result.optimal) << "baseline run did not reach optimality";
    EXPECT_TRUE(Shadow::failure.empty()) << Shadow::failure;
    // Both guard vacuity: a cutoff that never fires, or a run that never
    // compares a column, satisfies the assertion above for free.  The counts go
    // to the XML report (--gtest_output=xml) so the coverage the run achieved is
    // recoverable without making a passing suite noisier.
    ::testing::Test::RecordProperty("cutoff_fired", std::to_string(Shadow::fired));
    ::testing::Test::RecordProperty("columns_compared", std::to_string(Shadow::compared));
    EXPECT_GT(Shadow::fired, 0U) << "cutoff never fired; the comparison proved nothing";
    EXPECT_GT(Shadow::compared, 0U) << "no columns were compared";
}
}  // namespace cutoff_test

TEST(FeatureTests, PricingCutoffShadowTree) {
    using Shadow = cutoff_test::ShadowPricer<mcfcg::TreePricer>;
    Shadow::reset();
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar150");
    mcfcg::CGParams params;
    auto result = mcfcg::solve_cg<mcfcg::TreeMaster, Shadow>(
        inst, params, [](const mcfcg::TreeMaster& m) { return m.get_structural_duals(); },
        static_cast<uint32_t>(inst.sources.size()));
    cutoff_test::expect_shadow_agreed<Shadow>(result);
}

// The path bound (max π over unsettled sinks) is a different formula from the
// tree's demand-weighted budget and had no equivalent column-identity guard.
TEST(FeatureTests, PricingCutoffShadowPath) {
    using Shadow = cutoff_test::ShadowPricer<mcfcg::PathPricer>;
    Shadow::reset();
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar150");
    mcfcg::CGParams params;
    auto result = mcfcg::solve_cg<mcfcg::PathMaster, Shadow>(
        inst, params, [](const mcfcg::PathMaster& m) { return m.get_structural_duals(); },
        static_cast<uint32_t>(inst.commodities.size()));
    cutoff_test::expect_shadow_agreed<Shadow>(result);
}

// The family the cutoff is meant for, and the one where switching it on
// measurably moves the trajectory.  PricerHeavy so the source pricing
// filter — the mechanism that makes a cut source's stale _source_arcs matter —
// is live, and long paths so the integer-scaling allowance is under real load.
TEST(FeatureTests, PricingCutoffShadowIntermodalTree) {
    auto path = data_dir("intermodal") + "/BUS-2632-0.txt.gz";
    if (!fs::exists(path)) {
        GTEST_SKIP() << "data/intermodal not found";
    }
    using Shadow = cutoff_test::ShadowPricer<mcfcg::TreePricer>;
    Shadow::reset();
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.strategy = mcfcg::CGStrategy::PricerHeavy;
    auto result = mcfcg::solve_cg<mcfcg::TreeMaster, Shadow>(
        inst, params, [](const mcfcg::TreeMaster& m) { return m.get_structural_duals(); },
        static_cast<uint32_t>(inst.sources.size()));
    cutoff_test::expect_shadow_agreed<Shadow>(result);
}

// The π-free Lagrangian LB must be (a) always valid (≤ OPT) and (b) AVAILABLE
// while slacks are basic — the property that lets gap-termination fire on hard
// instances (the old clamped LB was suppressed whenever a slack was basic).
// Drives a manual tree CG loop on a capacity-binding instance, computing
// L(μ) = Σcap·μ + Σ_k d_k·sp_k(c−μ) − margin every priced iteration (with the
// capacity-dual term snapshotted pre-separation, exactly as cg_loop does), and
// asserts the bound never exceeds the reference optimum and that at least one
// slack-basic iteration produced a finite valid bound.
TEST(FeatureTests, LagrangianBoundValidWhileSlacksBasic) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar30");
    double ref = opt.at("planar30");

    mcfcg::TreeMaster master;
    master.init(inst);
    mcfcg::TreePricer pricer;
    pricer.init(inst);

    std::vector<double> big_pi(inst.sources.size(), std::numeric_limits<double>::infinity());
    auto empty_mu = inst.graph.create_arc_map<double>(0.0);
    auto init_cols = pricer.price(big_pi, empty_mu, true);
    if (!init_cols.empty()) {
        master.add_columns(std::move(init_cols));
    }
    pricer.reset_postponed();

    bool saw_slack_basic_lb = false;
    bool optimal = false;
    for (uint32_t iter = 0; iter < 10000; ++iter) {
        ASSERT_EQ(master.solve(), mcfcg::LPStatus::Optimal) << "LP not optimal at iter " << iter;
        auto primals = master.get_primals();
        auto pi_s = master.get_structural_duals();
        const auto& mu = master.get_capacity_duals();
        // Snapshot Σcap·μ against the solved row set, before separation mutates it.
        double cap_dual_term = master.compute_capacity_dual_term(mu);
        uint32_t slacks = master.count_active_slacks(primals);

        auto new_cap_arcs = master.add_violated_capacity_constraints(primals);
        uint32_t num_new_caps = static_cast<uint32_t>(new_cap_arcs.size());

        auto new_cols = pricer.price(pi_s, mu, false);
        if (new_cols.empty()) {
            new_cols = pricer.price(pi_s, mu, true);
        }
        if (!new_cols.empty()) {
            pricer.clear_postponed();
        }

        if (pricer.priced_all()) {
            double lb = cap_dual_term + pricer.lagrangian_path_sum() - pricer.lb_error_bound();
            EXPECT_TRUE(std::isfinite(lb)) << "LB not finite at iter " << iter;
            EXPECT_LE(lb, ref * (1.0 + 1e-6))
                << "LB exceeds OPT at iter " << iter << " (slacks basic=" << slacks << ")";
            if (slacks > 0 && std::isfinite(lb)) {
                saw_slack_basic_lb = true;
            }
        }

        if (new_cols.empty()) {
            if (num_new_caps == 0 && slacks == 0) {
                optimal = true;
                break;
            }
            if (slacks > 0) {
                (void)master.bump_active_slacks(primals, mcfcg::SLACK_BUMP_FACTOR);
            }
            pricer.reset_postponed();
            continue;
        }
        (void)master.bump_active_slacks(primals, mcfcg::SLACK_BUMP_FACTOR);
        master.add_columns(std::move(new_cols));
    }
    EXPECT_TRUE(optimal) << "manual tree loop did not converge";
    // This assertion is the whole point of the feature: it requires planar30
    // (a capacity-binding instance) to pass through at least one fully-priced
    // iteration with a slack still basic.  If a future pricing/slack change
    // makes planar30 reach feasibility before any priced iteration, swap in
    // another capacity-binding instance rather than dropping the check.
    EXPECT_TRUE(saw_slack_basic_lb)
        << "never observed a finite Lagrangian LB while a slack was basic";
}

// Repeated parallel runs must land within the design feasibility
// tolerance of the same value (the LP solver's basis choice still
// depends on the column arrival order across threads, so exact bitwise
// equality is not guaranteed — but the reported objective must be a
// valid UB within tolerance either way).
TEST(FeatureTests, ParallelReproducibility) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar80");
    mcfcg::CGParams params;
    params.num_threads = 4;
    auto r1 = mcfcg::solve_path_cg(inst, params);
    auto r2 = mcfcg::solve_path_cg(inst, params);
    EXPECT_TRUE(r1.optimal);
    EXPECT_TRUE(r2.optimal);
    double ref = opt.at("planar80");
    EXPECT_LT(std::abs(r1.objective - ref) / std::max(1.0, std::abs(ref)),
              mcfcg::RELATIVE_FEAS_TOL);
    EXPECT_LT(std::abs(r2.objective - ref) / std::max(1.0, std::abs(ref)),
              mcfcg::RELATIVE_FEAS_TOL);
}

// --- Compact source-based formulation (build_source_lp) ---
//
// Build the source LP, feed it to a HiGHS LPSolver (rows as bounds-only, then
// columns with their CSC coefficients), and check the optimum equals the paper
// reference. This guards the formulation end-to-end — RHS signs, conservation
// coefficients, and capacity rows — independently of the MPS text writer, which
// itself is exercised by writing and re-solving in the CLI.
static void check_source_lp(const std::string& family, const std::string& name) {
    auto opt = load_optimal(data_dir("commalab/" + family));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/" + family + "/" + name);
    auto slp = mcfcg::build_source_lp(inst);

    EXPECT_EQ(slp.num_cols, inst.sources.size() * inst.graph.num_arcs());
    ASSERT_EQ(slp.col_start.size(), slp.num_cols + 1);
    EXPECT_EQ(slp.col_start.back(), slp.value.size());
    EXPECT_EQ(slp.row_index.size(), slp.value.size());

    auto lp = mcfcg::create_lp_solver();
    std::vector<uint32_t> row_starts(slp.num_rows + 1, 0);
    lp->add_rows(slp.row_lower, slp.row_upper, row_starts, {}, {});
    lp->add_cols(slp.col_cost, slp.col_lower, slp.col_upper, slp.col_start, slp.row_index,
                 slp.value);
    ASSERT_EQ(lp->solve(), mcfcg::LPStatus::Optimal);
    double ref = opt.at(name);
    EXPECT_NEAR(lp->get_obj(), ref, std::abs(ref) * 1e-6);
}

TEST(SourceFormulation, Grid1) {
    check_source_lp("grid", "grid1");
}
TEST(SourceFormulation, Planar30) {
    check_source_lp("planar", "planar30");
}
