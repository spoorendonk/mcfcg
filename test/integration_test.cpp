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

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
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

// BUS-* tests stay disabled: they solve under HiGHS+tree+PricerHeavy
// in a few seconds (~6s for BUS-2632, scaling up), but running them
// on every CI run bloats the suite wall-time and the DISABLED_
// prefix keeps them accessible via --gtest_also_run_disabled_tests
// for manual runs and regression hunts.
TEST(IntermodalCorrectness, DISABLED_Bus2632) {
    auto path = data_dir("intermodal") + "/BUS-2632-0.txt.gz";
    if (!fs::exists(path))
        GTEST_SKIP() << "data/intermodal not found";
    auto opt = load_optimal(data_dir("intermodal"));
    auto inst = mcfcg::read_commalab(path);
    solve_intermodal_and_check(inst, opt.at("BUS-2632-0"));
}

TEST(IntermodalCorrectness, DISABLED_Bus7896) {
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

// --- cuOpt GPU solver tests ---

#ifdef MCFCG_USE_CUOPT

#include "mcfcg/lp/lp_solver.h"

static void solve_and_check_cuopt(const mcfcg::Instance& inst, double ref_obj, double tol = 0.001) {
    mcfcg::CGParams params;
    params.max_iterations = 10000;
    params.solver_factory = []() { return mcfcg::create_cuopt_solver(); };
    auto result = mcfcg::solve_path_cg(inst, params);
    EXPECT_TRUE(result.optimal) << "Did not reach optimality with cuOpt solver";
    EXPECT_GE(result.objective, ref_obj * (1.0 - tol)) << "Objective below reference";
    EXPECT_LE(result.objective, ref_obj * (1.0 + tol)) << "Objective above reference";
}

// cuOpt tests are slow (GPU barrier overhead per LP solve) — disabled by default.
// Run manually with: --gtest_also_run_disabled_tests --gtest_filter='CuOptCorrectness.*'
TEST(DISABLED_CuOptCorrectness, Grid1) {
    auto opt = load_optimal(data_dir("commalab/grid"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid1");
    solve_and_check_cuopt(inst, opt.at("grid1"));
}

TEST(DISABLED_CuOptCorrectness, Grid2) {
    auto opt = load_optimal(data_dir("commalab/grid"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/grid/grid2");
    solve_and_check_cuopt(inst, opt.at("grid2"));
}

TEST(DISABLED_CuOptCorrectness, Planar30) {
    auto opt = load_optimal(data_dir("commalab/planar"));
    auto inst = mcfcg::read_commalab(data_dir("commalab") + "/planar/planar30");
    solve_and_check_cuopt(inst, opt.at("planar30"));
}

#endif  // MCFCG_USE_CUOPT
