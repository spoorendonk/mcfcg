#include "mcfcg/cg/tree_cg.h"

#include "mcfcg/cg/master_base.h"
#include "mcfcg/cg/path_cg.h"
#include "mcfcg/cg/tree_master.h"
#include "mcfcg/graph/static_digraph_builder.h"
#include "mcfcg/instance.h"
#include "mcfcg/util/limits.h"
#include "test_paths.h"

#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using mcfcg::test::writeInstance;

// Helper: solve with both formulations, verify agreement
static void verifyPathTreeAgreement(const std::string& path, double expected_obj) {
    auto inst = mcfcg::read_commalab(path);

    mcfcg::CGParams params;
    params.max_iterations = 200;

    auto path_result = mcfcg::solve_path_cg(inst, params);
    auto tree_result = mcfcg::solve_tree_cg(inst, params);

    ASSERT_TRUE(path_result.optimal) << "path CG did not converge";
    ASSERT_TRUE(tree_result.optimal) << "tree CG did not converge";

    EXPECT_NEAR(path_result.objective, expected_obj, 1e-4) << "path CG wrong objective";
    EXPECT_NEAR(tree_result.objective, expected_obj, 1e-4) << "tree CG wrong objective";
    EXPECT_NEAR(path_result.objective, tree_result.objective, 1e-4) << "path and tree disagree";
}

// --- Single source, no capacity binding, obj* = 29 ---
TEST(TreeCGCorrectness, SingleSourceNoCap) {
    std::string path = mcfcg::test::unique_test_path("tree_ss_nocap.txt");
    writeInstance(path, 4, 5, 2, "1 2 1 10\n1 3 4 10\n2 3 2 10\n2 4 6 10\n3 4 1 10\n",
                  "1 4 5\n1 3 3\n");
    verifyPathTreeAgreement(path, 29.0);
    std::remove(path.c_str());
}

// --- Binding capacity forces split, obj* = 21 ---
TEST(TreeCGCorrectness, CapacityBinding) {
    std::string path = mcfcg::test::unique_test_path("tree_cap.txt");
    writeInstance(path, 4, 4, 1, "1 2 1 3\n1 3 5 10\n2 3 1 10\n3 4 1 10\n", "1 4 5\n");
    verifyPathTreeAgreement(path, 21.0);
    std::remove(path.c_str());
}

// --- Multiple sources, no capacity binding, obj* = 17 ---
TEST(TreeCGCorrectness, MultiSourceNoCap) {
    std::string path = mcfcg::test::unique_test_path("tree_ms_nocap.txt");
    writeInstance(path, 4, 3, 2, "1 3 1 10\n2 3 2 10\n3 4 1 10\n", "1 4 4\n2 4 3\n");
    verifyPathTreeAgreement(path, 17.0);
    std::remove(path.c_str());
}

// --- Multiple sources + capacity binding, obj* = 21 ---
TEST(TreeCGCorrectness, MultiSourceCapacity) {
    std::string path = mcfcg::test::unique_test_path("tree_ms_cap.txt");
    writeInstance(path, 4, 4, 2, "1 3 1 10\n2 3 2 10\n3 4 1 5\n1 4 4 10\n", "1 4 4\n2 4 3\n");
    verifyPathTreeAgreement(path, 21.0);
    std::remove(path.c_str());
}

// --- Single commodity, single path (trivial), obj* = 10 ---
TEST(TreeCGCorrectness, TrivialSinglePath) {
    std::string path = mcfcg::test::unique_test_path("tree_trivial.txt");
    writeInstance(path, 3, 2, 1, "1 2 2 10\n2 3 3 10\n", "1 3 2\n");
    verifyPathTreeAgreement(path, 10.0);
    std::remove(path.c_str());
}

// --- Many commodities, same source, different sinks ---
// obj* = 49
TEST(TreeCGCorrectness, ManyCommoditiesSameSource) {
    std::string path = mcfcg::test::unique_test_path("tree_many_k.txt");
    writeInstance(path, 4, 4, 3, "1 2 1 100\n1 3 3 100\n2 4 2 100\n3 4 1 100\n",
                  "1 2 10\n1 4 5\n1 3 8\n");
    verifyPathTreeAgreement(path, 49.0);
    std::remove(path.c_str());
}

// --- Row purging: verify aggressive purging does not change optimal objective ---

TEST(TreeCGRowPurge, CapacityBindingWithPurge) {
    std::string path = mcfcg::test::unique_test_path("tree_cap_purge.txt");
    writeInstance(path, 4, 4, 1, "1 2 1 3\n1 3 5 10\n2 3 1 10\n3 4 1 10\n", "1 4 5\n");
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.row_inactivity_threshold = 1;
    auto result = mcfcg::solve_tree_cg(inst, params);
    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 21.0, 1e-4);
    std::remove(path.c_str());
}

TEST(TreeCGRowPurge, MultiSourceCapWithPurge) {
    std::string path = mcfcg::test::unique_test_path("tree_ms_cap_purge.txt");
    writeInstance(path, 4, 4, 2, "1 3 1 10\n2 3 2 10\n3 4 1 5\n1 4 4 10\n", "1 4 4\n2 4 3\n");
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.row_inactivity_threshold = 1;
    auto result = mcfcg::solve_tree_cg(inst, params);
    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 21.0, 1e-4);
    std::remove(path.c_str());
}

// --- Column purging tests for tree formulation ---

TEST(TreeCGColPurge, PurgeDoesNotChangeObjective) {
    std::string path = mcfcg::test::unique_test_path("tree_purge_nocap.txt");
    writeInstance(path, 4, 5, 2, "1 2 1 10\n1 3 4 10\n2 3 2 10\n2 4 6 10\n3 4 1 10\n",
                  "1 4 5\n1 3 3\n");
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.col_age_limit = 3;
    auto path_result = mcfcg::solve_path_cg(inst, params);
    auto tree_result = mcfcg::solve_tree_cg(inst, params);
    ASSERT_TRUE(path_result.optimal);
    ASSERT_TRUE(tree_result.optimal);
    EXPECT_NEAR(path_result.objective, 29.0, 1e-4);
    EXPECT_NEAR(tree_result.objective, 29.0, 1e-4);
    std::remove(path.c_str());
}

TEST(TreeCGColPurge, PurgeWithCapacity) {
    std::string path = mcfcg::test::unique_test_path("tree_purge_cap.txt");
    writeInstance(path, 4, 4, 2, "1 3 1 10\n2 3 2 10\n3 4 1 5\n1 4 4 10\n", "1 4 4\n2 4 3\n");
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.col_age_limit = 3;
    auto path_result = mcfcg::solve_path_cg(inst, params);
    auto tree_result = mcfcg::solve_tree_cg(inst, params);
    ASSERT_TRUE(path_result.optimal);
    ASSERT_TRUE(tree_result.optimal);
    EXPECT_NEAR(path_result.objective, 21.0, 1e-4);
    EXPECT_NEAR(tree_result.objective, 21.0, 1e-4);
    std::remove(path.c_str());
}

// --- SlackMode::EdgeRows on TreeMaster
//
// Two arcs each from its own origin into a common sink.  Two sources
// (1, 2), two capacitated arcs — the tree selector picks EdgeRows
// (2 > 2 is false).  No capacity binding (caps are loose), so this is
// a smoke test for the "no init slacks" code path plus warm-start
// feasibility.  Optimal: c1 on 1→3 = 3*2 = 6; c2 on 2→3 = 5*3 = 15;
// total 21.
class TreeCGEdgeRows : public ::testing::Test {
protected:
    std::string path = mcfcg::test::unique_test_path("tree_cg_edge_rows.txt");
    void SetUp() override {
        writeInstance(path, 3, 2, 2, "1 3 3 10\n2 3 5 10\n", "1 3 2\n2 3 3\n");
    }
    void TearDown() override { std::remove(path.c_str()); }
};

TEST_F(TreeCGEdgeRows, SelectorAndSolve) {
    auto inst = mcfcg::read_commalab(path);

    mcfcg::TreeMaster master;
    master.init(inst);
    EXPECT_EQ(master.slack_mode(), mcfcg::SlackMode::EdgeRows);

    auto result = mcfcg::solve_tree_cg(inst);
    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 21.0, 1e-4);
}

// Tree EdgeRows with column-purge and row-purge exercises the slack
// index remap in purge_aged_columns and delete_edge_row_slacks on the
// tree-column path.  Caps are loose so no cuts actually fire on this
// instance, but the purge calls still run and must not disturb the
// empty _slack_col_lp / _arc_to_slack_col state.  Mirrors the path
// ColPurge/RowPurge/AggressiveAging tests.
TEST_F(TreeCGEdgeRows, ColPurgeStaysFeasible) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.col_age_limit = 1;
    auto result = mcfcg::solve_tree_cg(inst, params);
    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 21.0, 1e-4);
}

TEST_F(TreeCGEdgeRows, RowPurgeStaysFeasible) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.row_inactivity_threshold = 1;
    auto result = mcfcg::solve_tree_cg(inst, params);
    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 21.0, 1e-4);
}

TEST_F(TreeCGEdgeRows, AggressiveAgingStaysFeasible) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.col_age_limit = 1;
    params.row_inactivity_threshold = 1;
    auto result = mcfcg::solve_tree_cg(inst, params);
    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 21.0, 1e-4);
}

// EdgeRows requires warm_start=true on the tree formulation too.  init()
// throws when that contract is violated so release builds surface the
// misconfiguration instead of silently producing an infeasible LP.
TEST_F(TreeCGEdgeRows, InitThrowsWithoutWarmStart) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::TreeMaster master;
    EXPECT_THROW(master.init(inst, nullptr, nullptr, /*warm_start=*/false), std::invalid_argument);
}

// The throw must also propagate through the full solve_tree_cg pipeline
// so callers using CGParams see the contract violation rather than a
// silently-infeasible solve result.
TEST_F(TreeCGEdgeRows, SolveTreeCGThrowsWithoutWarmStart) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.warm_start = false;
    EXPECT_THROW(mcfcg::solve_tree_cg(inst, params), std::invalid_argument);
}

// --- SlackMode::EdgeRows on TreeMaster with actual capacity binding
//
// Same arc geometry as PathCGEdgeRows but with arc 1→3 left uncapacitated
// (capacity = INF) so the master's selector sees only 2 capacitated arcs
// against 2 sources and picks EdgeRows.  Caps on 1→2 and 2→3 both bind,
// so add_violated_capacity_constraints fires during CG and must insert
// lazy capacity rows paired with slack columns on the tree formulation.
//
//   arcs:     (1→2, c=1, u=3)  (1→3, c=5, u=INF)  (2→3, c=3, u=4)
//   commods:  c1 (1→3, d=2),   c2 (1→3, d=3),     c3 (2→3, d=1)
//
// Optimal obj = 25.  For source 1 (two commodities into sink 3) the tree
// formulation picks a convex combination of the 1→2→3 and 1→3 trees so
// that aggregated arc flows honour the binding capacities; source 2's
// tree is the single arc 2→3.
TEST(TreeCGEdgeRowsBinding, EdgeRowsLazySlackInsertion) {
    using namespace mcfcg;
    // commalab's plain-numeric writer cannot round-trip std::numeric_limits
    // infinity() through operator>>, so build the Instance directly.
    static_digraph_builder<double, double> builder(3);
    builder.add_arc(0, 1, 1.0, 3.0);  // 1→2 binding
    builder.add_arc(0, 2, 5.0, INF);  // 1→3 uncapacitated
    builder.add_arc(1, 2, 3.0, 4.0);  // 2→3 binding
    auto [graph, cost_map, cap_map] = builder.build();

    std::vector<Commodity> commodities = {
        {0, 2, 2.0},  // source 1 → sink 3, demand 2
        {0, 2, 3.0},  // source 1 → sink 3, demand 3
        {1, 2, 1.0},  // source 2 → sink 3, demand 1
    };
    auto sources = group_by_source(commodities);

    Instance inst{std::move(graph), std::move(cost_map), std::move(cap_map), std::move(commodities),
                  std::move(sources)};

    TreeMaster master;
    master.init(inst);
    ASSERT_EQ(master.slack_mode(), SlackMode::EdgeRows);

    auto result = solve_tree_cg(inst);
    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 25.0, 1e-4);
}
