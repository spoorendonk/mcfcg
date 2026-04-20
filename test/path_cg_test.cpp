#include "mcfcg/cg/path_cg.h"

#include "mcfcg/cg/master.h"
#include "mcfcg/cg/master_base.h"
#include "mcfcg/instance.h"
#include "test_paths.h"

#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>

// Helper to write plain-numeric instance file
static void writeInstance(const std::string& path, uint32_t vertices, uint32_t arcs,
                          uint32_t commodities, const std::string& arc_lines,
                          const std::string& commodity_lines) {
    std::ofstream out(path);
    out << vertices << '\n' << arcs << '\n' << commodities << '\n';
    out << arc_lines << commodity_lines;
}

// --- Instance 1: Single source, no capacity binding ---
// Graph: 0→1(c=1,u=10), 0→2(c=4,u=10), 1→2(c=2,u=10),
//        1→3(c=6,u=10), 2→3(c=1,u=10)
// Commodities: (0→3, d=5), (0→2, d=3)
// obj* = 29

class PathCGSingleSource : public ::testing::Test {
protected:
    std::string path = mcfcg::test::unique_test_path("path_cg_single.txt");
    void SetUp() override {
        writeInstance(path, 4, 5, 2, "1 2 1 10\n1 3 4 10\n2 3 2 10\n2 4 6 10\n3 4 1 10\n",
                      "1 4 5\n1 3 3\n");
    }
    void TearDown() override { std::remove(path.c_str()); }
};

TEST_F(PathCGSingleSource, OptimalObjective) {
    auto inst = mcfcg::read_commalab(path);
    auto result = mcfcg::solve_path_cg(inst);

    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 29.0, 1e-4);
}

// --- Instance 2: Binding capacity forces flow splitting ---
// obj* = 21

class PathCGCapacityBinding : public ::testing::Test {
protected:
    std::string path = mcfcg::test::unique_test_path("path_cg_cap.txt");
    void SetUp() override {
        writeInstance(path, 4, 4, 1, "1 2 1 3\n1 3 5 10\n2 3 1 10\n3 4 1 10\n", "1 4 5\n");
    }
    void TearDown() override { std::remove(path.c_str()); }
};

TEST_F(PathCGCapacityBinding, OptimalWithSplitFlow) {
    auto inst = mcfcg::read_commalab(path);
    auto result = mcfcg::solve_path_cg(inst);

    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 21.0, 1e-4);
}

// --- Instance 3: Multiple sources ---
// obj* = 17

class PathCGMultiSource : public ::testing::Test {
protected:
    std::string path = mcfcg::test::unique_test_path("path_cg_multi.txt");
    void SetUp() override {
        writeInstance(path, 4, 3, 2, "1 3 1 10\n2 3 2 10\n3 4 1 10\n", "1 4 4\n2 4 3\n");
    }
    void TearDown() override { std::remove(path.c_str()); }
};

TEST_F(PathCGMultiSource, OptimalObjective) {
    auto inst = mcfcg::read_commalab(path);
    ASSERT_EQ(inst.sources.size(), 2u);

    auto result = mcfcg::solve_path_cg(inst);

    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 17.0, 1e-4);
}

// --- Instance 4: Multiple sources + capacity binding ---
// obj* = 21

class PathCGMultiSourceCap : public ::testing::Test {
protected:
    std::string path = mcfcg::test::unique_test_path("path_cg_multi_cap.txt");
    void SetUp() override {
        writeInstance(path, 4, 4, 2, "1 3 1 10\n2 3 2 10\n3 4 1 5\n1 4 4 10\n", "1 4 4\n2 4 3\n");
    }
    void TearDown() override { std::remove(path.c_str()); }
};

TEST_F(PathCGMultiSourceCap, OptimalWithCapacity) {
    auto inst = mcfcg::read_commalab(path);
    ASSERT_EQ(inst.sources.size(), 2u);

    auto result = mcfcg::solve_path_cg(inst);

    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 21.0, 1e-4);
}

// --- Row purging: verify aggressive purging does not change optimal objective ---

TEST_F(PathCGCapacityBinding, RowPurgeDoesNotChangeObjective) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.row_inactivity_threshold = 1;  // aggressive: purge after 1 iteration inactive
    auto result = mcfcg::solve_path_cg(inst, params);

    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 21.0, 1e-4);
}

TEST_F(PathCGMultiSourceCap, RowPurgeDoesNotChangeObjective) {
    auto inst = mcfcg::read_commalab(path);
    ASSERT_EQ(inst.sources.size(), 2u);

    mcfcg::CGParams params;
    params.row_inactivity_threshold = 1;
    auto result = mcfcg::solve_path_cg(inst, params);

    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 21.0, 1e-4);
}

// --- Column purging tests: verify purging does not change the optimal objective ---

TEST_F(PathCGSingleSource, ColPurgeDoesNotChangeObjective) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.col_age_limit = 3;  // aggressive purge
    auto result = mcfcg::solve_path_cg(inst, params);

    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 29.0, 1e-4);
}

TEST_F(PathCGCapacityBinding, ColPurgeWithCapacity) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.col_age_limit = 3;
    auto result = mcfcg::solve_path_cg(inst, params);

    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 21.0, 1e-4);
}

TEST_F(PathCGMultiSourceCap, ColPurgeWithMultiSourceCapacity) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.col_age_limit = 3;
    auto result = mcfcg::solve_path_cg(inst, params);

    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 21.0, 1e-4);
}

TEST_F(PathCGSingleSource, ColPurgeDisabledMatchesDefault) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.col_age_limit = 0;
    auto result = mcfcg::solve_path_cg(inst, params);

    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 29.0, 1e-4);
}

// --- SlackMode::EdgeRows: instance with few capacitated arcs
//
// Triangle graph with 3 arcs and 3 commodities trips the EdgeRows
// selector in MasterBase::init (3 ≤ 3).  The cheap 1→2→3 path (cost 4)
// is shorter than the 1→3 bypass (cost 5), so warm-start routes all
// c1+c2 demand through 1→2→3 and overflows both capacities — forcing
// CG to add capacity rows, pair slacks with them, re-price, and shift
// some flow to 1→3.  Exercises the EdgeRows lazy-slack insertion path.
//
//   arcs:     (1→2, c=1, u=3)  (1→3, c=5, u=10)  (2→3, c=3, u=4)
//   commods:  c1 (1→3, d=2),   c2 (1→3, d=3),    c3 (2→3, d=1)
//
// Optimal flow (one of many equivalents, obj = 25):
//   c1+c2: 3 units on 1→2→3 (arc 1→2 full), 2 units on 1→3
//   c3:    1 unit on 2→3 (arc 2→3 now at 4/4)
//   cost = 4*3 + 5*2 + 3*1 = 25
class PathCGEdgeRows : public ::testing::Test {
protected:
    std::string path = mcfcg::test::unique_test_path("path_cg_edge_rows.txt");
    void SetUp() override {
        writeInstance(path, 3, 3, 3, "1 2 1 3\n1 3 5 10\n2 3 3 4\n", "1 3 2\n1 3 3\n2 3 1\n");
    }
    void TearDown() override { std::remove(path.c_str()); }
};

TEST_F(PathCGEdgeRows, SelectorPicksEdgeRows) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::PathMaster master;
    master.init(inst);
    EXPECT_EQ(master.slack_mode(), mcfcg::SlackMode::EdgeRows);
}

TEST_F(PathCGEdgeRows, SolvesToOptimal) {
    auto inst = mcfcg::read_commalab(path);
    auto result = mcfcg::solve_path_cg(inst);

    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 25.0, 1e-4);
}

TEST_F(PathCGEdgeRows, ColPurgeDoesNotBreakEdgeRows) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.col_age_limit = 1;  // aggressive aging
    auto result = mcfcg::solve_path_cg(inst, params);

    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 25.0, 1e-4);
}

TEST_F(PathCGEdgeRows, RowPurgeDoesNotBreakEdgeRows) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.row_inactivity_threshold = 1;
    auto result = mcfcg::solve_path_cg(inst, params);

    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 25.0, 1e-4);
}

// EdgeRows requires warm_start=true.  init() throws when that contract
// is violated so release builds surface the misconfiguration instead of
// silently producing an infeasible LP.
TEST_F(PathCGEdgeRows, InitThrowsWithoutWarmStart) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::PathMaster master;
    EXPECT_THROW(master.init(inst, nullptr, nullptr, /*warm_start=*/false), std::invalid_argument);
}

// The throw must also propagate through the full solve_path_cg pipeline
// so callers using CGParams see the contract violation rather than a
// silently-infeasible solve result.
TEST_F(PathCGEdgeRows, SolvePathCGThrowsWithoutWarmStart) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.warm_start = false;
    EXPECT_THROW(mcfcg::solve_path_cg(inst, params), std::invalid_argument);
}

// EdgeRows + very aggressive aging (age_limit=1) on an instance small
// enough to finish in a few iterations was the original COPT-regression
// trigger: every column aged out between iters → demand row infeasible.
// update_column_ages now treats any column with rc within LP_FEAS_TOL
// of zero as a basis candidate (not aged out), so demand rows always
// retain at least one column and the LP stays feasible.
TEST_F(PathCGEdgeRows, AggressiveAgingStaysFeasible) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::CGParams params;
    params.col_age_limit = 1;
    params.row_inactivity_threshold = 1;
    auto result = mcfcg::solve_path_cg(inst, params);

    ASSERT_TRUE(result.optimal);
    EXPECT_NEAR(result.objective, 25.0, 1e-4);
}

// Verify that PathCGCapacityBinding — arcs (4) > commodities (1) — still
// selects CommodityRows, so the existing CommodityRows path remains
// exercised.
TEST_F(PathCGCapacityBinding, SelectorPicksCommodityRows) {
    auto inst = mcfcg::read_commalab(path);
    mcfcg::PathMaster master;
    master.init(inst);
    EXPECT_EQ(master.slack_mode(), mcfcg::SlackMode::CommodityRows);
}
