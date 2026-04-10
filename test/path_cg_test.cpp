#include "mcfcg/cg/path_cg.h"

#include "mcfcg/instance.h"
#include "test_paths.h"

#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>

// Helper to write plain-numeric instance file
static void writeInstance(const std::string& path, uint32_t vertices, uint32_t arcs,
                          uint32_t commodities, const std::string& arc_lines,
                          const std::string& commodity_lines) {
    std::ofstream f(path);
    f << vertices << '\n' << arcs << '\n' << commodities << '\n';
    f << arc_lines << commodity_lines;
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
