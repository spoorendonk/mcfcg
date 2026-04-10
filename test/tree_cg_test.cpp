#include "mcfcg/cg/tree_cg.h"

#include "mcfcg/cg/path_cg.h"
#include "mcfcg/instance.h"
#include "test_paths.h"

#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <string>

// Helper to write plain-numeric instance file
static void writeInstance(const std::string& path, uint32_t vertices, uint32_t arcs,
                          uint32_t commodities, const std::string& arc_lines,
                          const std::string& commodity_lines) {
    std::ofstream out(path);
    out << vertices << '\n' << arcs << '\n' << commodities << '\n';
    out << arc_lines << commodity_lines;
}

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
