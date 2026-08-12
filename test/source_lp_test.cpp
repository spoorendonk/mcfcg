#include "mcfcg/source/source_lp.h"

#include "mcfcg/instance.h"
#include "test_paths.h"

#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>

// CommaLab/UniPi plain-numeric: 4 vertices, 6 arcs, 2 commodities.
//
// Deliberately exercises both terms that make the compact source LP's nonzero
// count differ from the 3*|S|*|E| upper bound, because both are common on the
// intermodal families and neither shows up on grid/planar:
//   - arcs 2 and 4 carry the negative-capacity sentinel, so they are
//     uncapacitated (INF), get no capacity row, and contribute 2 entries;
//   - arc 6 is a self-loop, whose +1/-1 conservation pair cancels, leaving only
//     its capacity entry.
// The two commodities originate at different vertices, so |S| = 2.
static const char* MIXED_INSTANCE = R"(4
6
2
1 2 1 10
1 3 4 -1
2 3 2 10
2 4 6 -1
3 4 1 10
2 2 3 10
1 4 5
2 4 3
)";

class SourceLPSizeTest : public ::testing::Test {
protected:
    std::string path = mcfcg::test::unique_test_path("source_lp_size.txt");

    void SetUp() override {
        std::ofstream f(path);
        ASSERT_TRUE(f.is_open());
        f << MIXED_INSTANCE;
    }
    void TearDown() override { std::remove(path.c_str()); }
};

TEST_F(SourceLPSizeTest, CountsCapacitatedAndSelfLoopArcs) {
    auto inst = mcfcg::read_commalab(path);
    const auto size = mcfcg::source_lp_size(inst);

    ASSERT_EQ(inst.sources.size(), 2u);
    EXPECT_EQ(size.capacitated_arcs, 4u);  // arcs 1, 3, 5, 6 (2 and 4 are INF)
    EXPECT_EQ(size.self_loop_arcs, 1u);    // arc 6: 2 -> 2

    EXPECT_EQ(size.cols, 12u);  // |S| * |E| = 2 * 6
    EXPECT_EQ(size.rows, 12u);  // |S| * |V| + capacitated = 2 * 4 + 4
    // Per source: 3 + 2 + 3 + 2 + 3 + 1 = 14 entries, twice.
    EXPECT_EQ(size.nnz, 28u);
}

// 3*|S|*|E| is an upper bound, not the count: 36 here against an actual 28.
TEST_F(SourceLPSizeTest, ExactNnzIsBelowTheThreeTimesUpperBound) {
    auto inst = mcfcg::read_commalab(path);
    const auto size = mcfcg::source_lp_size(inst);

    const uint64_t upper_bound = 3ULL * inst.sources.size() * inst.graph.num_arcs();
    EXPECT_EQ(upper_bound, 36u);
    EXPECT_LT(size.nnz, upper_bound);
}

// The guarantee that makes source_lp_size quotable: it must agree exactly with
// what build_source_lp emits. If append_source_column ever changes which
// entries it writes, this fails rather than letting the predictor drift.
TEST_F(SourceLPSizeTest, SourceLPSizeMatchesBuild) {
    auto inst = mcfcg::read_commalab(path);
    const auto size = mcfcg::source_lp_size(inst);
    const auto lp = mcfcg::build_source_lp(inst);

    EXPECT_EQ(size.cols, lp.num_cols);
    EXPECT_EQ(size.rows, lp.num_rows);
    EXPECT_EQ(size.nnz, lp.value.size());
    EXPECT_EQ(size.nnz, lp.row_index.size());
    EXPECT_EQ(size.nnz, lp.col_start.back());
}
