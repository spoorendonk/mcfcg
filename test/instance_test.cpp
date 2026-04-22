#include "mcfcg/instance.h"

#include "test_paths.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>

// CommaLab/UniPi plain-numeric format:
// 4 vertices, 5 arcs, 3 commodities
static const char* SMALL_INSTANCE = R"(4
5
3
1 2 1 10
1 3 4 10
2 3 2 10
2 4 6 10
3 4 1 10
1 4 5
1 3 3
2 4 2
)";

class InstanceTest : public ::testing::Test {
protected:
    std::string path = mcfcg::test::unique_test_path("small_test.txt");

    void SetUp() override {
        std::ofstream f(path);
        ASSERT_TRUE(f.is_open());
        f << SMALL_INSTANCE;
    }
    void TearDown() override { std::remove(path.c_str()); }
};

TEST_F(InstanceTest, ReadCommalab) {
    auto inst = mcfcg::read_commalab(path);

    EXPECT_EQ(inst.graph.num_vertices(), 4u);
    EXPECT_EQ(inst.graph.num_arcs(), 5u);
    EXPECT_EQ(inst.commodities.size(), 3u);

    EXPECT_EQ(inst.commodities[0].source, 0u);
    EXPECT_EQ(inst.commodities[0].sink, 3u);
    EXPECT_DOUBLE_EQ(inst.commodities[0].demand, 5.0);

    EXPECT_EQ(inst.sources.size(), 2u);
}

TEST_F(InstanceTest, SourceGrouping) {
    auto inst = mcfcg::read_commalab(path);

    for (auto& src : inst.sources) {
        for (uint32_t k : src.commodity_indices) {
            EXPECT_EQ(inst.commodities[k].source, src.vertex);
        }
    }
}

TEST_F(InstanceTest, RoundTrip) {
    auto inst = mcfcg::read_commalab(path);

    std::string out_path = mcfcg::test::unique_test_path("small_test_roundtrip.txt");
    mcfcg::write_commalab(inst, out_path);

    auto inst2 = mcfcg::read_commalab(out_path);

    EXPECT_EQ(inst2.graph.num_vertices(), inst.graph.num_vertices());
    EXPECT_EQ(inst2.graph.num_arcs(), inst.graph.num_arcs());
    EXPECT_EQ(inst2.commodities.size(), inst.commodities.size());

    for (uint32_t a : inst.graph.arcs()) {
        EXPECT_EQ(inst2.graph.arc_source(a), inst.graph.arc_source(a));
        EXPECT_EQ(inst2.graph.arc_target(a), inst.graph.arc_target(a));
        EXPECT_DOUBLE_EQ(inst2.cost[a], inst.cost[a]);
        EXPECT_DOUBLE_EQ(inst2.capacity[a], inst.capacity[a]);
    }

    for (size_t i = 0; i < inst.commodities.size(); ++i) {
        EXPECT_EQ(inst2.commodities[i].source, inst.commodities[i].source);
        EXPECT_EQ(inst2.commodities[i].sink, inst.commodities[i].sink);
        EXPECT_DOUBLE_EQ(inst2.commodities[i].demand, inst.commodities[i].demand);
    }

    std::remove(out_path.c_str());
}

// Intermodal-format round-trip: write_commalab must preserve fractional
// costs (walking/waiting arcs in Lienkamp's temp network) and +INF
// capacities (uncapped arcs) end-to-end.  Previously write_commalab
// used llround(), which truncated 0.5 to 0 and mapped INF to the
// implementation-defined LLONG_MIN sentinel.
TEST_F(InstanceTest, RoundTripFractionalCostAndInfCap) {
    static const char* instance = R"(3
2
1
1 2 0.5 -1
2 3 1.5 6
1 3 1
)";
    std::string in_path = mcfcg::test::unique_test_path("inf_test.txt");
    {
        std::ofstream f(in_path);
        f << instance;
    }
    auto inst = mcfcg::read_commalab(in_path);

    ASSERT_EQ(inst.graph.num_arcs(), 2u);
    // -1 cap should have been mapped to INF on read.
    EXPECT_TRUE(std::isinf(inst.capacity[0]));
    EXPECT_DOUBLE_EQ(inst.cost[0], 0.5);
    EXPECT_DOUBLE_EQ(inst.capacity[1], 6.0);
    EXPECT_DOUBLE_EQ(inst.cost[1], 1.5);

    std::string out_path = mcfcg::test::unique_test_path("inf_test_roundtrip.txt");
    mcfcg::write_commalab(inst, out_path);

    auto inst2 = mcfcg::read_commalab(out_path);
    EXPECT_TRUE(std::isinf(inst2.capacity[0])) << "INF capacity lost in round-trip";
    EXPECT_DOUBLE_EQ(inst2.cost[0], 0.5) << "Fractional cost lost in round-trip";
    EXPECT_DOUBLE_EQ(inst2.cost[1], 1.5) << "Fractional cost lost in round-trip";

    std::remove(in_path.c_str());
    std::remove(out_path.c_str());
}
