#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "mcfcg/graph/static_digraph.h"
#include "mcfcg/graph/static_digraph_builder.h"

// Diamond graph: 0→1, 0→2, 1→3, 2→3
class DiamondGraph : public ::testing::Test {
   protected:
    mcfcg::static_digraph graph;

    void SetUp() override {
        mcfcg::static_digraph_builder<> builder(4);
        builder.add_arc(0, 1);
        builder.add_arc(0, 2);
        builder.add_arc(1, 3);
        builder.add_arc(2, 3);
        auto [g] = builder.build();
        graph = std::move(g);
    }
};

TEST_F(DiamondGraph, NumVerticesAndArcs) {
    EXPECT_EQ(graph.num_vertices(), 4u);
    EXPECT_EQ(graph.num_arcs(), 4u);
}

TEST_F(DiamondGraph, Vertices) {
    std::vector<uint32_t> verts;
    for (auto v : graph.vertices())
        verts.push_back(v);
    EXPECT_EQ(verts, (std::vector<uint32_t>{0, 1, 2, 3}));
}

TEST_F(DiamondGraph, OutArcs) {
    // Vertex 0 has 2 out-arcs
    std::vector<uint32_t> out0;
    for (auto a : graph.out_arcs(0))
        out0.push_back(a);
    EXPECT_EQ(out0.size(), 2u);

    // Check targets
    std::vector<uint32_t> targets;
    for (auto a : graph.out_arcs(0))
        targets.push_back(graph.arc_target(a));
    std::sort(targets.begin(), targets.end());
    EXPECT_EQ(targets, (std::vector<uint32_t>{1, 2}));

    // Vertex 3 has no out-arcs
    std::vector<uint32_t> out3;
    for (auto a : graph.out_arcs(3))
        out3.push_back(a);
    EXPECT_TRUE(out3.empty());
}

TEST_F(DiamondGraph, InArcs) {
    // Vertex 3 has 2 in-arcs
    std::vector<uint32_t> in3;
    for (auto a : graph.in_arcs(3))
        in3.push_back(a);
    EXPECT_EQ(in3.size(), 2u);

    std::vector<uint32_t> sources;
    for (auto a : graph.in_arcs(3))
        sources.push_back(graph.arc_source(a));
    std::sort(sources.begin(), sources.end());
    EXPECT_EQ(sources, (std::vector<uint32_t>{1, 2}));

    // Vertex 0 has no in-arcs
    std::vector<uint32_t> in0;
    for (auto a : graph.in_arcs(0))
        in0.push_back(a);
    EXPECT_TRUE(in0.empty());
}

TEST_F(DiamondGraph, ArcSourceTarget) {
    for (auto a : graph.arcs()) {
        uint32_t s = graph.arc_source(a);
        uint32_t t = graph.arc_target(a);
        EXPECT_TRUE(graph.is_valid_vertex(s));
        EXPECT_TRUE(graph.is_valid_vertex(t));
    }
}

TEST_F(DiamondGraph, CreateMaps) {
    auto vmap = graph.create_vertex_map<int>(0);
    EXPECT_EQ(vmap.size(), 4u);

    auto amap = graph.create_arc_map<double>(1.0);
    EXPECT_EQ(amap.size(), 4u);
    EXPECT_DOUBLE_EQ(amap[0], 1.0);
}

TEST(StaticDigraphBuilder, WithProperties) {
    mcfcg::static_digraph_builder<int32_t, double> builder(3);
    builder.add_arc(0, 1, 10, 1.5);
    builder.add_arc(0, 2, 20, 2.5);
    builder.add_arc(1, 2, 30, 3.5);

    auto [graph, cost_map, weight_map] = builder.build();

    EXPECT_EQ(graph.num_vertices(), 3u);
    EXPECT_EQ(graph.num_arcs(), 3u);

    // Verify property maps match arc order (sorted by source)
    // Arc 0: 0→1, Arc 1: 0→2, Arc 2: 1→2
    EXPECT_EQ(cost_map[0], 10);
    EXPECT_EQ(cost_map[1], 20);
    EXPECT_EQ(cost_map[2], 30);
    EXPECT_DOUBLE_EQ(weight_map[0], 1.5);
    EXPECT_DOUBLE_EQ(weight_map[1], 2.5);
    EXPECT_DOUBLE_EQ(weight_map[2], 3.5);
}
