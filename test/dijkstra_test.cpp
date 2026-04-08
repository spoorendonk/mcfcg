#include "mcfcg/graph/dijkstra.h"

#include "mcfcg/graph/dijkstra_workspace.h"
#include "mcfcg/graph/static_digraph_builder.h"

#include <gtest/gtest.h>
#include <vector>

// Diamond graph with costs: 0→1(1), 0→2(4), 1→3(6), 2→3(1)
// Shortest path 0→3: 0→2→3 with cost 5
class DijkstraTest : public ::testing::Test {
protected:
    mcfcg::static_digraph graph;
    mcfcg::static_map<uint32_t, int64_t> lengths;
    mcfcg::dijkstra_workspace ws;

    void SetUp() override {
        mcfcg::static_digraph_builder<int64_t> builder(4);
        builder.add_arc(0, 1, 1);
        builder.add_arc(0, 2, 4);
        builder.add_arc(1, 3, 6);
        builder.add_arc(2, 3, 1);
        auto [g, len] = builder.build();
        graph = std::move(g);
        lengths = std::move(len);
        ws = mcfcg::dijkstra_workspace(graph.num_vertices());
    }
};

TEST_F(DijkstraTest, ShortestDistances) {
    mcfcg::dijkstra<mcfcg::dijkstra_store_distances> d(graph, lengths, ws);
    d.add_source(0);
    d.run();

    EXPECT_TRUE(d.visited(0));
    EXPECT_TRUE(d.visited(1));
    EXPECT_TRUE(d.visited(2));
    EXPECT_TRUE(d.visited(3));

    EXPECT_EQ(d.dist(0), 0);
    EXPECT_EQ(d.dist(1), 1);
    EXPECT_EQ(d.dist(2), 4);
    EXPECT_EQ(d.dist(3), 5);  // 0→2→3
}

TEST_F(DijkstraTest, ShortestPaths) {
    mcfcg::dijkstra<mcfcg::dijkstra_store_paths> d(graph, lengths, ws);
    d.add_source(0);
    d.run();

    EXPECT_EQ(d.dist(3), 5);

    // Trace path from 3 to source
    std::vector<uint32_t> path;
    uint32_t v = 3;
    while (d.has_pred(v)) {
        uint32_t a = d.pred_arc(v);
        path.push_back(a);
        v = graph.arc_source(a);
    }
    EXPECT_EQ(v, 0u);  // Reached source

    // Path should be 2 arcs: arc(2→3) then arc(0→2)
    EXPECT_EQ(path.size(), 2u);
}

TEST_F(DijkstraTest, StepByStep) {
    mcfcg::dijkstra<mcfcg::dijkstra_store_distances> d(graph, lengths, ws);
    d.add_source(0);

    EXPECT_FALSE(d.finished());

    // First settled: vertex 0 with dist 0
    auto [v0, d0] = d.current();
    EXPECT_EQ(v0, 0u);
    EXPECT_EQ(d0, 0);
    d.advance();

    // Next: vertex 1 with dist 1
    auto [v1, d1] = d.current();
    EXPECT_EQ(v1, 1u);
    EXPECT_EQ(d1, 1);
    d.advance();

    // Next: vertex 2 with dist 4
    auto [v2, d2] = d.current();
    EXPECT_EQ(v2, 2u);
    EXPECT_EQ(d2, 4);
    d.advance();

    // Next: vertex 3 with dist 5
    auto [v3, d3] = d.current();
    EXPECT_EQ(v3, 3u);
    EXPECT_EQ(d3, 5);
    d.advance();

    EXPECT_TRUE(d.finished());
}

TEST_F(DijkstraTest, Reset) {
    mcfcg::dijkstra<mcfcg::dijkstra_store_distances> d(graph, lengths, ws);
    d.add_source(0);
    d.run();
    EXPECT_EQ(d.dist(3), 5);

    d.reset();
    d.add_source(1);
    d.run();
    EXPECT_EQ(d.dist(3), 6);
    EXPECT_FALSE(d.reached(0));
}

TEST(DijkstraDefault, DefaultTraitsReachability) {
    mcfcg::static_digraph_builder<int64_t> builder(2);
    builder.add_arc(0, 1, 5);
    auto [g, len] = builder.build();

    mcfcg::dijkstra_workspace ws(g.num_vertices());
    mcfcg::dijkstra<> d(g, len, ws);
    d.add_source(0);
    d.run();
    EXPECT_TRUE(d.visited(1));
}
