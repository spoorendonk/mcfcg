#pragma once

#include "mcfcg/graph/d_ary_heap.h"
#include "mcfcg/graph/static_map.h"

#include <cstdint>
#include <vector>

namespace mcfcg {

// Reusable memory for Dijkstra/A* runs on the same graph topology.
// Owns the dense arrays (heap, status, distances, predecessors) so that
// successive shortest-path calls avoid per-source heap allocation.
// For multi-threading, use one workspace per thread.
struct dijkstra_workspace {
    using vertex = uint32_t;
    using arc = uint32_t;
    using length_type = int64_t;

    enum vertex_status : char { PRE_HEAP = 0, IN_HEAP = 1, POST_HEAP = 2 };

    static constexpr arc NO_PRED = ~arc{0};

    d_ary_heap<4, length_type> heap;
    static_map<vertex, vertex_status> status;
    static_map<vertex, length_type> dist;
    static_map<vertex, arc> pred;

    dijkstra_workspace() = default;

    explicit dijkstra_workspace(uint32_t num_vertices)
        : heap(num_vertices),
          status(num_vertices, PRE_HEAP),
          dist(num_vertices),
          pred(num_vertices, NO_PRED) {}

    // Track a vertex whose status changed from PRE_HEAP.
    void touch(vertex v) noexcept { _touched.push_back(v); }

    // Reset for the next shortest-path run. Only clears heap and touched
    // vertices — O(touched) instead of O(V). dist/pred are not reset;
    // they are always written before being read, gated by status.
    void reset() noexcept {
        heap.clear();
        for (vertex v : _touched) {
            status[v] = PRE_HEAP;
        }
        _touched.clear();
    }

private:
    std::vector<vertex> _touched;
};

}  // namespace mcfcg
