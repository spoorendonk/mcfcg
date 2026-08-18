#pragma once

#include <cstdint>
#include <vector>

#include "mcfcg/graph/d_ary_heap.h"
#include "mcfcg/graph/static_map.h"

namespace mcfcg {

// Reusable memory for Dijkstra/A* runs on the same graph topology.
// Owns the dense arrays (heap, status, distances, predecessors) so that
// successive shortest-path calls avoid per-source heap allocation.
// For multi-threading, use one workspace per thread.
struct dijkstra_workspace {
    using vertex = uint32_t;
    using arc = uint32_t;
    using length_type = int64_t;

    enum class VertexStatus : char { PreHeap = 0, InHeap = 1, PostHeap = 2 };

    static constexpr arc NO_PRED = ~arc{0};

    d_ary_heap<4, length_type> heap;
    static_map<vertex, VertexStatus> status;
    static_map<vertex, length_type> dist;
    static_map<vertex, arc> pred;

    dijkstra_workspace() = default;

    explicit dijkstra_workspace(uint32_t num_vertices)
        : heap(num_vertices),
          status(num_vertices, VertexStatus::PreHeap),
          dist(num_vertices),
          pred(num_vertices, NO_PRED) {}

    // Track a vertex whose status changed from VertexStatus::PreHeap.
    void touch(vertex v) noexcept { _touched.push_back(v); }

    // Reset for the next shortest-path run. Only clears heap and touched
    // vertices — O(touched) instead of O(V). dist/pred are not reset;
    // they are always written before being read, gated by status.
    void reset() noexcept {
        heap.clear();
        for (vertex v : _touched) {
            status[v] = VertexStatus::PreHeap;
        }
        _touched.clear();
    }

private:
    std::vector<vertex> _touched;
};

}  // namespace mcfcg
