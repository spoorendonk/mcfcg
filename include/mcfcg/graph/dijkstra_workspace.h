#pragma once

#include "mcfcg/graph/d_ary_heap.h"
#include "mcfcg/graph/static_map.h"

#include <cstdint>
#include <optional>

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

    d_ary_heap<4, length_type> heap;
    static_map<vertex, vertex_status> status;
    static_map<vertex, length_type> dist;
    static_map<vertex, std::optional<arc>> pred;

    dijkstra_workspace() = default;

    explicit dijkstra_workspace(uint32_t num_vertices)
        : heap(num_vertices),
          status(num_vertices, PRE_HEAP),
          dist(num_vertices),
          pred(num_vertices, std::nullopt) {}

    // Only clears heap and status. dist/pred are not reset — they are
    // always written before being read, gated by status (PRE_HEAP → never read).
    void reset() noexcept {
        heap.clear();
        status.fill(PRE_HEAP);
    }
};

}  // namespace mcfcg
