#pragma once

#include "mcfcg/graph/dijkstra_workspace.h"
#include "mcfcg/graph/semiring.h"
#include "mcfcg/graph/static_digraph.h"
#include "mcfcg/graph/static_map.h"

#include <cassert>
#include <cstdint>

namespace mcfcg {

struct dijkstra_default_traits {
    static constexpr bool store_distances = false;
    static constexpr bool store_paths = false;
};

struct dijkstra_store_distances {
    static constexpr bool store_distances = true;
    static constexpr bool store_paths = false;
};

struct dijkstra_store_paths {
    static constexpr bool store_distances = true;
    static constexpr bool store_paths = true;
};

// Dijkstra's algorithm on static_digraph with int64_t arc lengths.
// Traits control whether distances and predecessor arcs are stored.
// Memory is borrowed from a dijkstra_workspace to avoid per-call allocation.
template <typename Traits = dijkstra_default_traits>
class dijkstra {
    static_assert(!Traits::store_paths || Traits::store_distances,
                  "store_paths requires store_distances");

public:
    using vertex = uint32_t;
    using arc = uint32_t;
    using length_type = int64_t;
    using semiring = shortest_path_semiring<length_type>;
    using vertex_status = dijkstra_workspace::vertex_status;

private:
    const static_digraph* _graph = nullptr;
    const static_map<arc, length_type>* _length_map = nullptr;
    dijkstra_workspace* _ws = nullptr;

public:
    dijkstra() = default;

    dijkstra(const static_digraph& g, const static_map<arc, length_type>& lengths,
             dijkstra_workspace& ws)
        : _graph(&g), _length_map(&lengths), _ws(&ws) {}

    void reset() noexcept { _ws->reset(); }

    void add_source(vertex s, length_type dist = semiring::zero) noexcept {
        assert(_ws->status[s] != vertex_status::IN_HEAP);
        _ws->heap.push(s, dist);
        _ws->status[s] = vertex_status::IN_HEAP;
        _ws->touch(s);
        if constexpr (Traits::store_distances) {
            _ws->dist[s] = dist;
        }
        if constexpr (Traits::store_paths) {
            _ws->pred[s] = dijkstra_workspace::NO_PRED;
        }
    }

    bool finished() const noexcept { return _ws->heap.empty(); }

    std::pair<vertex, length_type> current() const noexcept {
        assert(!finished());
        auto top = _ws->heap.top();
        return {top.v, top.p};
    }

    void advance() noexcept {
        assert(!finished());
        auto [u, u_dist] = current();
        if constexpr (Traits::store_distances) {
            _ws->dist[u] = u_dist;
        }
        _ws->status[u] = vertex_status::POST_HEAP;
        _ws->heap.pop();

        for (arc a : _graph->out_arcs(u)) {
            vertex w = _graph->arc_target(a);
            if (_ws->status[w] == vertex_status::POST_HEAP)
                continue;

            length_type new_dist = semiring::plus(u_dist, (*_length_map)[a]);

            if (_ws->status[w] == vertex_status::IN_HEAP) {
                if (semiring::less(new_dist, _ws->heap.priority(w))) {
                    _ws->heap.promote(w, new_dist);
                    if constexpr (Traits::store_paths) {
                        _ws->pred[w] = a;
                    }
                }
            } else {
                _ws->heap.push(w, new_dist);
                _ws->status[w] = vertex_status::IN_HEAP;
                _ws->touch(w);
                if constexpr (Traits::store_paths) {
                    _ws->pred[w] = a;
                }
            }
        }
    }

    void run() noexcept {
        while (!finished())
            advance();
    }

    // Run until the heap is empty or the minimum key exceeds bound.
    void run_until(length_type bound) noexcept {
        while (!finished() && current().second <= bound)
            advance();
    }

    bool reached(vertex u) const noexcept { return _ws->status[u] != vertex_status::PRE_HEAP; }
    bool visited(vertex u) const noexcept { return _ws->status[u] == vertex_status::POST_HEAP; }

    length_type dist(vertex u) const noexcept
        requires(Traits::store_distances)
    {
        assert(visited(u));
        return _ws->dist[u];
    }

    arc pred_arc(vertex u) const noexcept
        requires(Traits::store_paths)
    {
        assert(reached(u) && _ws->pred[u] != dijkstra_workspace::NO_PRED);
        return _ws->pred[u];
    }

    bool has_pred(vertex u) const noexcept
        requires(Traits::store_paths)
    {
        return reached(u) && _ws->pred[u] != dijkstra_workspace::NO_PRED;
    }
};

// A* search on static_digraph with int64_t arc lengths and a vertex heuristic.
// The heuristic h[v] must be admissible (non-negative lower bound on distance
// from v to the nearest target). Heap priority is f(v) = g(v) + h(v).
// Supports early termination via a target count: stop after settling a given
// number of target vertices.
// Memory is borrowed from a dijkstra_workspace to avoid per-call allocation.
template <typename Traits = dijkstra_store_paths>
class astar_dijkstra {
public:
    using vertex = uint32_t;
    using arc = uint32_t;
    using length_type = int64_t;
    using semiring = shortest_path_semiring<length_type>;
    using vertex_status = dijkstra_workspace::vertex_status;

private:
    const static_digraph* _graph = nullptr;
    const static_map<arc, length_type>* _length_map = nullptr;
    const static_map<vertex, length_type>* _heuristic = nullptr;
    dijkstra_workspace* _ws = nullptr;

public:
    astar_dijkstra() = default;

    astar_dijkstra(const static_digraph& g, const static_map<arc, length_type>& lengths,
                   const static_map<vertex, length_type>& heuristic, dijkstra_workspace& ws)
        : _graph(&g), _length_map(&lengths), _heuristic(&heuristic), _ws(&ws) {}

    void reset() noexcept { _ws->reset(); }

    void add_source(vertex s, length_type dist = semiring::zero) noexcept {
        assert(_ws->status[s] != vertex_status::IN_HEAP);
        _ws->dist[s] = dist;
        length_type f = semiring::plus(dist, (*_heuristic)[s]);
        _ws->heap.push(s, f);
        _ws->status[s] = vertex_status::IN_HEAP;
        _ws->touch(s);
        if constexpr (Traits::store_paths) {
            _ws->pred[s] = dijkstra_workspace::NO_PRED;
        }
    }

    bool finished() const noexcept { return _ws->heap.empty(); }

    // Returns the vertex being settled and its true distance g(v).
    std::pair<vertex, length_type> settle_next() noexcept {
        assert(!finished());
        auto top = _ws->heap.top();
        vertex u = top.v;
        length_type u_dist = _ws->dist[u];
        _ws->status[u] = vertex_status::POST_HEAP;
        _ws->heap.pop();

        for (arc a : _graph->out_arcs(u)) {
            vertex w = _graph->arc_target(a);
            if (_ws->status[w] == vertex_status::POST_HEAP)
                continue;

            length_type new_g = semiring::plus(u_dist, (*_length_map)[a]);
            length_type new_f = semiring::plus(new_g, (*_heuristic)[w]);

            if (_ws->status[w] == vertex_status::IN_HEAP) {
                if (semiring::less(new_g, _ws->dist[w])) {
                    _ws->dist[w] = new_g;
                    _ws->heap.promote(w, new_f);
                    if constexpr (Traits::store_paths) {
                        _ws->pred[w] = a;
                    }
                }
            } else {
                _ws->dist[w] = new_g;
                _ws->heap.push(w, new_f);
                _ws->status[w] = vertex_status::IN_HEAP;
                _ws->touch(w);
                if constexpr (Traits::store_paths) {
                    _ws->pred[w] = a;
                }
            }
        }

        return {u, u_dist};
    }

    void run() noexcept {
        while (!finished())
            settle_next();
    }

    // Run until num_targets target vertices have been settled.
    // Mutates is_target (sets settled targets to false) and decrements
    // targets_remaining.  Caller initializes both before calling.
    void run_until_targets(static_map<vertex, bool>& is_target,
                           uint32_t& targets_remaining) noexcept {
        while (!finished() && targets_remaining > 0) {
            auto [v, _] = settle_next();
            if (is_target[v]) {
                is_target[v] = false;
                --targets_remaining;
            }
        }
    }

    bool reached(vertex u) const noexcept { return _ws->status[u] != vertex_status::PRE_HEAP; }
    bool visited(vertex u) const noexcept { return _ws->status[u] == vertex_status::POST_HEAP; }

    length_type dist(vertex u) const noexcept
        requires(Traits::store_distances)
    {
        assert(visited(u));
        return _ws->dist[u];
    }

    arc pred_arc(vertex u) const noexcept
        requires(Traits::store_paths)
    {
        assert(reached(u) && _ws->pred[u] != dijkstra_workspace::NO_PRED);
        return _ws->pred[u];
    }

    bool has_pred(vertex u) const noexcept
        requires(Traits::store_paths)
    {
        return reached(u) && _ws->pred[u] != dijkstra_workspace::NO_PRED;
    }
};

}  // namespace mcfcg
