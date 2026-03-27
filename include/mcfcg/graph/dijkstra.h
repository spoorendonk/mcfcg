#pragma once

#include <cassert>
#include <cstdint>
#include <optional>

#include "mcfcg/graph/d_ary_heap.h"
#include "mcfcg/graph/semiring.h"
#include "mcfcg/graph/static_digraph.h"
#include "mcfcg/graph/static_map.h"

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
template <typename Traits = dijkstra_default_traits>
class dijkstra {
   public:
    using vertex = uint32_t;
    using arc = uint32_t;
    using length_type = int64_t;
    using semiring = shortest_path_semiring<length_type>;

    enum vertex_status : char { PRE_HEAP = 0, IN_HEAP = 1, POST_HEAP = 2 };

   private:
    const static_digraph * _graph = nullptr;
    const static_map<arc, length_type> * _length_map = nullptr;

    d_ary_heap<4, length_type> _heap;
    static_map<vertex, vertex_status> _status;

    struct dist_map_holder {
        static_map<vertex, length_type> _distances;
    };
    struct empty_dist_holder {
        // nothing
    };

    struct pred_map_holder {
        static_map<vertex, std::optional<arc>> _pred_arcs;
    };
    struct empty_pred_holder {
        // nothing
    };

    using dist_storage = std::conditional_t<Traits::store_distances,
                                            dist_map_holder, empty_dist_holder>;
    using pred_storage = std::conditional_t<Traits::store_paths,
                                            pred_map_holder, empty_pred_holder>;

    [[no_unique_address]] dist_storage _dist_s;
    [[no_unique_address]] pred_storage _pred_s;

   public:
    dijkstra() = default;

    dijkstra(const static_digraph & g,
             const static_map<arc, length_type> & lengths)
        : _graph(&g),
          _length_map(&lengths),
          _heap(g.num_vertices()),
          _status(g.num_vertices(), PRE_HEAP) {
        if constexpr (Traits::store_distances) {
            _dist_s._distances =
                static_map<vertex, length_type>(g.num_vertices());
        }
        if constexpr (Traits::store_paths) {
            _pred_s._pred_arcs =
                static_map<vertex, std::optional<arc>>(g.num_vertices());
        }
    }

    void reset() noexcept {
        _heap.clear();
        _status.fill(PRE_HEAP);
    }

    void add_source(vertex s, length_type dist = semiring::zero) noexcept {
        assert(_status[s] != IN_HEAP);
        _heap.push(s, dist);
        _status[s] = IN_HEAP;
        if constexpr (Traits::store_distances) {
            _dist_s._distances[s] = dist;
        }
        if constexpr (Traits::store_paths) {
            _pred_s._pred_arcs[s].reset();
        }
    }

    bool finished() const noexcept { return _heap.empty(); }

    std::pair<vertex, length_type> current() const noexcept {
        assert(!finished());
        auto top = _heap.top();
        return {top.v, top.p};
    }

    void advance() noexcept {
        assert(!finished());
        auto [u, u_dist] = current();
        if constexpr (Traits::store_distances) {
            _dist_s._distances[u] = u_dist;
        }
        _status[u] = POST_HEAP;
        _heap.pop();

        for (arc a : _graph->out_arcs(u)) {
            vertex w = _graph->arc_target(a);
            if (_status[w] == POST_HEAP)
                continue;

            length_type new_dist = semiring::plus(u_dist, (*_length_map)[a]);

            if (_status[w] == IN_HEAP) {
                if (semiring::less(new_dist, _heap.priority(w))) {
                    _heap.promote(w, new_dist);
                    if constexpr (Traits::store_paths) {
                        _pred_s._pred_arcs[w].emplace(a);
                    }
                }
            } else {
                _heap.push(w, new_dist);
                _status[w] = IN_HEAP;
                if constexpr (Traits::store_paths) {
                    _pred_s._pred_arcs[w].emplace(a);
                }
            }
        }
    }

    void run() noexcept {
        while (!finished())
            advance();
    }

    bool reached(vertex u) const noexcept { return _status[u] != PRE_HEAP; }
    bool visited(vertex u) const noexcept { return _status[u] == POST_HEAP; }

    length_type dist(vertex u) const noexcept
        requires(Traits::store_distances)
    {
        assert(visited(u));
        return _dist_s._distances[u];
    }

    arc pred_arc(vertex u) const noexcept
        requires(Traits::store_paths)
    {
        assert(reached(u) && _pred_s._pred_arcs[u].has_value());
        return _pred_s._pred_arcs[u].value();
    }

    bool has_pred(vertex u) const noexcept
        requires(Traits::store_paths)
    {
        return reached(u) && _pred_s._pred_arcs[u].has_value();
    }
};

}  // namespace mcfcg
