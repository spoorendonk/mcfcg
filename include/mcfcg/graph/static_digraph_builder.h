#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <vector>

#include "mcfcg/graph/static_digraph.h"
#include "mcfcg/graph/static_map.h"

namespace mcfcg {

template <typename... Props>
class static_digraph_builder {
   public:
    using vertex = uint32_t;
    using arc = uint32_t;

    struct arc_entry {
        vertex source;
        vertex target;
        std::tuple<Props...> props;
    };

   private:
    uint32_t _num_vertices;
    std::vector<arc_entry> _arcs;

   public:
    explicit static_digraph_builder(uint32_t num_vertices)
        : _num_vertices(num_vertices) {}

    void add_arc(vertex u, vertex v, Props... props) {
        _arcs.push_back({u, v, std::tuple<Props...>{std::move(props)...}});
    }

    auto build() {
        std::sort(_arcs.begin(), _arcs.end(),
                  [](const arc_entry & a, const arc_entry & b) {
                      return a.source < b.source;
                  });

        uint32_t m = static_cast<uint32_t>(_arcs.size());
        std::vector<vertex> sources(m);
        std::vector<vertex> targets(m);
        for (uint32_t i = 0; i < m; ++i) {
            sources[i] = _arcs[i].source;
            targets[i] = _arcs[i].target;
        }

        static_digraph graph(_num_vertices, sources, targets);

        auto prop_maps = build_prop_maps(std::index_sequence_for<Props...>{});

        return std::tuple_cat(std::make_tuple(std::move(graph)),
                              std::move(prop_maps));
    }

   private:
    template <std::size_t... Is>
    auto build_prop_maps(std::index_sequence<Is...>) {
        return std::make_tuple(build_one_map<Is>()...);
    }

    template <std::size_t I>
    auto build_one_map() {
        using T = std::tuple_element_t<I, std::tuple<Props...>>;
        uint32_t m = static_cast<uint32_t>(_arcs.size());
        static_map<arc, T> map(m);
        for (uint32_t i = 0; i < m; ++i) {
            map[i] = std::get<I>(_arcs[i].props);
        }
        return map;
    }
};

}  // namespace mcfcg
