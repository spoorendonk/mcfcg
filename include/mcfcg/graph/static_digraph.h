#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <ranges>
#include <span>

#include "mcfcg/graph/static_map.h"

namespace mcfcg {

class static_digraph {
   public:
    using vertex = uint32_t;
    using arc = uint32_t;

   private:
    static_map<vertex, arc> _out_arc_begin;
    static_map<arc, vertex> _arc_target;
    static_map<arc, vertex> _arc_source;
    static_map<vertex, arc> _in_arc_begin;
    static_map<arc, arc> _in_arcs;

   public:
    static_digraph() = default;
    static_digraph(const static_digraph &) = default;
    static_digraph(static_digraph &&) = default;
    static_digraph & operator=(const static_digraph &) = default;
    static_digraph & operator=(static_digraph &&) = default;

    template <std::ranges::forward_range S, std::ranges::forward_range T>
        requires std::convertible_to<std::ranges::range_value_t<S>, vertex> &&
                     std::convertible_to<std::ranges::range_value_t<T>, vertex>
    static_digraph(std::size_t num_verts, S && sources, T && targets)
        : _out_arc_begin(num_verts, arc{0}),
          _arc_target(std::forward<T>(targets)),
          _arc_source(std::forward<S>(sources)),
          _in_arc_begin(num_verts, arc{0}),
          _in_arcs(_arc_target.size()) {
        assert(std::ranges::is_sorted(sources));
        static_map<vertex, arc> in_count(num_verts, arc{0});
        for (auto && s : sources)
            ++_out_arc_begin[s];
        for (auto && t : targets)
            ++in_count[t];
        std::exclusive_scan(_out_arc_begin.data(),
                            _out_arc_begin.data() + num_verts,
                            _out_arc_begin.data(), arc{0});
        std::exclusive_scan(in_count.data(), in_count.data() + num_verts,
                            _in_arc_begin.data(), arc{0});
        for (auto a : arcs()) {
            vertex t = _arc_target[a];
            --in_count[t];
            _in_arcs[_in_arc_begin[t] + in_count[t]] = a;
        }
    }

    constexpr uint32_t num_vertices() const noexcept {
        return static_cast<uint32_t>(_out_arc_begin.size());
    }
    constexpr uint32_t num_arcs() const noexcept {
        return static_cast<uint32_t>(_arc_target.size());
    }

    constexpr bool is_valid_vertex(vertex u) const noexcept {
        return u < num_vertices();
    }
    constexpr bool is_valid_arc(arc a) const noexcept { return a < num_arcs(); }

    constexpr auto vertices() const noexcept {
        return std::views::iota(vertex{0}, num_vertices());
    }
    constexpr auto arcs() const noexcept {
        return std::views::iota(arc{0}, num_arcs());
    }

    constexpr auto out_arcs(vertex u) const noexcept {
        assert(is_valid_vertex(u));
        arc begin = _out_arc_begin[u];
        arc end = (u + 1 < num_vertices()) ? _out_arc_begin[u + 1] : num_arcs();
        return std::views::iota(begin, end);
    }

    constexpr auto in_arcs(vertex u) const noexcept {
        assert(is_valid_vertex(u));
        const arc * begin = _in_arcs.data() + _in_arc_begin[u];
        const arc * end = (u + 1 < num_vertices())
                              ? _in_arcs.data() + _in_arc_begin[u + 1]
                              : _in_arcs.data() + num_arcs();
        return std::span<const arc>(begin, end);
    }

    constexpr vertex arc_source(arc a) const noexcept {
        assert(is_valid_arc(a));
        return _arc_source[a];
    }
    constexpr vertex arc_target(arc a) const noexcept {
        assert(is_valid_arc(a));
        return _arc_target[a];
    }

    template <typename T>
    constexpr auto create_vertex_map() const noexcept {
        return static_map<vertex, T>(num_vertices());
    }
    template <typename T>
    constexpr auto create_vertex_map(const T & val) const noexcept {
        return static_map<vertex, T>(num_vertices(), val);
    }

    template <typename T>
    constexpr auto create_arc_map() const noexcept {
        return static_map<arc, T>(num_arcs());
    }
    template <typename T>
    constexpr auto create_arc_map(const T & val) const noexcept {
        return static_map<arc, T>(num_arcs(), val);
    }
};

}  // namespace mcfcg
