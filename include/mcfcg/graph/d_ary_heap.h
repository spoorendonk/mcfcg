#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "mcfcg/graph/static_map.h"

namespace mcfcg {

// Updatable d-ary min-heap keyed by vertex ID with priority type P.
// Index map uses static_map for O(1) lookup.
template <std::size_t D, typename P>
class d_ary_heap {
   public:
    using vertex = uint32_t;
    using priority_type = P;
    using size_type = std::size_t;

    static constexpr size_type INVALID = std::numeric_limits<size_type>::max();

   private:
    struct entry {
        vertex v;
        priority_type p;
    };

    std::vector<entry> _heap;
    static_map<vertex, size_type> _index;

   public:
    d_ary_heap() = default;

    explicit d_ary_heap(uint32_t num_vertices)
        : _index(num_vertices, INVALID) {}

    void init(uint32_t num_vertices) {
        _heap.clear();
        _index = static_map<vertex, size_type>(num_vertices, INVALID);
    }

    bool empty() const noexcept { return _heap.empty(); }
    size_type size() const noexcept { return _heap.size(); }

    entry top() const noexcept {
        assert(!empty());
        return _heap[0];
    }

    vertex top_vertex() const noexcept { return top().v; }
    priority_type top_priority() const noexcept { return top().p; }

    bool contains(vertex v) const noexcept { return _index[v] != INVALID; }

    priority_type priority(vertex v) const noexcept {
        assert(contains(v));
        return _heap[_index[v]].p;
    }

    void push(vertex v, priority_type p) noexcept {
        assert(!contains(v));
        size_type pos = _heap.size();
        _heap.push_back({v, p});
        _index[v] = pos;
        sift_up(pos);
    }

    void pop() noexcept {
        assert(!empty());
        _index[_heap[0].v] = INVALID;
        if (_heap.size() > 1) {
            _heap[0] = _heap.back();
            _index[_heap[0].v] = 0;
            _heap.pop_back();
            sift_down(0);
        } else {
            _heap.pop_back();
        }
    }

    void promote(vertex v, priority_type p) noexcept {
        assert(contains(v));
        size_type pos = _index[v];
        assert(p <= _heap[pos].p);
        _heap[pos].p = p;
        sift_up(pos);
    }

    void push_or_promote(vertex v, priority_type p) noexcept {
        if (contains(v)) {
            if (p < priority(v)) {
                promote(v, p);
            }
        } else {
            push(v, p);
        }
    }

    void clear() noexcept {
        for (auto & e : _heap) {
            _index[e.v] = INVALID;
        }
        _heap.clear();
    }

   private:
    static constexpr size_type parent_of(size_type i) noexcept {
        return (i - 1) / D;
    }
    static constexpr size_type first_child_of(size_type i) noexcept {
        return i * D + 1;
    }

    void sift_up(size_type pos) noexcept {
        entry e = _heap[pos];
        while (pos > 0) {
            size_type parent = parent_of(pos);
            if (!(e.p < _heap[parent].p)) {
                break;
            }
            _heap[pos] = _heap[parent];
            _index[_heap[pos].v] = pos;
            pos = parent;
        }
        _heap[pos] = e;
        _index[e.v] = pos;
    }

    void sift_down(size_type pos) noexcept {
        entry e = _heap[pos];
        size_type n = _heap.size();
        for (;;) {
            size_type child = first_child_of(pos);
            if (child >= n) {
                break;
            }
            size_type best = child;
            size_type last = std::min(child + D, n);
            for (size_type c = child + 1; c < last; ++c) {
                if (_heap[c].p < _heap[best].p) {
                    best = c;
                }
            }
            if (!(_heap[best].p < e.p)) {
                break;
            }
            _heap[pos] = _heap[best];
            _index[_heap[pos].v] = pos;
            pos = best;
        }
        _heap[pos] = e;
        _index[e.v] = pos;
    }
};

}  // namespace mcfcg
