#pragma once

#include "mcfcg/cg/column.h"
#include "mcfcg/graph/dijkstra.h"
#include "mcfcg/instance.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mcfcg {

enum class PricingMode { Dijkstra, AStar };

// Compute lower bounds from every vertex to the nearest sink using original
// (unscaled) arc costs on the reverse graph.  All unique sink vertices across
// all commodities are seeded with distance 0 and a single multi-source reverse
// Dijkstra is run.  The result is an admissible A* heuristic that is stable
// across CG iterations (original costs never change).
//
// Admissibility: capacity duals mu_e <= 0 (non-positive for <= constraints in
// minimization), so reduced costs c_e - mu_e >= c_e.  The heuristic using
// original costs c_e underestimates the shortest path under reduced costs,
// guaranteeing A* optimality.
inline static_map<uint32_t, int64_t> compute_lower_bounds_to_targets(const Instance& inst,
                                                                     double scale) {
    const auto& g = inst.graph;
    using semiring_t = shortest_path_semiring<int64_t>;

    // Scale original costs to int64_t (same scale as pricing arc costs).
    auto orig_cost_scaled = g.create_arc_map<int64_t>();
    for (auto a : g.arcs()) {
        double c = inst.cost[a];
        orig_cost_scaled[a] = (c <= 0.0) ? int64_t{0} : static_cast<int64_t>(std::round(c * scale));
    }

    // Collect all unique sink vertices.
    auto status = g.create_vertex_map<char>(0);  // 0=pre, 1=in, 2=post
    d_ary_heap<4, int64_t> heap(g.num_vertices());
    auto dist = g.create_vertex_map<int64_t>();

    for (const auto& comm : inst.commodities) {
        uint32_t sink = comm.sink;
        if (status[sink] == 0) {
            heap.push(sink, int64_t{0});
            status[sink] = 1;
            dist[sink] = int64_t{0};
        }
    }

    // Multi-source reverse Dijkstra using in_arcs (reverse direction).
    while (!heap.empty()) {
        auto top = heap.top();
        uint32_t u = top.v;
        int64_t u_dist = top.p;
        dist[u] = u_dist;
        status[u] = 2;
        heap.pop();

        for (uint32_t a : g.in_arcs(u)) {
            uint32_t w = g.arc_source(a);
            if (status[w] == 2)
                continue;

            int64_t new_dist = semiring_t::plus(u_dist, orig_cost_scaled[a]);

            if (status[w] == 1) {
                if (semiring_t::less(new_dist, heap.priority(w))) {
                    heap.promote(w, new_dist);
                }
            } else {
                heap.push(w, new_dist);
                status[w] = 1;
                dist[w] = new_dist;
            }
        }
    }

    // Unreached vertices get a large but overflow-safe bound.
    constexpr int64_t UNREACHED = semiring_t::infty / 2;
    for (uint32_t v : g.vertices()) {
        if (status[v] != 2) {
            dist[v] = UNREACHED;
        }
    }

    return dist;
}

// CRTP base class for path and tree pricers.  Shared logic: member
// variables, initialization, reduced-cost computation, source loop,
// A* target setup/cleanup, and utility methods.
//
// Derived must implement:
//   void price_source_dijkstra(s_idx, src, source_v, duals, mu, out)
//   void process_source(s_idx, src, duals, mu, dijk, out)  [auto& dijk]
template <typename Derived, typename ColumnT>
class PricerBase {
public:
    using vertex_t = uint32_t;

    static constexpr double SCALE = 1e9;
    static constexpr double NEG_RC_TOL = -1e-6;

protected:
    const Instance* _inst = nullptr;
    std::vector<bool> _source_postponed;
    std::vector<std::vector<uint32_t>> _source_arcs;
    bool _track_arcs = false;
    PricingMode _mode = PricingMode::AStar;
    static_map<vertex_t, int64_t> _lower_bounds;
    dijkstra_workspace _workspace;
    static_map<uint32_t, int64_t> _rc;
    static_map<uint32_t, bool> _is_target;

    Derived& self() noexcept { return static_cast<Derived&>(*this); }

public:
    PricerBase() = default;

    void init(const Instance& inst, PricingMode mode = PricingMode::AStar) {
        _inst = &inst;
        _source_postponed.assign(inst.sources.size(), false);
        _mode = mode;
        _workspace = dijkstra_workspace(inst.graph.num_vertices());
        _rc = inst.graph.create_arc_map<int64_t>();
        if (_mode == PricingMode::AStar) {
            _lower_bounds = compute_lower_bounds_to_targets(inst, SCALE);
            _is_target = inst.graph.create_vertex_map<bool>(false);
        }
    }

    void set_track_arcs(bool enabled) {
        _track_arcs = enabled;
        if (enabled) {
            _source_arcs.resize(_inst->sources.size());
        }
    }

    std::vector<ColumnT> price(const std::vector<double>& duals,
                               const std::unordered_map<uint32_t, double>& mu,
                               bool final_round = false) {
        std::unordered_set<uint32_t> no_forbidden;
        return price(duals, mu, no_forbidden, final_round);
    }

    std::vector<ColumnT> price(const std::vector<double>& duals,
                               const std::unordered_map<uint32_t, double>& mu,
                               const std::unordered_set<uint32_t>& forbidden_arcs,
                               bool final_round) {
        compute_rc(mu, forbidden_arcs);

        std::vector<ColumnT> new_columns;

        for (uint32_t s_idx = 0; s_idx < _inst->sources.size(); ++s_idx) {
            if (!final_round && _source_postponed[s_idx])
                continue;

            const auto& src = _inst->sources[s_idx];
            vertex_t source_v = src.vertex;

            if (_mode == PricingMode::AStar) {
                price_source_astar(s_idx, src, source_v, duals, mu, new_columns);
            } else {
                self().price_source_dijkstra(s_idx, src, source_v, duals, mu, new_columns);
            }
        }

        return new_columns;
    }

    void filter_for_new_caps(const std::vector<uint32_t>& new_cap_arcs) {
        assert(_track_arcs && "filter_for_new_caps requires set_track_arcs(true)");
        std::unordered_set<uint32_t> cap_set(new_cap_arcs.begin(), new_cap_arcs.end());
        for (uint32_t s = 0; s < _source_postponed.size(); ++s) {
            bool affected = std::any_of(_source_arcs[s].begin(), _source_arcs[s].end(),
                                        [&](uint32_t a) { return cap_set.contains(a); });
            _source_postponed[s] = !affected;
        }
    }

    void reset_postponed() { std::fill(_source_postponed.begin(), _source_postponed.end(), false); }

protected:
    void compute_rc(const std::unordered_map<uint32_t, double>& mu,
                    const std::unordered_set<uint32_t>& forbidden_arcs) {
        constexpr int64_t BIG = shortest_path_semiring<int64_t>::infty / 2;
        for (auto a : _inst->graph.arcs()) {
            if (forbidden_arcs.count(a) > 0) {
                _rc[a] = BIG;
                continue;
            }
            double cost_a = _inst->cost[a];
            double mu_a = 0.0;
            auto it = mu.find(a);
            if (it != mu.end())
                mu_a = it->second;
            double val = cost_a - mu_a;
            _rc[a] = (val <= 0.0) ? int64_t{0} : static_cast<int64_t>(std::round(val * SCALE));
        }
    }

    void price_source_astar(uint32_t s_idx, const Source& src, vertex_t source_v,
                            const std::vector<double>& duals,
                            const std::unordered_map<uint32_t, double>& mu,
                            std::vector<ColumnT>& new_columns) {
        // Set target sinks (O(commodities-per-source), not O(V)).
        uint32_t num_targets = 0;
        for (uint32_t k : src.commodity_indices) {
            vertex_t sink = _inst->commodities[k].sink;
            if (!_is_target[sink]) {
                _is_target[sink] = true;
                ++num_targets;
            }
        }

        _workspace.reset();
        astar_dijkstra<dijkstra_store_paths> dijk(_inst->graph, _rc, _lower_bounds, _workspace);
        dijk.add_source(source_v);
        dijk.run_until_targets(_is_target, num_targets);

        self().process_source(s_idx, src, duals, mu, dijk, new_columns);

        // Clear only the sinks we set (O(commodities-per-source), not O(V)).
        for (uint32_t k : src.commodity_indices)
            _is_target[_inst->commodities[k].sink] = false;
    }
};

}  // namespace mcfcg
