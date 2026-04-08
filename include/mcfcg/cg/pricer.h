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

class PathPricer {
public:
    using vertex_t = uint32_t;

    static constexpr double SCALE = 1e9;
    static constexpr double NEG_RC_TOL = -1e-6;

private:
    const Instance* _inst = nullptr;
    std::vector<bool> _source_postponed;
    std::vector<std::vector<uint32_t>> _source_arcs;  // arcs used per source in last pricing
    bool _track_arcs = false;
    PricingMode _mode = PricingMode::AStar;
    static_map<vertex_t, int64_t> _lower_bounds;
    dijkstra_workspace _workspace;
    static_map<uint32_t, int64_t> _rc;
    static_map<uint32_t, bool> _is_target;

public:
    PathPricer() = default;

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

    std::vector<Column> price(const std::vector<double>& pi,
                              const std::unordered_map<uint32_t, double>& mu,
                              bool final_round = false) {
        std::unordered_set<uint32_t> no_forbidden;
        return price(pi, mu, no_forbidden, final_round);
    }

    std::vector<Column> price(const std::vector<double>& pi,
                              const std::unordered_map<uint32_t, double>& mu,
                              const std::unordered_set<uint32_t>& forbidden_arcs,
                              bool final_round) {
        // Compute clamped reduced costs for Dijkstra (reuses _rc allocation).
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

        std::vector<Column> new_columns;

        for (uint32_t s_idx = 0; s_idx < _inst->sources.size(); ++s_idx) {
            if (!final_round && _source_postponed[s_idx])
                continue;

            const auto& src = _inst->sources[s_idx];
            vertex_t source_v = src.vertex;

            if (_mode == PricingMode::AStar) {
                price_source_astar(s_idx, src, source_v, pi, mu, new_columns);
            } else {
                price_source_dijkstra(s_idx, src, source_v, pi, mu, new_columns);
            }
        }

        return new_columns;
    }

    // After new capacity constraints are added, mark sources for re-pricing
    // based on whether their last shortest-path tree used any newly constrained
    // arc.  Affected sources are un-postponed (their reduced costs changed);
    // unaffected sources are postponed (their reduced costs are unchanged).
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

private:
    void extract_columns(uint32_t s_idx, const Source& src, const std::vector<double>& pi,
                         const std::unordered_map<uint32_t, double>& mu, auto& dijk,
                         std::vector<Column>& new_columns) {
        bool found_any = false;
        if (_track_arcs)
            _source_arcs[s_idx].clear();

        for (uint32_t k : src.commodity_indices) {
            vertex_t sink = _inst->commodities[k].sink;
            if (!dijk.visited(sink))
                continue;

            // Extract path and compute true reduced cost
            Column col;
            col.cost = 0.0;
            col.commodity = k;
            double true_rc = -pi[k];
            vertex_t v = sink;
            while (dijk.has_pred(v)) {
                uint32_t a = dijk.pred_arc(v);
                col.arcs.push_back(a);
                if (_track_arcs)
                    _source_arcs[s_idx].push_back(a);
                col.cost += _inst->cost[a];
                double mu_a = 0.0;
                auto mit = mu.find(a);
                if (mit != mu.end())
                    mu_a = mit->second;
                true_rc += _inst->cost[a] - mu_a;
                v = _inst->graph.arc_source(a);
            }

            if (true_rc >= NEG_RC_TOL)
                continue;

            found_any = true;
            std::reverse(col.arcs.begin(), col.arcs.end());
            new_columns.push_back(std::move(col));
        }

        _source_postponed[s_idx] = !found_any;
    }

    void price_source_dijkstra(uint32_t s_idx, const Source& src, vertex_t source_v,
                               const std::vector<double>& pi,
                               const std::unordered_map<uint32_t, double>& mu,
                               std::vector<Column>& new_columns) {
        constexpr auto MAX_BOUND = shortest_path_semiring<int64_t>::infty / 2;

        // Per-target cutoffs: cutoff[sink] = max pi[k]*SCALE over
        // commodities k with that sink. Dijkstra stops when min key
        // exceeds max_cutoff.
        std::unordered_map<vertex_t, int64_t> cutoff;
        int64_t max_cutoff = 0;
        for (uint32_t k : src.commodity_indices) {
            vertex_t sink = _inst->commodities[k].sink;
            double raw = pi[k] * SCALE;
            auto scaled = raw >= static_cast<double>(MAX_BOUND)
                              ? MAX_BOUND
                              : static_cast<int64_t>(std::ceil(raw));
            auto it = cutoff.find(sink);
            if (it == cutoff.end()) {
                cutoff[sink] = scaled;
            } else {
                it->second = std::max(it->second, scaled);
            }
            max_cutoff = std::max(max_cutoff, scaled);
        }

        uint32_t remaining = static_cast<uint32_t>(cutoff.size());

        _workspace.reset();
        dijkstra<dijkstra_store_paths> dijk(_inst->graph, _rc, _workspace);
        dijk.add_source(source_v);

        while (!dijk.finished() && remaining > 0) {
            auto [u, u_dist] = dijk.current();
            if (u_dist > max_cutoff)
                break;
            dijk.advance();

            auto cit = cutoff.find(u);
            if (cit != cutoff.end()) {
                --remaining;
                cutoff.erase(cit);
                if (remaining > 0) {
                    max_cutoff = 0;
                    for (auto& [_, c] : cutoff) {
                        max_cutoff = std::max(max_cutoff, c);
                    }
                }
            }
        }

        extract_columns(s_idx, src, pi, mu, dijk, new_columns);
    }

    void price_source_astar(uint32_t s_idx, const Source& src, vertex_t source_v,
                            const std::vector<double>& pi,
                            const std::unordered_map<uint32_t, double>& mu,
                            std::vector<Column>& new_columns) {
        // Build target set for early termination (reuses _is_target allocation).
        _is_target.fill(false);
        uint32_t num_targets = 0;
        for (uint32_t k : src.commodity_indices) {
            vertex_t sink = _inst->commodities[k].sink;
            if (!_is_target[sink]) {
                _is_target[sink] = true;
                ++num_targets;
            }
        }

        // Use precomputed lower bounds (original costs, computed once at init).
        _workspace.reset();
        astar_dijkstra<dijkstra_store_paths> dijk(_inst->graph, _rc, _lower_bounds, _workspace);
        dijk.add_source(source_v);
        dijk.run_until_targets(_is_target, num_targets);

        extract_columns(s_idx, src, pi, mu, dijk, new_columns);
    }
};

}  // namespace mcfcg
