#pragma once

#include "mcfcg/cg/column.h"
#include "mcfcg/graph/dijkstra.h"
#include "mcfcg/instance.h"
#include "mcfcg/util/limits.h"
#include "mcfcg/util/thread_pool.h"
#include "mcfcg/util/tolerances.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <unordered_set>
#include <vector>

namespace mcfcg {

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
// variables, initialization, reduced-cost computation, batched
// round-robin source loop with parallel execution, A* target
// setup/cleanup, and utility methods.
//
// Derived must implement:
//   void process_source(s_idx, src, duals, mu, dijk, out, thread_id)  [auto& dijk]
template <typename Derived, typename ColumnT>
class PricerBase {
public:
    using vertex_t = uint32_t;

    static constexpr double SCALE = 1e9;

protected:
    const Instance* _inst = nullptr;
    std::vector<uint8_t> _source_postponed;
    std::vector<std::vector<uint32_t>> _source_arcs;
    bool _track_arcs = false;
    double _neg_rc_tol = NEG_RC_TOL;
    static_map<vertex_t, int64_t> _lower_bounds;
    static_map<uint32_t, int64_t> _rc;

    // Per-thread state for parallel pricing
    std::vector<dijkstra_workspace> _workspaces;
    std::vector<static_map<uint32_t, bool>> _is_targets;
    std::vector<std::vector<ColumnT>> _thread_columns;  // reused across batches
    // Scratch arc→flow map used only by TreePricer::process_source.
    // Declared on the base so thread_id indexing is shared with the
    // other per-thread workspaces; PathPricer ignores it.  Reusing the
    // map across calls keeps bucket storage allocated — the iteration
    // order drifts with bucket-count history, but that drift is absorbed
    // by the loosened EXISTING_COL_RC_TOL in cg_test_util.h.
    std::vector<std::unordered_map<uint32_t, double>> _thread_arc_flow;
    thread_pool* _pool = nullptr;

    // Round-robin cursor: where to start pricing next iteration
    uint32_t _last_source_idx = 0;
    uint32_t _batch_size = 0;  // 0 = all sources in one batch

    Derived& self() noexcept { return static_cast<Derived&>(*this); }

public:
    PricerBase() = default;
    // Non-copyable: per-thread workspaces and Dijkstra state are not
    // meaningful to clone.  A default copy would compile but silently
    // share nothing useful.
    PricerBase(const PricerBase&) = delete;
    PricerBase& operator=(const PricerBase&) = delete;
    PricerBase(PricerBase&&) noexcept = default;
    PricerBase& operator=(PricerBase&&) noexcept = default;

    void init(const Instance& inst, thread_pool* pool = nullptr, uint32_t batch_size = 0,
              double neg_rc_tol = NEG_RC_TOL) {
        _inst = &inst;
        _source_postponed.assign(inst.sources.size(), 0);
        _neg_rc_tol = neg_rc_tol;
        _pool = pool;
        _batch_size = batch_size;
        _last_source_idx = 0;

        uint32_t num_ws = pool != nullptr ? pool->num_threads() : 1;
        _workspaces.clear();
        _workspaces.reserve(num_ws);
        for (uint32_t wi = 0; wi < num_ws; ++wi)
            _workspaces.emplace_back(inst.graph.num_vertices());
        _thread_columns.resize(num_ws);
        _thread_arc_flow.clear();
        _thread_arc_flow.resize(num_ws);

        _rc = inst.graph.create_arc_map<int64_t>();
        _lower_bounds = compute_lower_bounds_to_targets(inst, SCALE);
        _is_targets.clear();
        _is_targets.reserve(num_ws);
        for (uint32_t wi = 0; wi < num_ws; ++wi)
            _is_targets.push_back(inst.graph.create_vertex_map<bool>(false));
    }

    void set_track_arcs(bool enabled) {
        _track_arcs = enabled;
        if (enabled) {
            _source_arcs.resize(_inst->sources.size());
        }
    }

    std::vector<ColumnT> price(const std::vector<double>& duals,
                               const static_map<uint32_t, double>& mu, bool final_round = false,
                               uint32_t max_cols = 0) {
        compute_rc(mu);

        uint32_t n_sources = static_cast<uint32_t>(_inst->sources.size());
        if (n_sources == 0)
            return {};

        uint32_t effective_batch = (_batch_size > 0) ? _batch_size : n_sources;
        uint32_t start = final_round ? 0 : _last_source_idx;
        uint32_t sources_scanned = 0;

        std::vector<ColumnT> all_columns;
        std::vector<uint32_t> batch;
        batch.reserve(effective_batch);

        while (sources_scanned < n_sources) {
            // Collect next batch of active (non-postponed) sources
            batch.clear();
            while (batch.size() < effective_batch && sources_scanned < n_sources) {
                uint32_t s_idx = (start + sources_scanned) % n_sources;
                ++sources_scanned;
                if (!final_round && _source_postponed[s_idx])
                    continue;
                batch.push_back(s_idx);
            }

            if (batch.empty())
                continue;

            // Price batch (parallel if pool available)
            auto batch_cols = price_batch(batch, duals, mu);
            all_columns.insert(all_columns.end(), std::make_move_iterator(batch_cols.begin()),
                               std::make_move_iterator(batch_cols.end()));

            if (max_cols > 0 && all_columns.size() >= max_cols)
                break;
        }

        _last_source_idx = (start + sources_scanned) % n_sources;
        return all_columns;
    }

    void filter_for_new_caps(const std::vector<uint32_t>& new_cap_arcs) {
        assert(_track_arcs && "filter_for_new_caps requires set_track_arcs(true)");
        std::unordered_set<uint32_t> cap_set(new_cap_arcs.begin(), new_cap_arcs.end());
        uint32_t n = static_cast<uint32_t>(_source_postponed.size());
        auto body = [&](uint32_t s) {
            bool affected = std::any_of(_source_arcs[s].begin(), _source_arcs[s].end(),
                                        [&](uint32_t a) { return cap_set.contains(a); });
            _source_postponed[s] = affected ? 0 : 1;
        };
        if (_pool != nullptr && n >= PAR_SOURCE_THRESHOLD) {
            _pool->parallel_for(n, [&](uint32_t s, uint32_t /*tid*/) { body(s); });
        } else {
            for (uint32_t s = 0; s < n; ++s)
                body(s);
        }
    }

    void reset_postponed() {
        std::fill(_source_postponed.begin(), _source_postponed.end(), uint8_t{0});
        _last_source_idx = 0;
    }

protected:
    // Branch-free body so the compiler can auto-vectorize the dense
    // cost/mu/_rc loop under -march=native.
    void compute_rc(const static_map<uint32_t, double>& mu) {
        uint32_t n_arcs = _inst->graph.num_arcs();
        auto body = [&](uint32_t a) {
            double val = _inst->cost[a] - mu[a];
            _rc[a] = (val <= 0.0) ? int64_t{0} : static_cast<int64_t>(std::round(val * SCALE));
        };
        if (_pool != nullptr && n_arcs >= PAR_ARC_THRESHOLD) {
            _pool->parallel_for(n_arcs, [&](uint32_t a, uint32_t /*tid*/) { body(a); });
        } else {
            for (uint32_t a = 0; a < n_arcs; ++a)
                body(a);
        }
    }

    std::vector<ColumnT> price_batch(const std::vector<uint32_t>& batch,
                                     const std::vector<double>& duals,
                                     const static_map<uint32_t, double>& mu) {
        uint32_t batch_n = static_cast<uint32_t>(batch.size());

        if (!_pool || _pool->num_threads() <= 1 || batch_n <= 1) {
            // Sequential
            std::vector<ColumnT> cols;
            for (uint32_t s_idx : batch)
                price_one_source(s_idx, duals, mu, cols, 0);
            return cols;
        }

        // Parallel: each thread accumulates into its own vector
        for (auto& tc : _thread_columns)
            tc.clear();

        _pool->parallel_for(batch_n, [&](uint32_t task_i, uint32_t tid) {
            price_one_source(batch[task_i], duals, mu, _thread_columns[tid], tid);
        });

        // Concatenate
        size_t total = 0;
        for (auto& tc : _thread_columns)
            total += tc.size();
        std::vector<ColumnT> result;
        result.reserve(total);
        for (auto& tc : _thread_columns)
            result.insert(result.end(), std::make_move_iterator(tc.begin()),
                          std::make_move_iterator(tc.end()));
        return result;
    }

    void price_one_source(uint32_t s_idx, const std::vector<double>& duals,
                          const static_map<uint32_t, double>& mu, std::vector<ColumnT>& out,
                          uint32_t thread_id) {
        const auto& src = _inst->sources[s_idx];
        vertex_t source_v = src.vertex;
        price_source_astar(s_idx, src, source_v, duals, mu, out, thread_id);
    }

    void price_source_astar(uint32_t s_idx, const Source& src, vertex_t source_v,
                            const std::vector<double>& duals,
                            const static_map<uint32_t, double>& mu,
                            std::vector<ColumnT>& new_columns, uint32_t thread_id) {
        auto& ws = _workspaces[thread_id];
        auto& is_target = _is_targets[thread_id];

        // Set target sinks (O(commodities-per-source), not O(V)).
        uint32_t num_targets = 0;
        for (uint32_t k : src.commodity_indices) {
            vertex_t sink = _inst->commodities[k].sink;
            if (!is_target[sink]) {
                is_target[sink] = true;
                ++num_targets;
            }
        }

        ws.reset();
        astar_dijkstra<dijkstra_store_paths> dijk(_inst->graph, _rc, _lower_bounds, ws);
        dijk.add_source(source_v);
        dijk.run_until_targets(is_target, num_targets);

        self().process_source(s_idx, src, duals, mu, dijk, new_columns, thread_id);

        // Clear only the sinks we set (O(commodities-per-source), not O(V)).
        for (uint32_t k : src.commodity_indices)
            is_target[_inst->commodities[k].sink] = false;
    }
};

}  // namespace mcfcg
