#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <optional>
#include <unordered_set>
#include <vector>

#include "mcfcg/cg/column.h"
#include "mcfcg/graph/dijkstra.h"
#include "mcfcg/instance.h"
#include "mcfcg/util/limits.h"
#include "mcfcg/util/thread_pool.h"
#include "mcfcg/util/tolerances.h"

namespace mcfcg {

// Largest usable scaled distance, and the single source of truth for the
// coupling between two roles that must stay equal: it is what
// compute_lower_bounds_to_targets stores for a vertex that reaches no sink,
// and what PricerBase::scale_dual saturates a +inf dual to.  Halving INFTY
// keeps `g + h` addable without overflow.
inline constexpr int64_t UNREACHED_BOUND = shortest_path_semiring<int64_t>::INFTY / 2;

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
            if (status[w] == 2) {
                continue;
            }

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
    for (uint32_t v : g.vertices()) {
        if (status[v] != 2) {
            dist[v] = UNREACHED_BOUND;
        }
    }

    return dist;
}

// One target sink of the source being priced, as seen by bounded pricing.
// `key` is the per-sink stopping threshold (path: the scaled
// structural dual); `demand` is the sink's aggregated demand (tree).  Each
// derived pricer uses the field its bound needs and ignores the other; the
// shared layout lets both share one per-thread scratch buffer.
struct BoundEntry {
    uint32_t sink;
    int64_t key;
    double demand;
};

// CRTP base class for path and tree pricers.  Shared logic: member
// variables, initialization, reduced-cost computation, batched
// round-robin source loop with parallel execution, A* target
// setup/cleanup, and utility methods.
//
// Derived must implement:
//   void process_source(s_idx, src, duals, mu, dijk, out, tid, bound_f)
//     [auto& dijk]
//   auto make_bound(s_idx, src, duals, scratch)  -- returns a tracker with
//     `int64_t bound()` (a cached read, queried once per settled vertex) and
//     `void on_settle(dijk, sink, g)`, which is what refreshes it.  Required
//     unconditionally: _bounded_pricing is a runtime flag, so the call is
//     always instantiated even for a pricer that will never run bounded.
template <typename Derived, typename ColumnT>
class PricerBase {
public:
    using vertex_t = uint32_t;

    static constexpr double SCALE = 1e9;
    // Largest usable scaled distance.  Two roles: scale_dual saturates here, so
    // a +inf dual yields a bound no *reachable* A* key exceeds; and
    // compute_lower_bounds_to_targets uses the same value as its UNREACHED
    // heuristic, so a frontier at or above it means every frontier vertex is a
    // dead end and the search is effectively exhausted.  A* keys therefore CAN
    // exceed it (f = g + h saturates only at semiring::INFTY) — which is why
    // price_source_astar refuses to stop there.
    static constexpr int64_t MAX_BOUND = UNREACHED_BOUND;

protected:
    const Instance* _inst = nullptr;
    std::vector<uint8_t> _source_postponed;
    std::vector<std::vector<uint32_t>> _source_arcs;
    bool _track_arcs = false;
    double _neg_rc_tol = NEG_RC_TOL;
    static_map<vertex_t, int64_t> _lower_bounds;
    static_map<uint32_t, int64_t> _rc;

    // Bounded single-source pricing (manuscript §3.3): stop the A* as soon as
    // the frontier proves no negative-reduced-cost column can exist for this
    // source.  Off by default; see the CGParams::bounded_pricing doc.
    bool _bounded_pricing = false;
    // Per unit of demand, the worst-case gap between an integer-scaled path
    // length and SCALE·(its true cost): compute_rc rounds each scaled arc cost
    // to the nearest integer, so the error is at most 0.5 per arc, and a
    // shortest path under non-negative lengths is simple, so V bounds its arc
    // count.  Used twice, in opposite directions — subtracted in
    // salvage_lagr_term so a salvaged Lagrangian term stays a valid lower
    // bound, and added to each pricing bound (see the derivations in
    // PathPricer::make_bound and TreePricer::make_bound) so the bound is
    // provably no more aggressive than _neg_rc_tol.
    double _round_slack_per_demand = 0.0;

    // Per-thread state for parallel pricing
    std::vector<dijkstra_workspace> _workspaces;
    std::vector<static_map<uint32_t, bool>> _is_targets;
    std::vector<std::vector<BoundEntry>> _bound_scratch;
    std::vector<std::vector<ColumnT>> _thread_columns;  // reused across batches
    // Per-source LB accumulators.  Writing to a per-source slot (instead
    // of a per-thread slot) keeps the final sum order-independent of the
    // thread pool's task→thread dispatch, so the reductions below are
    // deterministic run-to-run.  Each source is processed by exactly one
    // thread in a given batch, so no race.
    std::vector<double> _source_rc_error;
    // Per-source accumulator for the π-free Lagrangian LB: Σ_{k in source}
    // d_k · sp_k(c−μ), the demand-weighted reduced-cost shortest-path sum
    // WITHOUT subtracting the structural dual π.
    std::vector<double> _source_lagr_sum;
    thread_pool* _pool = nullptr;

    // Round-robin cursor: where to start pricing next iteration
    uint32_t _last_source_idx = 0;
    uint32_t _batch_size = 0;  // 0 = all sources in one batch

    // π-free capacity-relaxation Lagrangian lower-bound support.
    // _last_lagr_path_sum is Σ_k d_k·sp_k(c−μ) over all priced convexity
    // entities (commodities for path, sources for tree); combined with
    // Σ cap·μ in cg_loop it gives a lower bound valid for any μ≤0 regardless
    // of slack/feasibility state (the structural duals cancel analytically).
    // _last_rc_error_bound bounds the scale-integer rounding gap (subtracted
    // to certify the bound).  Both are valid only when _last_priced_all is
    // true — every source visited in the last price() call (priced_count
    // equals n_sources; a max_cols break that fires exactly on sweep
    // completion still counts).
    double _last_rc_error_bound = 0.0;
    double _last_lagr_path_sum = 0.0;
    bool _last_priced_all = false;

    // _source_bound_fired[s] is 1 when the pricing bound stopped the most
    // recent A* run for source s.  Written only when s is actually priced; the per-call
    // counters below are accumulated over the priced batches.  All zeros
    // whenever bounded pricing is off.
    std::vector<uint8_t> _source_bound_fired;
    uint64_t _last_bounded_count = 0;
    uint64_t _last_priced_count = 0;

    Derived& self() noexcept { return static_cast<Derived&>(*this); }

private:
    // CRTP contract: the constructors are private and Derived is a friend,
    // so this base cannot be instantiated or inherited from as a plain
    // template class.  Derived's own implicit constructors still reach
    // them through the friendship.
    PricerBase() = default;
    PricerBase(PricerBase&&) noexcept = default;
    friend Derived;

public:
    // Non-copyable: per-thread workspaces and Dijkstra state are not
    // meaningful to clone.  A default copy would compile but silently
    // share nothing useful.  Deleted members stay public so the diagnostic
    // on a copy attempt is "deleted", not "inaccessible"; that is also what
    // modernize-use-equals-delete asks for.  The CRTP check disagrees, but
    // a deleted constructor constructs nothing, so it cannot be the escape
    // hatch that check exists to close.
    // NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
    PricerBase(const PricerBase&) = delete;
    PricerBase& operator=(const PricerBase&) = delete;
    PricerBase& operator=(PricerBase&&) noexcept = default;

    void init(const Instance& inst, thread_pool* pool = nullptr, uint32_t batch_size = 0,
              double neg_rc_tol = NEG_RC_TOL, bool bounded_pricing = false) {
        _inst = &inst;
        _source_postponed.assign(inst.sources.size(), 0);
        _neg_rc_tol = neg_rc_tol;
        _pool = pool;
        _batch_size = batch_size;
        _last_source_idx = 0;
        _bounded_pricing = bounded_pricing;
        _round_slack_per_demand = 0.5 * static_cast<double>(inst.graph.num_vertices()) / SCALE;
        _source_bound_fired.assign(inst.sources.size(), 0);
        _last_bounded_count = 0;
        _last_priced_count = 0;

        uint32_t num_ws = pool != nullptr ? pool->num_threads() : 1;
        _workspaces.clear();
        _workspaces.reserve(num_ws);
        for (uint32_t wi = 0; wi < num_ws; ++wi) {
            _workspaces.emplace_back(inst.graph.num_vertices());
        }
        _thread_columns.resize(num_ws);
        _source_rc_error.assign(inst.sources.size(), 0.0);
        _source_lagr_sum.assign(inst.sources.size(), 0.0);

        _rc = inst.graph.create_arc_map<int64_t>();
        _lower_bounds = compute_lower_bounds_to_targets(inst, SCALE);
        _is_targets.clear();
        _is_targets.reserve(num_ws);
        for (uint32_t wi = 0; wi < num_ws; ++wi) {
            _is_targets.push_back(inst.graph.create_vertex_map<bool>(false));
        }
        _bound_scratch.assign(num_ws, {});
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

        // Reset the per-call LB accumulators up front so every return
        // path (including n_sources==0 and early breaks) leaves them
        // in a defined state.
        _last_priced_all = false;
        _last_rc_error_bound = 0.0;
        _last_lagr_path_sum = 0.0;
        _last_bounded_count = 0;
        _last_priced_count = 0;

        auto n_sources = static_cast<uint32_t>(_inst->sources.size());
        if (n_sources == 0) {
            _last_priced_all = true;
            return {};
        }

        // Reset per-source accumulators for this price() call.  Sources
        // not revisited this call (postponed under non-final_round) keep
        // their zero and contribute nothing — correct since priced_all
        // will be false in that case and the LB gate skips reading.
        std::ranges::fill(_source_rc_error, 0.0);
        std::ranges::fill(_source_lagr_sum, 0.0);

        uint32_t effective_batch = (_batch_size > 0) ? _batch_size : n_sources;
        uint32_t start = final_round ? 0 : _last_source_idx;
        uint32_t sources_scanned = 0;
        uint32_t priced_count = 0;
        uint64_t fire_count = 0;

        std::vector<ColumnT> all_columns;
        std::vector<uint32_t> batch;
        batch.reserve(effective_batch);

        while (sources_scanned < n_sources) {
            // Collect next batch of active (non-postponed) sources
            batch.clear();
            while (batch.size() < effective_batch && sources_scanned < n_sources) {
                uint32_t s_idx = (start + sources_scanned) % n_sources;
                ++sources_scanned;
                if (!final_round && _source_postponed[s_idx] != 0) {
                    continue;
                }
                batch.push_back(s_idx);
                ++priced_count;
            }

            if (batch.empty()) {
                continue;
            }

            // Price batch (parallel if pool available)
            auto batch_cols = price_batch(batch, duals, mu);
            all_columns.insert(all_columns.end(), std::make_move_iterator(batch_cols.begin()),
                               std::make_move_iterator(batch_cols.end()));
            for (uint32_t s_idx : batch) {
                fire_count += _source_bound_fired[s_idx];
            }

            if (max_cols > 0 && all_columns.size() >= max_cols) {
                break;
            }
        }

        _last_source_idx = (start + sources_scanned) % n_sources;
        // Deterministic summation in source-index order — independent
        // of thread dispatch.
        _last_rc_error_bound =
            std::accumulate(_source_rc_error.begin(), _source_rc_error.end(), 0.0);
        _last_lagr_path_sum =
            std::accumulate(_source_lagr_sum.begin(), _source_lagr_sum.end(), 0.0);
        // priced_count == n_sources already proves every source was
        // visited (postponed sources are skipped without incrementing
        // it); a max-cols break that fires exactly on the last batch
        // still completes the sweep and should keep priced_all=true so
        // the LB gate in cg_loop can fire.  Under PricerHeavy this is
        // the common case — the col cap is tuned to num_entities and
        // tree's one-col-per-source emission hits it precisely at the
        // end of the sweep.
        _last_priced_all = (priced_count == n_sources);
        _last_priced_count = priced_count;
        _last_bounded_count = fire_count;
        return all_columns;
    }

    [[nodiscard]] bool priced_all() const noexcept { return _last_priced_all; }

    // Sources priced by the last price() call, and how many of those the
    // pricing bound stopped short.  Always 0/N when bounded pricing is off.
    [[nodiscard]] uint64_t last_priced_count() const noexcept { return _last_priced_count; }
    [[nodiscard]] uint64_t last_bounded_count() const noexcept { return _last_bounded_count; }

    // π-free capacity-relaxation Lagrangian path sum Σ_k d_k·sp_k(c−μ) from
    // the last price() call.  Add Σ_a cap_a·μ_a and subtract lb_error_bound()
    // to obtain L(μ) ≤ OPT, valid for any μ≤0 independent of slack state.
    // Valid only when priced_all() is true.
    [[nodiscard]] double lagrangian_path_sum() const noexcept { return _last_lagr_path_sum; }

    // Upper bound on the rounding error in lagrangian_path_sum().  Edge
    // weights are scaled to int64 at SCALE=1e9 in compute_rc, so A* returns
    // the path minimizing a rounded-integer cost which can differ from
    // the true-min-reduced-cost path by at most L/SCALE per path (with
    // L the path length in arcs, doubled to account for both the
    // chosen path's and the true-min path's rounding).  Subtract this
    // from the Lagrangian bound to certify it.
    [[nodiscard]] double lb_error_bound() const noexcept { return _last_rc_error_bound; }

    // Round-robin cursor parked by the last price() call; exposed for
    // tests that verify partial pricing (PricerHeavy) advances it
    // mid-sweep when the max_cols early break fires.
    [[nodiscard]] uint32_t last_source_idx() const noexcept { return _last_source_idx; }

    void filter_for_new_caps(const std::vector<uint32_t>& new_cap_arcs) {
        assert(_track_arcs && "filter_for_new_caps requires set_track_arcs(true)");
        std::unordered_set<uint32_t> cap_set(new_cap_arcs.begin(), new_cap_arcs.end());
        auto n = static_cast<uint32_t>(_source_postponed.size());
        auto body = [&](uint32_t s) {
            // Deliberately reads _source_arcs even when a bounded search left
            // it describing an older routing (see should_record_arcs): re-pricing
            // every bounded source instead costs more than stale evidence does.
            bool affected = std::any_of(_source_arcs[s].begin(), _source_arcs[s].end(),
                                        [&](uint32_t a) { return cap_set.contains(a); });
            _source_postponed[s] = affected ? 0 : 1;
        };
        if (_pool != nullptr && n >= PAR_SOURCE_THRESHOLD) {
            _pool->parallel_for(n, [&](uint32_t s, uint32_t /*tid*/) { body(s); });
        } else {
            for (uint32_t s = 0; s < n; ++s) {
                body(s);
            }
        }
    }

    void reset_postponed() {
        std::ranges::fill(_source_postponed, uint8_t{0});
        _last_source_idx = 0;
    }

    // Clear postponement flags only; keep the round-robin cursor.  Used
    // by the main CG loop after a successful pricing pass so partial
    // pricing resumes from where it parked next iter, rather than
    // restarting at source 0 every iter.  Calling reset_postponed here
    // would silently defeat partial pricing under CGStrategy::PricerHeavy.
    void clear_postponed() { std::ranges::fill(_source_postponed, uint8_t{0}); }

protected:
    // Scale a dual to the pricer's integer distance units, saturating instead
    // of overflowing.  +inf (the warm-start duals) and NaN both map to
    // MAX_BOUND, which no A* key can exceed — so the bound is inert there,
    // and the warm start still explores the full reachable graph.
    static int64_t scale_dual(double dual) noexcept {
        double raw = dual * SCALE;
        if (!(raw < static_cast<double>(MAX_BOUND))) {
            return MAX_BOUND;
        }
        if (raw <= -static_cast<double>(MAX_BOUND)) {
            return -MAX_BOUND;
        }
        return static_cast<int64_t>(std::ceil(raw));
    }

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
            for (uint32_t a = 0; a < n_arcs; ++a) {
                body(a);
            }
        }
    }

    std::vector<ColumnT> price_batch(const std::vector<uint32_t>& batch,
                                     const std::vector<double>& duals,
                                     const static_map<uint32_t, double>& mu) {
        auto batch_n = static_cast<uint32_t>(batch.size());

        if (_pool == nullptr || _pool->num_threads() <= 1 || batch_n <= 1) {
            // Sequential
            std::vector<ColumnT> cols;
            for (uint32_t s_idx : batch) {
                price_one_source(s_idx, duals, mu, cols, 0);
            }
            return cols;
        }

        // Parallel: each thread accumulates into its own vector
        for (auto& tc : _thread_columns) {
            tc.clear();
        }

        _pool->parallel_for(batch_n, [&](uint32_t task_i, uint32_t tid) {
            price_one_source(batch[task_i], duals, mu, _thread_columns[tid], tid);
        });

        // Concatenate
        size_t total = 0;
        for (auto& tc : _thread_columns) {
            total += tc.size();
        }
        std::vector<ColumnT> result;
        result.reserve(total);
        for (auto& tc : _thread_columns) {
            result.insert(result.end(), std::make_move_iterator(tc.begin()),
                          std::make_move_iterator(tc.end()));
        }
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
            // Bound soundness: min_f() bounds the final g of an unsettled
            // vertex only where h vanishes.  compute_lower_bounds_to_targets
            // seeds every sink at 0, so this holds for sinks — and would break
            // if the heuristic ever became per-source or non-sink-seeded.
            // Only bounded pricing depends on it, so it is not worth checking
            // on the default path even in a debug build.
            assert((!_bounded_pricing || _lower_bounds[sink] == 0) &&
                   "A* heuristic must vanish at every sink");
            if (!is_target[sink]) {
                is_target[sink] = true;
                ++num_targets;
            }
        }

        ws.reset();
        astar_dijkstra<dijkstra_store_paths> dijk(_inst->graph, _rc, _lower_bounds, ws);
        dijk.add_source(source_v);

        // Frontier f at which the pricing bound stopped the search, empty when
        // it did not fire.  Every sink the bound left unsettled has a true
        // reduced-cost distance of at least this — which is what lets
        // salvage_lagr_term keep the Lagrangian bound from collapsing.  Held as
        // an optional because 0 is a frontier value the bound really can stop
        // at (a non-positive dual makes it negative before anything is
        // settled), so it cannot double as "did not fire".
        std::optional<int64_t> bound_f;
        if (_bounded_pricing) {
            auto& scratch = _bound_scratch[thread_id];
            auto tracker = self().make_bound(s_idx, src, duals, scratch);
            while (!dijk.finished() && num_targets > 0) {
                int64_t frontier = dijk.min_f();
                // Every frontier vertex now carries the UNREACHED heuristic
                // (compute_lower_bounds_to_targets seeds dead ends at
                // MAX_BOUND), so no unsettled sink is reachable any more.  Stop
                // like the heap-exhausted case and NOT as a bound hit: those sinks
                // are genuinely unreachable, so salvaging ~MAX_BOUND/SCALE for
                // them would latch a Lagrangian bound orders of magnitude above
                // OPT into the monotone best_lb, and would make the tree pricer
                // suppress the partial column the unbounded path emits for a
                // disconnected source.  Reachable targets have h=0, hence a
                // frontier below MAX_BOUND, so all of them are already settled.
                if (frontier >= MAX_BOUND) {
                    break;
                }
                if (frontier > tracker.bound()) {
                    bound_f = frontier;
                    break;
                }
                auto [settled, g_dist] = dijk.settle_next();
                if (is_target[settled]) {
                    is_target[settled] = false;
                    --num_targets;
                    tracker.on_settle(dijk, settled, g_dist);
                }
            }
        } else {
            dijk.run_until_targets(is_target, num_targets);
        }
        _source_bound_fired[s_idx] = bound_f.has_value() ? 1 : 0;

        self().process_source(s_idx, src, duals, mu, dijk, new_columns, thread_id, bound_f);

        // Clear only the sinks we set (O(commodities-per-source), not O(V)).
        for (uint32_t k : src.commodity_indices) {
            is_target[_inst->commodities[k].sink] = false;
        }
    }

    // Whether this call should refresh _source_arcs[s_idx].  A bounded search
    // covers only the sinks it settled, so its arc set is not the evidence
    // filter_for_new_caps needs.  Leave the previous complete set standing
    // rather than overwrite it with a partial one: possibly stale arcs still
    // describe this source's routing, while a partial set would understate it
    // and postpone a source a new capacity row does affect.
    //
    // The consequence is that filter_for_new_caps decides postponement from an
    // older routing for every bounded source, so switching the bound on changes
    // which sources the loop prices — and hence the CG trajectory — even though
    // the columns the pricer emits are identical (pinned bit-for-bit by the
    // FeatureTests.BoundedPricingShadow* tests).  That is a convergence-speed
    // effect only: the pricing-exhausted final_round re-prices every source
    // regardless of postponement.
    //
    // Do NOT "fix" it by treating a bounded source as affected
    // (_source_bound_fired[s] is exactly that flag, sticky until the source is
    // priced unbounded).  Measured on
    // intermodal tree/PricerHeavy under COPT: +31% wall clock, because the 65-77%
    // fire rate makes almost every source affected and the filter stops
    // filtering — SBT-56295 alone paid +68% at an unchanged iteration count.
    // The per-instance table is the appendix of results/ablation/README.md; it
    // measured a reverted change, so no tracked log carries it.
    // (With warm_start=false a source the bound stops on its very first price
    // keeps an *empty* set, which reads as "unaffected"; same caveat.)
    [[nodiscard]] bool should_record_arcs(std::optional<int64_t> bound_f) const noexcept {
        return _track_arcs && !bound_f.has_value();
    }

    // Lagrangian contribution salvaged for a commodity the bound left
    // unsettled.  `bound_f` lower-bounds the integer-scaled reduced-cost
    // distance to that sink, and rounding can inflate an integer-scaled path
    // length by at most 0.5 per arc, so
    //     d_k·sp_k(c−μ) >= demand · (bound_f/SCALE − margin)
    // is a valid (if slack) stand-in for the term the truncated search never
    // computed.  Dropping the term outright is also valid — every sp_k ≥ 0 —
    // but at convergence the bound fires on nearly every source, and a best_lb
    // that then collapses to Σ cap·μ would trade pricing time for the gap exit
    // it exists to trigger.
    [[nodiscard]] double salvage_lagr_term(std::optional<int64_t> bound_f,
                                           double demand) const noexcept {
        if (!bound_f.has_value()) {
            return 0.0;  // heap exhausted: the sink is genuinely unreachable
        }
        double bound = (static_cast<double>(*bound_f) / SCALE) - _round_slack_per_demand;
        return bound > 0.0 ? demand * bound : 0.0;
    }
};

}  // namespace mcfcg
