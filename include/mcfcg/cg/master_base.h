#pragma once

#include "mcfcg/instance.h"
#include "mcfcg/lp/lp_solver.h"
#include "mcfcg/util/limits.h"
#include "mcfcg/util/thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mcfcg {

// CRTP base class for path and tree master problems.  Shared logic:
// LP init, column management, solve/duals, lazy capacity constraints,
// column aging, and row/column purging.
//
// Derived must implement the following hooks (private + friend, or public):
//   uint32_t num_structural_entities() const;
//   std::pair<double, double> structural_row_bounds(uint32_t k) const;
//   uint32_t structural_row_index(const ColumnT& col) const;
//   void for_each_arc_coeff(const ColumnT& col, auto&& callback) const;
//   void accumulate_flow(const ColumnT& col, double x,
//                        static_map<uint32_t, double>& flow) const;
//
// Invariant: for_each_arc_coeff must not yield the same arc twice for a
// given column.  Path columns are simple paths; tree columns aggregate
// per-arc flow into a unique-arc map in the pricer.
template <typename Derived, typename ColumnT>
class MasterBase {
public:
    static constexpr double BIG_M = 1e8;

    // Below this many arcs (or columns) the dispatch overhead of
    // parallel_for outweighs the per-element work.  Tuned for the
    // typical instance sizes the solver targets; keep these together so
    // all parallel paths use a consistent policy.
    static constexpr uint32_t PAR_ARC_THRESHOLD = 4096;
    static constexpr uint32_t PAR_COL_THRESHOLD = 256;

protected:
    const Instance* _inst = nullptr;
    std::unique_ptr<LPSolver> _lp;

    uint32_t _num_structural_rows = 0;

    // Column store with bidirectional LP mapping
    std::vector<ColumnT> _columns;
    std::vector<uint32_t> _col_to_lp;  // column index -> LP column index
    std::vector<uint32_t> _col_age;    // consecutive iterations with zero primal

    // Capacity constraint tracking
    std::unordered_map<uint32_t, uint32_t> _arc_to_cap_row;
    std::vector<uint32_t> _cap_row_to_arc;
    std::vector<uint32_t> _cap_row_last_active;  // last iteration each capacity row was active

    // Per-arc list of (column index, coefficient) pairs.  Maintained
    // incrementally in add_columns and rebuilt after purge_aged_columns
    // because column compaction shifts indices.  Used by the cut row
    // construction in add_violated_capacity_constraints to look up only
    // the columns that touch each newly-violated arc instead of scanning
    // every column.
    struct ArcColEntry {
        uint32_t col_idx;
        double coeff;
    };
    static_map<uint32_t, std::vector<ArcColEntry>> _arc_to_col_entries;

    // Optional thread pool for parallelizing per-iteration work
    // (column CSC build, flow accumulation, violation scan, cut CSR
    // build).  Owned by the caller; nullptr means run sequentially.
    thread_pool* _pool = nullptr;

    // Per-thread workspaces reused across iterations.
    // _thread_flow holds partial flow accumulators for
    // add_violated_capacity_constraints; sized [num_threads].
    std::vector<static_map<uint32_t, double>> _thread_flow;

    Derived& self() noexcept { return static_cast<Derived&>(*this); }
    const Derived& self() const noexcept { return static_cast<const Derived&>(*this); }

public:
    MasterBase() = default;

    void init(const Instance& inst, std::unique_ptr<LPSolver> lp = nullptr,
              thread_pool* pool = nullptr) {
        _inst = &inst;
        _pool = pool;
        _arc_to_col_entries = inst.graph.create_arc_map<std::vector<ArcColEntry>>();
        // Per-thread flow accumulators are only used by the parallel
        // path in compute_arc_flow.  Skip the allocation entirely when
        // there is no pool — the serial path uses a stack-local map.
        _thread_flow.clear();
        if (pool != nullptr) {
            uint32_t num_threads = pool->num_threads();
            _thread_flow.reserve(num_threads);
            for (uint32_t tid = 0; tid < num_threads; ++tid) {
                _thread_flow.push_back(inst.graph.create_arc_map<double>(0.0));
            }
        }
        _num_structural_rows = self().num_structural_entities();
        _columns.clear();
        _col_to_lp.clear();
        _col_age.clear();
        _arc_to_cap_row.clear();
        _cap_row_to_arc.clear();
        _cap_row_last_active.clear();

        // Create LP once
        _lp = lp ? std::move(lp) : create_lp_solver();

        // Add slack columns (one per structural row, no row coefficients yet)
        std::vector<double> slack_obj(_num_structural_rows, BIG_M);
        std::vector<double> slack_lb(_num_structural_rows, 0.0);
        std::vector<double> slack_ub(_num_structural_rows, INF);
        _lp->add_cols(slack_obj, slack_lb, slack_ub);

        // Add structural rows with slack column coefficients
        std::vector<double> row_lb;
        std::vector<double> row_ub;
        std::vector<uint32_t> starts;
        std::vector<uint32_t> indices;
        std::vector<double> values;

        for (uint32_t k = 0; k < _num_structural_rows; ++k) {
            auto [lb, ub] = self().structural_row_bounds(k);
            row_lb.push_back(lb);
            row_ub.push_back(ub);
            starts.push_back(static_cast<uint32_t>(indices.size()));
            indices.push_back(k);  // slack column k
            values.push_back(1.0);
        }

        _lp->add_rows(row_lb, row_ub, starts, indices, values);
    }

    uint32_t add_columns(std::vector<ColumnT> cols) {
        if (cols.empty())
            return 0;

        uint32_t n = static_cast<uint32_t>(cols.size());
        std::vector<double> obj(n);
        std::vector<double> lb(n, 0.0);
        std::vector<double> ub(n, INF);
        std::vector<uint32_t> starts;
        std::vector<uint32_t> row_indices;
        std::vector<double> values;

        if (_pool != nullptr && n > 1) {
            // Parallel two-pass: each task fills its own local buffer (no
            // shared writes), then a serial concat builds the flat CSC.
            // _arc_to_cap_row is read-only here, so concurrent reads are
            // safe.
            std::vector<std::vector<uint32_t>> local_indices(n);
            std::vector<std::vector<double>> local_values(n);

            _pool->parallel_for(n, [&](uint32_t i, uint32_t /*tid*/) {
                obj[i] = cols[i].cost;
                auto& l_idx = local_indices[i];
                auto& l_val = local_values[i];
                l_idx.push_back(self().structural_row_index(cols[i]));
                l_val.push_back(1.0);
                self().for_each_arc_coeff(cols[i], [&](uint32_t arc, double coeff) {
                    auto it = _arc_to_cap_row.find(arc);
                    if (it != _arc_to_cap_row.end()) {
                        l_idx.push_back(it->second);
                        l_val.push_back(coeff);
                    }
                });
            });

            starts.resize(n + 1);
            uint32_t total = 0;
            for (uint32_t i = 0; i < n; ++i) {
                starts[i] = total;
                total += static_cast<uint32_t>(local_indices[i].size());
            }
            starts[n] = total;

            row_indices.resize(total);
            values.resize(total);
            for (uint32_t i = 0; i < n; ++i) {
                std::copy(local_indices[i].begin(), local_indices[i].end(),
                          row_indices.begin() + starts[i]);
                std::copy(local_values[i].begin(), local_values[i].end(),
                          values.begin() + starts[i]);
            }
        } else {
            // Serial flat build: one shared pair of vectors with amortized
            // growth.  Avoids the per-column allocations the parallel path
            // pays for, which would otherwise regress the default
            // single-threaded config.
            starts.reserve(n + 1);
            for (uint32_t i = 0; i < n; ++i) {
                obj[i] = cols[i].cost;
                starts.push_back(static_cast<uint32_t>(row_indices.size()));
                row_indices.push_back(self().structural_row_index(cols[i]));
                values.push_back(1.0);
                self().for_each_arc_coeff(cols[i], [&](uint32_t arc, double coeff) {
                    auto it = _arc_to_cap_row.find(arc);
                    if (it != _arc_to_cap_row.end()) {
                        row_indices.push_back(it->second);
                        values.push_back(coeff);
                    }
                });
            }
            starts.push_back(static_cast<uint32_t>(row_indices.size()));
        }

        uint32_t first_lp = _lp->add_cols(obj, lb, ub, starts, row_indices, values);

        // Update mapping and the arc->columns reverse index used by
        // add_violated_capacity_constraints.
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t local_idx = static_cast<uint32_t>(_columns.size());
            _col_to_lp.push_back(first_lp + i);
            _columns.push_back(std::move(cols[i]));
            _col_age.push_back(0);
            self().for_each_arc_coeff(_columns.back(), [&](uint32_t arc, double coeff) {
                _arc_to_col_entries[arc].push_back({local_idx, coeff});
            });
        }

        return n;
    }

    LPStatus solve() { return _lp->solve(); }
    double get_obj() const { return _lp->get_obj(); }
    std::vector<double> get_primals() const { return _lp->get_primals(); }

    std::vector<double> get_structural_duals() const {
        auto all = _lp->get_duals();
        return std::vector<double>(all.begin(), all.begin() + _num_structural_rows);
    }

    std::unordered_map<uint32_t, double> get_capacity_duals() const {
        auto all = _lp->get_duals();
        std::unordered_map<uint32_t, double> result;
        for (uint32_t i = 0; i < _cap_row_to_arc.size(); ++i) {
            uint32_t row = _num_structural_rows + i;
            if (row < all.size()) {
                result[_cap_row_to_arc[i]] = all[row];
            }
        }
        return result;
    }

    std::vector<uint32_t> add_violated_capacity_constraints(const std::vector<double>& primals,
                                                            uint32_t current_iter = 0) {
        auto flow = compute_arc_flow(primals);
        auto new_arcs = find_violated_arcs(flow);

        if (new_arcs.empty())
            return {};

        // Build CSR for new capacity rows
        std::vector<double> row_lb;
        std::vector<double> row_ub;
        std::vector<uint32_t> starts;
        std::vector<uint32_t> indices;
        std::vector<double> values;

        for (uint32_t a : new_arcs) {
            row_lb.push_back(-INF);
            row_ub.push_back(_inst->capacity[a]);
            starts.push_back(static_cast<uint32_t>(indices.size()));
            for (const auto& entry : _arc_to_col_entries[a]) {
                indices.push_back(_col_to_lp[entry.col_idx]);
                values.push_back(entry.coeff);
            }
        }

        uint32_t first_row = _lp->add_rows(row_lb, row_ub, starts, indices, values);

        // Update mappings
        for (uint32_t i = 0; i < new_arcs.size(); ++i) {
            _arc_to_cap_row[new_arcs[i]] = first_row + i;
            _cap_row_to_arc.push_back(new_arcs[i]);
            _cap_row_last_active.push_back(current_iter);
        }

        return new_arcs;
    }

    // Mark capacity rows as active when their dual is non-zero.
    void update_capacity_row_activity(uint32_t current_iter) {
        auto duals = _lp->get_duals();
        for (uint32_t i = 0; i < _cap_row_to_arc.size(); ++i) {
            uint32_t row = _num_structural_rows + i;
            if (row < duals.size() && std::abs(duals[row]) > 1e-9) {
                _cap_row_last_active[i] = current_iter;
            }
        }
    }

    void update_column_ages(const std::vector<double>& primals) {
        if (_lp->has_basis()) {
            auto basic = _lp->get_basic_cols();
            for (uint32_t c = 0; c < _columns.size(); ++c) {
                if (basic[_col_to_lp[c]]) {
                    _col_age[c] = 0;  // basic columns are active
                } else {
                    ++_col_age[c];
                }
            }
        } else {
            // Barrier solver fallback: a column is active if primal > eps
            // or reduced cost < -eps (matches Flowty MCF model).
            constexpr double EPS = 1e-6;
            auto reduced_costs = _lp->get_reduced_costs();
            bool have_rc = !reduced_costs.empty();
            for (uint32_t c = 0; c < _columns.size(); ++c) {
                uint32_t lp_col = _col_to_lp[c];
                bool active = primals[lp_col] > EPS || (have_rc && reduced_costs[lp_col] < -EPS);
                if (active) {
                    _col_age[c] = 0;
                } else {
                    ++_col_age[c];
                }
            }
        }
    }

    // Remove capacity rows that have been non-binding for more than
    // inactivity_threshold iterations. Returns the number of rows purged.
    uint32_t purge_nonbinding_capacity_rows(uint32_t current_iter, uint32_t inactivity_threshold) {
        if (_cap_row_to_arc.empty())
            return 0;

        // Build delete mask for LP (size = total LP rows)
        uint32_t total_rows = _lp->num_rows();
        std::vector<int32_t> mask(total_rows, 0);
        uint32_t purge_count = 0;

        for (uint32_t i = 0; i < _cap_row_to_arc.size(); ++i) {
            uint32_t row = _num_structural_rows + i;
            if (current_iter > _cap_row_last_active[i] &&
                current_iter - _cap_row_last_active[i] > inactivity_threshold) {
                mask[row] = 1;
                ++purge_count;
            }
        }

        if (purge_count == 0)
            return 0;

        _lp->delete_rows(mask);

        // Rebuild internal data structures from surviving entries
        std::vector<uint32_t> new_cap_row_to_arc;
        std::vector<uint32_t> new_cap_row_last_active;
        _arc_to_cap_row.clear();

        for (uint32_t i = 0; i < _cap_row_to_arc.size(); ++i) {
            uint32_t old_row = _num_structural_rows + i;
            int32_t new_row = mask[old_row];
            if (new_row >= 0) {
                _arc_to_cap_row[_cap_row_to_arc[i]] = static_cast<uint32_t>(new_row);
                new_cap_row_to_arc.push_back(_cap_row_to_arc[i]);
                new_cap_row_last_active.push_back(_cap_row_last_active[i]);
            }
        }

        _cap_row_to_arc = std::move(new_cap_row_to_arc);
        _cap_row_last_active = std::move(new_cap_row_last_active);

        return purge_count;
    }

    uint32_t purge_aged_columns(uint32_t age_limit) {
        if (age_limit == 0)
            return 0;

        // Build LP-level deletion mask
        uint32_t num_lp = _lp->num_cols();
        std::vector<int32_t> mask(num_lp, 0);

        uint32_t purge_count = 0;
        for (uint32_t c = 0; c < _columns.size(); ++c) {
            if (_col_age[c] > age_limit) {
                mask[_col_to_lp[c]] = 1;
                ++purge_count;
            }
        }

        if (purge_count == 0)
            return 0;

        _lp->delete_cols(mask);

        // Compact internal vectors and record the old→new column index
        // mapping so we can remap _arc_to_col_entries in place instead
        // of clearing and re-walking every surviving column.  PURGED is
        // a sentinel meaning "this entry must be dropped".
        constexpr uint32_t PURGED = std::numeric_limits<uint32_t>::max();
        std::vector<uint32_t> old_to_new(_columns.size(), PURGED);
        uint32_t write = 0;
        for (uint32_t c = 0; c < _columns.size(); ++c) {
            int32_t new_lp = mask[_col_to_lp[c]];
            if (new_lp >= 0) {
                old_to_new[c] = write;
                _columns[write] = std::move(_columns[c]);
                _col_to_lp[write] = static_cast<uint32_t>(new_lp);
                _col_age[write] = _col_age[c];
                ++write;
            }
        }
        _columns.resize(write);
        _col_to_lp.resize(write);
        _col_age.resize(write);

        // Remap _arc_to_col_entries in place: for every entry, look up
        // the column's new index and either keep (with the new id) or
        // drop (if purged).  Skips the CRTP for_each_arc_coeff walk that
        // a full rebuild would do.
        for (auto& vec : _arc_to_col_entries) {
            size_t out = 0;
            for (size_t i = 0; i < vec.size(); ++i) {
                uint32_t new_idx = old_to_new[vec[i].col_idx];
                if (new_idx != PURGED) {
                    vec[out++] = {new_idx, vec[i].coeff};
                }
            }
            vec.resize(out);
        }

        return purge_count;
    }

    uint32_t num_columns() const { return static_cast<uint32_t>(_columns.size()); }
    const std::vector<ColumnT>& columns() const { return _columns; }

    uint32_t num_lp_cols() const { return _lp->num_cols(); }
    uint32_t num_lp_rows() const { return _lp->num_rows(); }

private:
    // Sum the flow contribution of every column on every arc.
    //
    // For determinism the parallel path uses a *static* block partition:
    // chunk index `t` always sees the same set of columns regardless of
    // which physical thread runs it, so the per-bucket sums are byte-
    // for-byte identical across runs.  The dynamic dispatcher is fine
    // for assigning chunks to threads — only the column→bucket mapping
    // needs to be deterministic.
    static_map<uint32_t, double> compute_arc_flow(const std::vector<double>& primals) {
        uint32_t num_cols = static_cast<uint32_t>(_columns.size());
        uint32_t num_arcs = _inst->graph.num_arcs();
        auto flow = _inst->graph.create_arc_map<double>(0.0);
        if (num_cols == 0)
            return flow;

        bool use_pool = _pool != nullptr && num_cols >= PAR_COL_THRESHOLD;

        if (!use_pool) {
            // Serial fast path: skip the per-thread workspaces entirely
            // and accumulate directly into the result.
            for (uint32_t c = 0; c < num_cols; ++c) {
                double x = primals[_col_to_lp[c]];
                if (x < 1e-10)
                    continue;
                self().accumulate_flow(_columns[c], x, flow);
            }
            return flow;
        }

        uint32_t num_threads = _pool->num_threads();
        uint32_t chunk = (num_cols + num_threads - 1) / num_threads;

        for (uint32_t tid = 0; tid < num_threads; ++tid) {
            _thread_flow[tid].fill(0.0);
        }

        // Each task owns chunk t = [t*chunk, (t+1)*chunk) and writes
        // into _thread_flow[t].  Note we index by the deterministic task
        // id, not the physical thread id.
        _pool->parallel_for(num_threads, [&](uint32_t task, uint32_t /*tid*/) {
            uint32_t start = task * chunk;
            uint32_t end = std::min(start + chunk, num_cols);
            auto& bucket = _thread_flow[task];
            for (uint32_t c = start; c < end; ++c) {
                double x = primals[_col_to_lp[c]];
                if (x < 1e-10)
                    continue;
                self().accumulate_flow(_columns[c], x, bucket);
            }
        });

        // Merge per-bucket partials into the final flow map.  Iterating
        // buckets in fixed [0..num_threads) order keeps the merge sum
        // deterministic too.
        if (num_arcs >= PAR_ARC_THRESHOLD) {
            _pool->parallel_for(num_arcs, [&](uint32_t a, uint32_t /*tid*/) {
                double sum = 0.0;
                for (uint32_t bucket = 0; bucket < num_threads; ++bucket) {
                    sum += _thread_flow[bucket][a];
                }
                flow[a] = sum;
            });
        } else {
            for (uint32_t a = 0; a < num_arcs; ++a) {
                double sum = 0.0;
                for (uint32_t bucket = 0; bucket < num_threads; ++bucket) {
                    sum += _thread_flow[bucket][a];
                }
                flow[a] = sum;
            }
        }
        return flow;
    }

    // Find arcs where flow exceeds capacity and no row exists yet.
    // Returns the violated arc ids in ascending order to keep the cut
    // ordering deterministic across pool/no-pool runs.
    std::vector<uint32_t> find_violated_arcs(const static_map<uint32_t, double>& flow) const {
        uint32_t num_arcs = _inst->graph.num_arcs();
        std::vector<uint32_t> new_arcs;

        if (_pool != nullptr && num_arcs >= PAR_ARC_THRESHOLD) {
            uint32_t num_threads = _pool->num_threads();
            std::vector<std::vector<uint32_t>> thread_new_arcs(num_threads);
            _pool->parallel_for(num_arcs, [&](uint32_t a, uint32_t tid) {
                if (flow[a] > _inst->capacity[a] + 1e-6 &&
                    _arc_to_cap_row.find(a) == _arc_to_cap_row.end()) {
                    thread_new_arcs[tid].push_back(a);
                }
            });
            size_t total = 0;
            for (auto& v : thread_new_arcs)
                total += v.size();
            new_arcs.reserve(total);
            for (auto& v : thread_new_arcs) {
                new_arcs.insert(new_arcs.end(), v.begin(), v.end());
            }
            std::sort(new_arcs.begin(), new_arcs.end());
        } else {
            for (uint32_t a = 0; a < num_arcs; ++a) {
                if (flow[a] > _inst->capacity[a] + 1e-6 &&
                    _arc_to_cap_row.find(a) == _arc_to_cap_row.end()) {
                    new_arcs.push_back(a);
                }
            }
        }
        return new_arcs;
    }
};

}  // namespace mcfcg
