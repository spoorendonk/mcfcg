#pragma once

#include "mcfcg/instance.h"
#include "mcfcg/lp/lp_solver.h"

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
    static constexpr double INF = std::numeric_limits<double>::infinity();

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

    Derived& self() noexcept { return static_cast<Derived&>(*this); }
    const Derived& self() const noexcept { return static_cast<const Derived&>(*this); }

public:
    MasterBase() = default;

    void init(const Instance& inst, std::unique_ptr<LPSolver> lp = nullptr) {
        _inst = &inst;
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

        // Build CSC matrix for new columns
        std::vector<double> obj;
        std::vector<double> lb;
        std::vector<double> ub;
        std::vector<uint32_t> starts;
        std::vector<uint32_t> row_indices;
        std::vector<double> values;

        for (auto& col : cols) {
            obj.push_back(col.cost);
            lb.push_back(0.0);
            ub.push_back(INF);
            starts.push_back(static_cast<uint32_t>(row_indices.size()));

            // Structural row coefficient
            row_indices.push_back(self().structural_row_index(col));
            values.push_back(1.0);

            // Capacity row coefficients
            self().for_each_arc_coeff(col, [&](uint32_t arc, double coeff) {
                auto it = _arc_to_cap_row.find(arc);
                if (it != _arc_to_cap_row.end()) {
                    row_indices.push_back(it->second);
                    values.push_back(coeff);
                }
            });
        }
        // Sentinel
        starts.push_back(static_cast<uint32_t>(row_indices.size()));

        uint32_t first_lp = _lp->add_cols(obj, lb, ub, starts, row_indices, values);

        // Update mapping
        uint32_t n = static_cast<uint32_t>(cols.size());
        for (uint32_t i = 0; i < n; ++i) {
            _col_to_lp.push_back(first_lp + i);
            _columns.push_back(std::move(cols[i]));
            _col_age.push_back(0);
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
        // Compute flow on each arc
        auto flow = _inst->graph.create_arc_map<double>(0.0);
        for (uint32_t c = 0; c < _columns.size(); ++c) {
            double x = primals[_col_to_lp[c]];
            if (x < 1e-10)
                continue;
            self().accumulate_flow(_columns[c], x, flow);
        }

        // Find newly violated arcs
        std::vector<uint32_t> new_arcs;
        for (auto a : _inst->graph.arcs()) {
            if (flow[a] > _inst->capacity[a] + 1e-6 &&
                _arc_to_cap_row.find(a) == _arc_to_cap_row.end()) {
                new_arcs.push_back(a);
            }
        }

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
            for (uint32_t c = 0; c < _columns.size(); ++c) {
                self().for_each_arc_coeff(_columns[c], [&](uint32_t arc, double coeff) {
                    if (arc == a) {
                        indices.push_back(_col_to_lp[c]);
                        values.push_back(coeff);
                    }
                });
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
            // Barrier solver fallback: active if non-zero primal
            for (uint32_t c = 0; c < _columns.size(); ++c) {
                if (primals[_col_to_lp[c]] < 1e-10) {
                    ++_col_age[c];
                } else {
                    _col_age[c] = 0;
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

        // Compact internal vectors, remapping surviving columns
        uint32_t write = 0;
        for (uint32_t c = 0; c < _columns.size(); ++c) {
            int32_t new_lp = mask[_col_to_lp[c]];
            if (new_lp >= 0) {
                _columns[write] = std::move(_columns[c]);
                _col_to_lp[write] = static_cast<uint32_t>(new_lp);
                _col_age[write] = _col_age[c];
                ++write;
            }
        }
        _columns.resize(write);
        _col_to_lp.resize(write);
        _col_age.resize(write);

        return purge_count;
    }

    uint32_t num_columns() const { return static_cast<uint32_t>(_columns.size()); }
    const std::vector<ColumnT>& columns() const { return _columns; }

    uint32_t num_lp_cols() const { return _lp->num_cols(); }
    uint32_t num_lp_rows() const { return _lp->num_rows(); }
};

}  // namespace mcfcg
