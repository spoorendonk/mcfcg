#pragma once

#include "mcfcg/cg/column.h"
#include "mcfcg/instance.h"
#include "mcfcg/lp/lp_solver.h"

#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

namespace mcfcg {

class PathMaster {
public:
    static constexpr double BIG_M = 1e8;
    static constexpr double INF = std::numeric_limits<double>::infinity();

private:
    const Instance* _inst = nullptr;
    std::unique_ptr<LPSolver> _lp;

    uint32_t _num_demand_rows = 0;

    // Column store with bidirectional LP mapping
    std::vector<Column> _columns;
    std::vector<uint32_t> _col_to_lp;  // column index -> LP column index

    // Capacity constraint tracking
    std::unordered_map<uint32_t, uint32_t> _arc_to_cap_row;
    std::vector<uint32_t> _cap_row_to_arc;

public:
    PathMaster() = default;

    void init(const Instance& inst) {
        _inst = &inst;
        _num_demand_rows = static_cast<uint32_t>(inst.commodities.size());
        _columns.clear();
        _col_to_lp.clear();
        _arc_to_cap_row.clear();
        _cap_row_to_arc.clear();

        // Create LP once
        _lp = create_lp_solver();

        // Add slack columns (one per commodity, no row coefficients yet)
        std::vector<double> slack_obj(_num_demand_rows, BIG_M);
        std::vector<double> slack_lb(_num_demand_rows, 0.0);
        std::vector<double> slack_ub(_num_demand_rows, INF);
        _lp->add_cols(slack_obj, slack_lb, slack_ub);

        // Add demand rows: sum_p x^k_p + slack_k >= d_k
        // Initially only slack columns have coefficients
        std::vector<double> row_lb;
        std::vector<double> row_ub;
        std::vector<uint32_t> starts;
        std::vector<uint32_t> indices;
        std::vector<double> values;

        for (uint32_t k = 0; k < _num_demand_rows; ++k) {
            row_lb.push_back(_inst->commodities[k].demand);
            row_ub.push_back(INF);
            starts.push_back(static_cast<uint32_t>(indices.size()));
            indices.push_back(k);  // slack column k
            values.push_back(1.0);
        }

        _lp->add_rows(row_lb, row_ub, starts, indices, values);
    }

    uint32_t add_columns(std::vector<Column> cols) {
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

            // Demand row coefficient
            row_indices.push_back(col.commodity);
            values.push_back(1.0);

            // Capacity row coefficients
            for (uint32_t arc : col.arcs) {
                auto it = _arc_to_cap_row.find(arc);
                if (it != _arc_to_cap_row.end()) {
                    row_indices.push_back(it->second);
                    values.push_back(1.0);
                }
            }
        }
        // Sentinel
        starts.push_back(static_cast<uint32_t>(row_indices.size()));

        uint32_t first_lp = _lp->add_cols(obj, lb, ub, starts, row_indices, values);

        // Update mapping
        uint32_t n = static_cast<uint32_t>(cols.size());
        for (uint32_t i = 0; i < n; ++i) {
            _col_to_lp.push_back(first_lp + i);
            _columns.push_back(std::move(cols[i]));
        }

        return n;
    }

    LPStatus solve() { return _lp->solve(); }
    double get_obj() const { return _lp->get_obj(); }
    std::vector<double> get_primals() const { return _lp->get_primals(); }

    std::vector<double> get_demand_duals() const {
        auto all = _lp->get_duals();
        return std::vector<double>(all.begin(), all.begin() + _num_demand_rows);
    }

    std::unordered_map<uint32_t, double> get_capacity_duals() const {
        auto all = _lp->get_duals();
        std::unordered_map<uint32_t, double> result;
        for (uint32_t i = 0; i < _cap_row_to_arc.size(); ++i) {
            uint32_t row = _num_demand_rows + i;
            if (row < all.size()) {
                result[_cap_row_to_arc[i]] = all[row];
            }
        }
        return result;
    }

    uint32_t add_violated_capacity_constraints(const std::vector<double>& primals) {
        // Compute flow on each arc
        auto flow = _inst->graph.create_arc_map<double>(0.0);
        for (uint32_t c = 0; c < _columns.size(); ++c) {
            double x = primals[_col_to_lp[c]];
            if (x < 1e-10)
                continue;
            for (uint32_t a : _columns[c].arcs) {
                flow[a] += x;
            }
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
            return 0;

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
                for (uint32_t arc : _columns[c].arcs) {
                    if (arc == a) {
                        indices.push_back(_col_to_lp[c]);
                        values.push_back(1.0);
                        break;
                    }
                }
            }
        }

        uint32_t first_row = _lp->add_rows(row_lb, row_ub, starts, indices, values);

        // Update mappings
        for (uint32_t i = 0; i < new_arcs.size(); ++i) {
            _arc_to_cap_row[new_arcs[i]] = first_row + i;
            _cap_row_to_arc.push_back(new_arcs[i]);
        }

        return static_cast<uint32_t>(new_arcs.size());
    }

    uint32_t num_columns() const { return static_cast<uint32_t>(_columns.size()); }
    const std::vector<Column>& columns() const { return _columns; }

    uint32_t num_lp_cols() const { return _lp->num_cols(); }
    uint32_t num_lp_rows() const { return _lp->num_rows(); }

};

}  // namespace mcfcg
