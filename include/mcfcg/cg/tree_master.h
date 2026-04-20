#pragma once

#include "mcfcg/cg/master_base.h"
#include "mcfcg/cg/tree_column.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace mcfcg {

class TreeMaster : public MasterBase<TreeMaster, TreeColumn> {
    friend class MasterBase<TreeMaster, TreeColumn>;

    uint32_t num_structural_entities() const {
        return static_cast<uint32_t>(_inst->sources.size());
    }

    std::pair<double, double> structural_row_bounds([[maybe_unused]] uint32_t k) const {
        return {1.0, 1.0};
    }

    uint32_t structural_row_index(const TreeColumn& col) const { return col.source_idx; }

    void for_each_arc_coeff(const TreeColumn& col, auto&& callback) const {
        for (const auto& af : col.arc_flows) {
            callback(af.arc, af.flow);
        }
    }

    // Tree column cost for source s = sum_{k in s} d_k × path_k_cost,
    // bounded above by (sum_{k in s} d_k) × (|V|-1) × max_arc_cost.
    // Use the max over sources so the slack ceiling is tight enough to
    // out-price the costliest real column but not needlessly large.
    double slack_cost_upper_bound() const {
        double max_src_demand_sum = 0.0;
        for (const auto& src : _inst->sources) {
            double sum = 0.0;
            for (uint32_t k : src.commodity_indices) {
                sum += _inst->commodities[k].demand;
            }
            max_src_demand_sum = std::max(max_src_demand_sum, sum);
        }
        if (max_src_demand_sum <= 0.0) {
            max_src_demand_sum = 1.0;
        }
        return _max_cost * static_cast<double>(_inst->graph.num_vertices()) * max_src_demand_sum;
    }
};

}  // namespace mcfcg
