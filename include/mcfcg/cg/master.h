#pragma once

#include "mcfcg/cg/column.h"
#include "mcfcg/cg/master_base.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace mcfcg {

class PathMaster : public MasterBase<PathMaster, Column> {
    friend class MasterBase<PathMaster, Column>;

    uint32_t num_structural_entities() const {
        return static_cast<uint32_t>(_inst->commodities.size());
    }

    std::pair<double, double> structural_row_bounds(uint32_t k) const {
        return {_inst->commodities[k].demand, INF};
    }

    uint32_t structural_row_index(const Column& col) const { return col.commodity; }

    void for_each_arc_coeff(const Column& col, auto&& callback) const {
        for (uint32_t arc : col.arcs) {
            callback(arc, 1.0);
        }
    }

    // Path column cost = sum over arcs of inst.cost[a], bounded above by
    // (|V|-1) × max_arc_cost (simple path).  Returned per column; the
    // ceiling is 10× this so a slack can always out-price any path.
    double slack_cost_upper_bound() const {
        return _max_cost * static_cast<double>(_inst->graph.num_vertices());
    }
};

}  // namespace mcfcg
