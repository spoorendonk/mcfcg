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

    void accumulate_flow(const TreeColumn& col, double x,
                         static_map<uint32_t, double>& flow) const {
        for (const auto& af : col.arc_flows) {
            flow[af.arc] += x * af.flow;
        }
    }

public:
    std::vector<double> get_source_duals() const { return get_structural_duals(); }
};

}  // namespace mcfcg
