#include "mcfcg/cg/tree_cg.h"

#include "mcfcg/cg/cg_loop.h"
#include "mcfcg/cg/tree_master.h"
#include "mcfcg/cg/tree_pricer.h"

namespace mcfcg {

CGResult solve_tree_cg(const Instance& inst, const CGParams& params) {
    return solve_cg<TreeMaster, TreePricer>(
        inst, params, [](const TreeMaster& m) { return m.get_source_duals(); },
        static_cast<uint32_t>(inst.sources.size()));
}

}  // namespace mcfcg
