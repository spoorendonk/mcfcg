#include "mcfcg/cg/path_cg.h"

#include "mcfcg/cg/cg_loop.h"
#include "mcfcg/cg/master.h"
#include "mcfcg/cg/pricer.h"

namespace mcfcg {

CGResult solve_path_cg(const Instance& inst, const CGParams& params) {
    return solve_cg<PathMaster, PathPricer>(
        inst, params, [](const PathMaster& m) { return m.get_demand_duals(); },
        static_cast<uint32_t>(inst.commodities.size()));
}

}  // namespace mcfcg
