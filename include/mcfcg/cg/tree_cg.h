#pragma once

#include "mcfcg/cg/path_cg.h"

namespace mcfcg {

CGResult solve_tree_cg(const Instance & inst, const CGParams & params = {});

}  // namespace mcfcg
