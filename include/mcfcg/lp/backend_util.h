#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

// Small shared helpers for the LP backend implementations (COPT / MOSEK / cuOpt
// / HiGHS). Header-only, internal — not part of the public LPSolver interface.
namespace mcfcg::detail {

// Bound magnitude at or beyond which a backend treats a bound as infinite. The
// master and tests pass 1e20 as an "infinity" stand-in; clamp anything with
// magnitude >= this to the backend's native infinity representation (MOSEK: a
// one-sided bound key; cuOpt: +/-CUOPT_INFINITY). Passing a large finite value
// as a genuine range bound would corrupt an interior-point starting point (the
// bound value enters the barrier), so only the infinite *side* may encode it.
inline constexpr double LP_BOUND_INF_THRESHOLD = 1e19;

// Collect the indices flagged for deletion (mask[i] == 1) from an LPSolver
// delete mask, narrowed to int32_t for the backend delete APIs
// (COPT_DelCols/DelRows, MSK_removevars/removecons, ...).
inline std::vector<int32_t> collect_delete_indices(const std::vector<int32_t>& mask) {
    std::vector<int32_t> del_list;
    for (std::size_t idx = 0; idx < mask.size(); ++idx) {
        if (mask[idx] == 1) {
            del_list.push_back(static_cast<int32_t>(idx));
        }
    }
    return del_list;
}

// Rewrite a delete mask in place to the LPSolver delete-mask output contract
// (see lp_solver.h): mask[i] = -1 if item i was deleted (input mask[i] == 1),
// else its new index in the compacted LP. Surviving items are renumbered
// densely in their original order.
inline void remap_delete_mask(std::vector<int32_t>& mask) {
    int32_t new_idx = 0;
    for (int32_t& entry : mask) {
        entry = (entry == 1) ? -1 : new_idx++;
    }
}

}  // namespace mcfcg::detail
