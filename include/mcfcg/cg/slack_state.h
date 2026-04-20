#pragma once

#include "mcfcg/lp/lp_solver.h"
#include "mcfcg/util/tolerances.h"

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <span>
#include <unordered_map>
#include <vector>

namespace mcfcg {

// Where slack columns live.  The master places slacks on whichever row
// set is smaller so the slack column count is min(num_structural_rows,
// num_capacitated_arcs) — the selector runs in MasterBase::init once the
// instance is known.
//   CommodityRows: one slack per structural row (commodity demand in
//                  path, source convexity in tree), added at init.
//                  Structural rows carry the slack with coeff +1.
//   EdgeRows:      one slack per capacity row, added lazily in
//                  add_violated_capacity_constraints.  Slack coeff is -1
//                  in its capacity row (flow_a - s_a ≤ cap_a).
//                  Structural rows have NO slack entry, so the LP is
//                  feasible only if the master is seeded with at least
//                  one column per structural row — EdgeRows requires
//                  warm_start=true; init throws when that contract is
//                  violated.  After warm-start, demand rows stay served
//                  because update_column_ages never ages out a basis
//                  candidate (reduced cost within LP_FEAS_TOL of zero),
//                  so purging aged columns cannot take the last basic
//                  column serving a row.
enum class SlackMode : uint8_t { CommodityRows, EdgeRows };

// gtest pretty-printer so EXPECT_EQ on SlackMode prints the enum name
// on failure instead of a raw byte.  Header-only; no link-time impact.
inline std::ostream& operator<<(std::ostream& out, SlackMode mode) {
    return out << (mode == SlackMode::EdgeRows ? "EdgeRows" : "CommodityRows");
}

// Slack-column bookkeeping for the master problem.  Owns the three
// vectors tracking every slack's LP column index, current objective
// coefficient, and (for EdgeRows) arc-to-LP-col index, plus the mode
// and the absolute cost ceiling enforced during bump_active.
//
// The mutation that creates slacks (at init in CommodityRows mode; in
// add_violated_capacity_constraints in EdgeRows mode) stays on
// MasterBase because it interleaves with structural rows and capacity
// rows.  The remapping through a delete_cols mask in purge_aged_columns
// / delete_edge_row_slacks also stays there because it needs the
// column mask that MasterBase itself manages.  What lives here is the
// pure per-iteration logic (empty / has_active / bump_active).
struct SlackState {
    SlackMode mode = SlackMode::CommodityRows;

    // col_lp[k] is the LP column index of slack k; cost[k] is its
    // current objective coefficient (grown by bump_active).  In
    // CommodityRows mode, slacks are columns 0..num_structural-1 and
    // never shift.  In EdgeRows mode, slacks interleave with user
    // columns in LP index space, so purge_aged_columns and
    // delete_edge_row_slacks in MasterBase remap these indices through
    // the delete mask.
    std::vector<uint32_t> col_lp;
    std::vector<double> cost;

    // EdgeRows-only: arc → LP column index of that arc's paired slack.
    // Empty in CommodityRows mode.
    std::unordered_map<uint32_t, uint32_t> arc_to_col;

    // Absolute ceiling on the geometric slack-cost bump.  Set by
    // MasterBase::init from a Derived-supplied upper bound on real
    // column cost (10× with clamps) to keep the LP basis numerically
    // stable.
    double cost_ceiling = 1e8;

    bool empty() const noexcept { return col_lp.empty(); }

    void clear() noexcept {
        col_lp.clear();
        cost.clear();
        arc_to_col.clear();
    }

    // Returns true if any slack column has primal > COL_ACTIVE_EPS in
    // the passed primal vector.  Used by cg_loop to decide whether to
    // keep bumping slack costs after pricing is exhausted.
    bool has_active(std::span<const double> primals) const noexcept {
        if (col_lp.empty()) {
            return false;
        }
        constexpr double SLACK_ACTIVE_EPS = COL_ACTIVE_EPS;
        return std::ranges::any_of(col_lp, [&](uint32_t lp_col) {
            return lp_col < primals.size() && primals[lp_col] > SLACK_ACTIVE_EPS;
        });
    }

    // Grow the cost of every active slack by `factor`, clamped to
    // `cost_ceiling`.  Pushes updated costs through to the LP via
    // set_col_cost.  Returns the number of slacks actually bumped.
    uint32_t bump_active(std::span<const double> primals, double factor, LPSolver& lpsolver) {
        if (col_lp.empty()) {
            return 0;
        }
        constexpr double SLACK_ACTIVE_EPS = COL_ACTIVE_EPS;
        uint32_t bumped = 0;
        for (uint32_t k = 0; k < col_lp.size(); ++k) {
            uint32_t lp_col = col_lp[k];
            if (lp_col >= primals.size() || primals[lp_col] <= SLACK_ACTIVE_EPS) {
                continue;
            }
            if (cost[k] >= cost_ceiling) {
                continue;  // already at LP-backend numerical ceiling
            }
            cost[k] = std::min(cost[k] * factor, cost_ceiling);
            lpsolver.set_col_cost(lp_col, cost[k]);
            ++bumped;
        }
        return bumped;
    }
};

}  // namespace mcfcg
