#pragma once

// Central numerical tolerances for the MCF column generation solver.
//
// All tolerances derive from a single design target: the MCF solution
// should be feasible to within RELATIVE_FEAS_TOL of demand and capacity
// bounds.  Every downstream threshold is set so that LP-solver noise at
// that precision cannot trigger false positives in slack detection,
// column acceptance, or constraint-violation checks.
//
// Tolerance chain (each layer ≥ 10× the one above to avoid chasing
// noise):
//
//   LP_FEAS_TOL  = 1e-4    LP primal/dual feasibility
//        │
//        ├── COL_ACTIVE_EPS = 1e-3   primal > eps ⇒ column/slack is
//        │                           genuinely basic, not residual noise
//        │
//        ├── NEG_RC_TOL     = -1e-3  RC must be this negative before a
//        │                           new column is accepted (noise from
//        │                           dual accumulation over ~10 arcs is
//        │                           ~10 × LP_FEAS_TOL ≈ 1e-3)
//        │
//        ├── CAP_VIOL_TOL   = 1e-4   flow must exceed capacity by this
//        │                           much before a lazy cut is added
//        │
//        └── DUAL_ACTIVE_EPS = 1e-4  |dual| must exceed this for a
//                                    capacity row to count as "active"
//                                    (purge protection)

namespace mcfcg {

// ── Design target ──────────────────────────────────────────────────
// The MCF solution is feasible when demand and capacity constraints
// are satisfied to within this relative tolerance.
inline constexpr double RELATIVE_FEAS_TOL = 1e-4;

// ── LP solver ──────────────────────────────────────────────────────
// Absolute primal/dual feasibility tolerance passed to every backend
// (HiGHS, COPT, cuOpt).  Equal to the design target — the LP should
// deliver solutions at least as tight as what we promise downstream.
inline constexpr double LP_FEAS_TOL = RELATIVE_FEAS_TOL;  // 1e-4

// ── Column / slack activity ────────────────────────────────────────
// A column (or slack) is considered genuinely basic / active when its
// LP primal exceeds this threshold.  Must be > LP_FEAS_TOL so that
// residual numerical noise from the LP solver does not register as
// "active".  Used by bump_active_slacks, has_active_slacks, and the
// barrier-solver fallback in update_column_ages.
inline constexpr double COL_ACTIVE_EPS = LP_FEAS_TOL * 10;  // 1e-3

// ── Reduced cost acceptance ────────────────────────────────────────
// A new column is accepted by the pricer only if its true reduced
// cost is below this threshold.  Dual noise accumulates over the arcs
// in a path/tree (~10 arcs × LP_FEAS_TOL ≈ 1e-3), so the threshold
// must be more negative than the noise floor.  The default matches
// COL_ACTIVE_EPS with a sign flip; barrier backends (cuOpt) may use
// the same value since their duals are already at LP_FEAS_TOL.
inline constexpr double NEG_RC_TOL = -COL_ACTIVE_EPS;  // -1e-3

// ── Capacity violation ─────────────────────────────────────────────
// Lazy capacity rows are added when flow exceeds capacity by more
// than this absolute amount.  Matches LP_FEAS_TOL: any violation
// larger than what the LP can resolve should be cut.
inline constexpr double CAP_VIOL_TOL = LP_FEAS_TOL;  // 1e-4

// ── Dual activity (capacity-row purge protection) ──────────────────
// A capacity row is marked "active" when |dual| exceeds this, which
// protects it from purge.  Matches LP_FEAS_TOL: a dual smaller than
// the LP's own precision is indistinguishable from zero.
inline constexpr double DUAL_ACTIVE_EPS = LP_FEAS_TOL;  // 1e-4

}  // namespace mcfcg
