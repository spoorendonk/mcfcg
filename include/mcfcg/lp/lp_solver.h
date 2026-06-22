#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace mcfcg {

enum class LPStatus { Optimal, Infeasible, Unbounded, Error };

// Backend-agnostic LP solver interface used by the CG master.
//
// Callers own the solver (via unique_ptr).  One solver handles one LP
// across its lifetime: the master incrementally adds columns and rows
// and triggers solve(), re-reading primals / duals / reduced costs
// between mutations.
//
// ## Delete-mask semantics
//
// delete_cols / delete_rows take an in-out mask:
//   * input:  mask[i] = 1 to delete item i, 0 to keep.
//   * output: mask[i] = new index of item i in the compacted LP, or -1
//             if the item was deleted.
// The master relies on this to remap its column-index bookkeeping
// (_col_to_lp, _slack.col_lp, _slack.arc_to_col, etc.) after a purge.
//
// ## Cached-solution invalidation
//
// HiGHS and COPT invalidate their cached primal/dual/reduced-cost
// vectors on delete_cols or delete_rows.  Callers that need to read
// those vectors after a purge MUST either capture them before the
// delete or re-solve first.  cg_loop.h orders activity updates and
// slack bumps BEFORE the purge for exactly this reason.
//
// ## `starts` convention (uniform across add_cols and add_rows)
//
// CSC for columns, CSR for rows.  In both calls, starts is sized
// n+1 where n is the number of new items, and starts[n] == values.size()
// is a mandatory sentinel.  The backend reads the i-th item's range as
// [starts[i], starts[i+1]).
class LPSolver {
public:
    virtual ~LPSolver() = default;

    // Add columns with no row coefficients. Returns index of first new column.
    virtual uint32_t add_cols(const std::vector<double>& obj, const std::vector<double>& lb,
                              const std::vector<double>& ub) = 0;

    // Add columns with row coefficients (CSC format).
    // Returns index of first new column.
    // starts: CSC column starts into row_indices/values, size = obj.size() + 1
    //         (the final sentinel must equal values.size()).
    virtual uint32_t add_cols(const std::vector<double>& obj, const std::vector<double>& lb,
                              const std::vector<double>& ub, const std::vector<uint32_t>& starts,
                              const std::vector<uint32_t>& row_indices,
                              const std::vector<double>& values) = 0;

    // Add rows (constraints) in CSR format. Returns index of first new row.
    // starts: CSR row starts into indices/values, size = lb.size() + 1
    //         (the final sentinel must equal values.size()).
    virtual uint32_t add_rows(const std::vector<double>& lb, const std::vector<double>& ub,
                              const std::vector<uint32_t>& starts,
                              const std::vector<uint32_t>& indices,
                              const std::vector<double>& values) = 0;

    // Bulk delete columns by mask. mask[i]=1 means delete column i.
    // After call, mask[i] contains the new index of column i (or -1 if
    // deleted).
    virtual void delete_cols(std::vector<int32_t>& mask) = 0;

    // Bulk delete rows by mask. mask[i]=1 means delete row i.
    // After call, mask[i] contains the new index of row i (or -1 if deleted).
    virtual void delete_rows(std::vector<int32_t>& mask) = 0;

    // Update a single column's objective coefficient. Used by the master's
    // bump-to-fixed-point slack-cost loop.
    virtual void set_col_cost(uint32_t col, double cost) = 0;

    // Solve the LP.  When `certify` is true the caller needs a *certified*
    // (vertex-quality) primal — slacks driven to exactly 0 where feasible — so
    // it can trust the objective as an MCF upper bound.  The CG loop sets it
    // only when it has stalled (pricing exhausted but slacks still basic),
    // which on a pure interior-point solution can be an artifact of the
    // non-vertex solution rather than true infeasibility.  Backends whose
    // barrier already yields a clean enough solution ignore it (no-op);
    // HiGHS (HiPO) enables crossover for that one solve.  Default false
    // preserves the fast per-iteration path.
    virtual LPStatus solve(bool certify = false) = 0;

    // Get solution info (valid after solve returns Optimal)
    virtual double get_obj() const = 0;
    virtual std::vector<double> get_primals() const = 0;
    virtual std::vector<double> get_duals() const = 0;

    // Returns per-column reduced costs from the last solve. Default
    // returns an empty vector for backends that do not expose them.
    virtual std::vector<double> get_reduced_costs() const { return {}; }

    virtual uint32_t num_cols() const = 0;
    virtual uint32_t num_rows() const = 0;

    // Returns true if a valid basis is available (simplex solvers).
    virtual bool has_basis() const { return false; }

    // Returns per-column basis status: true = basic, false = non-basic.
    // Only valid when has_basis() returns true.
    virtual std::vector<bool> get_basic_cols() const { return {}; }

    // Upper bound on the slack-cost ceiling this backend can tolerate without
    // numerical trouble (MasterBase::init clamps the per-instance ceiling to
    // this).  The default 1e7 suits HiGHS (its dual-simplex ratio test starts
    // failing near ~1e9) and cuOpt-barrier (whose IPM stalls on a wide
    // slack-vs-real dynamic range).  Robust CPU/GPU barriers (MOSEK, COPT)
    // override this higher so that instances with expensive columns (per-row
    // cost > 1e7, e.g. planar2500 tree) can still price their slacks out and
    // reach a slack-free upper bound.
    virtual double max_slack_cost() const { return 1e7; }
};

// Emit a one-line provenance banner to stderr at solver construction (captured
// in CG / benchmark logs). Reports the pinned barrier settings (presolve off,
// crossover off, tol = BARRIER_TOL) plus the effective execution mode and
// thread count, so every run is self-documenting. `threads <= 0` is reported as
// "auto(<hardware_concurrency>)"; `gpu` selects the CPU/GPU execution label.
void log_solver_config(const char* backend, const char* method, bool gpu, int threads);

std::unique_ptr<LPSolver> create_lp_solver(bool verbose = false);

#ifdef MCFCG_USE_CUOPT
// cuOpt always solves with its barrier (IPM) method. This research repo does not
// expose per-solver algorithm selection — each backend uses one fixed method
// (HiGHS: whatever it was compiled with; cuOpt: barrier; COPT: barrier).
std::unique_ptr<LPSolver> create_cuopt_solver(bool verbose = false);
#endif

#ifdef MCFCG_USE_COPT
// gpu_mode selects COPT's barrier execution: 0 = CPU, 1 = GPU mode 1, 2 = GPU
// mode 2. The default -1 (and any out-of-range value) means "GPU barrier (2)".
std::unique_ptr<LPSolver> create_copt_solver(bool verbose = false, int gpu_mode = -1);
#endif

#ifdef MCFCG_USE_MOSEK
// MOSEK always solves with its interior-point (barrier) method, presolve and
// crossover off — one fixed method per backend, like COPT/cuOpt.
std::unique_ptr<LPSolver> create_mosek_solver(bool verbose = false);
#endif

}  // namespace mcfcg
