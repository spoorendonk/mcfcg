#ifdef MCFCG_USE_MOSEK

#include <mosek.h>

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

#include "mcfcg/lp/backend_util.h"
#include "mcfcg/lp/lp_solver.h"
#include "mcfcg/util/tolerances.h"

namespace mcfcg {

namespace {

void check_mosek(MSKrescodee res, const char* msg) {
    if (res != MSK_RES_OK) {
        throw std::runtime_error(std::string("MOSEK error (") + std::to_string(res) + ") in " +
                                 msg);
    }
}

// MOSEK's infinity is a one-sided bound key, not a finite number. Treat any
// bound with magnitude >= LP_BOUND_INF_THRESHOLD as infinite when picking the
// MSKboundkeye; the numeric value handed to MOSEK for an infinite side is then
// ignored by the key (see backend_util.h for why the threshold matters).
//
// Derive a MOSEK bound key + clamped (lb, ub) pair from a generic (lb, ub).
struct BoundKey {
    MSKboundkeye bk;
    double lo;
    double hi;
};

BoundKey to_mosek_bound(double lb, double ub) {
    bool lb_inf = lb <= -detail::LP_BOUND_INF_THRESHOLD;
    bool ub_inf = ub >= detail::LP_BOUND_INF_THRESHOLD;
    if (lb_inf && ub_inf) {
        return {.bk = MSK_BK_FR, .lo = 0.0, .hi = 0.0};
    }
    if (lb_inf) {
        return {.bk = MSK_BK_UP, .lo = 0.0, .hi = ub};
    }
    if (ub_inf) {
        return {.bk = MSK_BK_LO, .lo = lb, .hi = 0.0};
    }
    if (lb == ub) {
        return {.bk = MSK_BK_FX, .lo = lb, .hi = ub};
    }
    return {.bk = MSK_BK_RA, .lo = lb, .hi = ub};
}

// Stream handler that drops MOSEK log output when the solver is non-verbose.
// When verbose, MSK_IPAR_LOG > 0 already prints to MOSEK's default stream;
// we simply do not attach this silencer in that case.
void MSKAPI mosek_silent_stream(MSKuserhandle_t /*handle*/, const char* /*str*/) {}

}  // namespace

class MosekSolver : public LPSolver {
private:
    MSKenv_t _env = nullptr;
    MSKtask_t _task = nullptr;
    // Authoritative solution slot the getters read. MOSEK keeps the interior
    // point in MSK_SOL_ITR; a certify solve (basis identification) writes the
    // rounded vertex into MSK_SOL_BAS, so solve() switches this to MSK_SOL_BAS
    // for that solve. Default MSK_SOL_ITR (steady-state, no crossover).
    MSKsoltypee _sol = MSK_SOL_ITR;

public:
    explicit MosekSolver(bool verbose = false) {
        check_mosek(MSK_makeenv(&_env, nullptr), "makeenv");
        check_mosek(MSK_maketask(_env, 0, 0, &_task), "maketask");

        if (!verbose) {
            // Suppress MOSEK's banner/log entirely.
            check_mosek(MSK_putintparam(_task, MSK_IPAR_LOG, 0), "Log=0");
            check_mosek(
                MSK_linkfunctotaskstream(_task, MSK_STREAM_LOG, nullptr, mosek_silent_stream),
                "linkfunctotaskstream");
        } else {
            check_mosek(MSK_putintparam(_task, MSK_IPAR_LOG, 10), "Log=10");
        }

        // Interior point (barrier), presolve off, no basis identification
        // (crossover off) — one fixed method per backend, matching COPT.
        check_mosek(MSK_putintparam(_task, MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_INTPNT),
                    "Optimizer=intpnt");
        check_mosek(MSK_putintparam(_task, MSK_IPAR_PRESOLVE_USE, MSK_PRESOLVE_MODE_OFF),
                    "Presolve=off");
        check_mosek(MSK_putintparam(_task, MSK_IPAR_INTPNT_BASIS, MSK_BI_NEVER),
                    "IntpntBasis=never");
        check_mosek(MSK_putobjsense(_task, MSK_OBJECTIVE_SENSE_MINIMIZE), "ObjSense=min");

        // Barrier convergence tolerances, pinned to BARRIER_TOL identically
        // across backends so cross-solver timings compare like for like.
        check_mosek(MSK_putdouparam(_task, MSK_DPAR_INTPNT_TOL_PFEAS, BARRIER_TOL), "TolPfeas");
        check_mosek(MSK_putdouparam(_task, MSK_DPAR_INTPNT_TOL_DFEAS, BARRIER_TOL), "TolDfeas");
        check_mosek(MSK_putdouparam(_task, MSK_DPAR_INTPNT_TOL_REL_GAP, BARRIER_TOL), "TolRelGap");
        int threads = 0;
        check_mosek(MSK_getintparam(_task, MSK_IPAR_NUM_THREADS, &threads), "GetNumThreads");
        int major = 0;
        int minor = 0;
        int revision = 0;
        // Degrade rather than throw, like the COPT and cuOpt version queries: a
        // banner field is provenance, and failing to read it must not take down
        // a solve that would otherwise run.
        std::string version = "unknown";
        if (MSK_getversion(&major, &minor, &revision) == MSK_RES_OK) {
            version = std::to_string(major) + "." + std::to_string(minor) + "." +
                      std::to_string(revision);
        }
        log_solver_config("mosek", version.c_str(), "barrier", /*gpu=*/false, threads);
    }

    ~MosekSolver() override {
        if (_task != nullptr) {
            MSK_deletetask(&_task);
        }
        if (_env != nullptr) {
            MSK_deleteenv(&_env);
        }
    }

    MosekSolver(const MosekSolver&) = delete;
    MosekSolver& operator=(const MosekSolver&) = delete;
    MosekSolver(MosekSolver&&) = delete;
    MosekSolver& operator=(MosekSolver&&) = delete;

    // Build parallel MOSEK bound-key / lo / hi arrays from generic (lb, ub)
    // for a slice of n items, ready for MSK_putvarboundslice/putconboundslice.
    static void build_bounds(const std::vector<double>& lb, const std::vector<double>& ub,
                             int32_t n, std::vector<MSKboundkeye>& bk, std::vector<double>& lo,
                             std::vector<double>& hi) {
        bk.resize(n);
        lo.resize(n);
        hi.resize(n);
        for (int32_t i = 0; i < n; ++i) {
            BoundKey b = to_mosek_bound(lb[i], ub[i]);
            bk[i] = b.bk;
            lo[i] = b.lo;
            hi[i] = b.hi;
        }
    }

    uint32_t add_cols(const std::vector<double>& obj, const std::vector<double>& lb,
                      const std::vector<double>& ub) override {
        uint32_t first = num_cols();
        auto n = static_cast<int32_t>(obj.size());
        if (n == 0) {
            return first;
        }
        check_mosek(MSK_appendvars(_task, n), "appendvars");
        auto lo_idx = static_cast<int32_t>(first);
        int32_t hi_idx = lo_idx + n;
        check_mosek(MSK_putcslice(_task, lo_idx, hi_idx, obj.data()), "putcslice");
        std::vector<MSKboundkeye> bk;
        std::vector<double> lo;
        std::vector<double> hi;
        build_bounds(lb, ub, n, bk, lo, hi);
        check_mosek(MSK_putvarboundslice(_task, lo_idx, hi_idx, bk.data(), lo.data(), hi.data()),
                    "putvarboundslice");
        return first;
    }

    uint32_t add_cols(const std::vector<double>& obj, const std::vector<double>& lb,
                      const std::vector<double>& ub, const std::vector<uint32_t>& starts,
                      const std::vector<uint32_t>& row_indices,
                      const std::vector<double>& values) override {
        assert(starts.size() == obj.size() + 1 && starts.back() == values.size() &&
               "add_cols requires starts.size() == n+1 with starts.back() == values.size()");
        uint32_t first = num_cols();
        auto n = static_cast<int32_t>(obj.size());
        if (n == 0) {
            return first;
        }
        check_mosek(MSK_appendvars(_task, n), "appendvars");
        auto lo_idx = static_cast<int32_t>(first);
        int32_t hi_idx = lo_idx + n;
        check_mosek(MSK_putcslice(_task, lo_idx, hi_idx, obj.data()), "putcslice");
        std::vector<MSKboundkeye> bk;
        std::vector<double> lo;
        std::vector<double> hi;
        build_bounds(lb, ub, n, bk, lo, hi);
        check_mosek(MSK_putvarboundslice(_task, lo_idx, hi_idx, bk.data(), lo.data(), hi.data()),
                    "putvarboundslice");

        // ptr[i]/ptr[i+1] are the begin/end offsets of column i into asub/aval,
        // so a single (size n+1) int32 copy of starts serves as both ptrb and
        // ptre = ptr+1. asub is the int32-narrowed row index array.
        std::vector<int32_t> ptr(starts.begin(), starts.end());
        std::vector<int32_t> asub(row_indices.begin(), row_indices.end());
        check_mosek(MSK_putacolslice(_task, lo_idx, hi_idx, ptr.data(), ptr.data() + 1, asub.data(),
                                     values.data()),
                    "putacolslice");
        return first;
    }

    uint32_t add_rows(const std::vector<double>& lb, const std::vector<double>& ub,
                      const std::vector<uint32_t>& starts, const std::vector<uint32_t>& indices,
                      const std::vector<double>& values) override {
        assert(starts.size() == lb.size() + 1 && starts.back() == values.size() &&
               "add_rows requires starts.size() == m+1 with starts.back() == values.size()");
        uint32_t first = num_rows();
        auto m = static_cast<int32_t>(lb.size());
        if (m == 0) {
            return first;
        }
        check_mosek(MSK_appendcons(_task, m), "appendcons");
        auto lo_idx = static_cast<int32_t>(first);
        int32_t hi_idx = lo_idx + m;
        std::vector<MSKboundkeye> bk;
        std::vector<double> lo;
        std::vector<double> hi;
        build_bounds(lb, ub, m, bk, lo, hi);
        check_mosek(MSK_putconboundslice(_task, lo_idx, hi_idx, bk.data(), lo.data(), hi.data()),
                    "putconboundslice");

        // ptr[i]/ptr[i+1]: begin/end offsets of row i into asub/aval (see add_cols).
        std::vector<int32_t> ptr(starts.begin(), starts.end());
        std::vector<int32_t> asub(indices.begin(), indices.end());
        check_mosek(MSK_putarowslice(_task, lo_idx, hi_idx, ptr.data(), ptr.data() + 1, asub.data(),
                                     values.data()),
                    "putarowslice");
        return first;
    }

    void delete_cols(std::vector<int32_t>& mask) override {
        auto del_list = detail::collect_delete_indices(mask);
        if (!del_list.empty()) {
            check_mosek(
                MSK_removevars(_task, static_cast<int32_t>(del_list.size()), del_list.data()),
                "removevars");
        }
        detail::remap_delete_mask(mask);
    }

    void delete_rows(std::vector<int32_t>& mask) override {
        auto del_list = detail::collect_delete_indices(mask);
        if (!del_list.empty()) {
            check_mosek(
                MSK_removecons(_task, static_cast<int32_t>(del_list.size()), del_list.data()),
                "removecons");
        }
        detail::remap_delete_mask(mask);
    }

    void set_col_cost(uint32_t col, double cost) override {
        check_mosek(MSK_putcj(_task, static_cast<int32_t>(col), cost), "putcj(set_col_cost)");
    }

    [[nodiscard]] bool certify_runs_crossover() const override { return true; }

    LPStatus solve(bool certify) override {
        // Steady state runs no basis identification (MSK_BI_NEVER, pinned). The
        // CG loop requests certify=true only on a stall: turn basis
        // identification on (MOSEK's crossover) so the interior point is rounded
        // to a vertex and basic slacks collapse to 0 — same recovery as HiGHS.
        check_mosek(
            MSK_putintparam(_task, MSK_IPAR_INTPNT_BASIS, certify ? MSK_BI_ALWAYS : MSK_BI_NEVER),
            "IntpntBasis");
        // A certify solve rounds the interior point to a vertex in MSK_SOL_BAS;
        // read that slot so the getters return the crossed-over (slack-cleared)
        // solution rather than the unchanged interior point in MSK_SOL_ITR.
        _sol = certify ? MSK_SOL_BAS : MSK_SOL_ITR;
        MSKrescodee trmcode = MSK_RES_OK;
        MSKrescodee res = MSK_optimizetrm(_task, &trmcode);
        if (res != MSK_RES_OK) {
            // A solve that fails outright (or terminates abnormally) must surface
            // as Error — never let a stale interior-point point pass as solved
            // (cf. issue #33: a swallowed barrier failure returns garbage).
            return LPStatus::Error;
        }

        // The solution status is the authoritative gate, not trmcode: a
        // non-converged termination (stall / max-iterations / numerical, all of
        // which still return res == MSK_RES_OK with trmcode set) leaves the
        // active slot's solsta non-OPTIMAL, so the default branch below maps it
        // to Error. A trmcode warning that nonetheless reached optimal tolerance
        // yields solsta == OPTIMAL and is a usable solution (cf. COPT's
        // IMPRECISE -> Optimal); gating on trmcode would spuriously fail those.
        // Either way an unconverged barrier can never leak out as Optimal (#33).
        MSKsolstae solsta = MSK_SOL_STA_UNKNOWN;
        if (MSK_getsolsta(_task, _sol, &solsta) != MSK_RES_OK) {
            return LPStatus::Error;
        }

        switch (solsta) {
            case MSK_SOL_STA_OPTIMAL:
                return LPStatus::Optimal;
            case MSK_SOL_STA_PRIM_INFEAS_CER:
                return LPStatus::Infeasible;
            case MSK_SOL_STA_DUAL_INFEAS_CER:
                return LPStatus::Unbounded;
            default:
                return LPStatus::Error;
        }
    }

    [[nodiscard]] double get_obj() const override {
        double val = 0.0;
        check_mosek(MSK_getprimalobj(_task, _sol, &val), "getprimalobj");
        return val;
    }

    [[nodiscard]] std::vector<double> get_primals() const override {
        std::vector<double> vals(num_cols());
        if (!vals.empty()) {
            check_mosek(MSK_getxx(_task, _sol, vals.data()), "getxx");
        }
        return vals;
    }

    [[nodiscard]] std::vector<double> get_duals() const override {
        std::vector<double> vals(num_rows());
        if (!vals.empty()) {
            check_mosek(MSK_gety(_task, _sol, vals.data()), "gety");
        }
        return vals;
    }

    [[nodiscard]] std::vector<double> get_reduced_costs() const override {
        uint32_t n = num_cols();
        std::vector<double> vals(n);
        if (n > 0) {
            // MSK_getreducedcosts returns slx - sux = c_j - (A'y)_j, the same
            // reduced-cost convention COPT/HiGHS expose.
            check_mosek(MSK_getreducedcosts(_task, _sol, 0, static_cast<int32_t>(n), vals.data()),
                        "getreducedcosts");
        }
        return vals;
    }

    [[nodiscard]] uint32_t num_cols() const override {
        int32_t n = 0;
        check_mosek(MSK_getnumvar(_task, &n), "getnumvar");
        return static_cast<uint32_t>(n);
    }

    [[nodiscard]] uint32_t num_rows() const override {
        int32_t m = 0;
        check_mosek(MSK_getnumcon(_task, &m), "getnumcon");
        return static_cast<uint32_t>(m);
    }

    // MOSEK's interior-point method handles a much wider slack-vs-real-column
    // dynamic range than HiGHS/cuOpt, so allow a higher slack-cost ceiling.
    // This lets instances whose per-row column cost exceeds 1e7 (e.g.
    // planar2500 tree, ~1.7e7/source) price their slacks out and reach a
    // slack-free upper bound, which the 1e7 default would never permit.
    [[nodiscard]] double max_slack_cost() const override { return 1e9; }
};

std::unique_ptr<LPSolver> create_mosek_solver(bool verbose) {
    return std::make_unique<MosekSolver>(verbose);
}

}  // namespace mcfcg

#endif  // MCFCG_USE_MOSEK
