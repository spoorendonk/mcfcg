#ifdef MCFCG_USE_MOSEK

#include "mcfcg/lp/lp_solver.h"
#include "mcfcg/util/tolerances.h"

#include <cassert>
#include <mosek.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace mcfcg {

namespace {

void check_mosek(MSKrescodee res, const char* msg) {
    if (res != MSK_RES_OK) {
        throw std::runtime_error(std::string("MOSEK error (") + std::to_string(res) + ") in " +
                                 msg);
    }
}

// MOSEK's infinity is a one-sided bound key, not a finite number: the master
// and tests pass 1e20 as an "infinity" stand-in. Treat any bound with
// magnitude >= 1e19 as infinite when picking the MSKboundkeye, mirroring the
// cuOpt backend's CUOPT_BOUND_INF_THRESHOLD. Passing 1e20 to MOSEK as a finite
// range bound would corrupt the interior-point starting point (the bound value
// enters the barrier), so only the bound *key* is allowed to encode infinity;
// the numeric value handed to MOSEK for an infinite side is ignored by the key.
constexpr double MOSEK_BOUND_INF_THRESHOLD = 1e19;

// Derive a MOSEK bound key + clamped (lb, ub) pair from a generic (lb, ub).
struct BoundKey {
    MSKboundkeye bk;
    double lo;
    double hi;
};

BoundKey to_mosek_bound(double lb, double ub) {
    bool lb_inf = lb <= -MOSEK_BOUND_INF_THRESHOLD;
    bool ub_inf = ub >= MOSEK_BOUND_INF_THRESHOLD;
    if (lb_inf && ub_inf) {
        return {MSK_BK_FR, 0.0, 0.0};
    }
    if (lb_inf) {
        return {MSK_BK_UP, 0.0, ub};
    }
    if (ub_inf) {
        return {MSK_BK_LO, lb, 0.0};
    }
    if (lb == ub) {
        return {MSK_BK_FX, lb, ub};
    }
    return {MSK_BK_RA, lb, ub};
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

        // Keep barrier tolerances one order tighter than LP_FEAS_TOL so the
        // interior-point duals are precise enough for the pricer's NEG_RC_TOL
        // (same rationale as the COPT backend's FEASTOL/DUALTOL).
        check_mosek(MSK_putdouparam(_task, MSK_DPAR_INTPNT_TOL_PFEAS, LP_FEAS_TOL / 10),
                    "TolPfeas");
        check_mosek(MSK_putdouparam(_task, MSK_DPAR_INTPNT_TOL_DFEAS, LP_FEAS_TOL / 10),
                    "TolDfeas");
        check_mosek(MSK_putdouparam(_task, MSK_DPAR_INTPNT_TOL_REL_GAP, LP_FEAS_TOL / 10),
                    "TolRelGap");
    }

    ~MosekSolver() override {
        if (_task) {
            MSK_deletetask(&_task);
        }
        if (_env) {
            MSK_deleteenv(&_env);
        }
    }

    MosekSolver(const MosekSolver&) = delete;
    MosekSolver& operator=(const MosekSolver&) = delete;
    MosekSolver(MosekSolver&&) = delete;
    MosekSolver& operator=(MosekSolver&&) = delete;

    uint32_t add_cols(const std::vector<double>& obj, const std::vector<double>& lb,
                      const std::vector<double>& ub) override {
        uint32_t first = num_cols();
        auto n = static_cast<int32_t>(obj.size());
        check_mosek(MSK_appendvars(_task, n), "appendvars");
        for (int32_t i = 0; i < n; ++i) {
            int32_t j = static_cast<int32_t>(first) + i;
            check_mosek(MSK_putcj(_task, j, obj[i]), "putcj");
            BoundKey b = to_mosek_bound(lb[i], ub[i]);
            check_mosek(MSK_putvarbound(_task, j, b.bk, b.lo, b.hi), "putvarbound");
        }
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
        check_mosek(MSK_appendvars(_task, n), "appendvars");

        std::vector<int32_t> sub;
        for (int32_t i = 0; i < n; ++i) {
            int32_t j = static_cast<int32_t>(first) + i;
            check_mosek(MSK_putcj(_task, j, obj[i]), "putcj");
            BoundKey b = to_mosek_bound(lb[i], ub[i]);
            check_mosek(MSK_putvarbound(_task, j, b.bk, b.lo, b.hi), "putvarbound");

            uint32_t beg = starts[i];
            uint32_t end = starts[i + 1];
            auto nz = static_cast<int32_t>(end - beg);
            sub.resize(nz);
            for (int32_t k = 0; k < nz; ++k) {
                sub[k] = static_cast<int32_t>(row_indices[beg + k]);
            }
            check_mosek(MSK_putacol(_task, j, nz, sub.data(), values.data() + beg), "putacol");
        }
        return first;
    }

    uint32_t add_rows(const std::vector<double>& lb, const std::vector<double>& ub,
                      const std::vector<uint32_t>& starts, const std::vector<uint32_t>& indices,
                      const std::vector<double>& values) override {
        assert(starts.size() == lb.size() + 1 && starts.back() == values.size() &&
               "add_rows requires starts.size() == m+1 with starts.back() == values.size()");
        uint32_t first = num_rows();
        auto m = static_cast<int32_t>(lb.size());
        check_mosek(MSK_appendcons(_task, m), "appendcons");

        std::vector<int32_t> sub;
        for (int32_t i = 0; i < m; ++i) {
            int32_t row = static_cast<int32_t>(first) + i;
            BoundKey b = to_mosek_bound(lb[i], ub[i]);
            check_mosek(MSK_putconbound(_task, row, b.bk, b.lo, b.hi), "putconbound");

            uint32_t beg = starts[i];
            uint32_t end = starts[i + 1];
            auto nz = static_cast<int32_t>(end - beg);
            sub.resize(nz);
            for (int32_t k = 0; k < nz; ++k) {
                sub[k] = static_cast<int32_t>(indices[beg + k]);
            }
            check_mosek(MSK_putarow(_task, row, nz, sub.data(), values.data() + beg), "putarow");
        }
        return first;
    }

    void delete_cols(std::vector<int32_t>& mask) override {
        std::vector<int32_t> del_list;
        for (size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] == 1) {
                del_list.push_back(static_cast<int32_t>(i));
            }
        }

        if (!del_list.empty()) {
            check_mosek(
                MSK_removevars(_task, static_cast<int32_t>(del_list.size()), del_list.data()),
                "removevars");
        }

        uint32_t new_idx = 0;
        for (size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] == 1) {
                mask[i] = -1;
            } else {
                mask[i] = static_cast<int32_t>(new_idx++);
            }
        }
    }

    void delete_rows(std::vector<int32_t>& mask) override {
        std::vector<int32_t> del_list;
        for (size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] == 1) {
                del_list.push_back(static_cast<int32_t>(i));
            }
        }

        if (!del_list.empty()) {
            check_mosek(
                MSK_removecons(_task, static_cast<int32_t>(del_list.size()), del_list.data()),
                "removecons");
        }

        uint32_t new_idx = 0;
        for (size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] == 1) {
                mask[i] = -1;
            } else {
                mask[i] = static_cast<int32_t>(new_idx++);
            }
        }
    }

    void set_col_cost(uint32_t col, double cost) override {
        check_mosek(MSK_putcj(_task, static_cast<int32_t>(col), cost), "putcj(set_col_cost)");
    }

    LPStatus solve() override {
        MSKrescodee trmcode = MSK_RES_OK;
        MSKrescodee res = MSK_optimizetrm(_task, &trmcode);
        if (res != MSK_RES_OK) {
            // A solve that fails outright (or terminates abnormally) must surface
            // as Error — never let a stale interior-point point pass as solved
            // (cf. issue #33: a swallowed barrier failure returns garbage).
            return LPStatus::Error;
        }

        MSKsolstae solsta = MSK_SOL_STA_UNKNOWN;
        if (MSK_getsolsta(_task, MSK_SOL_ITR, &solsta) != MSK_RES_OK) {
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

    double get_obj() const override {
        double val = 0.0;
        check_mosek(MSK_getprimalobj(_task, MSK_SOL_ITR, &val), "getprimalobj");
        return val;
    }

    std::vector<double> get_primals() const override {
        std::vector<double> vals(num_cols());
        if (!vals.empty()) {
            check_mosek(MSK_getxx(_task, MSK_SOL_ITR, vals.data()), "getxx");
        }
        return vals;
    }

    std::vector<double> get_duals() const override {
        std::vector<double> vals(num_rows());
        if (!vals.empty()) {
            check_mosek(MSK_gety(_task, MSK_SOL_ITR, vals.data()), "gety");
        }
        return vals;
    }

    std::vector<double> get_reduced_costs() const override {
        uint32_t n = num_cols();
        std::vector<double> vals(n);
        if (n > 0) {
            // MSK_getreducedcosts returns slx - sux = c_j - (A'y)_j, the same
            // reduced-cost convention COPT/HiGHS expose.
            check_mosek(
                MSK_getreducedcosts(_task, MSK_SOL_ITR, 0, static_cast<int32_t>(n), vals.data()),
                "getreducedcosts");
        }
        return vals;
    }

    uint32_t num_cols() const override {
        int32_t n = 0;
        check_mosek(MSK_getnumvar(_task, &n), "getnumvar");
        return static_cast<uint32_t>(n);
    }

    uint32_t num_rows() const override {
        int32_t m = 0;
        check_mosek(MSK_getnumcon(_task, &m), "getnumcon");
        return static_cast<uint32_t>(m);
    }
};

std::unique_ptr<LPSolver> create_mosek_solver(bool verbose) {
    return std::make_unique<MosekSolver>(verbose);
}

}  // namespace mcfcg

#endif  // MCFCG_USE_MOSEK
