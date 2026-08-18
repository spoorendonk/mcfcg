#ifdef MCFCG_USE_COPT

#include <copt.h>

#include <cassert>
#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "mcfcg/lp/backend_util.h"
#include "mcfcg/lp/lp_solver.h"
#include "mcfcg/util/tolerances.h"

namespace mcfcg {

namespace {

void check_copt(int status, const char* msg) {
    if (status != COPT_RETCODE_OK) {
        throw std::runtime_error(std::string("COPT error (") + std::to_string(status) + ") in " +
                                 msg);
    }
}

// COPT exposes no numeric version getter, only a multi-line human banner whose
// first line reads "Cardinal Optimizer v8.0.1. Build date Oct 22 2025". Pull the
// dotted version out of it rather than reading COPT_VERSION_* from the header:
// the header's macros describe what we compiled against, and the whole point of
// the banner field is to name the library actually loaded. Falls back to
// "unknown" rather than guessing if the format ever changes.
std::string copt_version() {
    // COPT_BUFFSIZE is the buffer size its message getters document; anything
    // smaller is below the API's contract. The +1 keeps a NUL available even if
    // COPT ever filled the buffer exactly, since string_view below assumes one.
    char banner[COPT_BUFFSIZE + 1] = {0};
    if (COPT_GetBanner(banner, COPT_BUFFSIZE) != COPT_RETCODE_OK) {
        return "unknown";
    }
    // isdigit takes an int that must be representable as unsigned char; a plain
    // char is signed here, so a non-ASCII byte in the banner would be UB.
    auto is_digit = [](char chr) { return std::isdigit(static_cast<unsigned char>(chr)) != 0; };
    std::string_view text(banner);
    for (size_t i = 0; i + 1 < text.size(); ++i) {
        bool at_token_start = (i == 0) || text[i - 1] == ' ';
        if (!at_token_start || text[i] != 'v' || !is_digit(text[i + 1])) {
            continue;
        }
        size_t end = i + 1;
        while (end < text.size() && (is_digit(text[end]) || text[end] == '.')) {
            ++end;
        }
        // Drop a trailing '.' — it ends the banner's sentence, not the version.
        while (end > i + 1 && text[end - 1] == '.') {
            --end;
        }
        return std::string(text.substr(i + 1, end - (i + 1)));
    }
    return "unknown";
}

}  // namespace

class CoptSolver : public LPSolver {
private:
    copt_env* _env = nullptr;
    copt_prob* _prob = nullptr;

public:
    explicit CoptSolver(bool verbose = false, int gpu_mode = -1) {
        check_copt(COPT_CreateEnv(&_env), "CreateEnv");
        check_copt(COPT_CreateProb(_env, &_prob), "CreateProb");

        check_copt(COPT_SetIntParam(_prob, COPT_INTPARAM_LPMETHOD, 2), "LpMethod=barrier");
        // GPUMode 2 requests the GPU barrier; COPT falls back to CPU when no GPU
        // is present, so this is safe on a GPU-less host (it does not crash).
        // gpu_mode comes from --copt-gpu-mode (0=CPU, 1=GPU mode 1, 2=GPU mode
        // 2); the -1 sentinel (flag absent) and any out-of-range value default
        // to the GPU barrier (2).
        if (gpu_mode < 0 || gpu_mode > 2) {
            gpu_mode = 2;
        }
        check_copt(COPT_SetIntParam(_prob, COPT_INTPARAM_GPUMODE, gpu_mode),
                   ("GPUMode=" + std::to_string(gpu_mode)).c_str());
        check_copt(COPT_SetIntParam(_prob, COPT_INTPARAM_PRESOLVE, 0), "Presolve=off");
        // Crossover off by default; solve(certify=true) flips it on for that one
        // solve (the CG loop's stall recovery), like HiGHS.
        check_copt(COPT_SetIntParam(_prob, COPT_INTPARAM_CROSSOVER, 0), "Crossover=off");
        check_copt(COPT_SetIntParam(_prob, COPT_INTPARAM_LOGGING, verbose ? 1 : 0), "Logging");
        // Barrier feasibility/optimality tolerance, pinned to BARRIER_TOL
        // identically across backends. The small obj gap on tiny tree
        // instances (e.g. planar30) is inherent to barrier interior-point
        // convergence without crossover, not feastol.
        check_copt(COPT_SetDblParam(_prob, COPT_DBLPARAM_FEASTOL, BARRIER_TOL), "FeasTol");
        check_copt(COPT_SetDblParam(_prob, COPT_DBLPARAM_DUALTOL, BARRIER_TOL), "DualTol");
        int threads = 0;
        check_copt(COPT_GetIntParam(_prob, COPT_INTPARAM_THREADS, &threads), "GetThreads");
        log_solver_config("copt", copt_version().c_str(), "barrier", /*gpu=*/gpu_mode != 0,
                          threads);
    }

    ~CoptSolver() override {
        if (_prob) {
            COPT_DeleteProb(&_prob);
        }
        if (_env) {
            COPT_DeleteEnv(&_env);
        }
    }

    CoptSolver(const CoptSolver&) = delete;
    CoptSolver& operator=(const CoptSolver&) = delete;
    CoptSolver(CoptSolver&&) = delete;
    CoptSolver& operator=(CoptSolver&&) = delete;

    uint32_t add_cols(const std::vector<double>& obj, const std::vector<double>& lb,
                      const std::vector<double>& ub) override {
        uint32_t first = num_cols();
        int n = static_cast<int>(obj.size());
        std::vector<char> types(n, COPT_CONTINUOUS);
        check_copt(COPT_AddCols(_prob, n, obj.data(), nullptr, nullptr, nullptr, nullptr,
                                types.data(), lb.data(), ub.data(), nullptr),
                   "AddCols");
        return first;
    }

    uint32_t add_cols(const std::vector<double>& obj, const std::vector<double>& lb,
                      const std::vector<double>& ub, const std::vector<uint32_t>& starts,
                      const std::vector<uint32_t>& row_indices,
                      const std::vector<double>& values) override {
        assert(starts.size() == obj.size() + 1 && starts.back() == values.size() &&
               "add_cols requires starts.size() == n+1 with starts.back() == values.size()");
        uint32_t first = num_cols();
        int n = static_cast<int>(obj.size());

        // Convert sentinel-based starts to colMatBeg + colMatCnt
        std::vector<int> col_beg(n);
        std::vector<int> col_cnt(n);
        for (int i = 0; i < n; ++i) {
            col_beg[i] = static_cast<int>(starts[i]);
            col_cnt[i] = static_cast<int>(starts[i + 1] - starts[i]);
        }

        std::vector<int> indices(row_indices.size());
        for (size_t i = 0; i < row_indices.size(); ++i) {
            indices[i] = static_cast<int>(row_indices[i]);
        }

        std::vector<char> types(n, COPT_CONTINUOUS);
        check_copt(
            COPT_AddCols(_prob, n, obj.data(), col_beg.data(), col_cnt.data(), indices.data(),
                         values.data(), types.data(), lb.data(), ub.data(), nullptr),
            "AddCols(CSC)");
        return first;
    }

    uint32_t add_rows(const std::vector<double>& lb, const std::vector<double>& ub,
                      const std::vector<uint32_t>& starts, const std::vector<uint32_t>& indices,
                      const std::vector<double>& values) override {
        assert(starts.size() == lb.size() + 1 && starts.back() == values.size() &&
               "add_rows requires starts.size() == m+1 with starts.back() == values.size()");
        uint32_t first = num_rows();
        int m = static_cast<int>(lb.size());

        // Convert starts (size m+1 with sentinel) to rowMatBeg + rowMatCnt
        std::vector<int> row_beg(m);
        std::vector<int> row_cnt(m);
        for (int i = 0; i < m; ++i) {
            row_beg[i] = static_cast<int>(starts[i]);
            row_cnt[i] = static_cast<int>(starts[i + 1] - starts[i]);
        }

        std::vector<int> col_indices(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            col_indices[i] = static_cast<int>(indices[i]);
        }

        // NULL sense: COPT treats rowBound/rowUpper as lower/upper bounds directly
        check_copt(COPT_AddRows(_prob, m, row_beg.data(), row_cnt.data(), col_indices.data(),
                                values.data(), nullptr, lb.data(), ub.data(), nullptr),
                   "AddRows");
        return first;
    }

    void delete_cols(std::vector<int32_t>& mask) override {
        auto del_list = detail::collect_delete_indices(mask);
        if (!del_list.empty()) {
            check_copt(COPT_DelCols(_prob, static_cast<int>(del_list.size()), del_list.data()),
                       "DelCols");
        }
        detail::remap_delete_mask(mask);
    }

    void delete_rows(std::vector<int32_t>& mask) override {
        auto del_list = detail::collect_delete_indices(mask);
        if (!del_list.empty()) {
            check_copt(COPT_DelRows(_prob, static_cast<int>(del_list.size()), del_list.data()),
                       "DelRows");
        }
        detail::remap_delete_mask(mask);
    }

    void set_col_cost(uint32_t col, double cost) override {
        int idx = static_cast<int>(col);
        check_copt(COPT_SetColObj(_prob, 1, &idx, &cost), "SetColObj");
    }

    bool certify_runs_crossover() const override { return true; }

    LPStatus solve(bool certify) override {
        // Steady state runs crossover off (pinned, fast). The CG loop requests
        // certify=true only on a stall, where crossover rounds the barrier point
        // to a vertex so basic slacks collapse to 0 and a slack-free UB can be
        // recorded — same recovery as HiGHS.
        check_copt(COPT_SetIntParam(_prob, COPT_INTPARAM_CROSSOVER, certify ? 1 : 0), "Crossover");
        int status = COPT_SolveLp(_prob);
        if (status != COPT_RETCODE_OK) {
            return LPStatus::Error;
        }

        int lp_status = 0;
        check_copt(COPT_GetIntAttr(_prob, COPT_INTATTR_LPSTATUS, &lp_status), "GetLpStatus");

        switch (lp_status) {
            case COPT_LPSTATUS_OPTIMAL:
            // IMPRECISE = solved within relaxed tolerances; acceptable for CG duals
            case COPT_LPSTATUS_IMPRECISE:
                return LPStatus::Optimal;
            case COPT_LPSTATUS_INFEASIBLE:
                return LPStatus::Infeasible;
            case COPT_LPSTATUS_UNBOUNDED:
                return LPStatus::Unbounded;
            default:
                return LPStatus::Error;
        }
    }

    double get_obj() const override {
        double val = 0.0;
        check_copt(COPT_GetDblAttr(_prob, COPT_DBLATTR_LPOBJVAL, &val), "GetLpObjval");
        return val;
    }

    std::vector<double> get_primals() const override {
        std::vector<double> vals(num_cols());
        check_copt(COPT_GetLpSolution(_prob, vals.data(), nullptr, nullptr, nullptr),
                   "GetLpSolution(primals)");
        return vals;
    }

    std::vector<double> get_duals() const override {
        std::vector<double> vals(num_rows());
        check_copt(COPT_GetLpSolution(_prob, nullptr, nullptr, vals.data(), nullptr),
                   "GetLpSolution(duals)");
        return vals;
    }

    std::vector<double> get_reduced_costs() const override {
        std::vector<double> vals(num_cols());
        check_copt(COPT_GetLpSolution(_prob, nullptr, nullptr, nullptr, vals.data()),
                   "GetLpSolution(redCost)");
        return vals;
    }

    uint32_t num_cols() const override {
        int n = 0;
        check_copt(COPT_GetIntAttr(_prob, COPT_INTATTR_COLS, &n), "GetIntAttr(Cols)");
        return static_cast<uint32_t>(n);
    }

    uint32_t num_rows() const override {
        int m = 0;
        check_copt(COPT_GetIntAttr(_prob, COPT_INTATTR_ROWS, &m), "GetIntAttr(Rows)");
        return static_cast<uint32_t>(m);
    }

    // COPT's barrier tolerates a wide slack-vs-real dynamic range; allow a
    // higher slack-cost ceiling than the 1e7 default so expensive-column
    // instances (per-row cost > 1e7) can price their slacks out.  See
    // LPSolver::max_slack_cost and the MOSEK override.
    double max_slack_cost() const override { return 1e9; }
};

std::unique_ptr<LPSolver> create_copt_solver(bool verbose, int gpu_mode) {
    return std::make_unique<CoptSolver>(verbose, gpu_mode);
}

}  // namespace mcfcg

#endif  // MCFCG_USE_COPT
