#ifdef MCFCG_USE_COPT

#include "mcfcg/lp/lp_solver.h"

#include <copt.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace mcfcg {

namespace {

void check_copt(int status, const char* msg) {
    if (status != COPT_RETCODE_OK) {
        throw std::runtime_error(std::string("COPT error (") + std::to_string(status) + ") in " +
                                 msg);
    }
}

}  // namespace

class CoptSolver : public LPSolver {
private:
    copt_env* _env = nullptr;
    copt_prob* _prob = nullptr;

public:
    explicit CoptSolver(bool verbose = false) {
        check_copt(COPT_CreateEnv(&_env), "CreateEnv");
        check_copt(COPT_CreateProb(_env, &_prob), "CreateProb");

        check_copt(COPT_SetIntParam(_prob, COPT_INTPARAM_LPMETHOD, 2), "LpMethod=barrier");
        check_copt(COPT_SetIntParam(_prob, COPT_INTPARAM_GPUMODE, 2), "GPUMode=2");
        check_copt(COPT_SetIntParam(_prob, COPT_INTPARAM_PRESOLVE, 0), "Presolve=off");
        check_copt(COPT_SetIntParam(_prob, COPT_INTPARAM_CROSSOVER, 0), "Crossover=off");
        check_copt(COPT_SetIntParam(_prob, COPT_INTPARAM_LOGGING, verbose ? 1 : 0), "Logging");
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
        uint32_t first = num_rows();
        int m = static_cast<int>(lb.size());

        // Convert starts to rowMatBeg + rowMatCnt
        std::vector<int> row_beg(m);
        std::vector<int> row_cnt(m);
        for (int i = 0; i < m; ++i) {
            row_beg[i] = static_cast<int>(starts[i]);
            uint32_t end = (static_cast<size_t>(i + 1) < starts.size())
                               ? starts[i + 1]
                               : static_cast<uint32_t>(values.size());
            row_cnt[i] = static_cast<int>(end - starts[i]);
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
        // Build list of indices to delete
        std::vector<int> del_list;
        for (size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] == 1) {
                del_list.push_back(static_cast<int>(i));
            }
        }

        if (!del_list.empty()) {
            check_copt(COPT_DelCols(_prob, static_cast<int>(del_list.size()), del_list.data()),
                       "DelCols");
        }

        // Recompute mask: new index for surviving columns
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
        std::vector<int> del_list;
        for (size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] == 1) {
                del_list.push_back(static_cast<int>(i));
            }
        }

        if (!del_list.empty()) {
            check_copt(COPT_DelRows(_prob, static_cast<int>(del_list.size()), del_list.data()),
                       "DelRows");
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

    LPStatus solve() override {
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
};

std::unique_ptr<LPSolver> create_copt_solver(bool verbose) {
    return std::make_unique<CoptSolver>(verbose);
}

}  // namespace mcfcg

#endif  // MCFCG_USE_COPT
