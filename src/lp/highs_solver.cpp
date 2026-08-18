#include <Highs.h>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "mcfcg/lp/lp_solver.h"
#include "mcfcg/util/tolerances.h"

namespace mcfcg {

// Shared across every backend (declared in lp_solver.h, defined once here in the
// always-compiled HiGHS TU). One concise provenance line per solver
// construction — i.e. once per CG solve, since the loop builds one solver.
void log_solver_config(const char* backend, const char* version, const char* method, bool gpu,
                       int threads) {
    char threads_buf[32];
    if (gpu) {
        std::snprintf(threads_buf, sizeof(threads_buf), "n/a(GPU)");
    } else if (threads > 0) {
        std::snprintf(threads_buf, sizeof(threads_buf), "%d", threads);
    } else {
        std::snprintf(threads_buf, sizeof(threads_buf), "auto(%u)",
                      std::thread::hardware_concurrency());
    }
    std::fprintf(stderr,
                 "[lp-config] backend=%s version=%s method=%s exec=%s presolve=off crossover=off "
                 "tol=%g threads=%s\n",
                 backend, version, method, gpu ? "GPU" : "CPU", BARRIER_TOL, threads_buf);
}

class HiGHSSolver : public LPSolver {
private:
    Highs _highs;
    uint32_t _num_cols = 0;
    uint32_t _num_rows = 0;

public:
    explicit HiGHSSolver(bool verbose = false) {
        _highs.setOptionValue("output_flag", verbose);
        // Pinned identically across all backends (presolve off, crossover off,
        // tol = BARRIER_TOL) so cross-solver timings compare like for like.
        // Crossover off yields a pure interior-point solution (no basis); the
        // column-aging path falls back to the COL_ACTIVE_EPS primal threshold.
        _highs.setOptionValue("presolve", "off");
        // Crossover off by default (pure interior-point, fast per CG iter); the
        // CG loop re-requests it via solve(certify=true) only when it stalls
        // with basic slacks. HiPO's interior solution is then non-vertex and
        // leaves the demand-row slacks at O(tol) > 0, which the path-CG
        // feasibility gate can't certify; crossover rounds it to a vertex
        // (slacks exactly 0) so a slack-free UB can be recorded. See README.
        _highs.setOptionValue("run_crossover", "off");
        _highs.setOptionValue("primal_feasibility_tolerance", BARRIER_TOL);
        _highs.setOptionValue("dual_feasibility_tolerance", BARRIER_TOL);
        // LP method: default to the HiPO interior-point solver. On these
        // column-generation masters HiPO is ~2x faster than HiGHS' simplex
        // default (grid15 tree: HiPO 70s vs simplex 148s) and matches the
        // MOSEK/COPT barrier objectives. Override via
        // MCFCG_HIGHS_SOLVER = simplex | ipm | hipo | pdlp.
        const char* highs_method = std::getenv("MCFCG_HIGHS_SOLVER");
        const char* method = highs_method != nullptr ? highs_method : "hipo";
        _highs.setOptionValue("solver", method);
        HighsModel model;
        model.lp_.sense_ = ObjSense::kMinimize;
        model.lp_.offset_ = 0.0;
        _highs.passModel(std::move(model));
        HighsInt threads = 0;
        _highs.getOptionValue("threads", threads);
        // Githash identifies the upstream tag only — it cannot reveal whether
        // cmake/patches/highs-hipo-refine-status.patch is applied, so a run's
        // full HiGHS provenance is this banner plus PROVENANCE.txt.
        char version[64];
        std::snprintf(version, sizeof(version), "%s-%s", highsVersion(), highsGithash());
        log_solver_config("highs", version, method, /*gpu=*/false, static_cast<int>(threads));
    }

    uint32_t add_cols(const std::vector<double>& obj, const std::vector<double>& lb,
                      const std::vector<double>& ub) override {
        uint32_t first = _num_cols;
        auto n = static_cast<HighsInt>(obj.size());
        _highs.addCols(n, obj.data(), lb.data(), ub.data(), 0, nullptr, nullptr, nullptr);
        _num_cols += static_cast<uint32_t>(n);
        return first;
    }

    uint32_t add_cols(const std::vector<double>& obj, const std::vector<double>& lb,
                      const std::vector<double>& ub, const std::vector<uint32_t>& starts,
                      const std::vector<uint32_t>& row_indices,
                      const std::vector<double>& values) override {
        assert(starts.size() == obj.size() + 1 && starts.back() == values.size() &&
               "add_cols requires starts.size() == n+1 with starts.back() == values.size()");
        uint32_t first = _num_cols;
        auto n = static_cast<HighsInt>(obj.size());
        auto nnz = static_cast<HighsInt>(values.size());

        // Convert uint32_t starts/indices to HighsInt
        std::vector<HighsInt> h_starts(starts.size());
        for (size_t i = 0; i < starts.size(); ++i) {
            h_starts[i] = static_cast<HighsInt>(starts[i]);
        }
        std::vector<HighsInt> h_indices(row_indices.size());
        for (size_t i = 0; i < row_indices.size(); ++i) {
            h_indices[i] = static_cast<HighsInt>(row_indices[i]);
        }

        _highs.addCols(n, obj.data(), lb.data(), ub.data(), nnz, h_starts.data(), h_indices.data(),
                       values.data());
        _num_cols += static_cast<uint32_t>(n);
        return first;
    }

    uint32_t add_rows(const std::vector<double>& lb, const std::vector<double>& ub,
                      const std::vector<uint32_t>& starts, const std::vector<uint32_t>& indices,
                      const std::vector<double>& values) override {
        assert(starts.size() == lb.size() + 1 && starts.back() == values.size() &&
               "add_rows requires starts.size() == m+1 with starts.back() == values.size()");
        uint32_t first = _num_rows;
        auto m = static_cast<HighsInt>(lb.size());
        auto nnz = static_cast<HighsInt>(values.size());

        std::vector<HighsInt> h_starts(starts.size());
        for (size_t i = 0; i < starts.size(); ++i) {
            h_starts[i] = static_cast<HighsInt>(starts[i]);
        }

        std::vector<HighsInt> h_indices(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            h_indices[i] = static_cast<HighsInt>(indices[i]);
        }

        _highs.addRows(m, lb.data(), ub.data(), nnz, h_starts.data(), h_indices.data(),
                       values.data());
        _num_rows += static_cast<uint32_t>(m);
        return first;
    }

    void delete_cols(std::vector<int32_t>& mask) override {
        std::vector<HighsInt> h_mask(mask.begin(), mask.end());
        _highs.deleteCols(h_mask.data());
        // HiGHS writes new indices into the mask: -1 for deleted, new index
        // otherwise
        uint32_t surviving = 0;
        for (size_t i = 0; i < h_mask.size(); ++i) {
            mask[i] = static_cast<int32_t>(h_mask[i]);
            if (mask[i] >= 0) {
                ++surviving;
            }
        }
        _num_cols = surviving;
    }

    void set_col_cost(uint32_t col, double cost) override {
        _highs.changeColCost(static_cast<HighsInt>(col), cost);
    }

    void delete_rows(std::vector<int32_t>& mask) override {
        std::vector<HighsInt> h_mask(mask.begin(), mask.end());
        _highs.deleteRows(h_mask.data());
        uint32_t surviving = 0;
        for (size_t i = 0; i < h_mask.size(); ++i) {
            mask[i] = static_cast<int32_t>(h_mask[i]);
            if (mask[i] >= 0) {
                ++surviving;
            }
        }
        _num_rows = surviving;
    }

    bool certify_runs_crossover() const override { return true; }

    LPStatus solve(bool certify) override {
        // Crossover is requested only on a certify solve (the CG loop's stall
        // recovery): round the interior point to a vertex so basic slacks
        // collapse to exactly 0 and a slack-free UB can be certified.
        _highs.setOptionValue("run_crossover", certify ? "on" : "off");
        auto status = _highs.run();
        if (status != HighsStatus::kOk) {
            return LPStatus::Error;
        }
        auto model_status = _highs.getModelStatus();
        switch (model_status) {
            case HighsModelStatus::kOptimal:
                return LPStatus::Optimal;
            case HighsModelStatus::kInfeasible:
                return LPStatus::Infeasible;
            case HighsModelStatus::kUnbounded:
                return LPStatus::Unbounded;
            default:
                return LPStatus::Error;
        }
    }

    double get_obj() const override {
        double val = 0.0;
        _highs.getInfoValue("objective_function_value", val);
        return val;
    }

    std::vector<double> get_primals() const override {
        const auto& sol = _highs.getSolution();
        return sol.col_value;
    }

    std::vector<double> get_duals() const override {
        const auto& sol = _highs.getSolution();
        return sol.row_dual;
    }

    std::vector<double> get_reduced_costs() const override {
        const auto& sol = _highs.getSolution();
        return sol.col_dual;
    }

    uint32_t num_cols() const override { return _num_cols; }
    uint32_t num_rows() const override { return _num_rows; }

    bool has_basis() const override { return _highs.getBasis().valid; }

    std::vector<bool> get_basic_cols() const override {
        const auto& basis = _highs.getBasis();
        std::vector<bool> result(basis.col_status.size());
        for (size_t i = 0; i < basis.col_status.size(); ++i) {
            result[i] = (basis.col_status[i] == HighsBasisStatus::kBasic);
        }
        return result;
    }
};

std::unique_ptr<LPSolver> create_lp_solver(bool verbose) {
    return std::make_unique<HiGHSSolver>(verbose);
}

}  // namespace mcfcg
