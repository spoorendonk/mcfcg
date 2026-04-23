#ifdef MCFCG_USE_CUOPT

#include "mcfcg/lp/lp_solver.h"
#include "mcfcg/util/tolerances.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/cuopt_c.h>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MCFCG_CUOPT_DELTA_API
#include <cuopt/linear_programming/cuopt_c_delta.h>
#endif

namespace mcfcg {

namespace {

void check_cuopt(cuopt_int_t status, const char* msg) {
    if (status != CUOPT_SUCCESS) {
        throw std::runtime_error(std::string("cuOpt error in ") + msg);
    }
}

// Extract primal / dual / reduced-cost vectors from a cuOptSolution.
// Returns LPStatus::Optimal on success, or a status reflecting termination.
LPStatus extract_solution(cuOptSolution solution, uint32_t n, uint32_t m, double& obj,
                          std::vector<double>& primals, std::vector<double>& duals,
                          std::vector<double>& reduced_costs) {
    cuopt_int_t term_status = 0;
    if (cuOptGetTerminationStatus(solution, &term_status) != CUOPT_SUCCESS) {
        return LPStatus::Error;
    }
    // Note: cuOpt API has "TERIMINATION" typo in constant names
    if (term_status == CUOPT_TERIMINATION_STATUS_INFEASIBLE)
        return LPStatus::Infeasible;
    if (term_status == CUOPT_TERIMINATION_STATUS_UNBOUNDED)
        return LPStatus::Unbounded;
    if (term_status != CUOPT_TERIMINATION_STATUS_OPTIMAL)
        return LPStatus::Error;

    cuopt_float_t obj_val = 0;
    if (cuOptGetObjectiveValue(solution, &obj_val) != CUOPT_SUCCESS)
        return LPStatus::Error;
    obj = static_cast<double>(obj_val);

    std::vector<cuopt_float_t> f_primals(n);
    if (cuOptGetPrimalSolution(solution, f_primals.data()) != CUOPT_SUCCESS)
        return LPStatus::Error;
    primals.assign(f_primals.begin(), f_primals.end());

    std::vector<cuopt_float_t> f_duals(m);
    if (cuOptGetDualSolution(solution, f_duals.data()) != CUOPT_SUCCESS)
        return LPStatus::Error;
    duals.assign(f_duals.begin(), f_duals.end());

    std::vector<cuopt_float_t> f_rc(n);
    if (cuOptGetReducedCosts(solution, f_rc.data()) != CUOPT_SUCCESS)
        return LPStatus::Error;
    reduced_costs.assign(f_rc.begin(), f_rc.end());

    return LPStatus::Optimal;
}

}  // namespace

// cuOpt's public C API has no incremental mutators — cuOptCreateRangedProblem
// takes a fully-formed problem and cuOptDestroyProblem tears it down, so every
// solve() rebuilds from scratch. The delta C API (see cuopt_c_delta.h on the
// spoorendonk/cuopt fork, tracked by spoorendonk/mcfcg #20 and sub-issues
// #22/#23/#24) introduces a persistent problem handle plus
// cuOptAddColumns / cuOptAddRows / cuOptDeleteColumns / cuOptDeleteRows /
// cuOptSetVariableObjectiveCoefficient / cuOptResolve.
//
// This file compiles two shapes, selected at build time:
//
//   * default (MCFCG_CUOPT_DELTA_API undefined): rebuild path — mutators buffer
//     into host vectors, solve() creates + destroys a cuOptOptimizationProblem
//     every call. No dependency on the fork.
//   * opt-in (MCFCG_CUOPT_DELTA_API defined): delta path — mutators forward to
//     the fork's delta API after the first solve, solve() uses cuOptResolve on
//     a persistent handle. Requires a cuOpt build that ships cuopt_c_delta.h.
class CuOptSolver : public LPSolver {
private:
    // Column data
    std::vector<double> _obj;
    std::vector<double> _col_lb;
    std::vector<double> _col_ub;

    // Row data
    std::vector<double> _row_lb;
    std::vector<double> _row_ub;

    // Constraint matrix stored in CSC (column-major) format.
    // We convert to CSR for cuOpt at solve time.
    struct CSCEntry {
        uint32_t row;
        double value;
    };
    std::vector<std::vector<CSCEntry>> _col_entries;

    // Cached solution
    double _cached_obj = 0.0;
    std::vector<double> _cached_primals;
    std::vector<double> _cached_duals;
    std::vector<double> _cached_reduced_costs;

    bool _verbose = false;

#ifdef MCFCG_CUOPT_DELTA_API
    // Persistent cuOpt handles, populated by the first solve and reused on
    // subsequent mutations + resolves. Null until first solve.
    cuOptOptimizationProblem _problem = nullptr;
    cuOptSolverSettings _settings = nullptr;
    cuOptSolution _solution = nullptr;
#endif

public:
    CuOptSolver() = default;
    explicit CuOptSolver(bool verbose) : _verbose(verbose) {}

    CuOptSolver(const CuOptSolver&) = delete;
    CuOptSolver& operator=(const CuOptSolver&) = delete;
    CuOptSolver(CuOptSolver&&) = delete;
    CuOptSolver& operator=(CuOptSolver&&) = delete;

    ~CuOptSolver() override {
#ifdef MCFCG_CUOPT_DELTA_API
        if (_solution)
            cuOptDestroySolution(&_solution);
        if (_settings)
            cuOptDestroySolverSettings(&_settings);
        if (_problem)
            cuOptDestroyProblem(&_problem);
#endif
    }

    uint32_t add_cols(const std::vector<double>& obj, const std::vector<double>& lb,
                      const std::vector<double>& ub) override {
        uint32_t first = num_cols();
        for (size_t i = 0; i < obj.size(); ++i) {
            _obj.push_back(obj[i]);
            _col_lb.push_back(lb[i]);
            _col_ub.push_back(ub[i]);
            _col_entries.emplace_back();
        }
#ifdef MCFCG_CUOPT_DELTA_API
        if (_problem) {
            // No coefficients — empty CSC (starts = {0, 0, ..., 0}).
            std::vector<cuopt_int_t> starts(obj.size() + 1, 0);
            std::vector<cuopt_float_t> f_obj(obj.begin(), obj.end());
            std::vector<cuopt_float_t> f_lb(lb.begin(), lb.end());
            std::vector<cuopt_float_t> f_ub(ub.begin(), ub.end());
            check_cuopt(
                cuOptAddColumns(_problem, static_cast<cuopt_int_t>(obj.size()), f_obj.data(),
                                f_lb.data(), f_ub.data(), starts.data(), nullptr, nullptr, nullptr),
                "cuOptAddColumns");
        }
#endif
        return first;
    }

    uint32_t add_cols(const std::vector<double>& obj, const std::vector<double>& lb,
                      const std::vector<double>& ub, const std::vector<uint32_t>& starts,
                      const std::vector<uint32_t>& row_indices,
                      const std::vector<double>& values) override {
        assert(starts.size() == obj.size() + 1 && starts.back() == values.size() &&
               "add_cols requires starts.size() == n+1 with starts.back() == values.size()");
        uint32_t first = num_cols();
        auto n = static_cast<uint32_t>(obj.size());
        for (uint32_t i = 0; i < n; ++i) {
            _obj.push_back(obj[i]);
            _col_lb.push_back(lb[i]);
            _col_ub.push_back(ub[i]);

            std::vector<CSCEntry> entries;
            uint32_t begin = starts[i];
            uint32_t end = starts[i + 1];  // caller includes sentinel
            for (uint32_t j = begin; j < end; ++j) {
                entries.push_back({row_indices[j], values[j]});
            }
            _col_entries.push_back(std::move(entries));
        }
#ifdef MCFCG_CUOPT_DELTA_API
        if (_problem) {
            std::vector<cuopt_int_t> f_starts(starts.begin(), starts.end());
            std::vector<cuopt_int_t> f_rows(row_indices.begin(), row_indices.end());
            std::vector<cuopt_float_t> f_vals(values.begin(), values.end());
            std::vector<cuopt_float_t> f_obj(obj.begin(), obj.end());
            std::vector<cuopt_float_t> f_lb(lb.begin(), lb.end());
            std::vector<cuopt_float_t> f_ub(ub.begin(), ub.end());
            check_cuopt(cuOptAddColumns(_problem, static_cast<cuopt_int_t>(n), f_obj.data(),
                                        f_lb.data(), f_ub.data(), f_starts.data(), f_rows.data(),
                                        f_vals.data(), nullptr),
                        "cuOptAddColumns");
        }
#endif
        return first;
    }

    uint32_t add_rows(const std::vector<double>& lb, const std::vector<double>& ub,
                      const std::vector<uint32_t>& starts, const std::vector<uint32_t>& indices,
                      const std::vector<double>& values) override {
        assert(starts.size() == lb.size() + 1 && starts.back() == values.size() &&
               "add_rows requires starts.size() == m+1 with starts.back() == values.size()");
        uint32_t first = num_rows();
        auto m = static_cast<uint32_t>(lb.size());

        for (uint32_t i = 0; i < m; ++i) {
            _row_lb.push_back(lb[i]);
            _row_ub.push_back(ub[i]);

            uint32_t begin = starts[i];
            uint32_t end = starts[i + 1];  // caller includes sentinel

            uint32_t row = first + i;
            for (uint32_t j = begin; j < end; ++j) {
                uint32_t col = indices[j];
                _col_entries[col].push_back({row, values[j]});
            }
        }
#ifdef MCFCG_CUOPT_DELTA_API
        if (_problem) {
            std::vector<cuopt_int_t> f_starts(starts.begin(), starts.end());
            std::vector<cuopt_int_t> f_cols(indices.begin(), indices.end());
            std::vector<cuopt_float_t> f_vals(values.begin(), values.end());
            std::vector<cuopt_float_t> f_lb(lb.begin(), lb.end());
            std::vector<cuopt_float_t> f_ub(ub.begin(), ub.end());
            check_cuopt(cuOptAddRows(_problem, static_cast<cuopt_int_t>(m), f_lb.data(),
                                     f_ub.data(), f_starts.data(), f_cols.data(), f_vals.data()),
                        "cuOptAddRows");
        }
#endif
        return first;
    }

    void delete_cols(std::vector<int32_t>& mask) override {
        uint32_t n = num_cols();
#ifdef MCFCG_CUOPT_DELTA_API
        std::vector<cuopt_int_t> delta_mask;
        if (_problem) {
            delta_mask.assign(mask.begin(), mask.end());
        }
#endif
        std::vector<uint32_t> old_to_new(n);
        uint32_t new_idx = 0;
        for (uint32_t i = 0; i < n; ++i) {
            if (mask[i] == 1) {
                old_to_new[i] = UINT32_MAX;
            } else {
                old_to_new[i] = new_idx++;
            }
        }

        std::vector<double> new_obj;
        std::vector<double> new_col_lb;
        std::vector<double> new_col_ub;
        std::vector<std::vector<CSCEntry>> new_col_entries;

        for (uint32_t i = 0; i < n; ++i) {
            if (old_to_new[i] != UINT32_MAX) {
                new_obj.push_back(_obj[i]);
                new_col_lb.push_back(_col_lb[i]);
                new_col_ub.push_back(_col_ub[i]);
                new_col_entries.push_back(std::move(_col_entries[i]));
            }
        }

        _obj = std::move(new_obj);
        _col_lb = std::move(new_col_lb);
        _col_ub = std::move(new_col_ub);
        _col_entries = std::move(new_col_entries);

        for (uint32_t i = 0; i < n; ++i) {
            mask[i] = (old_to_new[i] == UINT32_MAX) ? -1 : static_cast<int32_t>(old_to_new[i]);
        }

#ifdef MCFCG_CUOPT_DELTA_API
        if (_problem) {
            check_cuopt(cuOptDeleteColumns(_problem, delta_mask.data()), "cuOptDeleteColumns");
        }
#endif
    }

    void set_col_cost(uint32_t col, double cost) override {
        assert(col < _obj.size());
        _obj[col] = cost;
#ifdef MCFCG_CUOPT_DELTA_API
        if (_problem) {
            check_cuopt(
                cuOptSetVariableObjectiveCoefficient(_problem, static_cast<cuopt_int_t>(col),
                                                     static_cast<cuopt_float_t>(cost)),
                "cuOptSetVariableObjectiveCoefficient");
        }
#endif
    }

    void delete_rows(std::vector<int32_t>& mask) override {
        uint32_t m = num_rows();
#ifdef MCFCG_CUOPT_DELTA_API
        std::vector<cuopt_int_t> delta_mask;
        if (_problem) {
            delta_mask.assign(mask.begin(), mask.end());
        }
#endif
        std::vector<uint32_t> old_to_new(m);
        uint32_t new_idx = 0;
        for (uint32_t i = 0; i < m; ++i) {
            if (mask[i] == 1) {
                old_to_new[i] = UINT32_MAX;
            } else {
                old_to_new[i] = new_idx++;
            }
        }

        std::vector<double> new_row_lb;
        std::vector<double> new_row_ub;
        for (uint32_t i = 0; i < m; ++i) {
            if (old_to_new[i] != UINT32_MAX) {
                new_row_lb.push_back(_row_lb[i]);
                new_row_ub.push_back(_row_ub[i]);
            }
        }
        _row_lb = std::move(new_row_lb);
        _row_ub = std::move(new_row_ub);

        for (auto& entries : _col_entries) {
            std::erase_if(entries,
                          [&](const CSCEntry& e) { return old_to_new[e.row] == UINT32_MAX; });
            for (auto& e : entries) {
                e.row = old_to_new[e.row];
            }
        }

        for (uint32_t i = 0; i < m; ++i) {
            mask[i] = (old_to_new[i] == UINT32_MAX) ? -1 : static_cast<int32_t>(old_to_new[i]);
        }

#ifdef MCFCG_CUOPT_DELTA_API
        if (_problem) {
            check_cuopt(cuOptDeleteRows(_problem, delta_mask.data()), "cuOptDeleteRows");
        }
#endif
    }

    LPStatus solve() override {
        uint32_t n = num_cols();
        uint32_t m = num_rows();

        if (n == 0) {
            return LPStatus::Error;
        }

#ifdef MCFCG_CUOPT_DELTA_API
        if (_problem) {
            // Resolve the persistent problem. cuOptResolve may reallocate the
            // solution handle; pass the previous pointer so the implementation
            // can reuse or free it.
            auto status = cuOptResolve(_problem, _settings, &_solution);
            if (status != CUOPT_SUCCESS)
                return LPStatus::Error;
            return extract_solution(_solution, n, m, _cached_obj, _cached_primals, _cached_duals,
                                    _cached_reduced_costs);
        }
#endif

        // First-solve path (also the only path when delta API is off): build
        // the problem from the buffered host state, solve, and extract.
        // Convert internal CSC storage to CSR for cuOpt.
        std::vector<std::vector<std::pair<uint32_t, double>>> row_entries(m);
        for (uint32_t c = 0; c < n; ++c) {
            for (const auto& e : _col_entries[c]) {
                row_entries[e.row].emplace_back(c, e.value);
            }
        }

        std::vector<cuopt_int_t> row_offsets;
        std::vector<cuopt_int_t> col_indices;
        std::vector<cuopt_float_t> coeff_values;
        row_offsets.reserve(m + 1);

        cuopt_int_t offset = 0;
        for (uint32_t r = 0; r < m; ++r) {
            row_offsets.push_back(offset);
            // Sort by column index for cuOpt
            auto& re = row_entries[r];
            std::sort(re.begin(), re.end());
            for (const auto& [col, val] : re) {
                col_indices.push_back(static_cast<cuopt_int_t>(col));
                coeff_values.push_back(static_cast<cuopt_float_t>(val));
            }
            offset += static_cast<cuopt_int_t>(re.size());
        }
        row_offsets.push_back(offset);  // sentinel

        std::vector<cuopt_float_t> f_obj(_obj.begin(), _obj.end());
        std::vector<cuopt_float_t> f_col_lb(_col_lb.begin(), _col_lb.end());
        std::vector<cuopt_float_t> f_col_ub(_col_ub.begin(), _col_ub.end());
        std::vector<cuopt_float_t> f_row_lb(_row_lb.begin(), _row_lb.end());
        std::vector<cuopt_float_t> f_row_ub(_row_ub.begin(), _row_ub.end());

        std::vector<char> var_types(n, CUOPT_CONTINUOUS);

        cuOptOptimizationProblem problem = nullptr;
        auto status = cuOptCreateRangedProblem(
            static_cast<cuopt_int_t>(m), static_cast<cuopt_int_t>(n), CUOPT_MINIMIZE,
            static_cast<cuopt_float_t>(0.0), f_obj.data(), row_offsets.data(), col_indices.data(),
            coeff_values.data(), f_row_lb.data(), f_row_ub.data(), f_col_lb.data(), f_col_ub.data(),
            var_types.data(), &problem);
        if (status != CUOPT_SUCCESS) {
            return LPStatus::Error;
        }

        cuOptSolverSettings settings = nullptr;
        status = cuOptCreateSolverSettings(&settings);
        if (status != CUOPT_SUCCESS) {
            cuOptDestroyProblem(&problem);
            return LPStatus::Error;
        }

        cuOptSetParameter(settings, CUOPT_METHOD, std::to_string(CUOPT_METHOD_BARRIER).c_str());
        auto tol_str = std::to_string(LP_FEAS_TOL);
        cuOptSetParameter(settings, CUOPT_RELATIVE_GAP_TOLERANCE, tol_str.c_str());
        cuOptSetParameter(settings, CUOPT_RELATIVE_PRIMAL_TOLERANCE, tol_str.c_str());
        cuOptSetParameter(settings, CUOPT_RELATIVE_DUAL_TOLERANCE, tol_str.c_str());

        cuOptSetParameter(settings, CUOPT_LOG_TO_CONSOLE, _verbose ? "1" : "0");

        cuOptSolution solution = nullptr;
        status = cuOptSolve(problem, settings, &solution);
        if (status != CUOPT_SUCCESS) {
            cuOptDestroySolverSettings(&settings);
            cuOptDestroyProblem(&problem);
            return LPStatus::Error;
        }

        LPStatus result = extract_solution(solution, n, m, _cached_obj, _cached_primals,
                                           _cached_duals, _cached_reduced_costs);

#ifdef MCFCG_CUOPT_DELTA_API
        // Retain the handles for subsequent delta-API calls.
        _problem = problem;
        _settings = settings;
        _solution = solution;
#else
        cuOptDestroySolution(&solution);
        cuOptDestroySolverSettings(&settings);
        cuOptDestroyProblem(&problem);
#endif
        return result;
    }

    double get_obj() const override { return _cached_obj; }
    std::vector<double> get_primals() const override { return _cached_primals; }
    std::vector<double> get_duals() const override { return _cached_duals; }
    std::vector<double> get_reduced_costs() const override { return _cached_reduced_costs; }

    uint32_t num_cols() const override { return static_cast<uint32_t>(_obj.size()); }
    uint32_t num_rows() const override { return static_cast<uint32_t>(_row_lb.size()); }
};

std::unique_ptr<LPSolver> create_cuopt_solver(bool verbose) {
    return std::make_unique<CuOptSolver>(verbose);
}

}  // namespace mcfcg

#endif  // MCFCG_USE_CUOPT
