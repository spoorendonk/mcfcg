#ifdef MCFCG_USE_CUOPT

#include "mcfcg/lp/lp_solver.h"

#include <algorithm>
#include <cstdint>
#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/cuopt_c.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace mcfcg {

namespace {

void check_cuopt(cuopt_int_t status, const char* msg) {
    if (status != CUOPT_SUCCESS) {
        throw std::runtime_error(std::string("cuOpt error in ") + msg);
    }
}

}  // namespace

// cuOpt has no incremental API (no add_cols/add_rows/delete_cols/delete_rows).
// This solver maintains the full problem state internally and rebuilds the
// cuOpt problem from scratch on each solve() call.
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
    // For each column, list of (row, value) entries
    std::vector<std::vector<CSCEntry>> _col_entries;

    // Cached solution
    double _cached_obj = 0.0;
    std::vector<double> _cached_primals;
    std::vector<double> _cached_duals;
    std::vector<double> _cached_reduced_costs;

    bool _verbose = false;

public:
    CuOptSolver() = default;
    explicit CuOptSolver(bool verbose) : _verbose(verbose) {}

    uint32_t add_cols(const std::vector<double>& obj, const std::vector<double>& lb,
                      const std::vector<double>& ub) override {
        uint32_t first = num_cols();
        for (size_t i = 0; i < obj.size(); ++i) {
            _obj.push_back(obj[i]);
            _col_lb.push_back(lb[i]);
            _col_ub.push_back(ub[i]);
            _col_entries.emplace_back();
        }
        return first;
    }

    uint32_t add_cols(const std::vector<double>& obj, const std::vector<double>& lb,
                      const std::vector<double>& ub, const std::vector<uint32_t>& starts,
                      const std::vector<uint32_t>& row_indices,
                      const std::vector<double>& values) override {
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
        return first;
    }

    uint32_t add_rows(const std::vector<double>& lb, const std::vector<double>& ub,
                      const std::vector<uint32_t>& starts, const std::vector<uint32_t>& indices,
                      const std::vector<double>& values) override {
        uint32_t first = num_rows();
        auto m = static_cast<uint32_t>(lb.size());

        for (uint32_t i = 0; i < m; ++i) {
            _row_lb.push_back(lb[i]);
            _row_ub.push_back(ub[i]);

            uint32_t begin = starts[i];
            // Callers don't include sentinel; derive end from next start or nnz
            uint32_t end = (i + 1 < m) ? starts[i + 1] : static_cast<uint32_t>(values.size());

            uint32_t row = first + i;
            for (uint32_t j = begin; j < end; ++j) {
                uint32_t col = indices[j];
                _col_entries[col].push_back({row, values[j]});
            }
        }
        return first;
    }

    void delete_cols(std::vector<int32_t>& mask) override {
        uint32_t n = num_cols();
        // Build list of surviving columns and their new indices
        std::vector<uint32_t> old_to_new(n);
        uint32_t new_idx = 0;
        for (uint32_t i = 0; i < n; ++i) {
            if (mask[i] == 1) {
                old_to_new[i] = UINT32_MAX;  // deleted
            } else {
                old_to_new[i] = new_idx++;
            }
        }

        // Compact column data
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

        // Write new indices back into mask (-1 for deleted)
        for (uint32_t i = 0; i < n; ++i) {
            mask[i] = (old_to_new[i] == UINT32_MAX) ? -1 : static_cast<int32_t>(old_to_new[i]);
        }
    }

    void delete_rows(std::vector<int32_t>& mask) override {
        uint32_t m = num_rows();
        // Build old-to-new row mapping
        std::vector<uint32_t> old_to_new(m);
        uint32_t new_idx = 0;
        for (uint32_t i = 0; i < m; ++i) {
            if (mask[i] == 1) {
                old_to_new[i] = UINT32_MAX;  // deleted
            } else {
                old_to_new[i] = new_idx++;
            }
        }

        // Compact row bounds
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

        // Update row indices in column entries and remove deleted entries
        for (auto& entries : _col_entries) {
            std::erase_if(entries,
                          [&](const CSCEntry& e) { return old_to_new[e.row] == UINT32_MAX; });
            for (auto& e : entries) {
                e.row = old_to_new[e.row];
            }
        }

        // Write new indices back into mask
        for (uint32_t i = 0; i < m; ++i) {
            mask[i] = (old_to_new[i] == UINT32_MAX) ? -1 : static_cast<int32_t>(old_to_new[i]);
        }
    }

    LPStatus solve() override {
        uint32_t n = num_cols();
        uint32_t m = num_rows();

        if (n == 0) {
            return LPStatus::Error;
        }

        // Convert internal CSC storage to CSR for cuOpt.
        // Build CSR row_offsets, col_indices, coeff_values.
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

        // Convert bounds and objective to cuopt_float_t
        std::vector<cuopt_float_t> f_obj(n);
        std::vector<cuopt_float_t> f_col_lb(n);
        std::vector<cuopt_float_t> f_col_ub(n);
        for (uint32_t i = 0; i < n; ++i) {
            f_obj[i] = static_cast<cuopt_float_t>(_obj[i]);
            f_col_lb[i] = static_cast<cuopt_float_t>(_col_lb[i]);
            f_col_ub[i] = static_cast<cuopt_float_t>(_col_ub[i]);
        }

        std::vector<cuopt_float_t> f_row_lb(m);
        std::vector<cuopt_float_t> f_row_ub(m);
        for (uint32_t i = 0; i < m; ++i) {
            f_row_lb[i] = static_cast<cuopt_float_t>(_row_lb[i]);
            f_row_ub[i] = static_cast<cuopt_float_t>(_row_ub[i]);
        }

        // All variables are continuous
        std::vector<char> var_types(n, CUOPT_CONTINUOUS);

        // Create problem
        cuOptOptimizationProblem problem = nullptr;
        auto status = cuOptCreateRangedProblem(
            static_cast<cuopt_int_t>(m), static_cast<cuopt_int_t>(n), CUOPT_MINIMIZE,
            static_cast<cuopt_float_t>(0.0), f_obj.data(), row_offsets.data(), col_indices.data(),
            coeff_values.data(), f_row_lb.data(), f_row_ub.data(), f_col_lb.data(), f_col_ub.data(),
            var_types.data(), &problem);
        if (status != CUOPT_SUCCESS) {
            return LPStatus::Error;
        }

        // Create solver settings
        cuOptSolverSettings settings = nullptr;
        status = cuOptCreateSolverSettings(&settings);
        if (status != CUOPT_SUCCESS) {
            cuOptDestroyProblem(&problem);
            return LPStatus::Error;
        }

        // Use GPU barrier solver with tight tolerances (1e-8 relative)
        cuOptSetParameter(settings, CUOPT_METHOD, std::to_string(CUOPT_METHOD_BARRIER).c_str());
        cuOptSetParameter(settings, CUOPT_RELATIVE_GAP_TOLERANCE, "1e-8");
        cuOptSetParameter(settings, CUOPT_RELATIVE_PRIMAL_TOLERANCE, "1e-8");
        cuOptSetParameter(settings, CUOPT_RELATIVE_DUAL_TOLERANCE, "1e-8");

        cuOptSetParameter(settings, CUOPT_LOG_TO_CONSOLE, _verbose ? "1" : "0");

        // Solve
        cuOptSolution solution = nullptr;
        status = cuOptSolve(problem, settings, &solution);
        if (status != CUOPT_SUCCESS) {
            cuOptDestroySolverSettings(&settings);
            cuOptDestroyProblem(&problem);
            return LPStatus::Error;
        }

        // Cleanup helper — ensures cuOpt resources are freed on all paths.
        auto cleanup = [&] {
            if (solution) {
                cuOptDestroySolution(&solution);
            }
            cuOptDestroySolverSettings(&settings);
            cuOptDestroyProblem(&problem);
        };

        // Check termination status
        cuopt_int_t term_status = 0;
        if (cuOptGetTerminationStatus(solution, &term_status) != CUOPT_SUCCESS) {
            cleanup();
            return LPStatus::Error;
        }

        LPStatus result = LPStatus::Error;
        // Note: cuOpt API has "TERIMINATION" typo in constant names
        if (term_status == CUOPT_TERIMINATION_STATUS_OPTIMAL) {
            // Extract objective
            cuopt_float_t obj_val = 0;
            if (cuOptGetObjectiveValue(solution, &obj_val) != CUOPT_SUCCESS) {
                cleanup();
                return LPStatus::Error;
            }
            _cached_obj = static_cast<double>(obj_val);

            // Extract primals
            std::vector<cuopt_float_t> f_primals(n);
            if (cuOptGetPrimalSolution(solution, f_primals.data()) != CUOPT_SUCCESS) {
                cleanup();
                return LPStatus::Error;
            }
            _cached_primals.resize(n);
            for (uint32_t i = 0; i < n; ++i) {
                _cached_primals[i] = static_cast<double>(f_primals[i]);
            }

            // Extract duals
            std::vector<cuopt_float_t> f_duals(m);
            if (cuOptGetDualSolution(solution, f_duals.data()) != CUOPT_SUCCESS) {
                cleanup();
                return LPStatus::Error;
            }
            _cached_duals.resize(m);
            for (uint32_t i = 0; i < m; ++i) {
                _cached_duals[i] = static_cast<double>(f_duals[i]);
            }

            // Extract reduced costs
            std::vector<cuopt_float_t> f_rc(n);
            if (cuOptGetReducedCosts(solution, f_rc.data()) != CUOPT_SUCCESS) {
                cleanup();
                return LPStatus::Error;
            }
            _cached_reduced_costs.resize(n);
            for (uint32_t i = 0; i < n; ++i) {
                _cached_reduced_costs[i] = static_cast<double>(f_rc[i]);
            }

            result = LPStatus::Optimal;
        } else if (term_status == CUOPT_TERIMINATION_STATUS_INFEASIBLE) {
            result = LPStatus::Infeasible;
        } else if (term_status == CUOPT_TERIMINATION_STATUS_UNBOUNDED) {
            result = LPStatus::Unbounded;
        }

        cleanup();
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
