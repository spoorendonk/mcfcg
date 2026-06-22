#ifdef MCFCG_USE_CUOPT

#include "mcfcg/lp/lp_solver.h"
#include "mcfcg/util/tolerances.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuopt/linear_programming/constants.h>
#include <cuopt/linear_programming/cuopt_c.h>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MCFCG_CUOPT_DELTA_API
#include <cuopt/linear_programming/cuopt_c_delta.h>
#endif

// cuOpt corrected the "TERIMINATION" typo in the termination-status constants
// in a release after 26.04 (26.04 still ships the typo; the 26.06+ base the
// delta-api fork tracks has it fixed). Accept either spelling so this backend
// builds against both an older installed cuOpt and the fork.
#ifndef CUOPT_TERMINATION_STATUS_OPTIMAL
#define CUOPT_TERMINATION_STATUS_OPTIMAL CUOPT_TERIMINATION_STATUS_OPTIMAL
#define CUOPT_TERMINATION_STATUS_INFEASIBLE CUOPT_TERIMINATION_STATUS_INFEASIBLE
#define CUOPT_TERMINATION_STATUS_UNBOUNDED CUOPT_TERIMINATION_STATUS_UNBOUNDED
#endif

// CUOPT_PRESOLVE_OFF (the integer value 0 of the "presolve" parameter) is only
// defined in newer cuOpt headers; older installs declare the CUOPT_PRESOLVE
// parameter name but not the value enum. Fall back to its documented value so
// this backend builds against both.
#ifndef CUOPT_PRESOLVE_OFF
#define CUOPT_PRESOLVE_OFF 0
#endif

namespace mcfcg {

namespace {

void check_cuopt(cuopt_int_t status, const char* msg) {
    if (status != CUOPT_SUCCESS) {
        throw std::runtime_error(std::string("cuOpt error in ") + msg);
    }
}

// cuOpt's infinity is IEEE inf (CUOPT_INFINITY). Coerce any bound at or beyond a
// large-magnitude sentinel to +/-CUOPT_INFINITY before handing it to cuOpt. A
// large FINITE bound (e.g. a 1e20 "infinity" stand-in) is a genuine two-sided
// range to cuOpt: its value enters the barrier starting point and breaks
// the solve down numerically (search-direction NaN, returns the origin as
// "Optimal"), whereas +/-inf is handled as a one-sided constraint. mcfcg's own
// INF is already +inf, so this is a no-op for the master; it guards any caller
// that passes a finite infinity sentinel.
constexpr double CUOPT_BOUND_INF_THRESHOLD = 1e19;
cuopt_float_t to_cuopt_bound(double v) {
    if (v >= CUOPT_BOUND_INF_THRESHOLD) {
        return CUOPT_INFINITY;
    }
    if (v <= -CUOPT_BOUND_INF_THRESHOLD) {
        return -CUOPT_INFINITY;
    }
    return static_cast<cuopt_float_t>(v);
}
std::vector<cuopt_float_t> to_cuopt_bounds(const std::vector<double>& v) {
    std::vector<cuopt_float_t> out;
    out.reserve(v.size());
    for (double x : v) {
        out.push_back(to_cuopt_bound(x));
    }
    return out;
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
    if (term_status == CUOPT_TERMINATION_STATUS_INFEASIBLE) {
        return LPStatus::Infeasible;
    }
    if (term_status == CUOPT_TERMINATION_STATUS_UNBOUNDED) {
        return LPStatus::Unbounded;
    }
    if (term_status != CUOPT_TERMINATION_STATUS_OPTIMAL) {
        return LPStatus::Error;
    }

    cuopt_float_t obj_val = 0;
    if (cuOptGetObjectiveValue(solution, &obj_val) != CUOPT_SUCCESS) {
        return LPStatus::Error;
    }
    obj = static_cast<double>(obj_val);

    std::vector<cuopt_float_t> f_primals(n);
    if (cuOptGetPrimalSolution(solution, f_primals.data()) != CUOPT_SUCCESS) {
        return LPStatus::Error;
    }
    primals.assign(f_primals.begin(), f_primals.end());

    std::vector<cuopt_float_t> f_duals(m);
    if (cuOptGetDualSolution(solution, f_duals.data()) != CUOPT_SUCCESS) {
        return LPStatus::Error;
    }
    duals.assign(f_duals.begin(), f_duals.end());

    std::vector<cuopt_float_t> f_rc(n);
    if (cuOptGetReducedCosts(solution, f_rc.data()) != CUOPT_SUCCESS) {
        return LPStatus::Error;
    }
    reduced_costs.assign(f_rc.begin(), f_rc.end());

    // Defensive guard for #33: a failed cuOpt GPU barrier (cuDSS device-alloc /
    // numerical error) deterministically yields CUOPT_TERMINATION_STATUS_
    // NUMERICAL_ERROR, already rejected above. But cuOpt can also
    // nondeterministically report OPTIMAL while the failed factorization
    // collapsed the search direction to NaN and returned a garbage incumbent.
    // Reject a non-finite "optimal" solution rather than feeding it to the CG
    // loop. (Finite-but-wrong garbage from a mislabelled OPTIMAL is a
    // cuOpt-internal bug; see #33 for the upstream report.)
    auto all_finite = [](const std::vector<double>& vec) {
        return std::all_of(vec.begin(), vec.end(), [](double val) { return std::isfinite(val); });
    };
    if (!std::isfinite(obj) || !all_finite(primals) || !all_finite(duals) ||
        !all_finite(reduced_costs)) {
        return LPStatus::Error;
    }

    return LPStatus::Optimal;
}

}  // namespace

// cuOpt's public C API has no incremental mutators — cuOptCreateRangedProblem
// takes a fully-formed problem and cuOptDestroyProblem tears it down, so every
// solve() rebuilds from scratch. The delta C API (see cuopt_c_delta.h on the
// spoorendonk/cuopt fork, tracked by spoorendonk/mcfcg #20 and sub-issues
// #22/#23/#24) introduces a persistent problem handle plus
// cuOptAddColumns / cuOptAddRows / cuOptDeleteColumns / cuOptDeleteRows /
// cuOptSetObjectiveCoefficients / cuOptResolve.
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
            auto f_lb = to_cuopt_bounds(lb);
            auto f_ub = to_cuopt_bounds(ub);
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
            auto f_lb = to_cuopt_bounds(lb);
            auto f_ub = to_cuopt_bounds(ub);
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
            // cuOptAddRows requires each row's column indices sorted strictly
            // ascending — cuOpt does not re-sort delta-appended rows. The
            // caller's CSR is not guaranteed sorted: a lazily-added capacity
            // row (master_base.h add_violated_capacity_constraints) enumerates
            // columns in creation order, which is no longer ascending in
            // LP-index space once column purges have remapped _col_to_lp. Sort
            // each row's (col, value) pairs here, matching the rebuild path.
            std::vector<cuopt_int_t> f_starts;
            f_starts.reserve(m + 1);
            std::vector<cuopt_int_t> f_cols;
            f_cols.reserve(indices.size());
            std::vector<cuopt_float_t> f_vals;
            f_vals.reserve(values.size());
            std::vector<std::pair<cuopt_int_t, cuopt_float_t>> row_entries;
            for (uint32_t i = 0; i < m; ++i) {
                f_starts.push_back(static_cast<cuopt_int_t>(f_cols.size()));
                uint32_t begin = starts[i];
                uint32_t end = starts[i + 1];  // caller includes sentinel
                row_entries.clear();
                row_entries.reserve(end - begin);
                for (uint32_t j = begin; j < end; ++j) {
                    row_entries.emplace_back(static_cast<cuopt_int_t>(indices[j]),
                                             static_cast<cuopt_float_t>(values[j]));
                }
                std::sort(row_entries.begin(), row_entries.end());
                for (const auto& [col, val] : row_entries) {
                    f_cols.push_back(col);
                    f_vals.push_back(val);
                }
            }
            f_starts.push_back(static_cast<cuopt_int_t>(f_cols.size()));  // sentinel
            auto f_lb = to_cuopt_bounds(lb);
            auto f_ub = to_cuopt_bounds(ub);
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
        std::vector<cuopt_int_t> delta_indices;
        if (_problem) {
            for (uint32_t i = 0; i < n; ++i) {
                if (mask[i] == 1) {
                    delta_indices.push_back(static_cast<cuopt_int_t>(i));
                }
            }
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
            check_cuopt(cuOptDeleteColumns(_problem, static_cast<cuopt_int_t>(delta_indices.size()),
                                           delta_indices.data()),
                        "cuOptDeleteColumns");
        }
#endif
    }

    void set_col_cost(uint32_t col, double cost) override {
        assert(col < _obj.size());
        _obj[col] = cost;
#ifdef MCFCG_CUOPT_DELTA_API
        if (_problem) {
            const cuopt_int_t idx = static_cast<cuopt_int_t>(col);
            const cuopt_float_t v = static_cast<cuopt_float_t>(cost);
            check_cuopt(cuOptSetObjectiveCoefficients(_problem, 1, &idx, &v),
                        "cuOptSetObjectiveCoefficients");
        }
#endif
    }

    void delete_rows(std::vector<int32_t>& mask) override {
        uint32_t m = num_rows();
#ifdef MCFCG_CUOPT_DELTA_API
        std::vector<cuopt_int_t> delta_indices;
        if (_problem) {
            for (uint32_t i = 0; i < m; ++i) {
                if (mask[i] == 1) {
                    delta_indices.push_back(static_cast<cuopt_int_t>(i));
                }
            }
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
            check_cuopt(cuOptDeleteRows(_problem, static_cast<cuopt_int_t>(delta_indices.size()),
                                        delta_indices.data()),
                        "cuOptDeleteRows");
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
        auto f_col_lb = to_cuopt_bounds(_col_lb);
        auto f_col_ub = to_cuopt_bounds(_col_ub);
        auto f_row_lb = to_cuopt_bounds(_row_lb);
        auto f_row_ub = to_cuopt_bounds(_row_ub);

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

        // cuOpt always solves with barrier (IPM). This repo does not expose
        // per-solver algorithm selection.
        cuOptSetParameter(settings, CUOPT_METHOD, std::to_string(CUOPT_METHOD_BARRIER).c_str());
        // Presolve OFF. The CG master mutates this LP incrementally and reads
        // duals / reduced costs keyed by column and row index every iteration,
        // and the delta C API appends/deletes rows and columns by index on a
        // persistent handle. cuOpt presolve aggregates and removes rows/columns,
        // which breaks that 1:1 index mapping and is wasted work on these
        // repeatedly-resolved warm-started LPs. _settings is created once and
        // reused by cuOptResolve, so setting this here covers the delta path too.
        cuOptSetParameter(settings, CUOPT_PRESOLVE, std::to_string(CUOPT_PRESOLVE_OFF).c_str());
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
