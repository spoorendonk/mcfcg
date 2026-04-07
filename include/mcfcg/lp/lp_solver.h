#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace mcfcg {

enum class LPStatus { Optimal, Infeasible, Unbounded, Error };

class LPSolver {
   public:
    virtual ~LPSolver() = default;

    // Add columns with no row coefficients. Returns index of first new column.
    virtual uint32_t add_cols(const std::vector<double> & obj,
                              const std::vector<double> & lb,
                              const std::vector<double> & ub) = 0;

    // Add columns with row coefficients (CSC format).
    // Returns index of first new column.
    // starts: CSC column starts into row_indices/values (size = obj.size()+1)
    virtual uint32_t add_cols(const std::vector<double> & obj,
                              const std::vector<double> & lb,
                              const std::vector<double> & ub,
                              const std::vector<uint32_t> & starts,
                              const std::vector<uint32_t> & row_indices,
                              const std::vector<double> & values) = 0;

    // Add rows (constraints) in CSR format. Returns index of first new row.
    virtual uint32_t add_rows(const std::vector<double> & lb,
                              const std::vector<double> & ub,
                              const std::vector<uint32_t> & starts,
                              const std::vector<uint32_t> & indices,
                              const std::vector<double> & values) = 0;

    // Bulk delete columns by mask. mask[i]=1 means delete column i.
    // After call, mask[i] contains the new index of column i (or -1 if
    // deleted).
    virtual void delete_cols(std::vector<int32_t> & mask) = 0;

    // Bulk delete rows by mask. mask[i]=1 means delete row i.
    // After call, mask[i] contains the new index of row i (or -1 if deleted).
    virtual void delete_rows(std::vector<int32_t> & mask) = 0;

    // Solve the LP
    virtual LPStatus solve() = 0;

    // Get solution info (valid after solve returns Optimal)
    virtual double get_obj() const = 0;
    virtual std::vector<double> get_primals() const = 0;
    virtual std::vector<double> get_duals() const = 0;

    virtual uint32_t num_cols() const = 0;
    virtual uint32_t num_rows() const = 0;
};

std::unique_ptr<LPSolver> create_lp_solver();

#ifdef MCFCG_USE_CUOPT
std::unique_ptr<LPSolver> create_cuopt_solver(bool verbose = false);
#endif

}  // namespace mcfcg
