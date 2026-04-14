#include "mcfcg/lp/lp_solver.h"

#include <Highs.h>

namespace mcfcg {

class HiGHSSolver : public LPSolver {
private:
    Highs _highs;
    uint32_t _num_cols = 0;
    uint32_t _num_rows = 0;

public:
    HiGHSSolver() {
        _highs.setOptionValue("output_flag", false);
        HighsModel model;
        model.lp_.sense_ = ObjSense::kMinimize;
        model.lp_.offset_ = 0.0;
        _highs.passModel(std::move(model));
    }

    uint32_t add_cols(const std::vector<double>& obj, const std::vector<double>& lb,
                      const std::vector<double>& ub) override {
        uint32_t first = _num_cols;
        HighsInt n = static_cast<HighsInt>(obj.size());
        _highs.addCols(n, obj.data(), lb.data(), ub.data(), 0, nullptr, nullptr, nullptr);
        _num_cols += static_cast<uint32_t>(n);
        return first;
    }

    uint32_t add_cols(const std::vector<double>& obj, const std::vector<double>& lb,
                      const std::vector<double>& ub, const std::vector<uint32_t>& starts,
                      const std::vector<uint32_t>& row_indices,
                      const std::vector<double>& values) override {
        uint32_t first = _num_cols;
        HighsInt n = static_cast<HighsInt>(obj.size());
        HighsInt nnz = static_cast<HighsInt>(values.size());

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
        uint32_t first = _num_rows;
        HighsInt m = static_cast<HighsInt>(lb.size());
        HighsInt nnz = static_cast<HighsInt>(values.size());

        // Convert to HighsInt
        std::vector<HighsInt> h_starts(starts.size());
        for (size_t i = 0; i < starts.size(); ++i) {
            h_starts[i] = static_cast<HighsInt>(starts[i]);
        }
        // add_rows needs starts of size m+1 for bulk call
        // If starts.size() == m, append nnz as sentinel
        if (h_starts.size() == static_cast<size_t>(m)) {
            h_starts.push_back(nnz);
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

    LPStatus solve() override {
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
        auto& sol = _highs.getSolution();
        return sol.col_value;
    }

    std::vector<double> get_duals() const override {
        auto& sol = _highs.getSolution();
        return sol.row_dual;
    }

    std::vector<double> get_reduced_costs() const override {
        auto& sol = _highs.getSolution();
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

std::unique_ptr<LPSolver> create_lp_solver() {
    return std::make_unique<HiGHSSolver>();
}

}  // namespace mcfcg
