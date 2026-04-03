#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <unordered_map>
#include <vector>

#include "mcfcg/cg/column.h"
#include "mcfcg/cg/master.h"
#include "mcfcg/cg/pricer.h"
#include "mcfcg/cg/tree_column.h"
#include "mcfcg/cg/tree_master.h"
#include "mcfcg/cg/tree_pricer.h"
#include "mcfcg/instance.h"

namespace {

constexpr double NEW_COL_RC_TOL = 1e-6;
constexpr double EXISTING_COL_RC_TOL = 1e-4;

static void write_instance(const std::string& path, uint32_t vertices, uint32_t arcs,
                           uint32_t commodities, const std::string& arc_lines,
                           const std::string& commodity_lines) {
    std::ofstream f(path);
    f << vertices << '\n' << arcs << '\n' << commodities << '\n';
    f << arc_lines << commodity_lines;
}

double recompute_path_rc(const mcfcg::Column& col, const std::vector<double>& pi,
                         const std::unordered_map<uint32_t, double>& mu) {
    double rc = col.cost - pi[col.commodity];
    for (uint32_t arc : col.arcs) {
        auto it = mu.find(arc);
        if (it != mu.end()) {
            rc -= it->second;
        }
    }
    return rc;
}

double recompute_tree_rc(const mcfcg::TreeColumn& col, const std::vector<double>& pi_s,
                         const std::unordered_map<uint32_t, double>& mu) {
    double rc = col.cost - pi_s[col.source_idx];
    for (auto& af : col.arc_flows) {
        auto it = mu.find(af.arc);
        if (it != mu.end()) {
            rc -= af.flow * it->second;
        }
    }
    return rc;
}

// Run path CG manually, validating RC at each iteration.
// Returns true if optimal, sets obj.
bool run_path_cg_with_validation(const mcfcg::Instance& inst, double& obj) {
    mcfcg::PathMaster master;
    master.init(inst);

    mcfcg::PathPricer pricer;
    pricer.init(inst);

    // Warm-start
    std::vector<double> big_pi(inst.commodities.size(), mcfcg::PathMaster::BIG_M);
    std::unordered_map<uint32_t, double> empty_mu;
    auto init_cols = pricer.price(big_pi, empty_mu, true);
    if (!init_cols.empty()) {
        master.add_columns(std::move(init_cols));
    }
    pricer.reset_postponed();

    for (uint32_t iter = 0; iter < 10000; ++iter) {
        auto status = master.solve();
        if (status != mcfcg::LPStatus::Optimal) {
            return false;
        }

        obj = master.get_obj();
        auto primals = master.get_primals();

        // Separation
        uint32_t new_caps = master.add_violated_capacity_constraints(primals);
        if (new_caps > 0) {
            continue;
        }

        // Get duals
        auto pi = master.get_demand_duals();
        auto mu = master.get_capacity_duals();

        // Validate existing columns have non-negative RC
        for (auto& col : master.columns()) {
            double rc = recompute_path_rc(col, pi, mu);
            EXPECT_GE(rc, -EXISTING_COL_RC_TOL) << "Existing column for commodity " << col.commodity
                                   << " has negative RC " << rc << " at iteration " << iter;
        }

        // Price
        auto new_cols = pricer.price(pi, mu, false);
        if (new_cols.empty()) {
            new_cols = pricer.price(pi, mu, true);
            if (new_cols.empty()) {
                return true;  // Optimal
            }
            pricer.reset_postponed();
        }

        // Validate new columns have negative RC
        for (auto& col : new_cols) {
            double rc = recompute_path_rc(col, pi, mu);
            EXPECT_LT(rc, NEW_COL_RC_TOL) << "New column for commodity " << col.commodity
                                  << " has non-negative RC " << rc << " at iteration " << iter;
        }

        // Check no duplicates
        for (auto& col : new_cols) {
            for (auto& existing : master.columns()) {
                if (existing.commodity == col.commodity && existing.arcs == col.arcs) {
                    ADD_FAILURE() << "Duplicate column for commodity " << col.commodity
                                 << " at iteration " << iter;
                }
            }
        }

        if (new_cols.size() > 1000) {
            new_cols.resize(1000);
        }
        master.add_columns(std::move(new_cols));
    }
    return false;
}

// Run tree CG manually, validating RC at each iteration.
bool run_tree_cg_with_validation(const mcfcg::Instance& inst, double& obj) {
    mcfcg::TreeMaster master;
    master.init(inst);

    mcfcg::TreePricer pricer;
    pricer.init(inst);

    // Warm-start
    std::vector<double> big_pi(inst.sources.size(), mcfcg::TreeMaster::BIG_M);
    std::unordered_map<uint32_t, double> empty_mu;
    auto init_cols = pricer.price(big_pi, empty_mu, true);
    if (!init_cols.empty()) {
        master.add_columns(std::move(init_cols));
    }
    pricer.reset_postponed();

    for (uint32_t iter = 0; iter < 10000; ++iter) {
        auto status = master.solve();
        if (status != mcfcg::LPStatus::Optimal) {
            return false;
        }

        obj = master.get_obj();
        auto primals = master.get_primals();

        uint32_t new_caps = master.add_violated_capacity_constraints(primals);
        if (new_caps > 0) {
            continue;
        }

        auto pi_s = master.get_source_duals();
        auto mu = master.get_capacity_duals();

        // Validate existing columns have non-negative RC
        for (auto& col : master.columns()) {
            double rc = recompute_tree_rc(col, pi_s, mu);
            EXPECT_GE(rc, -EXISTING_COL_RC_TOL) << "Existing tree column for source " << col.source_idx
                                   << " has negative RC " << rc << " at iteration " << iter;
        }

        // Price
        auto new_cols = pricer.price(pi_s, mu, false);
        if (new_cols.empty()) {
            new_cols = pricer.price(pi_s, mu, true);
            if (new_cols.empty()) {
                return true;
            }
            pricer.reset_postponed();
        }

        // Validate new columns have negative RC
        for (auto& col : new_cols) {
            double rc = recompute_tree_rc(col, pi_s, mu);
            EXPECT_LT(rc, NEW_COL_RC_TOL) << "New tree column for source " << col.source_idx
                                  << " has non-negative RC " << rc << " at iteration " << iter;
        }

        // Check no duplicates
        for (auto& col : new_cols) {
            for (auto& existing : master.columns()) {
                if (existing.source_idx != col.source_idx) {
                    continue;
                }
                if (existing.arc_flows.size() != col.arc_flows.size()) {
                    continue;
                }
                bool match = true;
                for (auto& af : col.arc_flows) {
                    bool found = false;
                    for (auto& eaf : existing.arc_flows) {
                        if (eaf.arc == af.arc && std::abs(eaf.flow - af.flow) < 1e-10) {
                            found = true;
                            break;
                        }
                    }
                    if (!found) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    ADD_FAILURE() << "Duplicate tree column for source " << col.source_idx
                                 << " at iteration " << iter;
                }
            }
        }

        if (new_cols.size() > 1000) {
            new_cols.resize(1000);
        }
        master.add_columns(std::move(new_cols));
    }
    return false;
}

}  // namespace

// --- Path formulation: capacity binding (single commodity) ---

class PathRCCapBinding : public ::testing::Test {
   protected:
    std::string path = "rc_cap_binding.txt";
    void SetUp() override {
        write_instance(path, 4, 4, 1, "1 2 1 3\n1 3 5 10\n2 3 1 10\n3 4 1 10\n", "1 4 5\n");
    }
    void TearDown() override { std::remove(path.c_str()); }
};

TEST_F(PathRCCapBinding, ReducedCostValidation) {
    auto inst = mcfcg::read_commalab(path);
    double obj = 0;
    ASSERT_TRUE(run_path_cg_with_validation(inst, obj));
    EXPECT_NEAR(obj, 21.0, 1e-4);
}

// --- Path formulation: multi-source + capacity ---

class PathRCMultiSourceCap : public ::testing::Test {
   protected:
    std::string path = "rc_multi_cap.txt";
    void SetUp() override {
        write_instance(path, 4, 4, 2, "1 3 1 10\n2 3 2 10\n3 4 1 5\n1 4 4 10\n",
                       "1 4 4\n2 4 3\n");
    }
    void TearDown() override { std::remove(path.c_str()); }
};

TEST_F(PathRCMultiSourceCap, ReducedCostValidation) {
    auto inst = mcfcg::read_commalab(path);
    double obj = 0;
    ASSERT_TRUE(run_path_cg_with_validation(inst, obj));
    EXPECT_NEAR(obj, 21.0, 1e-4);
}

// --- Tree formulation: capacity binding ---

TEST_F(PathRCCapBinding, TreeReducedCostValidation) {
    auto inst = mcfcg::read_commalab(path);
    double obj = 0;
    ASSERT_TRUE(run_tree_cg_with_validation(inst, obj));
    EXPECT_NEAR(obj, 21.0, 1e-4);
}

// --- Tree formulation: multi-source + capacity ---

TEST_F(PathRCMultiSourceCap, TreeReducedCostValidation) {
    auto inst = mcfcg::read_commalab(path);
    double obj = 0;
    ASSERT_TRUE(run_tree_cg_with_validation(inst, obj));
    EXPECT_NEAR(obj, 21.0, 1e-4);
}
