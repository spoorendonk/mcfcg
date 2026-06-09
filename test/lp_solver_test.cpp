#include "mcfcg/lp/lp_solver.h"

#include <cmath>
#include <gtest/gtest.h>

// Solve: min x + 2y  s.t. x + y >= 3, x >= 0, y >= 0
TEST(LPSolver, SimpleLP) {
    auto lp = mcfcg::create_lp_solver();

    // Add 2 columns: x and y
    lp->add_cols({1.0, 2.0},   // obj
                 {0.0, 0.0},   // lb
                 {1e20, 1e20}  // ub
    );

    // Add row: x + y >= 3  =>  3 <= x + y <= inf
    lp->add_rows({3.0},      // lb
                 {1e20},     // ub
                 {0, 2},     // starts (sentinel = nnz)
                 {0, 1},     // indices
                 {1.0, 1.0}  // values
    );

    auto status = lp->solve();
    ASSERT_EQ(status, mcfcg::LPStatus::Optimal);

    double obj = lp->get_obj();
    EXPECT_NEAR(obj, 3.0, 1e-6);  // x=3, y=0

    auto primals = lp->get_primals();
    EXPECT_NEAR(primals[0], 3.0, 1e-6);
    EXPECT_NEAR(primals[1], 0.0, 1e-6);
}

// Solve: min -x - 3y  s.t. x + y <= 5, 2x + y <= 8, x,y >= 0
// Unique optimal: x=0, y=5, obj=-15
TEST(LPSolver, TwoConstraints) {
    auto lp = mcfcg::create_lp_solver();

    lp->add_cols({-1.0, -3.0}, {0.0, 0.0}, {1e20, 1e20});

    lp->add_rows({-1e20, -1e20},       // lb
                 {5.0, 8.0},           // ub
                 {0, 2, 4},            // starts (sentinel = nnz)
                 {0, 1, 0, 1},         // indices
                 {1.0, 1.0, 2.0, 1.0}  // values
    );

    auto status = lp->solve();
    ASSERT_EQ(status, mcfcg::LPStatus::Optimal);

    double obj = lp->get_obj();
    EXPECT_NEAR(obj, -15.0, 1e-6);

    auto primals = lp->get_primals();
    EXPECT_NEAR(primals[0], 0.0, 1e-6);
    EXPECT_NEAR(primals[1], 5.0, 1e-6);
}

TEST(LPSolver, Duals) {
    auto lp = mcfcg::create_lp_solver();

    // min x  s.t. x >= 5
    lp->add_cols({1.0}, {0.0}, {1e20});
    lp->add_rows({5.0}, {1e20}, {0, 1}, {0}, {1.0});

    auto status = lp->solve();
    ASSERT_EQ(status, mcfcg::LPStatus::Optimal);

    auto duals = lp->get_duals();
    EXPECT_EQ(duals.size(), 1u);
    // Dual of x >= 5 should be 1.0 (shadow price)
    EXPECT_NEAR(std::abs(duals[0]), 1.0, 1e-6);
}

TEST(LPSolver, IncrementalColumns) {
    auto lp = mcfcg::create_lp_solver();

    // Start with: min x  s.t. x >= 5
    lp->add_cols({1.0}, {0.0}, {1e20});
    lp->add_rows({5.0}, {1e20}, {0, 1}, {0}, {1.0});

    auto status = lp->solve();
    ASSERT_EQ(status, mcfcg::LPStatus::Optimal);
    EXPECT_NEAR(lp->get_obj(), 5.0, 1e-6);

    // Add column y with obj=0.5, participates in same constraint
    // Now: min x + 0.5y  s.t. x + y >= 5
    // But adding a column to existing row requires re-adding the row
    // For now just verify column addition works
    lp->add_cols({0.5}, {0.0}, {1e20});
    EXPECT_EQ(lp->num_cols(), 2u);
}

#ifdef MCFCG_USE_CUOPT
// Smoke-test each selectable cuOpt method end-to-end on a small LP, giving the
// PDLP / dual-simplex resolve paths a real consumer (#30). PDLP is driven to a
// 1e-6 gap/feasibility tolerance inside the backend (#24); the solution-value
// assertions here use 1e-5 — loose enough to absorb first-order PDLP slack at
// that tolerance, tight enough that a coarse (e.g. 1e-2) regression fails.
// Requires a healthy GPU + the cuOpt fork at build time; skipped from the
// default (HiGHS-only) build because create_cuopt_solver is not compiled.
class CuOptMethodTest : public ::testing::TestWithParam<mcfcg::CuOptMethod> {};

// min x + 2y  s.t. x + y >= 3, x,y >= 0  =>  x=3, y=0, obj=3, dual=1
TEST_P(CuOptMethodTest, SolvesSimpleLP) {
    auto lp = mcfcg::create_cuopt_solver(/*verbose=*/false, GetParam());

    lp->add_cols({1.0, 2.0}, {0.0, 0.0}, {1e20, 1e20});
    lp->add_rows({3.0}, {1e20}, {0, 2}, {0, 1}, {1.0, 1.0});

    auto status = lp->solve();
    ASSERT_EQ(status, mcfcg::LPStatus::Optimal);

    EXPECT_NEAR(lp->get_obj(), 3.0, 1e-5);

    auto primals = lp->get_primals();
    EXPECT_NEAR(primals[0], 3.0, 1e-5);
    EXPECT_NEAR(primals[1], 0.0, 1e-5);

    auto duals = lp->get_duals();
    ASSERT_EQ(duals.size(), 1u);
    EXPECT_NEAR(std::abs(duals[0]), 1.0, 1e-5);
}

INSTANTIATE_TEST_SUITE_P(AllMethods, CuOptMethodTest,
                         ::testing::Values(mcfcg::CuOptMethod::Barrier, mcfcg::CuOptMethod::Pdlp,
                                           mcfcg::CuOptMethod::DualSimplex));
#endif  // MCFCG_USE_CUOPT
