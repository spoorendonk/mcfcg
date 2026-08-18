#include "mcfcg/lp/lp_solver.h"

#include <gtest/gtest.h>

#include <cmath>

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
// Single cuOpt correctness test: drive the backend (barrier — the only method
// this repo exposes) through a sequence of incremental mutations interleaved
// with solves, checking the objective after each step. The cuOpt backend uses
// the persistent-handle delta C API by default (cuOptAddColumns / cuOptAddRows /
// cuOptDeleteRows / cuOptDeleteColumns / cuOptSetObjectiveCoefficients /
// cuOptResolve), so this exercises it directly; a build with
// -DMCFCG_CUOPT_DELTA_API=OFF exercises the rebuild path instead. The optimum
// after each mutation is known in closed
// form, so a delta bug that corrupts the warm-started problem — a stale
// coefficient, a botched index remap after delete, an ignored cost update —
// surfaces as a wrong objective. This is the isolated counterpart to the CG
// integration test, which exercises the same delta calls under load.
//
// The 1e20 column/row bounds also guard the backend's infinity coercion: a
// FINITE 1e20 (a common "infinity" stand-in) must be coerced to CUOPT_INFINITY,
// otherwise cuOpt reads e.g. [5, 1e20] as a true two-sided range whose 1e20
// bound detonates the barrier IPM (NaN search direction -> origin as "Optimal").
//
// Requires a healthy GPU + the cuOpt fork at build time; skipped from the
// default (HiGHS-only) build because create_cuopt_solver is not compiled.
TEST(CuOptSolver, IncrementalMutationsTrackOptimum) {
    auto lp = mcfcg::create_cuopt_solver(/*verbose=*/false);
    const double tol = 1e-4;

    // 1) min x  s.t. x >= 5                         -> x=5, obj=5
    lp->add_cols({1.0}, {0.0}, {1e20});
    lp->add_rows({5.0}, {1e20}, {0, 1}, {0}, {1.0});
    ASSERT_EQ(lp->solve(), mcfcg::LPStatus::Optimal);
    EXPECT_NEAR(lp->get_obj(), 5.0, tol);
    // Also check the backend returns usable duals, not just a correct
    // objective: the CG loop consumes the dual vector every iteration, so an
    // obj-only check would miss a backend that solves but reports garbled
    // duals. Dual of the binding x >= 5 row is 1.0.
    auto duals = lp->get_duals();
    ASSERT_EQ(duals.size(), 1u);
    EXPECT_NEAR(std::abs(duals[0]), 1.0, tol);

    // 2) add y (obj 0.5) into row 0  -> min x+0.5y s.t. x+y>=5 -> y=5, obj=2.5
    lp->add_cols({0.5}, {0.0}, {1e20}, {0, 1}, {0}, {1.0});
    ASSERT_EQ(lp->solve(), mcfcg::LPStatus::Optimal);
    EXPECT_NEAR(lp->get_obj(), 2.5, tol);

    // 3) raise y's cost to 2.0                      -> x=5, y=0, obj=5
    lp->set_col_cost(1, 2.0);
    ASSERT_EQ(lp->solve(), mcfcg::LPStatus::Optimal);
    EXPECT_NEAR(lp->get_obj(), 5.0, tol);

    // 4) add row y >= 2            -> min x+2y s.t. x+y>=5, y>=2 -> x=3,y=2,obj=7
    lp->add_rows({2.0}, {1e20}, {0, 1}, {1}, {1.0});
    ASSERT_EQ(lp->solve(), mcfcg::LPStatus::Optimal);
    EXPECT_NEAR(lp->get_obj(), 7.0, tol);

    // 5) delete the y>=2 row (row index 1)          -> back to x=5, y=0, obj=5
    std::vector<int32_t> row_mask = {0, 1};
    lp->delete_rows(row_mask);
    EXPECT_EQ(row_mask[0], 0);
    EXPECT_EQ(row_mask[1], -1);
    ASSERT_EQ(lp->solve(), mcfcg::LPStatus::Optimal);
    EXPECT_NEAR(lp->get_obj(), 5.0, tol);

    // 6) add a fixed-at-zero column z, then delete it -> optimum unchanged (5)
    lp->add_cols({1.0}, {0.0}, {0.0});  // z in [0,0]
    ASSERT_EQ(lp->num_cols(), 3u);
    std::vector<int32_t> col_mask = {0, 0, 1};
    lp->delete_cols(col_mask);
    EXPECT_EQ(col_mask[2], -1);
    ASSERT_EQ(lp->num_cols(), 2u);
    ASSERT_EQ(lp->solve(), mcfcg::LPStatus::Optimal);
    EXPECT_NEAR(lp->get_obj(), 5.0, tol);
}
#endif  // MCFCG_USE_CUOPT

#ifdef MCFCG_USE_MOSEK
// MOSEK backend correctness: drive the barrier (the only method this backend
// exposes — presolve off, crossover off) through incremental mutations
// interleaved with solves, checking the objective after each step. The optimum
// after every mutation is known in closed form, so a bug in the bound-key
// derivation, the index remap after a delete, an ignored cost update, or a
// flipped dual/reduced-cost sign surfaces as a wrong value here.
//
// The first row is an equality (lb == ub) — the row type the mcfcg master
// actually emits for demand/convexity ('=' rows; capacity rows are '<='). The
// 1e20 column/row bounds guard the infinity coercion: a finite 1e20 handed to
// MOSEK as a range bound would corrupt the interior-point starting point, so it
// must be mapped to an open (MSK_BK_LO/UP) bound key instead.
//
// Requires the MOSEK SDK + a valid license at build/run time; skipped from the
// default (HiGHS-only) build because create_mosek_solver is not compiled.
TEST(MosekSolver, IncrementalMutationsTrackOptimum) {
    auto lp = mcfcg::create_mosek_solver(/*verbose=*/false);
    const double tol = 1e-3;

    // 1) min x  s.t. x == 5                         -> x=5, obj=5
    lp->add_cols({1.0}, {0.0}, {1e20});
    lp->add_rows({5.0}, {5.0}, {0, 1}, {0}, {1.0});
    ASSERT_EQ(lp->solve(), mcfcg::LPStatus::Optimal);
    EXPECT_NEAR(lp->get_obj(), 5.0, tol);
    // The CG loop consumes the dual vector every iteration, so verify duals are
    // usable, not just the objective. Dual of the binding x == 5 row is 1.0.
    auto duals = lp->get_duals();
    ASSERT_EQ(duals.size(), 1u);
    EXPECT_NEAR(duals[0], 1.0, tol);

    // 2) add y (obj 0.5) into row 0  -> min x+0.5y s.t. x+y==5 -> y=5, obj=2.5
    lp->add_cols({0.5}, {0.0}, {1e20}, {0, 1}, {0}, {1.0});
    ASSERT_EQ(lp->solve(), mcfcg::LPStatus::Optimal);
    EXPECT_NEAR(lp->get_obj(), 2.5, tol);

    // 3) raise y's cost to 2.0                      -> x=5, y=0, obj=5
    lp->set_col_cost(1, 2.0);
    ASSERT_EQ(lp->solve(), mcfcg::LPStatus::Optimal);
    EXPECT_NEAR(lp->get_obj(), 5.0, tol);
    // Reduced costs follow c - A'y. With the x==5 row dual at 1.0: x is basic so
    // RC 0; y sits at its lower bound with cost 2, so RC = 2 - 1 = 1. The pricer
    // consumes these — a flipped sign would make y's RC negative and mislead it.
    auto rc = lp->get_reduced_costs();
    ASSERT_EQ(rc.size(), 2u);
    EXPECT_NEAR(rc[0], 0.0, tol);
    EXPECT_NEAR(rc[1], 1.0, tol);

    // 4) add row y >= 2          -> min x+2y s.t. x+y==5, y>=2 -> x=3,y=2,obj=7
    lp->add_rows({2.0}, {1e20}, {0, 1}, {1}, {1.0});
    ASSERT_EQ(lp->solve(), mcfcg::LPStatus::Optimal);
    EXPECT_NEAR(lp->get_obj(), 7.0, tol);

    // 5) delete the y>=2 row (row index 1)          -> back to x=5, y=0, obj=5
    std::vector<int32_t> row_mask = {0, 1};
    lp->delete_rows(row_mask);
    EXPECT_EQ(row_mask[0], 0);
    EXPECT_EQ(row_mask[1], -1);
    ASSERT_EQ(lp->solve(), mcfcg::LPStatus::Optimal);
    EXPECT_NEAR(lp->get_obj(), 5.0, tol);

    // 6) add a fixed-at-zero column z, then delete it -> optimum unchanged (5)
    lp->add_cols({1.0}, {0.0}, {0.0});  // z in [0,0]
    ASSERT_EQ(lp->num_cols(), 3u);
    std::vector<int32_t> col_mask = {0, 0, 1};
    lp->delete_cols(col_mask);
    EXPECT_EQ(col_mask[2], -1);
    ASSERT_EQ(lp->num_cols(), 2u);
    ASSERT_EQ(lp->solve(), mcfcg::LPStatus::Optimal);
    EXPECT_NEAR(lp->get_obj(), 5.0, tol);

    // has_basis() is false for the barrier (no crossover); get_basic_cols empty.
    EXPECT_FALSE(lp->has_basis());
    EXPECT_TRUE(lp->get_basic_cols().empty());
}
#endif  // MCFCG_USE_MOSEK
