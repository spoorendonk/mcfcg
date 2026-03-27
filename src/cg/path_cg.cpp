#include "mcfcg/cg/path_cg.h"

#include "mcfcg/cg/master.h"
#include "mcfcg/cg/pricer.h"
#include "mcfcg/util/timer.h"

#include <limits>

namespace mcfcg {

CGResult solve_path_cg(const Instance& inst, const CGParams& params) {
    PathMaster master;
    master.init(inst);

    PathPricer pricer;
    pricer.init(inst);

    Timer timer;
    CGLogger logger(params.verbosity);
    logger.print_header();

    constexpr double INF = std::numeric_limits<double>::infinity();

    CGResult result{};
    result.optimal = false;
    bool solved = false;

    auto populate_timing = [&] {
        result.time_lp = timer.elapsed(TimerCat::LP);
        result.time_pricing = timer.elapsed(TimerCat::Pricing);
        result.time_separation = timer.elapsed(TimerCat::Separation);
        result.time_total = timer.elapsed(TimerCat::Total);
    };

    auto set_optimal = [&](double obj, uint32_t iter) {
        result.objective = obj;
        result.iterations = iter + 1;
        result.total_columns = master.num_columns();
        result.optimal = true;
        populate_timing();
        logger.print_summary(result.iterations, obj, true, result.time_lp, result.time_pricing,
                             result.time_separation, result.time_total);
    };

    timer.start(TimerCat::Total);

    if (params.warm_start) {
        // Price with large demand duals so all shortest-path columns pass the
        // negative reduced cost filter (true_rc = -pi[k] + cost < 0 for large pi).
        timer.start(TimerCat::Pricing);
        std::vector<double> big_pi(inst.commodities.size(), PathMaster::BIG_M);
        std::unordered_map<uint32_t, double> empty_mu;
        auto init_cols = pricer.price(big_pi, empty_mu, true);
        if (!init_cols.empty()) {
            master.add_columns(std::move(init_cols));
        }
        pricer.reset_postponed();
        timer.stop(TimerCat::Pricing);
    }

    for (uint32_t iter = 0; iter < params.max_iterations; ++iter) {
        Timer iter_timer;
        iter_timer.start(TimerCat::Total);

        // --- LP solve ---
        timer.start(TimerCat::LP);
        iter_timer.start(TimerCat::LP);
        auto status = master.solve();
        iter_timer.stop(TimerCat::LP);
        timer.stop(TimerCat::LP);

        if (status != LPStatus::Optimal)
            break;
        solved = true;

        double obj = master.get_obj();
        auto primals = master.get_primals();

        // --- Separation ---
        timer.start(TimerCat::Separation);
        iter_timer.start(TimerCat::Separation);
        uint32_t new_caps = master.add_violated_capacity_constraints(primals);
        iter_timer.stop(TimerCat::Separation);
        timer.stop(TimerCat::Separation);

        if (new_caps > 0) {
            iter_timer.stop(TimerCat::Total);
            logger.print_iteration(
                iter + 1, obj, -INF, obj, master.num_lp_cols(), master.num_lp_rows(), 0, 0,
                new_caps, 0, iter_timer.elapsed(TimerCat::LP),
                iter_timer.elapsed(TimerCat::Pricing), iter_timer.elapsed(TimerCat::Separation),
                iter_timer.elapsed(TimerCat::Total));
            continue;
        }

        // --- Pricing ---
        timer.start(TimerCat::Pricing);
        iter_timer.start(TimerCat::Pricing);
        auto pi = master.get_demand_duals();
        auto mu = master.get_capacity_duals();

        auto new_cols = pricer.price(pi, mu, false);

        if (new_cols.empty()) {
            new_cols = pricer.price(pi, mu, true);
            if (new_cols.empty()) {
                iter_timer.stop(TimerCat::Pricing);
                timer.stop(TimerCat::Pricing);
                timer.stop(TimerCat::Total);

                iter_timer.stop(TimerCat::Total);
                logger.print_iteration(
                    iter + 1, obj, -INF, obj, master.num_lp_cols(), master.num_lp_rows(), 0, 0, 0,
                    0, iter_timer.elapsed(TimerCat::LP), iter_timer.elapsed(TimerCat::Pricing),
                    iter_timer.elapsed(TimerCat::Separation), iter_timer.elapsed(TimerCat::Total));

                set_optimal(obj, iter);
                return result;
            }
            pricer.reset_postponed();
        }

        if (new_cols.size() > params.max_cols_per_iter) {
            new_cols.resize(params.max_cols_per_iter);
        }

        iter_timer.stop(TimerCat::Pricing);
        timer.stop(TimerCat::Pricing);

        uint32_t added = master.add_columns(std::move(new_cols));

        result.iterations = iter + 1;
        result.total_columns = master.num_columns();

        iter_timer.stop(TimerCat::Total);
        logger.print_iteration(
            iter + 1, obj, -INF, obj, master.num_lp_cols(), master.num_lp_rows(), added, 0, 0, 0,
            iter_timer.elapsed(TimerCat::LP), iter_timer.elapsed(TimerCat::Pricing),
            iter_timer.elapsed(TimerCat::Separation), iter_timer.elapsed(TimerCat::Total));
    }

    timer.stop(TimerCat::Total);

    if (solved)
        result.objective = master.get_obj();
    populate_timing();
    logger.print_summary(result.iterations, result.objective, result.optimal, result.time_lp,
                         result.time_pricing, result.time_separation, result.time_total);
    return result;
}

}  // namespace mcfcg
