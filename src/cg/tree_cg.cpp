#include "mcfcg/cg/tree_cg.h"

#include <unordered_set>

#include "mcfcg/cg/tree_master.h"
#include "mcfcg/cg/tree_pricer.h"

namespace mcfcg {

CGResult solve_tree_cg(const Instance & inst, const CGParams & params) {
    TreeMaster master;
    master.init(inst);

    TreePricer pricer;
    pricer.init(inst);

    CGResult result{};
    result.optimal = false;
    bool solved = false;

    auto set_optimal = [&](double obj, uint32_t iter) {
        result.objective = obj;
        result.iterations = iter + 1;
        result.total_columns = master.num_columns();
        result.optimal = true;
    };

    for (uint32_t iter = 0; iter < params.max_iterations; ++iter) {
        auto status = master.solve();
        if (status != LPStatus::Optimal)
            break;
        solved = true;

        double obj = master.get_obj();
        auto primals = master.get_primals();

        uint32_t new_caps = master.add_violated_capacity_constraints(primals);
        if (new_caps > 0)
            continue;

        auto pi_s = master.get_source_duals();
        auto mu = master.get_capacity_duals();

        auto new_cols = pricer.price(pi_s, mu, false);

        if (new_cols.empty()) {
            new_cols = pricer.price(pi_s, mu, true);
            if (new_cols.empty()) {
                set_optimal(obj, iter);
                return result;
            }
            pricer.reset_postponed();
        }

        uint32_t added = master.add_columns(std::move(new_cols));
        if (added == 0) {
            std::unordered_set<uint32_t> forbidden;
            for (auto & [arc, dual] : mu) {
                forbidden.insert(arc);
            }
            auto alt_cols = pricer.price(pi_s, mu, forbidden, true);
            added = master.add_columns(std::move(alt_cols));
            if (added == 0) {
                set_optimal(obj, iter);
                return result;
            }
            pricer.reset_postponed();
        }
        result.iterations = iter + 1;
        result.total_columns = master.num_columns();
    }

    if (solved)
        result.objective = master.get_obj();
    return result;
}

}  // namespace mcfcg
