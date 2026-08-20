#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <thread>
#include <unordered_map>

#include "mcfcg/cg/master.h"
#include "mcfcg/cg/master_base.h"
#include "mcfcg/cg/path_cg.h"
#include "mcfcg/cg/tree_cg.h"
#include "mcfcg/cg/tree_master.h"
#include "mcfcg/instance.h"
#include "mcfcg/source/source_lp.h"

// Demand coefficients for TNTP instances (from paper's coefs.csv)
static const std::unordered_map<std::string, double> TNTP_COEFS = {
    {"Austin", 6.0},       {"Barcelona", 5050.0},    {"BerlinCenter", 0.5},
    {"Birmingham", 0.9},   {"ChicagoRegional", 4.1}, {"ChicagoSketch", 2.4},
    {"Philadelphia", 7.0}, {"Sydney", 1.9},          {"Winnipeg", 2000.0},
};

// Extract city name from TNTP path: "some/dir/CityName_net.tntp" -> "CityName"
static std::string tntp_city_name(const std::string& net_path) {
    auto slash = net_path.rfind('/');
    auto start = (slash == std::string::npos) ? 0 : slash + 1;
    auto underscore = net_path.find('_', start);
    if (underscore == std::string::npos) {
        return "";
    }
    return net_path.substr(start, underscore - start);
}

// Derive trips path from net path, preserving .gz suffix:
//   CityName_net.tntp    -> CityName_trips.tntp
//   CityName_net.tntp.gz -> CityName_trips.tntp.gz
static std::string tntp_trips_path(const std::string& net_path) {
    std::string suffix = "_net.tntp.gz";
    auto pos = net_path.rfind(suffix);
    if (pos != std::string::npos) {
        return net_path.substr(0, pos) + "_trips.tntp.gz";
    }
    suffix = "_net.tntp";
    pos = net_path.rfind(suffix);
    if (pos != std::string::npos) {
        return net_path.substr(0, pos) + "_trips.tntp";
    }
    return "";
}

static bool is_tntp_net(const std::string& path) {
    return path.ends_with("_net.tntp") || path.ends_with("_net.tntp.gz");
}

static void print_usage(std::FILE* out) {
    std::fprintf(
        out,
        "Usage: mcfcg_cli <instance_path> [options]\n"
        "Options:\n"
        "  --formulation path|tree  (default: path)\n"
        "  --max-iters N            (default: 10000)\n"
        "  --trips PATH             TNTP trips file\n"
        "  --coef N                 TNTP demand coefficient\n"
        "  --threads N              Number of pricing threads (default: 0=auto, 1=serial)\n"
        "  --batch-size N           Pricing batch size (0=all)\n"
        "  --solver NAME            LP solver: highs (default), copt, cuopt, mosek\n"
        "  --copt-gpu-mode N        COPT barrier execution: 0=CPU, 1/2=GPU (default: 2).\n"
        "                           Only affects --solver copt.\n"
        "  --write-mps PATH         Write the compact source-based LP as MPS to\n"
        "                           PATH (gz if .gz) and exit; does not solve.\n"
        "  --verbose-solver         Enable LP solver's own log output\n"
        "  --stats-only             Print instance cost-scale + slack-ceiling stats\n"
        "                           (CSV) for the chosen --solver and exit; no solve.\n"
        "  --col-age-limit N        Purge columns after N idle iters (default: 5, 0=off)\n"
        "  --row-inactivity N       Purge cap rows after N idle iters (default: 5, 0=off)\n"
        "  --neg-rc-tol X           Reduced cost tolerance (default: -1e-3)\n"
        "  --time-limit S           Wall-clock budget in seconds (0=off); stops the CG\n"
        "                           loop at the next iter and reports best UB/LB.\n"
        "  --strategy S             pricer-light (default) or pricer-heavy\n"
        "  --bounded-pricing        Stop each source's A* once the duals prove no\n"
        "                           improving column remains (default: off)\n"
        "  -h, --help               Print this help message and exit.\n");
}

// Map a --solver name to params.solver_factory. Returns false (after printing a
// diagnostic) if the name is unknown or the requested backend was not compiled
// in. Extracted from main() so the argument-parsing loop stays within the
// clang-tidy cognitive-complexity budget — solver selection is its own concern.
static bool configure_solver(const std::string& solver, bool verbose_solver, int copt_gpu_mode,
                             mcfcg::CGParams& params) {
    if (solver == "cuopt") {
#ifdef MCFCG_USE_CUOPT
        params.solver_factory = [verbose_solver] {
            return mcfcg::create_cuopt_solver(verbose_solver);
        };
#else
        std::fprintf(stderr, "cuOpt not available. Rebuild with -DMCFCG_USE_CUOPT=ON.\n");
        return false;
#endif
    } else if (solver == "copt") {
#ifdef MCFCG_USE_COPT
        params.solver_factory = [verbose_solver, copt_gpu_mode] {
            return mcfcg::create_copt_solver(verbose_solver, copt_gpu_mode);
        };
#else
        (void)copt_gpu_mode;
        std::fprintf(stderr, "COPT not available. Rebuild with -DMCFCG_USE_COPT=ON.\n");
        return false;
#endif
    } else if (solver == "mosek") {
#ifdef MCFCG_USE_MOSEK
        params.solver_factory = [verbose_solver] {
            return mcfcg::create_mosek_solver(verbose_solver);
        };
#else
        std::fprintf(stderr, "MOSEK not available. Rebuild with -DMCFCG_USE_MOSEK=ON.\n");
        return false;
#endif
    } else if (solver == "highs") {
        params.solver_factory = [verbose_solver] {
            return mcfcg::create_lp_solver(verbose_solver);
        };
    } else {
        std::fprintf(stderr, "Unknown solver '%s'. Valid: highs, copt, cuopt, mosek\n",
                     solver.c_str());
        return false;
    }
    return true;
}

namespace {

// One flat argument-parsing dispatch followed by the solve it configures.
// The one part that was its own concern -- backend selection -- is already
// extracted as configure_solver above; splitting the rest would only pass
// the same dozen option locals between functions.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
int run_cli(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(stderr);
        return EXIT_FAILURE;
    }

    std::string instance_path;
    std::string formulation = "path";
    uint32_t max_iters = 10000;
    uint32_t num_threads = 0;
    uint32_t batch_size = 0;
    std::string solver = "highs";
    int copt_gpu_mode = -1;  // -1 = COPT default (GPU barrier, mode 2)
    std::string trips_path;
    std::string write_mps_path;
    double coef = 0.0;
    bool verbose_solver = false;
    bool stats_only = false;
    mcfcg::CGParams params;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(stdout);
            return EXIT_SUCCESS;
        }
        if (std::strcmp(argv[i], "--verbose-solver") == 0) {
            verbose_solver = true;
            continue;
        }
        if (std::strcmp(argv[i], "--stats-only") == 0) {
            stats_only = true;
            continue;
        }
        // Value-less flags must be handled before the "requires a value" guard
        // below, which would otherwise reject them in trailing position.
        if (std::strcmp(argv[i], "--bounded-pricing") == 0) {
            params.bounded_pricing = true;
            continue;
        }
        if (i == 1) {
            instance_path = argv[i];
            continue;
        }
        if (i + 1 >= argc) {
            std::fprintf(stderr, "Option '%s' requires a value.\n", argv[i]);
            return EXIT_FAILURE;
        }
        if (std::strcmp(argv[i], "--formulation") == 0) {
            formulation = argv[++i];
        } else if (std::strcmp(argv[i], "--max-iters") == 0) {
            max_iters = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--trips") == 0) {
            trips_path = argv[++i];
        } else if (std::strcmp(argv[i], "--write-mps") == 0) {
            write_mps_path = argv[++i];
        } else if (std::strcmp(argv[i], "--coef") == 0) {
            coef = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--threads") == 0) {
            num_threads = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--batch-size") == 0) {
            batch_size = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--solver") == 0) {
            solver = argv[++i];
        } else if (std::strcmp(argv[i], "--copt-gpu-mode") == 0) {
            copt_gpu_mode = std::atoi(argv[++i]);
            if (copt_gpu_mode < 0 || copt_gpu_mode > 2) {
                std::fprintf(stderr, "Invalid --copt-gpu-mode '%s'. Valid: 0 (CPU), 1, 2 (GPU).\n",
                             argv[i]);
                return EXIT_FAILURE;
            }
        } else if (std::strcmp(argv[i], "--col-age-limit") == 0) {
            params.col_age_limit = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--row-inactivity") == 0) {
            params.row_inactivity_threshold = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "--neg-rc-tol") == 0) {
            params.neg_rc_tol = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--time-limit") == 0) {
            params.time_limit_seconds = std::atof(argv[++i]);
        } else if (std::strcmp(argv[i], "--strategy") == 0) {
            std::string s = argv[++i];
            if (s == "pricer-heavy") {
                params.strategy = mcfcg::CGStrategy::PricerHeavy;
            } else if (s == "pricer-light") {
                params.strategy = mcfcg::CGStrategy::PricerLight;
            } else {
                std::fprintf(stderr, "Unknown strategy '%s'. Valid: pricer-light, pricer-heavy\n",
                             s.c_str());
                return EXIT_FAILURE;
            }
        } else {
            std::fprintf(stderr, "Unknown option '%s'.\n", argv[i]);
            print_usage(stderr);
            return EXIT_FAILURE;
        }
    }

    mcfcg::Instance inst;

    if (is_tntp_net(instance_path)) {
        // TNTP format — auto-detect trips path and coefficient
        if (trips_path.empty()) {
            trips_path = tntp_trips_path(instance_path);
        }
        if (coef == 0.0) {
            auto city = tntp_city_name(instance_path);
            auto it = TNTP_COEFS.find(city);
            if (it != TNTP_COEFS.end()) {
                coef = it->second;
            } else {
                std::fprintf(stderr, "Unknown TNTP city '%s' — use --coef\n", city.c_str());
                return EXIT_FAILURE;
            }
        }
        std::fprintf(stderr, "TNTP: net=%s trips=%s coef=%.1f\n", instance_path.c_str(),
                     trips_path.c_str(), coef);
        inst = mcfcg::read_tntp(instance_path, trips_path, coef);
    } else {
        inst = mcfcg::read_commalab(instance_path);
    }

    std::fprintf(stderr,
                 "Instance: %u vertices, %u arcs, %zu commodities, "
                 "%zu sources\n",
                 inst.graph.num_vertices(), inst.graph.num_arcs(), inst.commodities.size(),
                 inst.sources.size());

    // Echo the selected backend + formulation up front (the backend also prints
    // its own [lp-config] banner at construction once the solve starts).
    if (solver == "copt") {
        std::fprintf(stderr, "Solver: copt (gpu-mode %d), formulation: %s\n",
                     copt_gpu_mode < 0 ? 2 : copt_gpu_mode, formulation.c_str());
    } else {
        std::fprintf(stderr, "Solver: %s, formulation: %s\n", solver.c_str(), formulation.c_str());
    }

    // --stats-only short-circuits: report the instance cost scale and the
    // slack-cost ceiling (the headroom diagnostic) and exit before any solve.
    // Builds the chosen master via init() so the reported ceiling /
    // slack_cost_upper_bound come from the same code path a real solve uses
    // (single source of truth).  The master is built with the SELECTED backend
    // so slack_cost_ceiling reflects that backend's LPSolver::max_slack_cost
    // (e.g. 1e9 for MOSEK/COPT vs 1e7 for HiGHS).  init() only adds rows/cols —
    // it never optimizes — but creating a non-HiGHS backend still needs its
    // license/GPU; the default (HiGHS) needs neither.
    if (stats_only) {
        if (!configure_solver(solver, verbose_solver, copt_gpu_mode, params)) {
            return EXIT_FAILURE;
        }
        double sum_arc_costs = 0.0;
        double max_arc = 0.0;
        for (uint32_t a : inst.graph.arcs()) {
            double c = inst.cost[a];
            sum_arc_costs += c;
            max_arc = std::max(max_arc, c);
        }
        double max_src_demand_sum = 0.0;
        double total_demand = 0.0;
        for (const auto& src : inst.sources) {
            double sum = 0.0;
            for (uint32_t k : src.commodity_indices) {
                sum += inst.commodities[k].demand;
            }
            max_src_demand_sum = std::max(max_src_demand_sum, sum);
            total_demand += sum;
        }
        double ub_val = 0.0;
        double ceiling = 0.0;
        mcfcg::SlackMode mode = mcfcg::SlackMode::CommodityRows;
        if (formulation == "tree") {
            mcfcg::TreeMaster master;
            master.init(inst, params.solver_factory(), nullptr, /*warm_start=*/true);
            ub_val = master.slack_cost_upper_bound_value();
            ceiling = master.slack_cost_ceiling();
            mode = master.slack_mode();
        } else {
            mcfcg::PathMaster master;
            master.init(inst, params.solver_factory(), nullptr, /*warm_start=*/true);
            ub_val = master.slack_cost_upper_bound_value();
            ceiling = master.slack_cost_ceiling();
            mode = master.slack_mode();
        }
        // Compact source LP dimensions come from source_lp_size, which sizes the
        // model without building it — so these columns are reported even for the
        // instances --write-mps refuses as too large, which is exactly where a
        // hand-derived formula would otherwise have to be trusted.
        const mcfcg::SourceLPSize slp_size = mcfcg::source_lp_size(inst);
        std::printf(
            "instance,formulation,vertices,arcs,capacitated_arcs,self_loop_arcs,"
            "commodities,sources,max_arc_cost,"
            "sum_arc_costs,max_src_demand_sum,total_demand,slack_cost_upper_bound,"
            "slack_cost_ceiling,slack_mode,source_lp_cols,source_lp_rows,source_lp_nnz\n");
        std::printf("%s,%s,%u,%u,%u,%u,%zu,%zu,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%s,%llu,%llu,%llu\n",
                    instance_path.c_str(), formulation.c_str(), inst.graph.num_vertices(),
                    inst.graph.num_arcs(), slp_size.capacitated_arcs, slp_size.self_loop_arcs,
                    inst.commodities.size(), inst.sources.size(), max_arc, sum_arc_costs,
                    max_src_demand_sum, total_demand, ub_val, ceiling,
                    mode == mcfcg::SlackMode::EdgeRows ? "EdgeRows" : "CommodityRows",
                    static_cast<unsigned long long>(slp_size.cols),
                    static_cast<unsigned long long>(slp_size.rows),
                    static_cast<unsigned long long>(slp_size.nnz));
        return EXIT_SUCCESS;
    }

    // --write-mps short-circuits: emit the compact source-based LP and exit
    // without configuring a solver or running column generation.
    if (!write_mps_path.empty()) {
        if (!mcfcg::write_source_mps(inst, write_mps_path)) {
            std::fprintf(stderr, "Failed to write MPS to %s\n", write_mps_path.c_str());
            return EXIT_FAILURE;
        }
        std::fprintf(stderr, "Wrote compact source LP to %s\n", write_mps_path.c_str());
        return EXIT_SUCCESS;
    }

    uint32_t effective_threads =
        num_threads == 0 ? std::max(1U, std::thread::hardware_concurrency()) : num_threads;
    std::fprintf(stderr, "Pricing threads: %u%s\n", effective_threads,
                 num_threads == 0 ? " (auto)" : "");

    // Print the slack-placement selection in the preamble (before LP
    // backend init chatter).  Single source of truth is select_slack_mode
    // in master_base.h — MasterBase::init calls the same helper.
    uint32_t num_capped_arcs = mcfcg::count_capacitated_arcs(inst);
    auto num_structural_rows = static_cast<uint32_t>(
        formulation == "tree" ? inst.sources.size() : inst.commodities.size());
    mcfcg::SlackMode chosen_mode = mcfcg::select_slack_mode(num_capped_arcs, num_structural_rows);
    std::fprintf(stderr, "Slack mode: %s (struct=%u, capped_arcs=%u)\n",
                 chosen_mode == mcfcg::SlackMode::EdgeRows ? "EdgeRows" : "CommodityRows",
                 num_structural_rows, num_capped_arcs);

    params.max_iterations = max_iters;
    params.num_threads = num_threads;
    params.pricing_batch_size = batch_size;
    params.verbosity = mcfcg::Verbosity::Iteration;

    if (!configure_solver(solver, verbose_solver, copt_gpu_mode, params)) {
        return EXIT_FAILURE;
    }

    auto start = std::chrono::steady_clock::now();
    mcfcg::CGResult result;

    if (formulation == "path") {
        result = mcfcg::solve_path_cg(inst, params);
    } else if (formulation == "tree") {
        result = mcfcg::solve_tree_cg(inst, params);
    } else {
        std::fprintf(stderr, "Unknown formulation: %s\n", formulation.c_str());
        return EXIT_FAILURE;
    }

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    // Provenance for the bounded-pricing evaluation (gh #41): a wall-clock
    // difference means nothing without the fire rate behind it.  Printed in
    // both arms — `priced` is the total pricing work the run did, and comparing
    // it on vs off is what separates "the bound saved search" from "the bound
    // changed how many sources get priced".  stderr, next to the other
    // one-line banners, so the stdout CSV contract is untouched.
    double fire_rate = result.priced_sources > 0
                           ? 100.0 * static_cast<double>(result.bounded_sources) /
                                 static_cast<double>(result.priced_sources)
                           : 0.0;
    std::fprintf(stderr, "[bounded-pricing] enabled=%d fired=%llu priced=%llu rate=%.1f%%\n",
                 params.bounded_pricing ? 1 : 0,
                 static_cast<unsigned long long>(result.bounded_sources),
                 static_cast<unsigned long long>(result.priced_sources), fire_rate);

    // CSV output.  lower_bound is empty when LB tracking never fired
    // (best_lb stayed at -INF) so downstream parsers don't have to
    // handle "-inf" literals — empty cell reads as NaN in pandas.
    std::printf(
        "instance,formulation,iterations,columns,objective,lower_bound,"
        "optimal,time,time_lp,time_pricing,time_separation\n");
    if (std::isfinite(result.lower_bound)) {
        std::printf("%s,%s,%u,%u,%.6f,%.6f,%d,%.3f,%.3f,%.3f,%.3f\n", instance_path.c_str(),
                    formulation.c_str(), result.iterations, result.total_columns, result.objective,
                    result.lower_bound, result.optimal ? 1 : 0, elapsed, result.time_lp,
                    result.time_pricing, result.time_separation);
    } else {
        std::printf("%s,%s,%u,%u,%.6f,,%d,%.3f,%.3f,%.3f,%.3f\n", instance_path.c_str(),
                    formulation.c_str(), result.iterations, result.total_columns, result.objective,
                    result.optimal ? 1 : 0, elapsed, result.time_lp, result.time_pricing,
                    result.time_separation);
    }

    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char* argv[]) {
    // MasterBase::init and the instance readers throw; without this a bad
    // instance path aborts via std::terminate with no message.
    try {
        return run_cli(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
