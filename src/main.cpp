#include "mcfcg/cg/path_cg.h"
#include "mcfcg/cg/tree_cg.h"
#include "mcfcg/instance.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

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
    if (underscore == std::string::npos)
        return "";
    return net_path.substr(start, underscore - start);
}

// Derive trips path from net path, preserving .gz suffix:
//   CityName_net.tntp    -> CityName_trips.tntp
//   CityName_net.tntp.gz -> CityName_trips.tntp.gz
static std::string tntp_trips_path(const std::string& net_path) {
    std::string suffix = "_net.tntp.gz";
    auto pos = net_path.rfind(suffix);
    if (pos != std::string::npos)
        return net_path.substr(0, pos) + "_trips.tntp.gz";
    suffix = "_net.tntp";
    pos = net_path.rfind(suffix);
    if (pos != std::string::npos)
        return net_path.substr(0, pos) + "_trips.tntp";
    return "";
}

static bool ends_with(const std::string& s, const std::string& suffix) {
    if (suffix.size() > s.size())
        return false;
    return s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static bool is_tntp_net(const std::string& path) {
    return ends_with(path, "_net.tntp") || ends_with(path, "_net.tntp.gz");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "Usage: mcfcg_cli <instance_path> [options]\n"
                     "Options:\n"
                     "  --formulation path|tree  (default: path)\n"
                     "  --max-iters N            (default: 10000)\n"
                     "  --trips PATH             TNTP trips file\n"
                     "  --coef N                 TNTP demand coefficient\n"
                     "  --threads N              Number of pricing threads (0=auto)\n"
                     "  --batch-size N           Pricing batch size (0=all)\n");
        return EXIT_FAILURE;
    }

    std::string instance_path = argv[1];
    std::string formulation = "path";
    uint32_t max_iters = 10000;
    uint32_t num_threads = 1;
    uint32_t batch_size = 0;
    std::string trips_path;
    double coef = 0.0;

    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc)
            break;
        if (std::strcmp(argv[i], "--formulation") == 0)
            formulation = argv[i + 1];
        else if (std::strcmp(argv[i], "--max-iters") == 0)
            max_iters = static_cast<uint32_t>(std::atoi(argv[i + 1]));
        else if (std::strcmp(argv[i], "--trips") == 0)
            trips_path = argv[i + 1];
        else if (std::strcmp(argv[i], "--coef") == 0)
            coef = std::atof(argv[i + 1]);
        else if (std::strcmp(argv[i], "--threads") == 0)
            num_threads = static_cast<uint32_t>(std::atoi(argv[i + 1]));
        else if (std::strcmp(argv[i], "--batch-size") == 0)
            batch_size = static_cast<uint32_t>(std::atoi(argv[i + 1]));
    }

    mcfcg::Instance inst;

    if (is_tntp_net(instance_path)) {
        // TNTP format — auto-detect trips path and coefficient
        if (trips_path.empty())
            trips_path = tntp_trips_path(instance_path);
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

    mcfcg::CGParams params;
    params.max_iterations = max_iters;
    params.num_threads = num_threads;
    params.pricing_batch_size = batch_size;
    params.verbosity = mcfcg::Verbosity::Iteration;

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

    // CSV output
    std::printf(
        "instance,formulation,iterations,columns,objective,"
        "optimal,time,time_lp,time_pricing,time_separation\n");
    std::printf("%s,%s,%u,%u,%.6f,%d,%.3f,%.3f,%.3f,%.3f\n", instance_path.c_str(),
                formulation.c_str(), result.iterations, result.total_columns, result.objective,
                result.optimal ? 1 : 0, elapsed, result.time_lp, result.time_pricing,
                result.time_separation);

    return EXIT_SUCCESS;
}
