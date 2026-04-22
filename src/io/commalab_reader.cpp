#include "mcfcg/graph/static_digraph_builder.h"
#include "mcfcg/instance.h"
#include "mcfcg/io/gz_util.h"
#include "mcfcg/util/limits.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mcfcg {

// CommaLab/UniPi plain-numeric format:
//   Line 1: <num_vertices>
//   Line 2: <num_arcs>
//   Line 3: <num_commodities>
//   Next M lines: <src> <dst> <cost> <capacity>   (1-indexed)
//   Next K lines: <origin> <destination> <demand>  (1-indexed)
//
// A negative capacity encodes an uncapacitated arc and maps to INF, so
// count_capacitated_arcs excludes it and no capacity row is lazily
// added.  Mirrors the `-1` sentinel that scripts/generate_instances.py
// emits for intermodal instances (and Lienkamp & Schiffer's upstream
// start_run.py emits for `arc.capacity >= 9999`).
static Instance parse_commalab(std::istream& file, const std::string& path) {
    uint32_t num_vertices = 0, num_arcs = 0, num_commodities = 0;
    file >> num_vertices >> num_arcs >> num_commodities;
    if (file.fail()) {
        throw std::runtime_error("Failed to read header from: " + path);
    }

    struct ArcData {
        uint32_t src, dst;
        double cost, cap;
    };
    std::vector<ArcData> arc_data;
    arc_data.reserve(num_arcs);

    for (uint32_t i = 0; i < num_arcs; ++i) {
        uint32_t src, dst;
        double cost, cap;
        file >> src >> dst >> cost >> cap;
        if (cap < 0.0) {
            cap = INF;
        }
        arc_data.push_back({src - 1, dst - 1, cost, cap});
    }
    if (file.fail()) {
        throw std::runtime_error("Failed to read arcs from: " + path);
    }

    std::vector<Commodity> commodities;
    commodities.reserve(num_commodities);

    for (uint32_t i = 0; i < num_commodities; ++i) {
        uint32_t src, dst;
        double demand;
        file >> src >> dst >> demand;
        commodities.push_back({src - 1, dst - 1, demand});
    }
    if (file.fail()) {
        throw std::runtime_error("Failed to read commodities from: " + path);
    }

    // Build graph
    static_digraph_builder<double, double> builder(num_vertices);
    for (auto& a : arc_data) {
        builder.add_arc(a.src, a.dst, a.cost, a.cap);
    }
    auto [graph, cost_map, cap_map] = builder.build();

    auto sources = group_by_source(commodities);

    return Instance{
        std::move(graph),       std::move(cost_map), std::move(cap_map),
        std::move(commodities), std::move(sources),
    };
}

Instance read_commalab(const std::string& path) {
    if (ends_with_gz(path)) {
        auto data = decompress_gz(path);
        std::istringstream iss(std::move(data));
        return parse_commalab(iss, path);
    }
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    return parse_commalab(file, path);
}

std::vector<Source> group_by_source(const std::vector<Commodity>& commodities) {
    std::unordered_map<uint32_t, uint32_t> source_index;
    std::vector<Source> sources;
    for (uint32_t k = 0; k < commodities.size(); ++k) {
        uint32_t s = commodities[k].source;
        auto it = source_index.find(s);
        if (it == source_index.end()) {
            source_index[s] = static_cast<uint32_t>(sources.size());
            sources.push_back({s, {k}});
        } else {
            sources[it->second].commodity_indices.push_back(k);
        }
    }
    return sources;
}

}  // namespace mcfcg
