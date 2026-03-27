#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mcfcg/graph/static_digraph.h"
#include "mcfcg/graph/static_map.h"

namespace mcfcg {

struct Commodity {
    uint32_t source;
    uint32_t sink;
    double demand;
};

struct Source {
    uint32_t vertex;
    std::vector<uint32_t> commodity_indices;
};

struct Instance {
    static_digraph graph;
    static_map<uint32_t, double> cost;
    static_map<uint32_t, double> capacity;
    std::vector<Commodity> commodities;
    std::vector<Source> sources;
};

Instance read_commalab(const std::string & path);
void write_commalab(const Instance & inst, const std::string & path);
std::vector<Source> group_by_source(const std::vector<Commodity> & commodities);

Instance read_tntp(const std::string & net_path, const std::string & trips_path,
                   double demand_coef);

}  // namespace mcfcg
