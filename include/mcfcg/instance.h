#pragma once

#include "mcfcg/graph/static_digraph.h"
#include "mcfcg/graph/static_map.h"

#include <cstdint>
#include <string>
#include <vector>

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

// Connectivity contract: every commodity's source→sink pair should be
// reachable in `graph`.  The pricer handles a disconnected commodity
// gracefully in `CommodityRows` slack mode (the demand-row slack
// absorbs the unmet demand; the CG loop terminates with
// `optimal=false` once slacks saturate at the ceiling), but in
// `EdgeRows` mode the first LP solve returns infeasible because no
// demand-row slack exists.  Callers should preprocess disconnected
// instances via `mcfcg_clean` (drops commodities whose sink is
// unreachable from their source) before handing them to the solver.
struct Instance {
    static_digraph graph;
    static_map<uint32_t, double> cost;
    static_map<uint32_t, double> capacity;
    std::vector<Commodity> commodities;
    std::vector<Source> sources;
};

// Read an instance in CommaLab/UniPi plain-numeric format (1-indexed
// vertices and arcs).  See `data/commalab/grid/format.doc` or
// `data/commalab/planar/format.doc` for the schema.  Paths ending in
// `.gz` are decompressed transparently via zlib.
Instance read_commalab(const std::string& path);
void write_commalab(const Instance& inst, const std::string& path);
std::vector<Source> group_by_source(const std::vector<Commodity>& commodities);

// Read an instance in TNTP transportation-network format; see
// https://github.com/bstabler/TransportationNetworks for the spec.
// Arc cost is taken from `free_flow_time`; raw OD demands are divided
// by `demand_coef` (city-specific, see TNTP_COEFS in src/main.cpp).
// `.gz` paths are decompressed transparently via zlib.
Instance read_tntp(const std::string& net_path, const std::string& trips_path, double demand_coef);

}  // namespace mcfcg
