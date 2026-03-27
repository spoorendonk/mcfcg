#include "mcfcg/instance.h"

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace mcfcg {

void write_commalab(const Instance & inst, const std::string & path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }

    file << inst.graph.num_vertices() << '\n';
    file << inst.graph.num_arcs() << '\n';
    file << inst.commodities.size() << '\n';

    for (uint32_t a : inst.graph.arcs()) {
        file << (inst.graph.arc_source(a) + 1) << ' '
             << (inst.graph.arc_target(a) + 1) << ' '
             << std::llround(inst.cost[a]) << ' '
             << std::llround(inst.capacity[a]) << '\n';
    }

    for (auto & k : inst.commodities) {
        file << (k.source + 1) << ' ' << (k.sink + 1) << ' '
             << std::llround(k.demand) << '\n';
    }
}

}  // namespace mcfcg
