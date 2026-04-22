#include "mcfcg/instance.h"

#include <cmath>
#include <fstream>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace mcfcg {

namespace {

// Emit integer-valued doubles compactly ("1" not "1"), fractional
// doubles with full float precision.  Matches the compact formatting
// used by scripts/generate_instances.py so round-tripping a fractional
// cost (e.g. 0.5 walking-arc cost in Lienkamp's intermodal temp
// networks) preserves the value.  std::llround would drop the
// fraction and — for an uncapped arc with capacity=+INF — return an
// implementation-defined sentinel (LLONG_MIN on glibc) that later
// reads back as a nonsensical binding capacity.
void write_double(std::ostream& os, double value) {
    if (std::isinf(value)) {
        // -1 is the uncapacitated sentinel (matches Lienkamp &
        // Schiffer start_run.py::write_instance).  read_commalab maps
        // negative caps back to INF on read.
        os << -1;
        return;
    }
    auto as_int = static_cast<long long>(value);
    if (static_cast<double>(as_int) == value) {
        os << as_int;
    } else {
        auto old_prec = os.precision(std::numeric_limits<double>::max_digits10);
        os << value;
        os.precision(old_prec);
    }
}

}  // namespace

void write_commalab(const Instance& inst, const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }

    file << inst.graph.num_vertices() << '\n';
    file << inst.graph.num_arcs() << '\n';
    file << inst.commodities.size() << '\n';

    for (uint32_t a : inst.graph.arcs()) {
        file << (inst.graph.arc_source(a) + 1) << ' ' << (inst.graph.arc_target(a) + 1) << ' ';
        write_double(file, inst.cost[a]);
        file << ' ';
        write_double(file, inst.capacity[a]);
        file << '\n';
    }

    for (auto& k : inst.commodities) {
        file << (k.source + 1) << ' ' << (k.sink + 1) << ' ';
        write_double(file, k.demand);
        file << '\n';
    }
}

}  // namespace mcfcg
