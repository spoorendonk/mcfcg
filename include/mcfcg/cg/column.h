#pragma once

#include <cstdint>
#include <vector>

namespace mcfcg {

struct Column {
    double cost;
    uint32_t commodity;
    std::vector<uint32_t> arcs;  // path arcs from source to sink
};

}  // namespace mcfcg
