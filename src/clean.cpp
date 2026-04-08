#include "mcfcg/graph/dijkstra.h"
#include "mcfcg/graph/dijkstra_workspace.h"
#include "mcfcg/instance.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: mcfcg_clean <input> [--output cleaned.txt]\n");
        return EXIT_FAILURE;
    }

    std::string input_path = argv[1];
    std::string output_path;

    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc)
            break;
        if (std::strcmp(argv[i], "--output") == 0)
            output_path = argv[i + 1];
    }

    if (output_path.empty()) {
        // Default: input stem + "_cleaned.txt"
        auto dot = input_path.rfind('.');
        if (dot != std::string::npos)
            output_path = input_path.substr(0, dot) + "_cleaned.txt";
        else
            output_path = input_path + "_cleaned.txt";
    }

    auto inst = mcfcg::read_commalab(input_path);

    std::fprintf(stderr, "Input: %u vertices, %u arcs, %zu commodities\n",
                 inst.graph.num_vertices(), inst.graph.num_arcs(), inst.commodities.size());

    // Build unit-weight arc map for reachability via Dijkstra
    auto lengths = inst.graph.create_arc_map<int64_t>();
    for (uint32_t a : inst.graph.arcs()) {
        lengths[a] = 1;
    }

    std::vector<mcfcg::Commodity> kept;
    uint32_t removed = 0;

    mcfcg::dijkstra_workspace ws(inst.graph.num_vertices());
    mcfcg::dijkstra<mcfcg::dijkstra_store_distances> dijk(inst.graph, lengths, ws);

    for (auto& src_group : inst.sources) {
        dijk.reset();
        dijk.add_source(src_group.vertex);
        dijk.run();

        for (uint32_t k_idx : src_group.commodity_indices) {
            auto& k = inst.commodities[k_idx];
            if (dijk.visited(k.sink)) {
                kept.push_back(k);
            } else {
                ++removed;
            }
        }
    }

    std::fprintf(stderr, "Removed %u unreachable commodities, kept %zu\n", removed, kept.size());

    // Rebuild instance with kept commodities
    auto sources = mcfcg::group_by_source(kept);
    mcfcg::Instance cleaned{
        std::move(inst.graph), std::move(inst.cost), std::move(inst.capacity),
        std::move(kept),       std::move(sources),
    };

    mcfcg::write_commalab(cleaned, output_path);
    std::fprintf(stderr, "Written to %s\n", output_path.c_str());

    return EXIT_SUCCESS;
}
