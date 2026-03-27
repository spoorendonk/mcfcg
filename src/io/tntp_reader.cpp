#include "mcfcg/instance.h"

#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mcfcg/graph/static_digraph_builder.h"
#include "mcfcg/io/gz_util.h"

namespace mcfcg {

// Parse *_net.tntp: skip metadata, read arcs.
// Format: init_node term_node capacity length free_flow_time b power speed toll
// type ;
static void parse_net(std::istream & file, const std::string & path,
                      uint32_t & num_nodes, std::vector<uint32_t> & src,
                      std::vector<uint32_t> & dst, std::vector<double> & cost,
                      std::vector<double> & capacity) {
    std::string line;
    num_nodes = 0;
    bool past_metadata = false;
    bool past_header = false;

    while (std::getline(file, line)) {
        // Parse metadata
        if (!past_metadata) {
            if (line.find("<NUMBER OF NODES>") != std::string::npos) {
                std::istringstream iss(line.substr(line.find('>') + 1));
                iss >> num_nodes;
            }
            if (line.find("<END OF METADATA>") != std::string::npos) {
                past_metadata = true;
            }
            continue;
        }

        // Skip blank lines and the ~ header line
        if (line.empty() || line[0] == '\n' || line[0] == '\r')
            continue;
        // Skip lines starting with ~ (column header)
        bool is_tilde = false;
        for (char c : line) {
            if (c == ' ' || c == '\t')
                continue;
            if (c == '~')
                is_tilde = true;
            break;
        }
        if (is_tilde) {
            past_header = true;
            continue;
        }
        if (!past_header)
            continue;

        // Parse arc line: init_node term_node capacity length free_flow_time
        // ...;
        std::istringstream iss(line);
        uint32_t s, t;
        double cap, length, fft;
        if (!(iss >> s >> t >> cap >> length >> fft))
            continue;

        // 1-indexed -> 0-indexed
        src.push_back(s - 1);
        dst.push_back(t - 1);
        cost.push_back(fft);
        capacity.push_back(cap);
    }

    if (num_nodes == 0)
        throw std::runtime_error("Failed to parse number of nodes from: " +
                                 path);
}

// Parse *_trips.tntp: skip metadata, read Origin blocks with dest:demand pairs.
static void parse_trips(std::istream & file, double demand_coef,
                        std::vector<Commodity> & commodities) {
    std::string line;
    bool past_metadata = false;
    uint32_t current_origin = 0;

    while (std::getline(file, line)) {
        if (!past_metadata) {
            if (line.find("<END OF METADATA>") != std::string::npos)
                past_metadata = true;
            continue;
        }

        // Check for "Origin N"
        if (line.find("Origin") != std::string::npos) {
            std::istringstream iss(line.substr(line.find("Origin") + 6));
            iss >> current_origin;
            continue;
        }

        if (current_origin == 0)
            continue;

        // Parse dest : demand ; pairs
        // Format: "  59 : 14 ;  72 : 5 ;"  or with floats "  59 : 0.14;"
        std::istringstream iss(line);
        std::string token;
        while (iss >> token) {
            // token should be the destination number
            uint32_t dest = 0;
            try {
                dest = static_cast<uint32_t>(std::stoul(token));
            } catch (...) {
                continue;
            }

            // Next should be ":"
            if (!(iss >> token) || token != ":")
                continue;

            // Next is the demand value (possibly with trailing ;)
            if (!(iss >> token))
                continue;

            // Remove trailing semicolons
            while (!token.empty() && token.back() == ';')
                token.pop_back();
            if (token.empty())
                continue;

            double demand = 0.0;
            try {
                demand = std::stod(token);
            } catch (...) {
                continue;
            }

            if (demand <= 0.0)
                continue;

            // 1-indexed -> 0-indexed, apply coefficient
            commodities.push_back(
                {current_origin - 1, dest - 1, demand / demand_coef});
        }
    }
}

static std::unique_ptr<std::istream> open_tntp(const std::string & path,
                                               std::string & storage) {
    if (ends_with_gz(path)) {
        storage = decompress_gz(path);
        return std::make_unique<std::istringstream>(std::move(storage));
    }
    auto f = std::make_unique<std::ifstream>(path);
    if (!f->is_open())
        throw std::runtime_error("Cannot open file: " + path);
    return f;
}

Instance read_tntp(const std::string & net_path, const std::string & trips_path,
                   double demand_coef) {
    uint32_t num_nodes = 0;
    std::vector<uint32_t> src, dst;
    std::vector<double> arc_cost, arc_cap;

    {
        std::string buf;
        auto stream = open_tntp(net_path, buf);
        parse_net(*stream, net_path, num_nodes, src, dst, arc_cost, arc_cap);
    }

    std::vector<Commodity> commodities;
    {
        std::string buf;
        auto stream = open_tntp(trips_path, buf);
        parse_trips(*stream, demand_coef, commodities);
    }

    if (src.empty())
        throw std::runtime_error("No arcs parsed from: " + net_path);
    if (commodities.empty())
        throw std::runtime_error("No commodities parsed from: " + trips_path);

    static_digraph_builder<double, double> builder(num_nodes);
    for (size_t i = 0; i < src.size(); ++i) {
        builder.add_arc(src[i], dst[i], arc_cost[i], arc_cap[i]);
    }
    auto [graph, cost_map, cap_map] = builder.build();

    auto sources = group_by_source(commodities);

    return Instance{
        std::move(graph),       std::move(cost_map), std::move(cap_map),
        std::move(commodities), std::move(sources),
    };
}

}  // namespace mcfcg
