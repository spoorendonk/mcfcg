#!/usr/bin/env python3
"""Generate MMCF instances from the tumBAIS intermodal repo in CommaLab format.

Usage:
    python3 generate_instances.py --repo data/intermodal-repo --output data/intermodal/raw

Requires:
    - The cloned repo with LFS data files (network_pt_road.xml, schedule.xml, etc.)
    - Python dependencies: networkx, pandas, numpy, geopandas, shapely, lxml
    - Note: Gurobi is NOT needed (we only build instances, not solve them)

Reimplements build_instance() from the upstream start_run.py, importing only
the Generator modules (no Gurobi/Solver dependency). Skips solver-only steps
(incidence matrix, heuristics).
"""

import argparse
import os
import sys

# Instance configs: passenger count -> subset fraction
# From the upstream start_run.py
SUBWAY_CONFIGS = {
    132: 0.06,
    308: 0.14,
    486: 0.22,
    662: 0.30,
}
SUBWAY_TIME_WINDOW = (420, 435)

BUS_CONFIGS = {
    2632: 0.1,
    7896: 0.3,
    13160: 0.5,
    18424: 0.7,
    23688: 0.9,
}
BUS_TIME_WINDOW = (420, 540)

SBT_CONFIGS = {
    6255: 0.1,
    18765: 0.3,
    31275: 0.5,
    43785: 0.7,
    56295: 0.9,
}
SBT_TIME_WINDOW = (420, 540)


def build_instance(seed, capacity, modes, subset, time_window):
    """Build instance using upstream Generator modules. No Gurobi needed."""
    import Generator.instance_generator as ig
    import Generator.XML_parser as xml
    import Generator.geodata as gd

    tw = ig.TimeWindow(time_window[0], time_window[1])
    modes_network = []
    modes_trips = []
    if "b" in modes:
        modes_network.append("bus")
        modes_trips.append("bus")
    if "t" in modes:
        modes_network.append("tram")
        modes_trips.append("tramOrMetro")
    if "s" in modes:
        modes_network.append("subway")
        if "tramOrMetro" not in modes_trips:
            modes_trips.append("tramOrMetro")
    if "r" in modes:
        modes_network.append("rail")
        modes_trips.append("train")

    network_file = xml.read_xml("data/network_pt_road.xml")
    city = gd.get_city_gdf("Munich")
    schedule_file = "data/schedule.xml"
    trip_file = "data/trips_munich_public_transport.csv"

    instance = ig.Instance(
        tw, modes_network, modes_trips, network_file, schedule_file, city, subset
    )
    instance.get_flat_network()
    instance.join_nodes("data/gtfs/stops.txt")
    instance.build_temporal_network()
    instance.add_waiting_layers()
    instance.add_transit_arcs()
    instance.add_walking_arcs()
    instance.get_trips(trip_file, seed)
    instance.set_arc_capacities(capacity)
    # Skip calculate_incidence_matrix, build_heuristic_dist, get_heuristic_fast
    # — those are only needed by the solver, not for instance export.
    return instance


def write_commalab(path, num_vertices, arcs, commodities):
    """Write instance in CommaLab/UniPi plain-numeric format (1-indexed)."""
    with open(path, "w") as f:
        f.write(f"{num_vertices}\n")
        f.write(f"{len(arcs)}\n")
        f.write(f"{len(commodities)}\n")
        for src, dst, cost, cap in arcs:
            f.write(f"{src + 1} {dst + 1} {cost} {cap}\n")
        for src, dst, demand in commodities:
            f.write(f"{src + 1} {dst + 1} {demand}\n")


def generate_family(prefix, mode_str, configs, time_window, seeds, output, gg):
    """Generate all instances for one mode family."""
    for passengers, subset in sorted(configs.items()):
        for seed in seeds:
            name = f"{prefix}-{passengers}-{seed}"
            out_path = os.path.join(output, f"{name}.txt")

            if os.path.exists(out_path):
                print(f"  {name}: already exists, skipping")
                continue

            print(f"  {name}: generating ...")
            try:
                instance = build_instance(
                    seed, 1.0, mode_str, subset, time_window
                )
                graph = gg.build_graph(instance)

                # Remap node IDs to contiguous 0..N-1
                nodes = list(graph.nodes())
                node_map = {n: i for i, n in enumerate(nodes)}
                num_vertices = len(nodes)

                # Extract arcs
                arcs = []
                for u, v, data in graph.edges(data=True):
                    cost = int(round(data.get("weight", 1)))
                    cap = int(round(data.get("cap", 1000)))
                    arcs.append((node_map[u], node_map[v], cost, cap))

                # Extract commodities: sources/sinks are dicts keyed by trip ID
                commodities = []
                for trip_id in instance.sources:
                    src = instance.sources[trip_id]
                    sink = instance.sinks[trip_id]
                    if src in node_map and sink in node_map:
                        commodities.append((node_map[src], node_map[sink], 1))

                if not commodities:
                    print(f"  {name}: no commodities found, skipping")
                    continue

                write_commalab(out_path, num_vertices, arcs, commodities)
                print(
                    f"  {name}: {num_vertices} vertices, {len(arcs)} arcs, "
                    f"{len(commodities)} commodities"
                )

            except Exception as e:
                print(f"  {name}: error - {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                continue


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo", required=True, help="Path to cloned intermodal repo"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for raw instances"
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Random seeds (default: 0)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["subway", "bus", "sbt"],
        help="Mode families to generate (default: subway bus sbt)",
    )
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    output = os.path.abspath(args.output)

    # The upstream code uses relative paths like 'data/network_pt_road.xml',
    # so we must chdir to the repo root.
    os.chdir(repo)
    sys.path.insert(0, repo)

    import Generator.graph_generator as gg

    os.makedirs(output, exist_ok=True)

    families = {
        "subway": ("SUBWAY", "s", SUBWAY_CONFIGS, SUBWAY_TIME_WINDOW),
        "bus": ("BUS", "b", BUS_CONFIGS, BUS_TIME_WINDOW),
        "sbt": ("SBT", "sbt", SBT_CONFIGS, SBT_TIME_WINDOW),
    }

    for mode in args.modes:
        if mode not in families:
            print(f"Unknown mode: {mode}", file=sys.stderr)
            continue
        prefix, mode_str, configs, tw = families[mode]
        print(f"\n=== {prefix} ===")
        generate_family(prefix, mode_str, configs, tw, args.seeds, output, gg)

    print(f"\nDone. Raw instances in {output}")


if __name__ == "__main__":
    main()
