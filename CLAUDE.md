# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@.dev-std/standards/cpp.md

## Build & Test

```clean
rm -rf build
```

```build
cmake -B build && cmake --build build -j$(nproc)
```

```test
ctest --test-dir build --output-on-failure -j$(nproc)
```

Run a single test by name:
```
./build/mcfcg_tests --gtest_filter='PathCGSingleSource.OptimalObjective'
./build/mcfcg_integration_tests --gtest_filter='GridCorrectness.Grid1'
```

## What This Is

Column generation solver for minimum-cost multicommodity flow (MCF). Supports path-based and tree-based Dantzig-Wolfe decompositions. Accompanies a paper targeting Mathematical Programming Computation (MPC) that reverses the 30-year finding of Jones et al. (1993) that path decomposition beats tree decomposition — at modern scale with |K| >> |S|, the tree formulation wins.

## Architecture

The CG loop (`src/cg/path_cg.cpp`, `src/cg/tree_cg.cpp`) drives the interaction between three components:

1. **Master problem** (`include/mcfcg/cg/master.h`, `tree_master.h`) — restricted LP with incremental column/row addition. Path formulation has one demand row per commodity; tree formulation has one convexity row per source. Both use BIG_M slack variables and lazy capacity constraints (added on violation).

2. **Pricer** (`include/mcfcg/cg/pricer.h`, `tree_pricer.h`) — computes reduced costs using dual values from the master. Runs Dijkstra from each source with clamped integer-scaled arc costs (SCALE=1e9). Source postponement skips sources that produced no negative-RC column last round. Path pricer extracts one column per commodity; tree pricer builds a single tree column per source aggregating demand-weighted arc flows.

3. **LP backend** (`include/mcfcg/lp/lp_solver.h`, `src/lp/highs_solver.cpp`) — abstract interface, currently HiGHS. CSC format for columns, CSR for rows. The `starts` convention differs: `add_cols` callers include a sentinel, `add_rows` callers don't (the HiGHS adapter appends it).

### Graph layer
`include/mcfcg/graph/` — CSR static digraph with typed arc/vertex maps (`static_map`), d-ary min-heap, Dijkstra with compile-time trait selection (store distances, paths, or neither via `[[no_unique_address]]`).

### I/O
Two instance formats: CommaLab/UniPi plain-numeric (1-indexed) and TNTP transportation networks. Both support `.gz` via zlib. TNTP uses `free_flow_time` as cost and divides demands by a city-specific coefficient.

## Instance Data

Four families in `data/`: grid, planar (CommaLab format), transportation (TNTP format, gz-compressed), intermodal (CommaLab format, gz-compressed). Integration tests run correctness checks against paper reference values for small instances from each family.

## Key Design Decisions

- **Integer-scaled Dijkstra costs**: Reduced costs are scaled by 1e9 and clamped to non-negative int64_t. Negative reduced costs (attractive arcs) become 0-length. True reduced cost is recomputed in floating point after path extraction.
- **Lazy capacity constraints**: The master starts with demand/convexity rows only. Capacity rows are added when flow exceeds capacity by >1e-6, avoiding a huge initial LP.
- **Forbidden-arc fallback**: When all new columns are duplicates, the CG loop re-prices with binding-capacity arcs forbidden, forcing alternative paths.
