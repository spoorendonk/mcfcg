@.devkit/standards/cpp.md

Navigation: LSP → narrow Grep → sliced Read. See `.devkit/standards/common.md` "Code Navigation".

# Project: mcfcg

## Build & Test

```clean
rm -rf build
```

```build
cmake -B build -DCMAKE_INSTALL_MESSAGE=LAZY && cmake --build build -j$(nproc)
```

```test
GTEST_BRIEF=1 ctest --test-dir build --output-on-failure --progress -j$(nproc)
```

Run a single test by name:
```
./build/mcfcg_tests --gtest_filter='PathCGSingleSource.OptimalObjective'
./build/mcfcg_integration_tests --gtest_filter='GridCorrectness.Grid1'
```

## What This Is

Column generation solver for minimum-cost multicommodity flow (MCF). Supports path-based and tree-based Dantzig-Wolfe decompositions.

## Architecture

The CG loop (`include/mcfcg/cg/cg_loop.h`) is a single template function `solve_cg<Master, Pricer, GetDuals>` shared by both formulations (`GetDuals` is the per-formulation callable that extracts the pricing dual vector from the master — demand duals for path, convexity duals for tree). It drives the interaction between three components:

1. **Master problem** (`include/mcfcg/cg/master.h`, `tree_master.h`) — restricted LP with incremental column/row addition. Path formulation has one demand row per commodity; tree formulation has one convexity row per source. Capacity rows are lazy (added on violation). Slack placement is selected per instance by `MasterBase::init` (`enum SlackMode` in `master_base.h`): `CommodityRows` puts one slack per structural row at init with coeff +1; `EdgeRows` pairs a slack with each lazily-added capacity row with coeff -1. The selector picks whichever side has fewer rows, so the slack count is `min(num_structural_rows, num_capacitated_arcs)`. EdgeRows requires `CGParams::warm_start=true` (no init-time slacks means the LP is only feasible once warm-start seeds at least one column per structural row) — `init` throws on violation. Every slack starts at initial cost = max arc cost, grown by `MasterBase::bump_active_slacks` every CG iteration while any slack is basic — the LP pivots each slack out once its cost exceeds whatever column serves the row. Bumps happen at end-of-iter, before purges (bumps read `get_primals()` which HiGHS/COPT invalidate on delete; a bump-to-fixed-point loop wrapping `solve()` would also not terminate when lazy capacities force a slack basic until pricing adds a new column). `SLACK_BUMP_FACTOR` in `cg_loop.h` plus a per-instance ceiling on `SlackState::cost_ceiling` bound the growth. `MasterBase::init` sets the ceiling to `clamp(10 * Derived::slack_cost_upper_bound(), 1e8, 1e9)` — path-master returns `num_vertices * max_arc_cost`; tree-master multiplies by the largest per-source demand sum so the slack can always out-price the costliest real tree column while staying below the HiGHS dual-simplex ratio-test failure regime.

2. **Pricer** (`include/mcfcg/cg/pricer.h`, `tree_pricer.h`) — computes reduced costs using dual values from the master. Runs Dijkstra from each source with clamped integer-scaled arc costs (SCALE=1e9). Source postponement skips sources that produced no negative-RC column last round. Path pricer extracts one column per commodity; tree pricer builds a single tree column per source aggregating demand-weighted arc flows. Unreachable commodity sinks (A* heap exhausts without settling the sink) are skipped: path pricer emits columns for the reachable commodities only; tree pricer emits a partial tree column covering reachable sinks only. Graceful only in `CommodityRows` slack mode (demand-row slacks absorb unmet demand) — in `EdgeRows` mode there is no demand slack and a disconnected source→sink surfaces as LP infeasibility. Preprocess disconnected instances via `mcfcg_clean` before solving.

3. **LP backend** (`include/mcfcg/lp/lp_solver.h`) — abstract interface with three implementations: HiGHS (default, always available, incremental), cuOpt (optional, GPU barrier, rebuild-from-scratch), and COPT (optional, GPU barrier, incremental). Enable optional backends with `-DMCFCG_USE_CUOPT=ON` or `-DMCFCG_USE_COPT=ON`. CSC format for columns, CSR for rows. The `starts` convention is uniform: `add_cols` and `add_rows` both require `starts.size() == n+1` with `starts[n] == values.size()`. The CG pricer uses a single `neg_rc_tol` (default `NEG_RC_TOL = -1e-3`, see `include/mcfcg/util/tolerances.h`); no backend overrides it today.

### Strategy presets
`CGParams::strategy` (enum `CGStrategy` in `include/mcfcg/cg/path_cg.h`) is a high-level preset that bundles several lower-level CGParams knobs. The preset is consulted in `solve_cg` to compute `effective_*` locals at the top of the function — the bundle supersedes the corresponding raw fields where documented. Today there are two values, named for how expensive the pricer is relative to the master: `PricerLight` (default — pricer is cheap, so push lots of cols/rows at the master: large col cap, column aging on, cuts and cols added in the same iteration) and `PricerHeavy` (pricer is expensive, so throttle it: cap cols/iter at num_entities, disable col aging, force the source pricing filter, defer pricing in iterations that added new lazy capacity rows). When adding a new tunable, decide whether it should be a raw CGParams field, part of an existing bundle, or motivate a new strategy value.

### Graph layer
`include/mcfcg/graph/` — CSR static digraph with typed arc/vertex maps (`static_map`), d-ary min-heap, Dijkstra/A* borrowing a `dijkstra_workspace` for reusable memory. Compile-time traits control which workspace fields are written (`if constexpr`).

### I/O
Two instance formats: CommaLab/UniPi plain-numeric (1-indexed) and TNTP transportation networks. Both support `.gz` via zlib. TNTP uses `free_flow_time` as cost and divides demands by a city-specific coefficient.

## Instance Data

Four families in `data/`: grid, planar (CommaLab format), transportation (TNTP format, gz-compressed), intermodal (CommaLab format, gz-compressed). Integration tests run correctness checks against paper reference values for small instances from each family.

## Key Design Decisions

- **Integer-scaled Dijkstra costs**: Reduced costs are scaled by 1e9 and clamped to non-negative int64_t. Negative reduced costs (attractive arcs) become 0-length. True reduced cost is recomputed in floating point after path extraction.
- **Lazy capacity constraints**: The master starts with demand/convexity rows only. Capacity rows are added when flow exceeds capacity by >1e-6, avoiding a huge initial LP.
- **No duplicate columns**: The pricer must never generate duplicate columns. If duplicates appear, it indicates a bug in pricing or reduced-cost computation.
- **Dense capacity duals `mu`**: Capacity duals are stored as a dense arc-indexed `static_map<uint32_t, double>` (default 0.0 for arcs without a capacity row), cached on `MasterBase` and reset incrementally each iteration. This lets `compute_rc` run a contiguous `cost[a] - mu[a]` loop that auto-vectorizes under `-march=native`. Build with `-DMCFCG_NATIVE_ARCH=OFF` to disable host CPU tuning for portable binaries.
