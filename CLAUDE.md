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

**Default to the tree formulation.** The path master has one demand row per commodity, which on high-commodity instances (transportation up to ~1.15M OD pairs, intermodal, planar2500) blows the master up to ~1M rows and the barrier LP solve dominates wall-clock — e.g. Philadelphia path is 6,436s (99% in the LP) vs 810s tree, ChicagoRegional 5,946s vs 97s (61×), Sydney 7,315s vs 72s (102×). Tree has one convexity row per source instead, keeping the master small. Objectives match between formulations; tree even solves instances path times out on (Austin). On grid/planar the two are roughly equal (tree occasionally slightly slower, e.g. planar2500). `scripts/benchmark_solvers.py` defaults every family to tree; pass `--formulations path,tree` to compare.

## Architecture

The CG loop (`include/mcfcg/cg/cg_loop.h`) is a single template function `solve_cg<Master, Pricer, GetDuals>` shared by both formulations (`GetDuals` is the per-formulation callable that extracts the pricing dual vector from the master — demand duals for path, convexity duals for tree). It drives the interaction between three components:

1. **Master problem** (`include/mcfcg/cg/master.h`, `tree_master.h`) — restricted LP with incremental column/row addition. Path formulation has one demand row per commodity; tree formulation has one convexity row per source. Capacity rows are lazy (added on violation). Slack placement is selected per instance by `MasterBase::init` (`enum SlackMode` in `master_base.h`): `CommodityRows` puts one slack per structural row at init with coeff +1; `EdgeRows` pairs a slack with each lazily-added capacity row with coeff -1. The selector picks whichever side has fewer rows, so the slack count is `min(num_structural_rows, num_capacitated_arcs)`. EdgeRows requires `CGParams::warm_start=true` (no init-time slacks means the LP is only feasible once warm-start seeds at least one column per structural row) — `init` throws on violation. Every slack starts at initial cost = max arc cost, grown by `MasterBase::bump_active_slacks` every CG iteration while any slack is basic — the LP pivots each slack out once its cost exceeds whatever column serves the row. Bumps happen at end-of-iter, before purges (bumps read `get_primals()` which COPT invalidates on delete; a bump-to-fixed-point loop wrapping `solve()` would also not terminate when lazy capacities force a slack basic until pricing adds a new column). `SLACK_BUMP_FACTOR` in `cg_loop.h` plus a per-instance ceiling on `SlackState::cost_ceiling` bound the growth. `MasterBase::init` creates the LP backend first, then sets the ceiling to `clamp(10 * Derived::slack_cost_upper_bound(), 1e6, LPSolver::max_slack_cost())` — path-master returns `num_vertices * max_arc_cost`; tree-master multiplies by the largest per-source demand sum. The ceiling is **backend-aware**: `max_slack_cost()` is `1e7` for HiGHS (the base default) and the cuOpt barrier (whose IPM stalls on a wide slack-vs-real dynamic range — slacks at 1e9 once caused a 100–1000s pathology on transportation), and `1e9` for the robust MOSEK/COPT barriers. **The cap matters for correctness, not just speed:** if the real per-row column cost exceeds the ceiling, that row's slack is cheaper than any real column and stays basic forever — no slack-free upper bound is ever reached and the penalized LP objective can sit *below* the true optimum (observed on `planar2500` tree, ~1.7e7/source vs a 1e7 cap: it never converged). Raising MOSEK/COPT to 1e9 lets those slacks price out; planar2500 tree then converges in ~140 iters / ~1700s. The 1e6 floor prevents the clamp from inverting on small instances. A reliable runtime signal that the ceiling is too low: the (valid) `best_lb` exceeds the current penalized LP objective while slacks are basic — the penalty isn't dominating real routing cost. The worst-case `slack_cost_upper_bound()` is far looser than typical per-row cost (esp. tree, where it is `demand_sum × full-path`), so it is a poor a-priori risk predictor; `scripts/slack_headroom.py` reports it plus a realistic per-row proxy (`optimum/total_demand` for path, `optimum/sources` for tree), but the proxy is an average and can miss expensive outlier rows.

2. **Pricer** (`include/mcfcg/cg/pricer.h`, `tree_pricer.h`) — computes reduced costs using dual values from the master. Runs Dijkstra from each source with clamped integer-scaled arc costs (SCALE=1e9). Source postponement skips sources that produced no negative-RC column last round. Path pricer extracts one column per commodity; tree pricer builds a single tree column per source aggregating demand-weighted arc flows. Unreachable commodity sinks (A* heap exhausts without settling the sink) are skipped: path pricer emits columns for the reachable commodities only; tree pricer emits a partial tree column covering reachable sinks only. Graceful only in `CommodityRows` slack mode (demand-row slacks absorb unmet demand) — in `EdgeRows` mode there is no demand slack and a disconnected source→sink surfaces as LP infeasibility. Preprocess disconnected instances via `mcfcg_clean` before solving.

3. **LP backend** (`include/mcfcg/lp/lp_solver.h`) — abstract interface. HiGHS is the always-available default (`--solver highs`; compiled in unconditionally, incremental, CPU — needs no license or GPU, so it works on non-GPU hosts). It uses the **HiPO interior-point method** by default (`MCFCG_HIGHS_SOLVER=simplex|ipm|hipo|pdlp` overrides). HiPO is ~2× faster than HiGHS' simplex default on these CG masters (grid15 tree: 70s vs 148s) and matches the MOSEK/COPT barrier objectives (~7–12× slower than those barriers, but license/GPU-free). HiPO had a bug: it discarded a near-optimal solution as an `internal error` when its IPX refinement step failed on ill-conditioned path masters (wide objective range from the slack penalty costs). We carry a local fix in `cmake/patches/highs-hipo-refine-status.patch`, applied to the FetchContent'd HiGHS source via `PATCH_COMMAND` in `CMakeLists.txt` (idempotent); see the gh issue tracking the upstream report. With the patch, HiPO solves those masters correctly and the full suite passes with no simplex fallback. Optional backends: cuOpt (GPU barrier, rebuild-from-scratch), COPT (barrier, GPU by default — overridable via `MCFCG_COPT_GPUMODE`, incremental), and MOSEK (CPU barrier, incremental); enable with `-DMCFCG_USE_CUOPT=ON`, `-DMCFCG_USE_COPT=ON`, or `-DMCFCG_USE_MOSEK=ON` (select at runtime with `--solver`). CSC format for columns, CSR for rows. The `starts` convention is uniform: `add_cols` and `add_rows` both require `starts.size() == n+1` with `starts[n] == values.size()`. The CG pricer uses a single `neg_rc_tol` (default `NEG_RC_TOL = -1e-3`, see `include/mcfcg/util/tolerances.h`); no backend overrides it today.

### Bounds and early termination
The CG loop tracks two monotone bounds. `best_ub` is the minimum LP objective over iterations whose LP primal is MCF-feasible (no slack basic, no new capacity row found this iter). `best_lb` is the maximum **π-free capacity-relaxation Lagrangian** lower bound over iterations where the pricer visited every source (`pricer.priced_all()` — no source postponed; a `max_cols` break that fires exactly on sweep completion still counts): `LB = Σ_a cap_a·μ_a + Σ_k d_k · sp_k(c−μ) − rounding_margin`, where `sp_k(c−μ)` is the reduced-cost shortest path the pricer found (`pricer.lagrangian_path_sum()`, accumulated WITHOUT subtracting the structural dual π_k), `Σ cap·μ` is `master.compute_capacity_dual_term(mu)` snapshotted before separation mutates the row set, and the rounding margin bounds the scale-integer vs true-RC gap. This is the textbook Lagrangian relaxation of the capacity (coupling) constraints — valid for **any** μ≤0 by weak duality, hence **independent of slack/feasibility state**, so unlike the old `dual_obj + Σ d_k·min(0,rc*_k)` form it is no longer gated on `num_active_slacks==0` and advances every priced iteration (the structural duals cancel analytically; accumulating sp_k directly avoids the catastrophic cancellation a basic slack's huge π_k would cause). Path weights by `d_k` (demand); tree's convexity RHS=1 collapses the weight. Early exit fires when `best_ub − best_lb < RELATIVE_FEAS_TOL · max(1, |best_ub|)` and the gap is non-negative. The `#slk` log column counts basic slacks — when non-zero the LP obj is a feasibility penalty and `best_ub` is not updated (but `best_lb` now is). Because the LB tracks from iteration 1, the gap closes as soon as the last slack clears, which is why early termination requires the ceiling to be high enough for every slack to price out (see the slack-ceiling note above).

### Strategy presets
`CGParams::strategy` (enum `CGStrategy` in `include/mcfcg/cg/path_cg.h`) is a high-level preset that bundles several lower-level CGParams knobs. `solve_cg` reads `params.strategy` and computes `effective_*` locals at the top of the function; the bundle supersedes the corresponding raw fields where documented. Two values today, named for how expensive the pricer is relative to the master:

- `PricerLight` (default): pricer is cheap, so push lots of cols/rows at the master. Large col cap, column aging on, cuts and cols added in the same iteration.
- `PricerHeavy`: pricer is expensive, so throttle it:
  - cap cols/iter at `num_entities`
  - disable column aging (overrides `CGParams::col_age_limit`)
  - force the source pricing filter on
  - defer pricing in iterations that added lazy capacity rows
  - partial pricing via `compute_partial_pricing_batch_size` (in path_cg.h): when `n_sources > pool_threads` the batch is `max(pool_threads, n_sources/4)` so the column cap short-circuits mid-sweep and `pricer._last_source_idx` parks for the next iter; when `n_sources <= pool_threads` partial pricing can't engage (one batch covers everyone anyway) and we return 0 (single big batch) rather than pretending otherwise. Scaled by `n_sources` not `num_entities` because the pricer batches over sources, and for path `num_entities = n_commodities` typically dominates. The success path uses `pricer.clear_postponed()` (flags only, cursor survives); warm-start and pricing-exhausted branches use `reset_postponed()` (flags + cursor). Overridden if the caller sets `CGParams::pricing_batch_size > 0` explicitly.

When adding a new tunable, decide whether it should be a raw CGParams field, part of an existing bundle, or motivate a new strategy value.

### Graph layer
`include/mcfcg/graph/` — CSR static digraph with typed arc/vertex maps (`static_map`), d-ary min-heap, Dijkstra/A* borrowing a `dijkstra_workspace` for reusable memory. Compile-time traits control which workspace fields are written (`if constexpr`).

### I/O
Two instance formats: CommaLab/UniPi plain-numeric (1-indexed) and TNTP transportation networks. Both support `.gz` via zlib. TNTP uses `free_flow_time` as cost and divides demands by a city-specific coefficient. CommaLab uses a negative capacity as the uncapacitated sentinel — `read_commalab` maps it to `+INF` so `count_capacitated_arcs` excludes those arcs and no capacity row is lazily added; `write_commalab` emits `-1` for `isinf(cap)`. This matches Lienkamp & Schiffer's `start_run.py::write_instance` for intermodal instances (`cap >= 9999 -> -1`) and is generated by `scripts/generate_instances.py`.

### Intermodal pitfalls
Intermodal instances (SUBWAY / BUS / SBT, generated from the Lienkamp & Schiffer repo) are much larger than grid/planar — hundreds of thousands of vertices, millions of arcs after time-expansion.
- **Use the tree formulation with `PricerHeavy`.** `solve_tree_cg` is robust on BUS/SBT under COPT and MOSEK, both PricerLight and PricerHeavy — converging to the paper's LP optimum on BUS-2632. (Under MOSEK the path formulation also solves intermodal in comparable time, but the per-commodity master makes it the wrong default on high-commodity instances generally — see "Default to the tree formulation" above.) Integration tests for intermodal use `solve_tree_cg` + `PricerHeavy` via the `solve_intermodal_and_check` helper.
- **Writer precision matters.** `write_commalab` emits `-1` for `isinf(capacity)` and preserves fractional costs (no `llround`) — mirroring Lienkamp & Schiffer's `start_run.py::write_instance`. Round-tripping a cleaned instance through the older truncating writer would drop fractional walking-arc costs (0.5 → 0, 1.5 → 2) and produce a platform-dependent `LLONG_MIN` sentinel that the reader accidentally still maps to INF; the `RoundTripFractionalCostAndInfCap` test guards this.

### cuOpt GPU pitfalls
- **Path master OOMs the GPU barrier on high-commodity instances.** The path formulation has one demand row per commodity, so instances with ~80k+ commodities (e.g. `planar2500` at 81,430) build a large ADAT normal-equations system. cuOpt's RMM pool plus the cuDSS factorization can exceed available VRAM (observed: `cudss_device_alloc` failure on a 16 GiB card with a desktop session holding ~6.5 GiB). **Use the tree formulation** — one convexity row per source (2,500 for planar2500) shrinks the barrier system ~30× and solves correctly under cuOpt. This generalizes the intermodal "use tree" rule to any high-commodity instance. (COPT's CPU barrier handles the 81k-row path master fine, ~1000s.)
- **A failed cuOpt barrier returns garbage as "solved".** On a cuDSS device-alloc / numerical-error termination, the cuOpt backend currently returns a non-optimal incumbent instead of erroring, so CG can converge on a ~100×-wrong objective (nondeterministically reported `optimal=1`). Tracked in #33. If a cuOpt run produces an objective far above the COPT/tree value, suspect a swallowed barrier failure, not a real optimum.

## Instance Data

Four families in `data/`: grid, planar (CommaLab format), transportation (TNTP format, gz-compressed), intermodal (CommaLab format, gz-compressed). Integration tests run correctness checks against paper reference values for small instances from each family.

## Key Design Decisions

- **Integer-scaled Dijkstra costs**: Reduced costs are scaled by 1e9 and clamped to non-negative int64_t. Negative reduced costs (attractive arcs) become 0-length. True reduced cost is recomputed in floating point after path extraction.
- **Lazy capacity constraints**: The master starts with demand/convexity rows only. Capacity rows are added when flow exceeds capacity by >1e-6, avoiding a huge initial LP.
- **No duplicate columns**: The pricer must never generate duplicate columns. If duplicates appear, it indicates a bug in pricing or reduced-cost computation.
- **Dense capacity duals `mu`**: Capacity duals are stored as a dense arc-indexed `static_map<uint32_t, double>` (default 0.0 for arcs without a capacity row), cached on `MasterBase` and reset incrementally each iteration. This lets `compute_rc` run a contiguous `cost[a] - mu[a]` loop that auto-vectorizes under `-march=native`. Build with `-DMCFCG_NATIVE_ARCH=OFF` to disable host CPU tuning for portable binaries.
