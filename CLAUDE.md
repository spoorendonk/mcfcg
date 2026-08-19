Navigation: LSP → narrow Grep → sliced Read.

# Standards

## Communication Style

Be terse. No preamble. No filler.

## Code Navigation

Prefer narrow queries over full-file reads:

1. **LSP** for symbol questions — `goToDefinition`, `hover`, `documentSymbol`, `workspaceSymbol`. Use before `Read`.
2. **Grep** with `-n` and a small `head_limit` (start at 20); raise only if inconclusive.
3. **Read** with `offset`/`limit` for a slice around the hit. Full-file `Read` is fine under ~200 lines or when structure matters.

Know the symbol → LSP. Know a string, not its location → Grep. Full-file Read is the last mile. This is a preference, not a prohibition: shelling out to `grep`/`rg` is fine when the built-in can't do the job (filtering a pipe like `git log | grep`, or a session without the `Grep` tool). What matters is bounding output, not which binary produces it.

Setup: install `clangd-lsp@claude-plugins-official` plus `clangd` (`apt install clangd`, or from LLVM). `.clangd` points at `build/compile_commands.json`, produced by `CMAKE_EXPORT_COMPILE_COMMANDS ON`.

## C++

- Target C++23. Use modern features (`std::expected`, concepts, ranges, `constexpr`).
- Use `#pragma once`. Minimize header includes; forward-declare where possible.
- **Formatting** is Google, via `.clang-format`. **Naming is not Google** — it is STL-flavoured, and `.clang-tidy` enforces it:

  | Kind | Style | Example |
  |---|---|---|
  | Functions (free and member) | `lower_case` | `solve_cg`, `compute_rc` |
  | Locals, parameters, public members | `lower_case` | `pricer_heavy`, `demand` |
  | Private members | **leading** `_` | `_source_arcs`, `_last_source_idx` |
  | Constants (global, class, static) | `UPPER_CASE` | `NEG_RC_TOL`, `MAX_BOUND` |
  | Enums and enumerators | `CamelCase` | `SlackMode::EdgeRows` |
  | Namespaces | `lower_case` | `mcfcg::detail` |
  | Macros | `UPPER_CASE` | |
  | Files | `snake_case.h` / `.cpp` | |

  Protected members are deliberately **unconstrained**: gtest fixtures use bare `protected:` members, so the `_` prefix is a library-code convention `.clang-tidy` does not enforce.

  Type names are deliberately two-tier and **not** enforced: `CamelCase` for the domain layer (`MasterBase`, `LPSolver`, `TreePricer`), `snake_case` for STL-like containers and graph algorithms (`static_map`, `d_ary_heap`, `thread_pool`, `dijkstra`). clang-tidy cannot express "either", so `ClassCase` is intentionally absent from the config — match the layer you are in.

## clang-tidy

`cmake --build build --target tidy` — the gate. Stamped per translation unit: ~30s at `-j` from cold, a no-op when nothing changed, and it needs only `cmake -B build`, not a compiled tree. `tidy-fix` applies fix-its serially and refuses to run outside a git checkout or with a dirty worktree under `include/`, `src/` or `test/`. Narrow a bulk fix with `-DMCFCG_TIDY_FIX_CHECKS='-*,some-check'`.

`WarningsAsErrors` covers `clang-diagnostic-*`, `bugprone-*`, `performance-*` and `readability-identifier-naming`; those block. Everything else — notably `readability-function-cognitive-complexity` — is advisory. Suppress an advisory finding with `NOLINTNEXTLINE(check)` plus a comment saying why, never by widening the config. `NOLINTNEXTLINE` applies to the *literal* next line: put the prose above it and the pragma last, and for a template put it between the `template<...>` line and the signature.

**Invoking clang-tidy by hand needs two flags the config cannot supply.** `HeaderFilterRegex` is silently ignored when clang-tidy auto-discovers `.clang-tidy`: `clang-tidy -p build src/cg/tree_cg.cpp` reports 2 diagnostics where the same command with `--config-file=.clang-tidy` reports 481 — every header, which is where nearly all of this codebase lives. A bare `.*` header filter is also wrong even with the config file, because it matches `build/_deps/highs-src/**`. So:

```
clang-tidy -p build --config-file=.clang-tidy \
  --header-filter="^$PWD/(include|src|test)/" --quiet <file.cpp>
```

Prefer the `tidy` target, which does both for you.

Two bulk-fix traps. `run-clang-tidy -fix` was measured reporting a set of warnings then silently applying only a fraction — use the serial `tidy-fix` target. And never run `clang-tidy --fix` in parallel here: every template lives in a header under `include/mcfcg`, so concurrent processes rewrite the same file and corrupt it.

## Complexity

When a complexity warning fires, don't extract methods mechanically. Ask what the independent responsibilities are and split along those boundaries. If the function is genuinely complex because the domain is, add a comment explaining why and suppress the warning.

## CMake

- `set(CMAKE_EXPORT_COMPILE_COMMANDS ON)` for clang-tidy.
- Use FetchContent for dependencies.
- A single root `CMakeLists.txt`; per-directory files would only add indirection.

## Testing (GoogleTest)

- Test files: `<module>_test.cpp` in `test/`.
- Name tests descriptively: `TEST_F(SolverTest, ReturnsOptimalForFeasibleInput)`.
- Terse output: `GTEST_BRIEF=1` prints only failures, `ctest --progress` collapses the running list, `CMAKE_INSTALL_MESSAGE=LAZY` suppresses install chatter. Don't remove these.

## Development Workflow

```
plan (non-trivial) → implement → test → push to main
```

Nothing formats on save — there are no Claude Code hooks. `clang-format` runs at commit time: `pre-commit` formats the staged C++, applies safe `clang-tidy` fixes, and re-stages the result, so what you commit is canonical even though the file you just edited is not. Don't hand-tune formatting — let the hook normalize it, or run it yourself:

```
clang-format -i <files>                       # normalize in place
clang-format --dry-run --Werror <files>       # check only, non-zero if unformatted
git diff --name-only --diff-filter=d HEAD | grep -E '\.(cpp|h)$' | xargs -r clang-format -i
```

The same step covers shell and Python via `shfmt` / `ruff`, but **neither is installed here**, so those are silent no-ops — treat Python edits as unformatted and unchecked. Run tests locally before considering work done, even on trivial-looking changes. The pre-push hook is the final gate; it too skips steps whose tool is missing, so its shellcheck, ruff and mypy passes currently report nothing rather than warn.

The hooks are **tracked in `.githooks/`** (`commit-msg`, `pre-commit`, `pre-push`, and the sourced `resolve-venv.sh`) — edit them there, not in `.git/hooks/`, which is empty of ours. Git only runs them when `core.hooksPath` points at that directory, and that setting is per-checkout config which cannot be tracked; `cmake -B build` sets it (option `MCFCG_INSTALL_GIT_HOOKS`, ON), leaving an existing `.githooks` value alone and warning rather than clobbering a hooksPath someone else set. To wire a clone up by hand: `git config core.hooksPath .githooks`. `.claude/` is still gitignored, owned by this repo, and not part of the published artifact — edit it in place; it holds only permission rules and a statusline; a new Claude Code hook must read its file path and command from the hook JSON **on stdin**, never `$CLAUDE_FILE_PATH` (unset by Claude Code — a hook reading it no-ops silently rather than failing). The clang-tidy gate deliberately lives outside the hooks, as the tracked `tidy` target plus the CI lint job, so a fresh clone can still run it — via CI, or `--target tidy` by hand. **Never `git push --no-verify` or `git commit --no-verify`** unless asked; a failing hook is a signal, so fix the root cause.

## Git Workflow

Trunk-based, linear history on main. Commit directly to main and push when local gates pass.

Feature branches are optional for larger changes: always branch from main (`git checkout main && git pull` first), never from another feature branch, keep them short-lived, and rebase or squash merge — no merge commits on main.

After a successful push:
- **Close any gh issue the work resolved**: `gh issue close <num> -c "<one-line note>"`, for every issue the push covers.
- **Delete the feature branch** if one was used: `git branch -d <branch>`, plus `git push origin --delete <branch>` if pushed.

## Commit Messages

Conventional Commits; the commit-msg hook enforces format.

- `type: description` or `type(scope): description`
- Types: `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `perf`, `chore`, `build`, `ci`
- Subject ≤72 chars. Focus on **why**, not what.

## Issue Tracking

GitHub Issues, via the `gh` CLI.

- **Default to HTTPS** for GitHub remotes, not SSH.
- **Read an issue** with `gh issue view <num> --json title,body,labels,state,comments`; plain `gh issue view <num>` is deprecated for programmatic use.
- Don't defer work into a new issue unless it is substantial. Fix small follow-ups inline or leave them alone.

Issues get picked up cold, in fresh sessions, often by an agent with no access to this machine. So: keep the body **self-contained** (problem, motivation, acceptance criteria, repro steps); use **no local references** (`/home/user/...`, "see my other checkout" — dead links in a fresh session); prefer **stable external links** (GitHub permalinks, papers, RFCs); and **describe local code by concept, not path**, hinting that the agent can search under `..`, `../..`, or `~/code/`.

## Working Rules

- **CLAUDE.md discipline.** When Claude gets something wrong, fix CLAUDE.md in the same commit. It's a living document — update it whenever better instructions would have prevented the mistake.
- **Follow the agreed plan.** If a plan should change, stop and discuss — don't silently diverge. Same outside a written plan: if the current approach isn't working, say so rather than quietly switching strategies. Implement everything specified; no TODO placeholders or stubs unless asked.
- **Match references exactly.** Implementing from papers, pseudocode, or open source: no early exits, iteration limits, size caps, or "optimization" shortcuts that change behaviour. Introduce heuristic approximations only when asked. Implement the edge cases rather than simplifying them away. When in doubt, be faithful and let tests verify.
- **Don't invent APIs.** Verify functions, flags, and methods exist before using them.
- **Don't ignore type errors.** If mypy/clang-tidy flags something, fix the root cause — don't suppress.
- **Don't use deprecated patterns.** Check current docs, not training data.
- **Performance matters.** Most of this is solvers: profile before micro-optimizing, but don't sacrifice perf for "clean code".

# Project: mcfcg

Column generation solver for minimum-cost multicommodity flow (MCF), with path-based and tree-based Dantzig-Wolfe decompositions.

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

Single test by name:
```
./build/mcfcg_tests --gtest_filter='PathCGSingleSource.OptimalObjective'
./build/mcfcg_integration_tests --gtest_filter='GridCorrectness.Grid1'
```

`scripts/` has a stdlib-unittest tier in `test/python/`, run by ctest as `Python.Scripts`. It exists because those scripts carry the provenance of every committed result, and their failure mode is a silently blank or mislabelled column rather than a crash. Add tests as `test/python/<module>_test.py` — the `_test.py` suffix matches the C++ convention and is why ctest passes `-p "*_test.py"`; unittest's default `test*.py` would discover nothing. Files must sit directly in `test/python/`: discovery does not recurse, and the directory is not a package, so `python3 -m unittest test.python.foo` will not work.

```
python3 -m unittest discover -s test/python -p '*_test.py'                # all
python3 -m unittest discover -s test/python -p '*_test.py' -k preserves   # one
```

## Default to the tree formulation

The path master has one demand row per commodity, which on high-commodity instances (transportation up to ~1.15M OD pairs, intermodal, planar2500) blows the master to ~1M rows and lets the barrier LP dominate wall clock: Philadelphia 6,436s path (99% in the LP) vs 810s tree, ChicagoRegional 5,946s vs 97s (61×), Sydney 7,315s vs 72s (102×). Tree has one convexity row per source instead. Objectives match; tree even solves instances path times out on (Austin). Grid/planar are roughly equal (tree occasionally slightly slower, e.g. planar2500). `scripts/benchmark_solvers.py` defaults every family to tree; pass `--formulations path,tree` to compare.

## Architecture

`solve_cg<Master, Pricer, GetDuals>` (`include/mcfcg/cg/cg_loop.h`) is one template function shared by both formulations, where `GetDuals` extracts the pricing dual vector from the master (demand duals for path, convexity duals for tree). It drives three components.

### 1. Master problem

`include/mcfcg/cg/master.h`, `tree_master.h` — restricted LP with incremental column/row addition. One demand row per commodity (path) or convexity row per source (tree); capacity rows are lazy, added on violation.

**Slack placement** is selected per instance by `MasterBase::init` (`enum SlackMode` in `master_base.h`):
- `CommodityRows` — one slack per structural row at init, coeff +1.
- `EdgeRows` — a slack paired with each lazily-added capacity row, coeff −1. Requires `CGParams::warm_start=true`, because with no init-time slacks the LP is only feasible once warm-start seeds at least one column per structural row; `init` throws on violation.

The selector picks whichever side has fewer rows, so slack count is `min(num_structural_rows, num_capacitated_arcs)`.

**Slack cost growth.** Each slack starts at max arc cost and is grown by `MasterBase::bump_active_slacks` every iteration while any slack is basic; the LP pivots a slack out once its cost exceeds whatever column serves the row. Bumps run at end-of-iter, before purges — they read `get_primals()`, which COPT invalidates on delete, and a bump-to-fixed-point loop around `solve()` would not terminate when lazy capacities keep a slack basic until pricing adds a column. `SLACK_BUMP_FACTOR` (`cg_loop.h`) plus `SlackState::cost_ceiling` bound the growth.

**The ceiling is backend-aware and matters for correctness, not just speed.** `MasterBase::init` creates the LP backend first, then sets `clamp(10 * Derived::slack_cost_upper_bound(), 1e6, LPSolver::max_slack_cost())` — path-master returns `num_vertices * max_arc_cost`, tree-master multiplies by the largest per-source demand sum. `max_slack_cost()` is `1e7` for HiGHS (the base default) and the cuOpt barrier, whose IPM stalls on a wide slack-vs-real dynamic range (slacks at 1e9 once caused a 100–1000s pathology on transportation), and `1e9` for the robust MOSEK/COPT barriers.

If real per-row column cost exceeds the ceiling, that row's slack is cheaper than any real column and stays basic forever: no slack-free upper bound is ever reached, and the penalized LP objective can sit *below* the true optimum. Observed on `planar2500` tree (~1.7e7/source against a 1e7 cap: never converged); at 1e9 those slacks price out and it converges in ~140 iters / ~1700s. The 1e6 floor keeps the clamp from inverting on small instances.

Runtime signal that the ceiling is too low: the (valid) `best_lb` exceeds the current penalized LP objective while slacks are basic — the penalty isn't dominating real routing cost. `slack_cost_upper_bound()` is a poor a-priori predictor, being far looser than typical per-row cost (especially tree, where it is `demand_sum × full-path`); `scripts/slack_headroom.py` reports it plus a realistic per-row proxy (`optimum/total_demand` path, `optimum/sources` tree), though that proxy is an average and can miss expensive outlier rows.

### 2. Pricer

`include/mcfcg/cg/pricer.h`, `tree_pricer.h` — reduced costs from master duals, via Dijkstra from each source with clamped integer-scaled arc costs (SCALE=1e9). Source postponement skips sources that produced no negative-RC column last round. Path pricer extracts one column per commodity; tree pricer builds one tree column per source aggregating demand-weighted arc flows.

Unreachable sinks (A* heap exhausts without settling the sink) are skipped: path emits columns for reachable commodities only, tree emits a partial tree covering reachable sinks only. Graceful **only** in `CommodityRows` mode, where demand-row slacks absorb unmet demand; in `EdgeRows` there is no demand slack and a disconnected source→sink surfaces as LP infeasibility. The tree's partial column is genuinely lossy — its master counts trees, not demand, so nothing notices the under-served commodities — tolerable only because a disconnected commodity makes the MCF infeasible anyway. Preprocess disconnected instances with `mcfcg_clean`.

#### Bounded single-source pricing — evaluated and rejected

`CGParams::bounded_pricing` / CLI `--bounded-pricing` (manuscript §3.3) stops a
source's A* once the frontier proves no negative-RC column remains, instead of
settling every sink. It is **exact** — the column set is identical bit-for-bit —
but not faster: best case **−2.4%** wall clock on intermodal (85.4% pricing share
× −2.8% per price), a wash or a loss everywhere else, because pricing share and
per-price saving are anticorrelated across families. grid/planar tree saves up to
30% per price and still *loses* 0.9% of wall clock, because pricing is 1.5% of it.
**Off by default everywhere, including the benchmark driver.** Measure it with
`--extra-args=--bounded-pricing`.

A family's raw wall-clock delta is **not** the effect: where the trajectory moves
it is ±Δiterations, running −27% to +20% across intermodal cells. Quote
`per_price_us` on cells with `traj_moved=0 AND traj_stable=1`, and quote them from
**copt-cpu** — copt-gpu runs identical trajectories and prices identical source
counts yet reports twice the saving, off an inflated `t_PR` baseline in its OFF
arm.

It is a *bound*, not a cutoff (gh #42): no incumbent is involved, and "cutoff" in
this codebase already means the reduced-cost acceptance threshold `NEG_RC_TOL`
and the LP backends' objective-cutoff parameters. The rename is **forward-only** —
nothing parses the old `[pricing-cutoff]` banner.

The evidence lives in `results/ablation/`, split into rounds named for the axis
each varies; round (a) (`families/`, gh #43) is 444 tracked logs and 74 paired
cells, re-derivable with `scripts/analyze_bounded_pricing_ablation.py` and pinned
by `CommittedAblationTest`.

**Read `results/ablation/README.md` before re-opening the question or touching
the bounded-pricing code**, and `results/ablation/families/README.md` before
citing a number. Together they carry the gain model and why it loses, the
three correctness traps the implementation must respect
(`FeatureTests.BoundedPricing*`), the two channels by which the flag shifts the
CG trajectory without changing columns, and which backends the measurement is
valid on — a single-backend result is worthless, copt-gpu cannot reproduce itself
on grid/planar or transportation, and its per-price numbers are not quotable.

### 3. LP backend

`include/mcfcg/lp/lp_solver.h` — abstract interface. CSC for columns, CSR for rows; `add_cols` and `add_rows` both require `starts.size() == n+1` with `starts[n] == values.size()`. The pricer uses a single `neg_rc_tol` (`NEG_RC_TOL = -1e-3`, `include/mcfcg/util/tolerances.h`); no backend overrides it today.

**HiGHS** is the always-available default (`--solver highs`) — compiled in unconditionally, incremental, CPU, no licence or GPU. It uses the **HiPO interior-point method** by default (`MCFCG_HIGHS_SOLVER=simplex|ipm|hipo|pdlp` overrides), ~2× faster than HiGHS' simplex default on these masters (grid15 tree: 70s vs 148s) and matching MOSEK/COPT barrier objectives, though ~7–12× slower than those barriers.

Two HiPO traps:
- It discarded a near-optimal solution as an `internal error` when its IPX refinement step failed on ill-conditioned path masters (wide objective range from slack penalty costs). We carry `cmake/patches/highs-hipo-refine-status.patch`, applied to the FetchContent'd source via `PATCH_COMMAND` (idempotent); see the gh issue tracking the upstream report. With it, the full suite passes with no simplex fallback.
- **HiPO's numerical backend (SuiteSparse AMD / METIS / SPARSEPAK-RCM / BLAS) must be linked statically.** HiGHS ≥1.15 ships it as a separate "extras" library defaulting to `BUILD_SHARED_EXTRAS_LIB=ON` (→ `#define HIGHS_SHARED_EXTRAS_LIBRARY`), resolved by loading `libhighs_extras.so` at runtime. Since we link statically (`BUILD_SHARED_LIBS OFF`) and never ship that `.so`, the extras go unresolved and HiGHS **silently falls back to dual simplex** while the `[lp-config]` banner still reads `method=hipo` — it echoes only the *requested* solver. `CMakeLists.txt` sets `BUILD_SHARED_EXTRAS_LIB OFF` to compile them in. (1.14 had no extras split; the 1.15 packaging introduced this.) **Verify with `mcfcg_cli … --verbose-solver`:** the HiGHS log on stdout must show `Running HiPO` / `IPX reports: ipm optimal`, not `Using dual simplex solver` or `features unavailable: amd, blas, metis, rcm`.

Optional backends, selected at runtime with `--solver`:
- **cuOpt** (`-DMCFCG_USE_CUOPT=ON`) — GPU barrier, **incremental delta C API by default**. `MCFCG_CUOPT_DELTA_API` defaults ON and needs the `spoorendonk/cuopt` fork's `cuopt_c_delta.h` (CMake errors with a fork/opt-out hint if absent). `-DMCFCG_CUOPT_DELTA_API=OFF` targets stock cuOpt via the rebuild-from-scratch path — a **serious** perf degradation that recreates the whole LP every iteration.
- **COPT** (`-DMCFCG_USE_COPT=ON`) — barrier, GPU by default, overridable per-run with `--copt-gpu-mode 0|1|2`, incremental.
- **MOSEK** (`-DMCFCG_USE_MOSEK=ON`) — CPU barrier, incremental.

**Pinned barrier config for fair comparison:** every backend runs presolve off, crossover off, convergence tol `BARRIER_TOL = 1e-4` (`tolerances.h`), and logs a one-line `[lp-config]` provenance banner (backend/method/exec/threads) to stderr at construction, captured in CG and benchmark logs.

**Crossover on certify.** Off per iteration for speed, but `LPSolver::solve(bool certify)` lets the loop re-request one crossover solve when it stalls — pricing exhausted but not optimal, or an interior solve spuriously infeasible after cuts. Without it, HiPO's non-vertex interior point leaves path-master demand slacks at O(tol)>0 and central duals that fail to price, so large path instances (Barcelona, Winnipeg) never certify a slack-free UB. COPT/MOSEK also run crossover (basis identification) on certify; the cuOpt GPU barrier has none, so `certify` is a no-op there — `LPSolver::certify_runs_crossover()` returns false and the loop skips the retry to avoid a redundant re-solve.

### Bounds and early termination

Two monotone bounds. `best_ub` is the minimum LP objective over iterations whose LP primal is MCF-feasible (no slack basic, no new capacity row that iteration). `best_lb` is the maximum **π-free capacity-relaxation Lagrangian** bound over iterations where the pricer visited every source (`pricer.priced_all()` — no source postponed; a `max_cols` break landing exactly on sweep completion still counts):

```
LB = Σ_a cap_a·μ_a + Σ_k d_k · sp_k(c−μ) − rounding_margin
```

`sp_k(c−μ)` is the reduced-cost shortest path the pricer found (`pricer.lagrangian_path_sum()`, accumulated WITHOUT subtracting the structural dual π_k), `Σ cap·μ` is `master.compute_capacity_dual_term(mu)` snapshotted before separation mutates the row set, and the margin bounds the scale-integer vs true-RC gap. Path weights by `d_k`; tree's convexity RHS=1 collapses the weight.

This is the textbook Lagrangian relaxation of the capacity (coupling) constraints — valid for **any** μ≤0 by weak duality, hence **independent of slack/feasibility state**. Unlike the old `dual_obj + Σ d_k·min(0,rc*_k)` form it is no longer gated on `num_active_slacks==0` and advances every priced iteration; the structural duals cancel analytically, and accumulating `sp_k` directly avoids the catastrophic cancellation a basic slack's huge π_k would cause.

Early exit fires when `best_ub − best_lb < RELATIVE_FEAS_TOL · max(1, |best_ub|)` and the gap is non-negative. **Both optimal exits report `best_ub`, never the terminating iteration's `obj`** — gap test and pricing exhaustion alike (`set_optimal(best_ub, iter)`) — so the summary's `UB=` always agrees with the `UB` column of the last iteration row. The pricing-exhaustion branch fires under exactly the UB-update guard (`num_new_caps == 0 && num_active_slacks == 0`), so `best_ub <= obj` holds by construction (asserted); reporting `obj` there used to return a value worse than CG's own incumbent whenever a barrier landed above an earlier solve on a strictly larger column set — cuOpt and MOSEK do this, COPT and HiGHS never did (#40).

The `#slk` log column counts basic slacks; when non-zero the LP obj is a feasibility penalty and `best_ub` is not updated (but `best_lb` is). Because the LB tracks from iteration 1, the gap closes as soon as the last slack clears — which is why early termination needs a ceiling high enough for every slack to price out (see slack ceiling above).

### Strategy presets

`CGParams::strategy` (enum `CGStrategy` in `include/mcfcg/cg/path_cg.h`) bundles several lower-level CGParams knobs. `solve_cg` reads it and computes `effective_*` locals at the top of the function; the bundle supersedes the corresponding raw fields where documented. Two values, named for how expensive the pricer is relative to the master:

- **`PricerLight`** (default): pricer is cheap, so push lots of cols/rows at the master — large col cap, column aging on, cuts and cols added in the same iteration.
- **`PricerHeavy`**: pricer is expensive, so throttle it — cap cols/iter at `num_entities`, disable column aging (overriding `CGParams::col_age_limit`), force the source pricing filter on, defer pricing in iterations that added lazy capacity rows, and use partial pricing via `compute_partial_pricing_batch_size` (`path_cg.h`).

Partial pricing: when `n_sources > pool_threads` the batch is `max(pool_threads, n_sources/4)`, so the column cap short-circuits mid-sweep and `pricer._last_source_idx` parks for the next iteration; when `n_sources <= pool_threads` it can't engage (one batch covers everyone), so we return 0 for a single big batch rather than pretending otherwise. Scaled by `n_sources`, not `num_entities`, because the pricer batches over sources and for path `num_entities = n_commodities` typically dominates. The success path uses `pricer.clear_postponed()` (flags only, cursor survives); warm-start and pricing-exhausted branches use `reset_postponed()` (flags + cursor). An explicit `CGParams::pricing_batch_size > 0` overrides all of it.

When adding a tunable, decide whether it belongs as a raw CGParams field, in an existing bundle, or as a new strategy value.

### Graph layer

`include/mcfcg/graph/` — CSR static digraph with typed arc/vertex maps (`static_map`), d-ary min-heap, Dijkstra/A* borrowing a `dijkstra_workspace` for reusable memory. Compile-time traits control which workspace fields are written (`if constexpr`).

### I/O

Two instance formats, both `.gz`-capable via zlib: CommaLab/UniPi plain-numeric (1-indexed) and TNTP transportation networks. TNTP uses `free_flow_time` as cost and divides demands by a city-specific coefficient. CommaLab uses negative capacity as the uncapacitated sentinel — `read_commalab` maps it to `+INF` so `count_capacitated_arcs` excludes those arcs and no capacity row is lazily added; `write_commalab` emits `-1` for `isinf(cap)`. This matches Lienkamp & Schiffer's `start_run.py::write_instance` for intermodal (`cap >= 9999 -> -1`) and is generated by `scripts/generate_instances.py`.

### Intermodal pitfalls

SUBWAY / BUS / SBT instances (from the Lienkamp & Schiffer repo) are far larger than grid/planar — hundreds of thousands of vertices, millions of arcs after time-expansion.
- **Use the tree formulation with `PricerHeavy`.** `solve_tree_cg` is robust on BUS/SBT under COPT and MOSEK in both presets, converging to the paper's LP optimum on BUS-2632. Intermodal spends 73–83% of wall clock in the pricer under COPT/MOSEK/cuOpt (only 40–43% under HiGHS, whose LP is far slower) — hence the family where bounded pricing was worth evaluating; see above for why it is still off. (MOSEK solves intermodal path in comparable time, but the per-commodity master is the wrong default on high-commodity instances generally.) Integration tests use `solve_tree_cg` + `PricerHeavy` via `solve_intermodal_and_check`.
- **Writer precision matters.** `write_commalab` emits `-1` for `isinf(capacity)` and preserves fractional costs (no `llround`), mirroring `start_run.py::write_instance`. Round-tripping a cleaned instance through the older truncating writer would drop fractional walking-arc costs (0.5 → 0, 1.5 → 2) and produce a platform-dependent `LLONG_MIN` sentinel that the reader accidentally still maps to INF; `RoundTripFractionalCostAndInfCap` guards this.

### cuOpt GPU pitfalls

- **Path master OOMs the GPU barrier on high-commodity instances.** One demand row per commodity means ~80k+ commodities (e.g. `planar2500` at 81,430) build a large ADAT normal-equations system, and cuOpt's RMM pool plus the cuDSS factorization can exceed VRAM (observed: `cudss_device_alloc` failure on a 16 GiB card with a desktop session holding ~6.5 GiB). **Use the tree formulation** — 2,500 convexity rows shrink the barrier system ~30× and solve correctly, generalizing the intermodal "use tree" rule to any high-commodity instance. (COPT's CPU barrier handles the 81k-row path master fine, ~1000s.)
- **A failed cuOpt barrier returns garbage as "solved".** On a cuDSS device-alloc / numerical-error termination the backend returns a non-optimal incumbent instead of erroring, so CG can converge on a ~100×-wrong objective (nondeterministically reported `optimal=1`). Tracked in #33. An objective far above the COPT/tree value means a swallowed barrier failure, not a real optimum.

## Instance Data

Four families in `data/`: grid and planar (CommaLab), transportation (TNTP, gz), intermodal (CommaLab, gz). Integration tests check small instances from each against paper reference values.

## Key Design Decisions

- **Integer-scaled Dijkstra costs.** Reduced costs scaled by 1e9 and clamped to non-negative int64_t; negative (attractive) arcs become 0-length. True reduced cost is recomputed in floating point after path extraction.
- **Lazy capacity constraints.** The master starts with demand/convexity rows only; capacity rows are added when flow exceeds capacity by >1e-6, avoiding a huge initial LP.
- **No duplicate columns.** The pricer must never generate one. Duplicates indicate a bug in pricing or reduced-cost computation.
- **Dense capacity duals `mu`.** Stored as a dense arc-indexed `static_map<uint32_t, double>` (0.0 for arcs without a capacity row), cached on `MasterBase` and reset incrementally each iteration, so `compute_rc` runs a contiguous `cost[a] - mu[a]` loop that auto-vectorizes under `-march=native`. Build with `-DMCFCG_NATIVE_ARCH=OFF` for portable binaries.
