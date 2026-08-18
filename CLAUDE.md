Navigation: LSP → narrow Grep → sliced Read. See "Code Navigation" below.

# Standards

## Communication Style

Be terse. No preamble. No filler.

## Code Navigation

Prefer narrow queries over full-file reads:

1. **LSP** for symbol questions. `goToDefinition`, `hover`, `documentSymbol`, `workspaceSymbol` answer "where is X / what's its signature" in a few tokens. Use before `Read`.
2. **Grep with `head_limit` (small) + `-n`** to locate lines. Start with `head_limit: 20`; raise only if inconclusive.
3. **Read with `offset`/`limit`** to fetch a slice around the hit. Full-file `Read` is fine for files under ~200 lines or when structure matters.

Know the symbol → LSP. Know a string, not its location → Grep. Full-file Read is the last mile.

This is a preference, not a prohibition. Shelling out to `grep`/`rg` is fine when the built-in tool can't do the job — filtering a pipe (`git log | grep`), or a session where the `Grep` tool isn't available. What matters is bounding the output, not which binary produces it.

## C++

- Target C++23. Use modern features (`std::expected`, concepts, ranges, `constexpr`).
- **Formatting** is Google, via `.clang-format`. **Naming is not Google** — it
  is STL-flavoured, and `.clang-tidy` enforces it:

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

  Protected members are deliberately **unconstrained** — gtest fixtures use bare
  `protected:` members, so the `_` prefix is a library-code convention that
  `.clang-tidy` does not enforce.

  Type names are deliberately two-tier and **not** enforced: `CamelCase` for the
  domain layer (`MasterBase`, `LPSolver`, `TreePricer`), `snake_case` for the
  STL-like containers and graph algorithms (`static_map`, `d_ary_heap`,
  `thread_pool`, `dijkstra`). clang-tidy cannot express "either", so `ClassCase`
  is intentionally absent from the config — match the layer you are in.
- Use `#pragma once` for include guards.
- Minimize includes in headers. Forward-declare where possible.

## clang-tidy

`cmake --build build --target tidy` — the gate. Stamped per translation unit, so
it is ~30s at `-j` from cold and a no-op when nothing changed. It needs only
`cmake -B build`, not a compiled tree. `tidy-fix` applies fix-its serially and refuses to run
outside a git checkout or with a dirty worktree under `include/`, `src/` or
`test/`. Narrow a bulk fix with `-DMCFCG_TIDY_FIX_CHECKS='-*,some-check'`.

`WarningsAsErrors` covers `clang-diagnostic-*`, `bugprone-*`, `performance-*`
and `readability-identifier-naming`; those block. Everything else — notably
`readability-function-cognitive-complexity` — is advisory signal. Suppress an
advisory finding with a `NOLINTNEXTLINE(check)` plus a comment saying why, never
by widening the config. Note that `NOLINTNEXTLINE` applies to the *literal* next
line, so put the prose above it and the pragma last, and for a template put it
between the `template<...>` line and the signature.

**Invoking clang-tidy by hand needs two flags the config cannot supply.**
`HeaderFilterRegex` is silently ignored when clang-tidy auto-discovers
`.clang-tidy`: `clang-tidy -p build src/cg/tree_cg.cpp` reports 2 diagnostics
where the same command with `--config-file=.clang-tidy` reports 481 — every
header, which is where nearly all of this codebase lives. And a bare `.*`
header filter is wrong even with the config file, because it matches
`build/_deps/highs-src/**`. So:

```
clang-tidy -p build --config-file=.clang-tidy \
  --header-filter="^$PWD/(include|src|test)/" --quiet <file.cpp>
```

Prefer the `tidy` target, which does both for you.

Two traps when fixing in bulk. `run-clang-tidy -fix` was measured reporting a
set of warnings and then silently applying only a fraction of them — use the
serial `tidy-fix` target instead. And never run `clang-tidy --fix` in parallel
here: every template lives in a header under `include/mcfcg`, so concurrent
processes rewrite the same file and corrupt it.

## CMake

- `set(CMAKE_EXPORT_COMPILE_COMMANDS ON)` for clang-tidy.
- Use FetchContent for dependencies.
- A single root `CMakeLists.txt`; the project is small enough that per-directory
  files would only add indirection.

## Testing (GoogleTest)

- Test files: `<module>_test.cpp` in `test/`.
- Name tests descriptively: `TEST_F(SolverTest, ReturnsOptimalForFeasibleInput)`.
- Terse output: `GTEST_BRIEF=1` prints only failures, `ctest --progress` collapses the running list, `CMAKE_INSTALL_MESSAGE=LAZY` suppresses install chatter. Don't remove these.

## LSP

Install `clangd-lsp@claude-plugins-official` plus `clangd` itself (`apt install clangd` or from LLVM). The repo's `.clangd` points at `build/compile_commands.json` (produced by `CMAKE_EXPORT_COMPILE_COMMANDS ON`). Prefer `LSP` tool queries (`goToDefinition`, `hover`, `documentSymbol`) over `Read` for symbol questions.

## Development Workflow

```
plan (non-trivial) → implement → test → push to main
```

Hooks auto-format on save (and type-check Python; C++ gets formatting only) — don't fix formatting manually. Run tests locally before considering work done — don't skip the suite even on changes that look trivial. The pre-push hook is the final gate.

Git hooks (`.git/hooks/*`) and Claude Code hooks (`.claude/`) are local-only and gitignored — not part of the published artifact, and not cloned with it. They were originally installed from an external toolkit; that dependency is gone, and the scripts are now plain local copies owned by this repo. The clang-tidy gate deliberately does not live in them: its substance is the `tidy` CMake target plus the CI lint job, both tracked, so a fresh clone can run the gate — via CI automatically, or `--target tidy` by hand. **Never use `git push --no-verify` or `git commit --no-verify`** unless explicitly asked. A failing hook is a signal — fix the root cause.

## Git Workflow

Trunk-based development with linear history on main. Commit directly to main and push when local gates pass.

Feature branches are optional for larger changes:
- Always branch from main. Run `git checkout main && git pull` first.
- Never branch from another feature branch.
- Keep branches short-lived; rebase or squash merge — no merge commits on main.

After a successful push:
- **Close any gh issue the work resolved**: `gh issue close <num> -c "<one-line note>"`. Do this for every issue covered by the push.
- **Delete the feature branch** if one was used: `git branch -d <branch>` locally, plus `git push origin --delete <branch>` if it was pushed. Don't leave stale branches behind.

## Issue Tracking

GitHub Issues is the tracker. Use the `gh` CLI.

- **Default to HTTPS** for GitHub remotes (`https://github.com/...`), not SSH.
- **Read an issue** with `gh issue view <num> --json title,body,labels,state,comments`. Plain `gh issue view <num>` is deprecated for programmatic use.
- Don't propose deferring work via a new gh issue unless it is substantial. Small follow-ups should be either fixed inline or left alone — don't open an issue just because you noticed something.

### Writing Issues

Issues get picked up later in fresh sessions, often by a different agent with no access to the author's machine. Write them to be picked up cold:

- **Self-contained.** Body must carry all needed context: problem, motivation, acceptance criteria, repro steps. Don't assume the reader has the current conversation.
- **No local references.** No local file paths, local repo paths, or machine-specific locations (`/home/user/...`, `~/code/foo/bar.py`, "see my other checkout"). Dead links in a fresh session.
- **Prefer stable external links.** GitHub permalinks, paper URLs, RFCs, official docs.
- **Be vague about local code context.** Describe the concept rather than the path; hint that the agent can search under `..`, `../..`, or `~/code/`.

## Commit Messages

Conventional Commits. The commit-msg hook enforces format.

- `type: description` or `type(scope): description`
- Types: `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `perf`, `chore`, `build`, `ci`
- Subject ≤72 chars. Focus on **why**, not what.

## CLAUDE.md Discipline

When Claude gets something wrong, fix CLAUDE.md in the same commit. It's a living document — update it whenever better instructions would have prevented the mistake.

## Complexity

When a complexity warning fires, don't extract methods mechanically. Ask: what are the independent responsibilities here? Split along those boundaries. If the function is genuinely complex because the domain is, add a comment explaining why and suppress the warning.

## Plan Adherence

**Follow the agreed plan.** If you think a plan should change, stop and discuss — don't silently diverge. The same goes outside a written plan: if your current approach isn't working, say so out loud — don't quietly switch strategies. Implement everything specified; don't leave TODO placeholders or stub implementations unless explicitly asked.

## Reference Correctness

When implementing from papers, pseudocode, or open-source references:
- Match the reference algorithm exactly. No early exits, iteration limits, size caps, or "optimization" shortcuts that change behavior.
- Only introduce heuristic approximations when explicitly asked.
- Implement edge cases and special handling — don't simplify them away.
- When in doubt, be faithful to the reference and let tests verify correctness.

## Common Mistakes

- **Don't invent APIs — verify they exist.** Check that functions, flags, and methods actually exist before using them.
- **Don't ignore type errors.** If mypy/clang-tidy flags something, fix the root cause — don't suppress.
- **Don't use deprecated patterns.** Check current docs, not training data.
- **Performance matters.** Most of our code is solvers — profile before micro-optimizing, but don't sacrifice perf for "clean code".

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

`scripts/` has a stdlib-unittest tier in `test/python/`, run by ctest as
`Python.Scripts`. It exists because those scripts carry the provenance of every
committed result, and their failure mode is a silently blank or mislabelled
column rather than a crash. Add a test as `test/python/<module>_test.py` — the
`_test.py` suffix matches the C++ convention and is why ctest passes
`-p "*_test.py"`; unittest's default `test*.py` would discover nothing. Files
must sit directly in `test/python/` (discovery does not recurse, and the
directory is not a package, so `python3 -m unittest test.python.foo` will not
work).

```
python3 -m unittest discover -s test/python -p '*_test.py'                # all
python3 -m unittest discover -s test/python -p '*_test.py' -k preserves   # one
```

## What This Is

Column generation solver for minimum-cost multicommodity flow (MCF). Supports path-based and tree-based Dantzig-Wolfe decompositions.

**Default to the tree formulation.** The path master has one demand row per commodity, which on high-commodity instances (transportation up to ~1.15M OD pairs, intermodal, planar2500) blows the master up to ~1M rows and the barrier LP solve dominates wall-clock — e.g. Philadelphia path is 6,436s (99% in the LP) vs 810s tree, ChicagoRegional 5,946s vs 97s (61×), Sydney 7,315s vs 72s (102×). Tree has one convexity row per source instead, keeping the master small. Objectives match between formulations; tree even solves instances path times out on (Austin). On grid/planar the two are roughly equal (tree occasionally slightly slower, e.g. planar2500). `scripts/benchmark_solvers.py` defaults every family to tree; pass `--formulations path,tree` to compare.

## Architecture

The CG loop (`include/mcfcg/cg/cg_loop.h`) is a single template function `solve_cg<Master, Pricer, GetDuals>` shared by both formulations (`GetDuals` is the per-formulation callable that extracts the pricing dual vector from the master — demand duals for path, convexity duals for tree). It drives the interaction between three components:

1. **Master problem** (`include/mcfcg/cg/master.h`, `tree_master.h`) — restricted LP with incremental column/row addition. Path formulation has one demand row per commodity; tree formulation has one convexity row per source. Capacity rows are lazy (added on violation). Slack placement is selected per instance by `MasterBase::init` (`enum SlackMode` in `master_base.h`): `CommodityRows` puts one slack per structural row at init with coeff +1; `EdgeRows` pairs a slack with each lazily-added capacity row with coeff -1. The selector picks whichever side has fewer rows, so the slack count is `min(num_structural_rows, num_capacitated_arcs)`. EdgeRows requires `CGParams::warm_start=true` (no init-time slacks means the LP is only feasible once warm-start seeds at least one column per structural row) — `init` throws on violation. Every slack starts at initial cost = max arc cost, grown by `MasterBase::bump_active_slacks` every CG iteration while any slack is basic — the LP pivots each slack out once its cost exceeds whatever column serves the row. Bumps happen at end-of-iter, before purges (bumps read `get_primals()` which COPT invalidates on delete; a bump-to-fixed-point loop wrapping `solve()` would also not terminate when lazy capacities force a slack basic until pricing adds a new column). `SLACK_BUMP_FACTOR` in `cg_loop.h` plus a per-instance ceiling on `SlackState::cost_ceiling` bound the growth. `MasterBase::init` creates the LP backend first, then sets the ceiling to `clamp(10 * Derived::slack_cost_upper_bound(), 1e6, LPSolver::max_slack_cost())` — path-master returns `num_vertices * max_arc_cost`; tree-master multiplies by the largest per-source demand sum. The ceiling is **backend-aware**: `max_slack_cost()` is `1e7` for HiGHS (the base default) and the cuOpt barrier (whose IPM stalls on a wide slack-vs-real dynamic range — slacks at 1e9 once caused a 100–1000s pathology on transportation), and `1e9` for the robust MOSEK/COPT barriers. **The cap matters for correctness, not just speed:** if the real per-row column cost exceeds the ceiling, that row's slack is cheaper than any real column and stays basic forever — no slack-free upper bound is ever reached and the penalized LP objective can sit *below* the true optimum (observed on `planar2500` tree, ~1.7e7/source vs a 1e7 cap: it never converged). Raising MOSEK/COPT to 1e9 lets those slacks price out; planar2500 tree then converges in ~140 iters / ~1700s. The 1e6 floor prevents the clamp from inverting on small instances. A reliable runtime signal that the ceiling is too low: the (valid) `best_lb` exceeds the current penalized LP objective while slacks are basic — the penalty isn't dominating real routing cost. The worst-case `slack_cost_upper_bound()` is far looser than typical per-row cost (esp. tree, where it is `demand_sum × full-path`), so it is a poor a-priori risk predictor; `scripts/slack_headroom.py` reports it plus a realistic per-row proxy (`optimum/total_demand` for path, `optimum/sources` for tree), but the proxy is an average and can miss expensive outlier rows.

2. **Pricer** (`include/mcfcg/cg/pricer.h`, `tree_pricer.h`) — computes reduced costs using dual values from the master. Runs Dijkstra from each source with clamped integer-scaled arc costs (SCALE=1e9). Source postponement skips sources that produced no negative-RC column last round. Path pricer extracts one column per commodity; tree pricer builds a single tree column per source aggregating demand-weighted arc flows. Unreachable commodity sinks (A* heap exhausts without settling the sink) are skipped: path pricer emits columns for the reachable commodities only; tree pricer emits a partial tree column covering reachable sinks only. Graceful only in `CommodityRows` slack mode (demand-row slacks absorb unmet demand) — in `EdgeRows` mode there is no demand slack and a disconnected source→sink surfaces as LP infeasibility. Note the tree formulation's partial tree is genuinely lossy (its master counts trees, not demand, so nothing notices the under-served commodities), which is tolerable only because a disconnected commodity makes the MCF infeasible anyway. Preprocess disconnected instances via `mcfcg_clean` before solving.

**Dual pricing cutoff** (`CGParams::pricing_cutoff`, CLI `--pricing-cutoff`, default **off**) — manuscript §3.3: stop a source's A* once the frontier proves no negative-RC column remains, rather than running until every sink is settled. Path bound is `max π` over unsettled sinks; tree bound is the residual convexity budget over the *sum* of remaining demands. Both add an allowance so the cut need only prove `rc ≥ neg_rc_tol` — that allowance is what makes it fire at all (65–77% vs 0–32% without), since at a master optimum every structural row has a basic column at rc 0 and the frontier reaches the sink at almost exactly the dual. **Evaluated and rejected — off by default everywhere, including the benchmark driver.** Pass `--extra-args=--pricing-cutoff` to measure it. The paired on/off A/B is committed: `results/ablation/` (README + per-run and paired CSVs + the raw logs), regenerated by `scripts/analyze_pricing_cutoff_ablation.py`. Read that README before re-opening the question. Two supporting measurements are *not* archived (they live in gitignored `bench_runs/`): the all-backend cutoff-on pass behind the HiGHS discussion below, and the +31% stale-arc experiment.

The gain has two terms, `pricing_share × per-source-price saving` **minus** `LP_share × Δiterations`, and both are why it loses. The best case on the suite is intermodal at **−3.6%** wall clock; transportation is −2.4% (noise, see below), grid/planar a wash (−1.3%/+0.3%). The two factors of the first term are **anticorrelated across families**: the cutoff prunes a multi-target search's tail, so per-price saving grows with commodities per source (grid/planar tree saves up to 25% per price) — but those families spend **1.0–1.5%** of wall clock pricing. Intermodal is the mirror image: exactly **1 commodity per source**, so ~2% per price, but **71–85%** of wall clock pricing. The product never clears ~2%, and only on instances that finish in under a minute: across `results/cg_benchmark.csv` the pricing share collapses as instances get harder (planar2500 0.1%, Philadelphia 1.1%, Birmingham 1.6%, Austin 4.3%) while one extra CG iteration there costs 0.4–1.5% of wall. Scaling intermodal up does not open this up — it has 1 commodity per source *by construction* (each request is its own source), so the ~2% is structural, not a size effect. Validation of the model on the four copt-cpu intermodal cells whose trajectory the cutoff leaves alone (`traj_moved=0` **and** `traj_stable=1`): predicted −2.3/−2.5/−1.6/−2.1%, measured −2.3/−2.4/−1.7/−2.3% — within 0.13pp. Quote copt-cpu, not copt-gpu: the latter has 2 reps and the model misses SBT-6255-0 there.

The second term makes it backend-specific, so **a single-backend measurement of this flag is worthless**. Intermodal LP/pricing split, recomputable from `results/cg_benchmark.csv` (path and tree): copt-cpu 3–4%/81–83%, copt-gpu 8–10%/75–78%, mosek 6–7%/80%, cuopt 15%/73–74%, **highs 49–53%/40–43%** (that is genuine HiPO — re-verified with `--verbose-solver` on an intermodal cell, not a silent dual-simplex fallback). Where the LP is 3% of runtime a ±Δiterations shift is nearly free; at half the clock it dominates. The on-arm reads **+18…+32% on HiGHS** against the committed cells — direction credible, **magnitude not citable**: that comparison is cross-session, and across the 39 cells where iterations *and* columns are bit-identical the same-config drift spans −5.5% to +15.3% wall, with `t_LP` up 10.7% on a byte-identical LP sequence.

Beware the total-pricing-time metric: the −4.3% first reported for intermodal was mostly **fewer sources priced** (trajectory), not cheaper pricing. Compare `t_PR / priced_sources`, which the `[pricing-cutoff] cut=… priced=…` line makes computable and which the ablation CSV carries as `per_price_us`. Three corollaries: a family total inside its noise floor is noise (transportation's −2.4% is not a gain — Barcelona posts −2.2% wall while its *per-price* cost rose 4.7%, and pricing is 0.7% of its wall clock); a per-price number is only meaningful where pricing time is measurable at all (38 of 48 grid/planar cells price for <0.1s total, where the 3-digit log timing is quantization and the median delta is exactly +0.0%); and `traj_moved=0` alone does not license quoting one, because it compares *medians* — `traj_stable=1` says the reps within each arm actually agreed. Transportation has no cell satisfying both. Three traps it must respect, each of which was a live bug caught in review — see `test/integration_test.cpp` `FeatureTests.PricingCutoff*`:
- `MAX_BOUND` is both `scale_dual`'s saturation value and `compute_lower_bounds_to_targets`' `UNREACHED`, so a frontier at/above it means **dead ends**, not a dual proof. Cutting there salvages ~4.6e9 into `best_lb` and suppresses the tree's partial column.
- A zero-demand commodity drives the tree's remaining demand to 0 with budget left; cutting there suppresses a strictly improving column on every iteration and CG reports it as optimal. CommaLab keeps zero-demand rows (only TNTP filters them).
- The tree budget must stay `+inf` through the warm start's `+inf` duals — it *divides* by remaining demand, so a saturated finite budget becomes a reachable threshold and a source goes unseeded (fatal in `EdgeRows`).

**The cutoff is column-identical but not trajectory-neutral.** Switching it on moves intermodal iteration counts by up to ±10 and changes column counts; that is expected and not a dropped column. Columns are pinned bit-for-bit — cost, reduced cost, and the full arc list / arc-flow vector — by `FeatureTests.PricingCutoffShadow{Tree,Path,IntermodalTree}`, which run the real `solve_cg` with the cutoff off while shadowing every dual vector with a second cutoff-on pricer (25k fires / 3.2k columns compared on BUS-2632-0 alone). Two other channels move the trajectory, neither a correctness bug (the pricing-exhausted `final_round` re-prices every source regardless of postponement):
- **Stale arc sets.** A cut search does not refresh `_source_arcs[s]` (`should_record_arcs`: a partial set would understate the routing and postpone a source a new cap row does affect), so `filter_for_new_caps` decides postponement from the routing that source had at its last *complete* price. Different sources priced → different columns that iteration → different LP → different lazily separated cap rows. Live only when the filter is on (`pricer_heavy || pricing_filter`). Do **not** fix it by treating a cut source as affected (`_source_cut[s]` is already that flag): measured +31% wall clock on intermodal, because a 65–77% fire rate makes nearly every source affected and the filter stops filtering — SBT-56295 alone paid +68% at an unchanged iteration count.
- **A weaker LB.** `salvage_lagr_term` substitutes `d_k·(cutoff_f/SCALE − margin)` for the `sp_k` a truncated search never computed — valid but weaker, so `best_lb` differs and the gap exit fires on a different iteration. Live in every configuration; moves the iteration count with an identical column set.

**Do not A/B the cutoff on grid/planar under copt-gpu.** Re-running the *same* config there differs on 23/48 instances — the same rate as on-vs-off — because the GPU barrier's interior point shifts and separation then picks a different violated-capacity set (grid10 tree diverges at iteration 13 on `#row` 1032 vs 1033 with identical `#col`/`LP_obj`). Intermodal under copt-cpu is near-deterministic (9/10 cells repeat the same iteration *and* column counts across the off-arm reps, 10/10 on the on-arm) and is the arm to measure on. `transportation_tree` is also copt-gpu and no better: 4 of its 6 cells fail to reproduce their own counts.

3. **LP backend** (`include/mcfcg/lp/lp_solver.h`) — abstract interface. HiGHS is the always-available default (`--solver highs`; compiled in unconditionally, incremental, CPU — needs no license or GPU, so it works on non-GPU hosts). It uses the **HiPO interior-point method** by default (`MCFCG_HIGHS_SOLVER=simplex|ipm|hipo|pdlp` overrides). HiPO is ~2× faster than HiGHS' simplex default on these CG masters (grid15 tree: 70s vs 148s) and matches the MOSEK/COPT barrier objectives (~7–12× slower than those barriers, but license/GPU-free). HiPO had a bug: it discarded a near-optimal solution as an `internal error` when its IPX refinement step failed on ill-conditioned path masters (wide objective range from the slack penalty costs). We carry a local fix in `cmake/patches/highs-hipo-refine-status.patch`, applied to the FetchContent'd HiGHS source via `PATCH_COMMAND` in `CMakeLists.txt` (idempotent); see the gh issue tracking the upstream report. With the patch, HiPO solves those masters correctly and the full suite passes with no simplex fallback. **HiPO's numerical backend (SuiteSparse AMD / METIS / SPARSEPAK-RCM / BLAS) must be linked statically.** HiGHS ≥1.15 ships it as a separate "extras" library and defaults `BUILD_SHARED_EXTRAS_LIB=ON` (→ `#define HIGHS_SHARED_EXTRAS_LIBRARY`), resolving the feature set by loading `libhighs_extras.so` at runtime. Because we link HiGHS statically (`BUILD_SHARED_LIBS OFF`) and never ship that `.so`, the extras go unresolved and HiGHS **silently falls back to dual simplex** while the `[lp-config]` banner still reads `method=hipo` (it only echoes the *requested* solver) — HiGHS emits `features unavailable: amd, blas, metis, rcm` and `Using dual simplex solver` only under `--verbose-solver`. `CMakeLists.txt` sets `BUILD_SHARED_EXTRAS_LIB OFF` to compile the extras into libhighs so HiPO is actually available (1.14 had no extras split, so HiPO worked there by default; the 1.15 packaging introduced this trap). **Verify HiPO is really running** with `mcfcg_cli … --verbose-solver`: the HiGHS log (on stdout) must show `Running HiPO` / `IPX reports: ipm optimal`, not `Using dual simplex solver`. Optional backends: cuOpt (GPU barrier, **incremental delta C API by default** — `MCFCG_CUOPT_DELTA_API` defaults ON and requires the `spoorendonk/cuopt` fork's `cuopt_c_delta.h` (CMake errors with a fork/opt-out hint if absent); set `-DMCFCG_CUOPT_DELTA_API=OFF` for stock non-fork cuOpt, which falls back to the rebuild-from-scratch path — a **serious** perf degradation that recreates the whole LP every iteration), COPT (barrier, GPU by default — overridable per-run via `--copt-gpu-mode 0|1|2`, incremental), and MOSEK (CPU barrier, incremental); enable with `-DMCFCG_USE_CUOPT=ON`, `-DMCFCG_USE_COPT=ON`, or `-DMCFCG_USE_MOSEK=ON` (select at runtime with `--solver`). **Pinned barrier config for fair comparison:** every backend runs presolve off, crossover off, and convergence tol `BARRIER_TOL = 1e-4` (`tolerances.h`); each logs a one-line `[lp-config]` provenance banner (backend/method/exec/threads) to stderr at construction, captured in CG and benchmark logs. **HiGHS crossover-on-certify:** crossover is off per iteration (fast), but `LPSolver::solve(bool certify)` lets the CG loop re-request one crossover solve when it stalls (pricing exhausted but not optimal, or an interior solve spuriously infeasible after cuts) — HiPO's non-vertex interior point otherwise leaves path-master demand slacks at O(tol)>0 and central duals that fail to price, so without it large path instances (Barcelona, Winnipeg) never certify a slack-free UB. COPT/MOSEK run crossover (basis identification) on a certify solve too; the cuOpt GPU barrier has no crossover so `certify` is a no-op there (`LPSolver::certify_runs_crossover()` returns false and the loop skips its certify retry to avoid a redundant re-solve). CSC format for columns, CSR for rows. The `starts` convention is uniform: `add_cols` and `add_rows` both require `starts.size() == n+1` with `starts[n] == values.size()`. The CG pricer uses a single `neg_rc_tol` (default `NEG_RC_TOL = -1e-3`, see `include/mcfcg/util/tolerances.h`); no backend overrides it today.

### Bounds and early termination
The CG loop tracks two monotone bounds. `best_ub` is the minimum LP objective over iterations whose LP primal is MCF-feasible (no slack basic, no new capacity row found this iter). `best_lb` is the maximum **π-free capacity-relaxation Lagrangian** lower bound over iterations where the pricer visited every source (`pricer.priced_all()` — no source postponed; a `max_cols` break that fires exactly on sweep completion still counts): `LB = Σ_a cap_a·μ_a + Σ_k d_k · sp_k(c−μ) − rounding_margin`, where `sp_k(c−μ)` is the reduced-cost shortest path the pricer found (`pricer.lagrangian_path_sum()`, accumulated WITHOUT subtracting the structural dual π_k), `Σ cap·μ` is `master.compute_capacity_dual_term(mu)` snapshotted before separation mutates the row set, and the rounding margin bounds the scale-integer vs true-RC gap. This is the textbook Lagrangian relaxation of the capacity (coupling) constraints — valid for **any** μ≤0 by weak duality, hence **independent of slack/feasibility state**, so unlike the old `dual_obj + Σ d_k·min(0,rc*_k)` form it is no longer gated on `num_active_slacks==0` and advances every priced iteration (the structural duals cancel analytically; accumulating sp_k directly avoids the catastrophic cancellation a basic slack's huge π_k would cause). Path weights by `d_k` (demand); tree's convexity RHS=1 collapses the weight. Early exit fires when `best_ub − best_lb < RELATIVE_FEAS_TOL · max(1, |best_ub|)` and the gap is non-negative. **Both optimal exits report `best_ub`, never the terminating iteration's `obj`** — gap test and pricing exhaustion alike (`set_optimal(best_ub, iter)`), so the summary's `UB=` always agrees with the `UB` column of the last iteration row. The pricing-exhaustion branch fires under exactly the UB-update guard (`num_new_caps == 0 && num_active_slacks == 0`), so `best_ub <= obj` holds there by construction (asserted); reporting `obj` there used to hand back a value worse than an incumbent CG already had whenever a barrier landed above an earlier solve on a strictly larger column set — cuOpt and MOSEK both do this, COPT and HiGHS never did (#40). The `#slk` log column counts basic slacks — when non-zero the LP obj is a feasibility penalty and `best_ub` is not updated (but `best_lb` now is). Because the LB tracks from iteration 1, the gap closes as soon as the last slack clears, which is why early termination requires the ceiling to be high enough for every slack to price out (see the slack-ceiling note above).

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
- **Use the tree formulation with `PricerHeavy`.** `solve_tree_cg` is robust on BUS/SBT under COPT and MOSEK, both PricerLight and PricerHeavy — converging to the paper's LP optimum on BUS-2632. Intermodal spends 73–83% of its wall clock in the pricer under COPT/MOSEK/cuOpt (but only 40–43% under HiGHS, whose LP is far slower), which is why it is the family where the dual pricing cutoff is worth evaluating — see the cutoff notes above for why it is nonetheless still off by default. (Under MOSEK the path formulation also solves intermodal in comparable time, but the per-commodity master makes it the wrong default on high-commodity instances generally — see "Default to the tree formulation" above.) Integration tests for intermodal use `solve_tree_cg` + `PricerHeavy` via the `solve_intermodal_and_check` helper.
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
