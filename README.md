# mcfcg

Column generation solver for the minimum-cost multicommodity flow (MCF)
problem with path-based and tree-based Dantzig-Wolfe decompositions.

Based on: S. Spoorendonk and B. Petersen,
[Tree-based formulation for the multi-commodity flow problem](https://arxiv.org/abs/2509.24656),
2025.

## Problem and formulations

Given a directed graph $G=(V,A)$ with arc costs $c_a$ and capacities
$u_a$, and a set of commodities $K$ where commodity $k$ routes
$d_k$ units from source $o_k$ to sink $t_k$, find the min-cost feasible
multicommodity flow.

### Arc-flow (compact) formulation

$$
\begin{aligned}
\min\;& \sum_{k\in K}\sum_{a\in A} c_a\, x^k_a \\
\text{s.t.}\;& \sum_{a\in\delta^+(v)} x^k_a - \sum_{a\in\delta^-(v)} x^k_a
  = \begin{cases} d_k & v = o_k\\ -d_k & v = t_k\\ 0 & \text{otherwise}\end{cases}
  \quad\forall k\in K,\ v\in V\\
& \sum_{k\in K} x^k_a \le u_a \quad\forall a\in A\\
& x^k_a \ge 0
\end{aligned}
$$

### Path formulation (Dantzig-Wolfe)

Let $P_k$ be the set of $o_k \!\to\! t_k$ simple paths, with $\lambda^k_p \ge 0$
the flow on path $p \in P_k$ and $c_p = \sum_{a\in p} c_a$.  (This is the
flow convention the solver uses: the demand row bound is $d_k$ and the
capacity row coefficient is $1$ per arc used; the pricer's reduced cost
does not carry a $d_k$ factor.)

$$
\begin{aligned}
\min\;& \sum_{k\in K}\sum_{p\in P_k} c_p\, \lambda^k_p \\
\text{s.t.}\;& \sum_{p\in P_k} \lambda^k_p \ge d_k \quad\forall k\in K \quad[\pi_k \ge 0]\\
& \sum_{k\in K}\sum_{p\in P_k} \delta_{ap}\, \lambda^k_p \le u_a
  \quad\forall a\in A \quad[\mu_a \le 0]\\
& \lambda^k_p \ge 0
\end{aligned}
$$

Reduced cost of a path $p$ for commodity $k$:
$\bar c^k_p = \sum_{a\in p} (c_a - \mu_a) - \pi_k$.
Pricing reduces to a shortest path in $G$ with arc weights
$c_a - \mu_a$ (one Dijkstra per source, targeting that source's sinks).

### Tree formulation (Dantzig-Wolfe)

Group commodities by source: $S_s = \{k \in K : o_k = s\}$.  For each
source $s$ let $T_s$ be the set of trees (subgraphs) serving every sink
of $S_s$ from $s$, with $\xi^s_t$ the fraction used, aggregated arc flow
$f^{s,t}_a = \sum_{k\in S_s} d_k [a \in \text{path from } s \text{ to } t_k \text{ in } t]$,
and tree cost $c_t = \sum_{a\in A} c_a\, f^{s,t}_a$.

$$
\begin{aligned}
\min\;& \sum_{s}\sum_{t\in T_s} c_t\, \xi^s_t \\
\text{s.t.}\;& \sum_{t\in T_s} \xi^s_t = 1 \quad\forall s \quad[\pi_s]\\
& \sum_{s}\sum_{t\in T_s} f^{s,t}_a\, \xi^s_t \le u_a
  \quad\forall a\in A \quad[\mu_a \le 0]\\
& \xi^s_t \ge 0
\end{aligned}
$$

Reduced cost of a tree $t$ for source $s$:
$\bar c^s_t = \sum_{a\in A} f^{s,t}_a (c_a - \mu_a) - \pi_s$.
Pricing is a single Dijkstra from $s$ with arc weights $c_a - \mu_a$
that simultaneously finds all shortest paths to the sinks of $S_s$; the
tree column aggregates demand-weighted arc flow over those paths.

## Algorithm

Dantzig-Wolfe column generation. The restricted master starts with
demand/convexity rows and slacks sized by
$\min(\lvert\text{structural rows}\rvert, \lvert\text{capacitated arcs}\rvert)$;
capacity rows are added lazily on violation. Each iteration solves the
LP once, separates violated capacity cuts, prices columns against the
captured duals, and commits what survived.

$$
\begin{array}{l}
\textbf{Algorithm 1 } \textsf{SolveCG}(G, K, \text{params}) \\
\hline
\textsf{master.init}();\ \textsf{pricer.init}() \\
\textbf{if } \text{warm-start} \textbf{ then seed master: one column per source, priced at } \pi = +\infty \\
UB \leftarrow +\infty,\ LB \leftarrow -\infty \\
\textbf{for } it = 1, \ldots, it_{\max} \textbf{ do} \\
\quad (\pi, \mu,\ \mathit{obj}) \leftarrow \textsf{master.solveAndReadDuals}() \qquad \triangleright \text{duals read BEFORE any mutation} \\
\quad A^{\text{new}} \leftarrow \textsf{master.separateCapacityViolations}() \qquad \triangleright \text{new lazy rows; no re-solve} \\
\quad s \leftarrow \textsf{master.numBasicSlacks}() \\
\quad \textbf{if } s = 0 \wedge A^{\text{new}} = \emptyset \textbf{ then } UB \leftarrow \min(UB, \mathit{obj}) \\
\quad \textbf{if } \textsf{PricerHeavy} \wedge A^{\text{new}} \ne \emptyset \textbf{ then continue} \qquad \triangleright \text{defer pricing; next iter's LP uses fresh duals} \\
\quad C \leftarrow \textsf{pricer.price}(\pi, \mu,\ C_{\max}) \\
\quad \textbf{if } C = \emptyset \textbf{ then } C \leftarrow \textsf{pricer.price}(\pi, \mu,\ C_{\max},\ \text{final}=\top) \qquad \triangleright \text{full sweep ignoring postpone flags} \\
\quad \textbf{if } C \ne \emptyset \textbf{ then } \textsf{pricer.clearPostponed}() \qquad \triangleright \text{flags only; keep cursor for partial pricing} \\
\quad \textbf{if } \textsf{pricer.pricedAll} \textbf{ then} \qquad \triangleright \text{no gate on } s \text{ or } A^{\text{new}}\text{: valid for any } \mu \le 0 \\
\qquad LB \leftarrow \max\!\bigl(LB,\ \textstyle\sum_{a} u_a \mu_a + \sum_k d_k\, \mathit{sp}_k(c - \mu) - \varepsilon\bigr) \\
\quad \textbf{if } UB < \infty \wedge 0 \le UB - LB < \tau \cdot \max(1, \lvert UB \rvert) \textbf{ then return optimal}(UB) \\
\quad \textbf{if } C = \emptyset \textbf{ then} \\
\qquad \textbf{if } s = 0 \wedge A^{\text{new}} = \emptyset \textbf{ then return optimal}(UB) \\
\qquad \textbf{if } s > 0 \textbf{ then } \textsf{master.bumpSlacks}() \\
\qquad \textsf{pricer.resetPostponed}();\ \textbf{continue} \qquad \triangleright \text{fresh sweep next iter} \\
\quad \text{trim } C \text{ to the } C_{\max} \text{ columns with lowest reduced cost} \\
\quad \textsf{master.bumpSlacks}();\ \text{purge aged cols};\ \text{purge idle cap rows} \\
\quad \textsf{master.addColumns}(C) \\
\textbf{return stopped}\bigl(UB < \infty\ ?\ UB : LB\bigr) \qquad \triangleright \text{time limit / } it_{\max} \text{ / LP not optimal} \\
\end{array}
$$

The LB is the Lagrangian relaxation of the capacity (coupling)
constraints: $\mathit{sp}_k(c-\mu)$ is the reduced-cost shortest path
the pricer already computed for entity $k$, accumulated *without*
subtracting the structural dual $\pi_k$, and $\varepsilon$ is a
rounding-error budget for the scale-integer Dijkstra. By weak duality
$L(\mu) \le \mathrm{OPT}$ for **any** $\mu \le 0$, so the bound holds
whatever the master's feasibility state — it is gated only on the
pricer having visited every source, not on $s = 0$ or
$A^{\text{new}} = \emptyset$, and so it advances from the first
iteration rather than waiting for the last slack to leave the basis.
The $\pi$-free form matters numerically as well as logically: a basic
slack pins $\pi_k$ at the bumped slack cost, and reconstructing the
bound from $\pi^\top b$ would then lose it to catastrophic
cancellation. Demands weight the path sum for the path formulation;
under tree the convexity RHS is $1$ and the $d_k$ weighting collapses.

Both optimal exits return $UB$ — the incumbent — never the terminating
iteration's LP objective. The pricing-exhaustion exit fires under
exactly the guard that updates $UB$, so $UB \le \mathit{obj}$ holds
there by construction; returning $\mathit{obj}$ instead used to hand
back a value worse than one CG already had whenever a barrier landed
above an earlier solve on a strictly larger column set.

The **non-optimal** exit reports $UB$ when one was ever recorded and
falls back to the Lagrangian $LB$ otherwise: with slacks basic the LP
objective is a feasibility penalty rather than a routing cost, so it
would be worse than useless as a reported value. That fallback is not a
corner case in practice — of the 20 uncertified cells in
`results/cg_benchmark.csv`, 19 produced a number at all, and 17 of those
report a lower bound in their `objective` column for exactly this reason.

`pricer.price` is the source-level dispatcher; each per-source call
(`PriceOneSource`) is the A* inner body. Postponement is a
one-iter-ahead filter: a source that emits no negative-RC column is
skipped on the next non-final call. Flags are cleared whenever the
main iteration commits columns (`clearPostponed`, keeps the cursor so
partial pricing resumes), when pricing finally exhausts
(`resetPostponed`, rewinds to source 0), and after the warm-start
pass. `filter_for_new_caps` rewrites the flag vector wholesale after
a cut round: sources whose best-path arcs were touched by a new cap
are flipped in (`postponed=0`), all others are postponed until a
later sweep re-examines them.

$$
\begin{array}{l}
\textbf{Algorithm 2 } \textsf{pricer.price}(\pi, \mu;\ \text{final}=\bot,\ C_{\max}=\infty) \\
\hline
\text{compute } w_a \leftarrow \max(0,\ c_a - \mu_a) \cdot 10^9 \text{ for all } a \in A \qquad \triangleright \text{dense vectorized arc pass} \\
C \leftarrow \emptyset;\ \textit{pricedCount} \leftarrow 0 \\
\textbf{for each source } s \text{ (round-robin from cursor, in batches of } B \text{) } \textbf{do} \\
\quad \textbf{if } \neg\text{final} \wedge s \in \text{Postponed} \textbf{ then continue} \qquad \triangleright \text{skipped sources do not count} \\
\quad C \leftarrow C \cup \textsf{PriceOneSource}(s, \pi, \mu);\ \textit{pricedCount}{+}{+} \qquad \triangleright \text{parallel across batch (thread pool)} \\
\quad \textbf{if } |C| \ge C_{\max} \textbf{ then break} \\
\textit{pricedAll} \leftarrow (\textit{pricedCount} = \lvert \text{sources} \rvert) \qquad \triangleright \text{derived at end; sweep-completing break still counts} \\
\textbf{return } (C,\ \textit{pricedAll}) \\
\end{array}
$$

$$
\begin{array}{l}
\textbf{Algorithm 3 } \textsf{PriceOneSource}(s,\ \pi,\ \mu) \\
\hline
\text{run A* from } s \text{ with edge weights } w_a \text{ until every reachable sink of } S_s \text{ is settled} \\
\textbf{for each } k \in S_s \textbf{ do} \\
\quad \textbf{if } t_k \text{ unreachable then skip} \\
\quad p_k \leftarrow \text{shortest path};\ \bar c_k \leftarrow \textstyle\sum_{a \in p_k}(c_a - \mu_a) - \pi_k \qquad \triangleright \text{true RC, floating point} \\
\textbf{path: emit } \{p_k : \bar c_k < \tau_{\mathrm{rc}}\};\ \text{postpone } s \text{ if none emitted} \\
\textbf{tree: aggregate } f_a = \textstyle\sum_k d_k [a \in p_k];\ \text{emit tree column if RC} < \tau_{\mathrm{rc}},\ \text{else postpone } s \\
\end{array}
$$

### Iteration log

`Verbosity::Iteration` prints one row per CG iteration:

| column | meaning |
|--------|---------|
| `It` | iteration number |
| `UB` | running min LP obj over MCF-feasible iters |
| `LB` | best $\pi$-free capacity-relaxation Lagrangian bound so far |
| `LP_obj` | current LP objective (carries slack penalty while `#slk > 0`) |
| `#col`, `#row` | columns / rows in the LP right now |
| `#slk` | basic slack columns; non-zero means `LP_obj` is a penalty, not a bound |
| `+col`, `-col` | columns added / purged this iteration (`*N` = produced but not committed on gap exit) |
| `+cut`, `-cut` | capacity rows added / purged this iteration |
| `t_LP`, `t_PR`, `t_SP`, `t_Tot` | per-iter seconds (LP, pricing, separation, total) |

## Build

Requires C++23, CMake 3.20+, and zlib.  HiGHS ships as a FetchContent
dependency — no external install needed.

```bash
cmake -B build -DCMAKE_INSTALL_MESSAGE=LAZY
cmake --build build -j$(nproc)
```

### Optional

| Flag | Default | Effect |
|------|---------|--------|
| `-DMCFCG_USE_CUOPT=ON`   | OFF | Enable the NVIDIA cuOpt GPU LP backend. Defaults to cuOpt's incremental delta C API (`MCFCG_CUOPT_DELTA_API`, below), which requires the fork. |
| `-DMCFCG_CUOPT_DELTA_API=OFF` | ON | Opt out of the delta C API for stock (non-fork) cuOpt: falls back to the rebuild-from-scratch path, which recreates the whole LP every CG iteration — a serious performance degradation. Default ON requires the fork's `cuopt_c_delta.h` (configure errors if absent). Only meaningful with `-DMCFCG_USE_CUOPT=ON`. |
| `-DMCFCG_USE_COPT=ON`    | OFF | Enable the COPT LP backend (requires COPT installed, `COPT_HOME` set) |
| `-DMCFCG_USE_MOSEK=ON`   | OFF | Enable the MOSEK CPU barrier LP backend (requires MOSEK, `MOSEK_HOME` set) |
| `-DMCFCG_NATIVE_ARCH=OFF` | ON | Disable `-march=native`. Keep ON for SIMD auto-vectorization of the hot `cost[a] - mu[a]` pricing loop; only turn OFF for portable binaries. |

### cuOpt and the delta-API fork

The cuOpt backend mutates the restricted master incrementally (add/delete
columns and rows, re-solve), so by default it uses cuOpt's incremental delta C
API (`MCFCG_CUOPT_DELTA_API=ON`). Stock cuOpt has no such API; the default path
needs a cuOpt build that ships `cuopt_c_delta.h` — the
[`spoorendonk/cuopt`](https://github.com/spoorendonk/cuopt) fork (delta-api
branch). **Build that fork first** (configure errors out if the header is
missing), or reconfigure with `-DMCFCG_CUOPT_DELTA_API=OFF` to fall back to the
rebuild-from-scratch path on stock cuOpt — a serious performance degradation
(the whole LP is recreated every CG iteration), supported only as a
compatibility fallback. To use the fork, point the configure at it (an install
prefix or a source checkout both work):

```bash
# one combined build with both COPT and the cuOpt delta fork
cmake -B build -DCMAKE_INSTALL_MESSAGE=LAZY \
  -DMCFCG_USE_COPT=ON \
  -DMCFCG_USE_CUOPT=ON \
  -DCUOPT_INCLUDE_DIR=/path/to/cuopt/cpp/include \
  -DCUOPT_LIBRARY=/path/to/cuopt/cpp/build/libcuopt.so
cmake --build build -j$(nproc)
```

`libcuopt.so` (and its `librmm` / `rapids_logger` deps) must be reachable by
the dynamic loader at run time. The build **embeds the cuOpt library directory
as an RPATH** — derived from `CUOPT_LIBRARY` — so `build/mcfcg_cli`, the tests,
and the tools run without exporting `LD_LIBRARY_PATH`. If you later **move the
fork's build directory**, either reconfigure (so the RPATH updates) or put the
new location on `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/path/to/cuopt/cpp/build:$LD_LIBRARY_PATH
```

## LP backends

Four LP backends implement a common interface: **HiGHS** (default, FetchContent,
no license/GPU), **cuOpt** (GPU barrier), **COPT** (CPU/GPU barrier), and
**MOSEK** (CPU barrier). Select at run time with `--solver`.

**Pinned barrier configuration.** For fair cross-solver comparison every backend
runs the same regime: **presolve off, crossover off, convergence tolerance 1e-4**
(the MCF feasibility design target; see `include/mcfcg/util/tolerances.h`
`BARRIER_TOL`). Each solver prints a one-line provenance banner to stderr at
construction (captured in the CG / benchmark logs), e.g.:

```
[lp-config] backend=mosek version=11.0.30 method=barrier exec=CPU presolve=off crossover=off tol=0.0001 threads=auto(32)
```

`version=` is queried from the library actually loaded at run time, never from
the vendor header's compile-time macros, so a stale `LD_LIBRARY_PATH` pointing at
a second install shows up here instead of being silently misreported.
`PROVENANCE.txt` records the two things it cannot see: whether the HiGHS HiPO
patch is applied, and whether cuOpt is the delta-API fork.

`threads=auto(N)` reports the backend's effective thread count (`N` = hardware
concurrency when the backend auto-selects); `exec` is CPU or GPU. The banner
reports the steady-state pins — a stall-recovery certify solve transiently runs
crossover on the crossover-capable backends (HiGHS/COPT/MOSEK; see below).

**HiGHS crossover-on-certify (stall recovery).** HiGHS uses the HiPO
interior-point method with crossover **off** per iteration, so the bulk of CG is
fast. But a pure interior-point solution is not a vertex: on the **path**
formulation of large instances its central duals can fail to price an improving
column, and demand-row slacks settle at O(tol) > 0 rather than exactly 0, so the
CG loop cannot certify a slack-free upper bound. When the loop detects this stall
(pricing exhausted but not optimal, or an interior solve spuriously reporting
infeasible after cuts), it re-requests that one solve as a **certify** solve via
`LPSolver::solve(certify=true)` — HiGHS then runs crossover to round the interior
point to a vertex (discriminating duals, slacks exactly 0). COPT and MOSEK
likewise run crossover (basis identification) only on a certify solve; the cuOpt
GPU barrier has no crossover and treats `certify` as a no-op (the loop skips its
stall recovery there, via `certify_runs_crossover()`). This keeps "crossover off"
honest for the common case (every backend runs crossover-off steady-state, and
in practice only HiGHS ever stalls) while letting any crossover-capable backend
certify the cases that need a vertex — crossover fires only on the stalled
solves, not every iteration.

## Test

```bash
GTEST_BRIEF=1 ctest --test-dir build --output-on-failure --progress -j$(nproc)
```

A single test:

```bash
./build/mcfcg_tests --gtest_filter='PathCGSingleSource.OptimalObjective'
./build/mcfcg_integration_tests --gtest_filter='GridCorrectness.Grid1'
```

## CLI usage

```bash
./build/mcfcg_cli <instance_path> [options]
```

| Option | Default | Meaning |
|--------|---------|---------|
| `--formulation path|tree` | `path` | Decomposition to use |
| `--max-iters N`           | 10000 | CG iteration cap |
| `--trips PATH`            | auto  | TNTP trips file (auto-detected from net path) |
| `--coef N`                | auto  | TNTP demand coefficient (auto per city) |
| `--threads N`             | 0     | Pricing threads (`0` = hardware concurrency, `1` = serial) |
| `--batch-size N`          | 0     | Sources priced per batch (`0` = all) |
| `--solver NAME`           | highs | LP backend: `highs`, `cuopt`, `copt`, `mosek` |
| `--copt-gpu-mode N`       | 2     | COPT barrier execution: `0` = CPU, `1`/`2` = GPU (default 2). Only affects `--solver copt`. |
| `--verbose-solver`        | off   | Enable the LP backend's own log output |
| `--col-age-limit N`       | 5     | Purge columns after N idle iters (`0` disables) |
| `--row-inactivity N`      | 5     | Purge cap rows after N idle iters (`0` disables) |
| `--neg-rc-tol X`          | -1e-3 | Reduced-cost acceptance threshold |
| `--strategy S`            | pricer-light | `pricer-light` or `pricer-heavy` preset |

```bash
# CommaLab format
./build/mcfcg_cli data/commalab/grid/grid1

# TNTP transportation format (auto-detects trips file and demand coefficient)
./build/mcfcg_cli data/transportation/Winnipeg_net.tntp.gz

# Tree formulation
./build/mcfcg_cli data/commalab/grid/grid1 --formulation tree
```

## Instance data

Four instance families from public sources:

| Family | Format | Source |
|--------|--------|--------|
| Grid | CommaLab | [UniPi MCF benchmark](https://commalab.di.unipi.it/datasets/mmcf/) |
| Planar | CommaLab | [UniPi MCF benchmark](https://commalab.di.unipi.it/datasets/mmcf/) |
| Transportation | TNTP (gz) | [TransportationNetworks](https://github.com/bstabler/TransportationNetworks) |
| Intermodal | CommaLab (gz) | [Lienkamp & Schiffer 2024](https://doi.org/10.1016/j.ejor.2023.09.019) |

Every instance of all four families is committed to this repository, so
reproducing any published row needs no download step. Three of the four can
also be re-derived from their original sources rather than taken on trust:
`scripts/download_commalab.sh` refetches grid and planar, and
`scripts/prepare_intermodal.sh` regenerates the intermodal instances end to
end. The transportation instances have no fetch script — the committed
`.tntp.gz` files came from the linked TransportationNetworks repository
unmodified, and the CLI applies the per-city demand coefficient at read time
(the coefficients are tabulated in `scripts/README.md`).

## Reproducing the published results

The committed result tables are in `results/`; `PROVENANCE.txt` pins the exact
solver versions, host and patches behind them, and `scripts/README.md` documents
how each table is produced and consolidated.

**The standard build is HiGHS-only.** That is deliberate — it needs no licence,
no GPU and no external install — but it means the default `cmake -B build`
produces a binary that can reproduce one of the five benchmark configurations.
The other four are opt-in at configure time, and a `--solver` label whose backend
was not compiled in reports as an `error` row rather than silently disappearing.
To build the full matrix:

```bash
export MOSEK_HOME=/opt/mosek/<ver>/tools/platform/linux64x86 \
       COPT_HOME=/opt/copt80
cmake -B build -DCMAKE_INSTALL_MESSAGE=LAZY \
      -DMCFCG_USE_MOSEK=ON -DMCFCG_USE_COPT=ON -DMCFCG_USE_CUOPT=ON \
      -DCUOPT_INCLUDE_DIR=/path/to/cuopt/cpp/include \
      -DCUOPT_LIBRARY=/path/to/cuopt/cpp/build/libcuopt.so
cmake --build build -j$(nproc)
```

### What each configuration costs you

| Config | Licence | Hardware | Reproducible with open-source components? |
|--------|---------|----------|-------------------------------------------|
| `highs`    | none (MIT) | CPU | **Yes.** Fetched and built by CMake; nothing to install. |
| `cuopt`    | none (Apache-2.0) | **NVIDIA GPU** | **Yes**, given a GPU. Build the [`spoorendonk/cuopt`](https://github.com/spoorendonk/cuopt) fork at commit `8ea7a033a` first — the delta C API the default build needs is not in stock cuOpt. |
| `mosek`    | commercial (academic licences available) | CPU | No. |
| `copt-cpu` | commercial (academic licences available) | CPU | No. |
| `copt-gpu` | commercial (academic licences available) | NVIDIA GPU | No. |

Cite the cuOpt fork by commit, not by branch: `delta-api` is a moving ref.

### Verifying a reproduction

Objective values are the reproducible quantity — specifically, the objectives of
runs that proved optimality. Wall-clock time and peak RSS are properties of the
host, the GPU and the solver build and will not match on different hardware, and
a run stopped by the time limit reports a bound that likewise depends on how fast
the machine is. `scripts/check_reproduction.py` compares objectives only, gates
on the certified ones, and reports cells your sweep did not run as *not run*
rather than folding them into a pass:

```bash
python3 scripts/benchmark_solvers.py --families grid --solvers highs \
    --formulations path,tree --out /tmp/grid_highs.csv
python3 scripts/check_reproduction.py /tmp/grid_highs.csv
```

That is the smoke test: 30 cells, roughly 15 minutes, no licence and no GPU. It
exits non-zero on any mismatch.

## Citing

Cite the paper for the result and the archived release for the code.
[`CITATION.cff`](CITATION.cff) carries both in machine-readable form — GitHub's
"Cite this repository" button reads it.

The paper is **arXiv:2509.24656**, <https://arxiv.org/abs/2509.24656>:

```bibtex
@misc{spoorendonk2025treemcf,
  title        = {Tree-based formulation for the multi-commodity flow problem},
  author       = {Spoorendonk, Simon and Petersen, Bj{\o}rn},
  year         = {2025},
  eprint       = {2509.24656},
  archivePrefix= {arXiv},
  primaryClass = {math.OC},
  url          = {https://arxiv.org/abs/2509.24656}
}
```

Cite a tag, never `main`: the result tables under `results/` move with the code,
and `PROVENANCE.txt` pins the solver stack only as of the tag that carries it.

## License

MIT
