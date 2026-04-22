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
\quad \textbf{if } \textsf{pricer.pricedAll} \wedge s = 0 \wedge A^{\text{new}} = \emptyset \textbf{ then} \\
\qquad LB \leftarrow \max\!\bigl(LB,\ \pi^\top b + \mu^\top u + \textstyle\sum_k d_k \min(\bar c^*_k, 0) - \varepsilon\bigr) \\
\quad \textbf{if } UB < \infty \wedge 0 \le UB - LB < \tau \cdot \max(1, \lvert UB \rvert) \textbf{ then return optimal}(UB) \\
\quad \textbf{if } C = \emptyset \textbf{ then} \\
\qquad \textbf{if } s = 0 \wedge A^{\text{new}} = \emptyset \textbf{ then return optimal}(\mathit{obj}) \\
\qquad \textbf{if } s > 0 \textbf{ then } \textsf{master.bumpSlacks}() \\
\qquad \textsf{pricer.resetPostponed}();\ \textbf{continue} \qquad \triangleright \text{fresh sweep next iter} \\
\quad \text{trim } C \text{ to the } C_{\max} \text{ columns with lowest reduced cost} \\
\quad \textsf{master.bumpSlacks}();\ \text{purge aged cols};\ \text{purge idle cap rows} \\
\quad \textsf{master.addColumns}(C) \\
\textbf{return stopped}(UB) \\
\end{array}
$$

The LB uses the LP *dual* objective $\pi^\top b + \mu^\top u$ (exact at
LP optimum even when a barrier backend's primal and dual differ at
solver tolerance); $\bar c^*_k$ is the pricer's best reduced cost for
entity $k$, and $\varepsilon$ a rounding-error budget for the
scale-integer Dijkstra. The structural-row RHS $b$ is $d_k$ for the
path formulation and $1$ for the tree formulation, so the $d_k$
weighting collapses under tree.

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
| `LB` | best Lagrangian/Farley bound so far |
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
| `-DMCFCG_USE_CUOPT=ON`   | OFF | Enable the NVIDIA cuOpt GPU LP backend (requires cuOpt SDK) |
| `-DMCFCG_USE_COPT=ON`    | OFF | Enable the COPT LP backend (requires COPT installed) |
| `-DMCFCG_NATIVE_ARCH=OFF` | ON | Disable `-march=native`. Keep ON for SIMD auto-vectorization of the hot `cost[a] - mu[a]` pricing loop; only turn OFF for portable binaries. |

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
| `--solver NAME`           | highs | LP backend: `highs`, `cuopt`, `copt` |
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

Download scripts are in `scripts/`.

## License

MIT
