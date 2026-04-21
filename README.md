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
demand / convexity rows only; capacity rows are added lazily on
violation.  Every CG iteration solves the LP, separates violated
capacity cuts, then prices one or many columns against the current
duals.

```text
solve_cg(Instance, params):
  master.init()                          # structural rows, slacks per selected mode
  pricer.init()                          # A* heuristic, per-thread workspaces

  if params.warm_start:                  # EdgeRows slack mode requires this
    prime master with one column per structural entity
    (price against +inf duals so every source's tree/path is feasible)

  for iter in 0 .. max_iterations:
    status = master.solve()
    if status != Optimal: break

    # Separation: scan columns for capacity violations, add cuts
    new_cap_arcs = master.add_violated_capacity_constraints()
    if new_cap_arcs:
      master.solve()                     # re-solve so duals reflect cuts

    # Pricing: generate negative-reduced-cost columns
    pi = master.get_structural_duals()
    mu = master.get_capacity_duals()
    new_cols = pricer.price(pi, mu)

    if new_cols is empty:
      new_cols = pricer.price(pi, mu, final_round=True)  # retry postponed sources
      if new_cols is empty:
        if master.has_active_slacks():
          master.bump_active_slacks()     # grow slack cost, continue
        else:
          return optimal(master.get_obj())

    master.update_capacity_row_activity()
    master.update_column_ages()
    master.bump_active_slacks()           # end-of-iter bump for next solve
    master.purge_aged_columns()
    master.purge_nonbinding_capacity_rows()
    master.add_columns(new_cols)

  return stopped(last_obj)                # max_iterations reached
```

Pricing for a single source $s$:

```text
pricer.price_one_source(s, duals, mu):
  rc_a = max(0, (c_a - mu_a)) scaled by 1e9    # integer reduced costs
  run A* Dijkstra from s until every reachable sink of S_s is settled
  for each sink t_k of s:
    skip t_k if unreachable — A*'s heap exhausted without settling it
    extract the shortest path, compute the floating-point true RC
    PATH:  emit one column per reachable commodity with true_rc < -tol
    TREE:  aggregate demand-weighted arc flow across reachable sinks of s,
           emit one tree column (partial if some sinks were unreachable)
           if the aggregate RC is negative
```

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
| `-DMCFCG_NATIVE_ARCH=OFF` | ON | Disable `-march=native` tuning; produces a portable binary |

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
| `--threads N`             | 1     | Pricing threads (`0` = hardware concurrency) |
| `--batch-size N`          | 0     | Sources priced per batch (`0` = all) |
| `--solver NAME`           | highs | LP backend: `highs`, `cuopt`, `copt` |
| `--col-age-limit N`       | 5     | Purge columns after N idle iters (`0` disables) |
| `--row-inactivity N`      | 5     | Purge cap rows after N idle iters (`0` disables) |
| `--neg-rc-tol X`          | -1e-3 | Reduced-cost acceptance threshold |
| `--strategy S`            | pricer-heavy | `pricer-heavy` or `pricer-light` preset |

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
