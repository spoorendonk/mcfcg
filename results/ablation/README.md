# Bounded single-source pricing ablation (gh #41, manuscript section 3.3)

Bounded pricing (`mcfcg_cli --bounded-pricing`) stops a source's A* once
the frontier proves no negative-reduced-cost column remains, instead of running
until every sink is settled. It is **exact**: the column set it emits is
identical bit-for-bit — cost, reduced cost, full arc list / arc-flow vector —
pinned by `FeatureTests.BoundedPricingShadow{Tree,Path,IntermodalTree}`. So the
only open question was whether it is *faster*.

The flag was called `--pricing-cutoff` when these sweeps ran and was renamed in
gh #42 — it is a bound, not a cutoff. Every log in this directory therefore
carries the old `[pricing-cutoff]` banner, which the analyzer still parses.

It is not, by enough to matter. **The flag ships off by default** and this
directory is the evidence.

Scope: what is archived here is the paired on/off A/B across three families. Two
supporting measurements cited below are *not* — the all-backend on-arm pass
and the rejected stale-arc experiment (+31% wall clock) both live in the
gitignored `bench_runs/`. Neither is load-bearing for the
conclusion below; both are flagged where they are cited.

## How it works

`CGParams::bounded_pricing` (CLI `--bounded-pricing`) bounds the best reduced cost
still reachable from the frontier and stops once that bound clears
`neg_rc_tol`. The bound differs by formulation:

- **path** — `max π` over the unsettled sinks.
- **tree** — the residual convexity budget over the *sum* of the remaining
  demands.

Both add an allowance so the cut need only prove `rc ≥ neg_rc_tol` rather than
`rc ≥ 0`, and **that allowance is what makes it fire at all**: 65–77% of searches
cut with it, against 0–32% without. The reason is that at a master optimum every
structural row has a basic column at reduced cost 0, so the frontier reaches the
sink at almost exactly the dual — an exact-zero test almost never triggers.

## The result

| family (formulation, backend) | pricing share | per-price saving | total `t_PR` | wall clock |
|---|---|---|---|---|
| intermodal (tree, copt-cpu) | 85.3% | −2.5% (4 cells) | −4.3% | **−3.6%** |
| intermodal (tree, copt-gpu) | 71.5% | −1.7% (4 cells) | −4.7% | **−3.7%** |
| transportation (tree, copt-gpu) | 22.8% | *no qualifying cell* | −2.8% | **−2.4%** |
| grid + planar (tree, copt-gpu) | 1.5% | −2.7 … −25.6% | −16.3% | **−1.3%** |
| grid + planar (path, copt-gpu) | 1.0% | +6.0 … −10.4% | −0.9% | **+0.3%** |

A per-price saving is only quotable on a cell where the trajectory held still in
both senses — `traj_moved=0` (the arms' median iteration and column counts agree)
**and** `traj_stable=1` (the repetitions *within* each arm agreed too). Intermodal
has 4 such cells per backend. Transportation has **none**: its only `traj_moved=0`
cell is Sydney, whose three on-arm reps are not unanimous (one ran 24 iterations /
13,210 columns against 25 / 12,647 for the other two), so the medians agree by
coincidence rather than because nothing moved.

For grid/planar the entry is a range rather than a median because 38 of those 48
cells price for under 0.1 s in total, where the log's 3-digit timing is pure
quantization — their median delta is exactly +0.0%. The range covers the 10 cells
whose pricing time is large enough to measure, and those 10 are where the
mechanism shows: **tree** saves 2.7–25.6% per price because its bound tightens on
every settle, while **path** ranges +6.0 to −10.4% because its `max pi` bound
waits on the most expensive remaining commodity, which settles last.

The gain is bounded by

    wall gain  ~  pricing_share x per-price saving   -   LP_share x Delta-iterations

and the two factors of the first term are **anticorrelated across families**.
The bound prunes the tail of a multi-target search, so its per-price saving
grows with commodities per source: grid/planar tree, at 1.2–26 commodities per
source, saves up to 25% on every price. But those families spend **1.0–1.5%** of
wall clock pricing, so none of it reaches the clock. Intermodal is the mirror
image — exactly **one commodity per source**, so ~2% per price, but **71–85%** of
wall clock spent pricing. The product never exceeds ~2% on any family, and it is
~2% only on instances that finish in under a minute: across the committed
benchmark the pricing share collapses as instances get harder (planar2500 0.1%,
Philadelphia 1.1%, Birmingham 1.6%, Austin 4.3%), while one extra CG iteration
there costs 0.4–1.5% of wall clock. Scaling intermodal up does not open this
up: it has one commodity per source *by construction* — each request is its own
source — so the ~2% per-price saving is structural, not a size effect.

Two caveats that the raw wall-clock column will mislead you about:

- **`t_PR` alone is not a pricing measurement.** The bound shifts the CG
  trajectory (two channels, both under "Trajectory channels" below; neither is
  a correctness bug), so an arm can price fewer sources and post a lower `t_PR`
  with every individual price costing the same. The `per_price_us` column
  (`t_PR / priced_sources`) is the trajectory-immune metric — computable from
  each run's `[bounded-pricing] cut=… priced=…` log line (`[pricing-cutoff]` in
  these archived logs) — and the
  `traj_moved` / `traj_stable` pair flags the cells where it can be read.
- **Family totals inside the noise floor are noise.** transportation's −2.4%
  is not a gain: Barcelona shows −2.2% wall clock while its *per-price* cost rose
  4.7%, and pricing is 0.7% of Barcelona's wall clock — the change cannot have
  come from the pricer. The `spread_off_pct` column carries the rep-to-rep noise
  floor per cell.

Where the model holds still it is accurate. On the four qualifying copt-cpu
intermodal cells (SBT-31275-0, SBT-43785-0, SBT-56295-0, SBT-6255-0), predicted
−2.3/−2.5/−1.6/−2.1% vs measured −2.3/−2.4/−1.7/−2.3% — within 0.13pp everywhere
(compare the `pred_wall_pct` and `d_t_tot_pct` columns). Quote the copt-cpu arm:
the same four cells under copt-gpu have only 2 reps each, and there the model
misses SBT-6255-0 badly (+0.1% predicted, −2.2% measured).

The second term is why a single-backend measurement of this flag is worthless.
Intermodal LP/pricing split by backend, computed from `results/cg_benchmark.csv`
(path and tree, the committed unbounded configuration):

| backend | LP share | pricing share |
|---|---|---|
| copt-cpu | 3–4% | 81–83% |
| copt-gpu | 8–10% | 75–78% |
| mosek | 6–7% | 80% |
| cuopt | 15% | 73–74% |
| **highs (HiPO)** | **49–53%** | **40–43%** |

Where the LP is 3% of runtime the trajectory shift is nearly free; where it is
half the clock it dominates. (These shares are of the whole family; per instance
HiGHS ranges wider still. The ablation's own sweeps show slightly different
shares — 85.3% for copt-cpu, 71.5% for copt-gpu — because they cover tree only
and a different session; the table above is the one an outside reader can
recompute from the release.)

## Implementation traps

Three traps the implementation must respect, each a live bug caught in review.
All three are pinned by `FeatureTests.BoundedPricing*` in
`test/integration_test.cpp`; read them before touching the bounded-pricing code.

- **`MAX_BOUND` is overloaded.** It is both `scale_dual`'s saturation value and
  `compute_lower_bounds_to_targets`' `UNREACHED` sentinel, so a frontier at or
  above it means **dead ends**, not a dual proof. Cutting there salvages ~4.6e9
  into `best_lb` and suppresses the tree's partial column.
- **Zero-demand commodities.** One drives the tree's remaining demand to 0 with
  budget left; cutting there suppresses a strictly improving column on every
  iteration and CG reports the result as optimal. CommaLab keeps zero-demand
  rows — only TNTP filters them — so this is reachable on real instances.
- **The tree budget must stay `+inf` through the warm start's `+inf` duals.** It
  *divides* by remaining demand, so a saturated finite budget becomes a
  reachable threshold and a source goes unseeded — fatal in `EdgeRows`, which
  has no demand slack to absorb it.

The column set is pinned bit-for-bit — cost, reduced cost, and the full arc list
/ arc-flow vector — by
`FeatureTests.BoundedPricingShadow{Tree,Path,IntermodalTree}`, which run the real
`solve_cg` with the bound **off** while shadowing every dual vector with a
second bounded-on pricer (25k fires / 3.2k columns compared on BUS-2632-0 alone).

## Trajectory channels

**The bound is column-identical but not trajectory-neutral.** Switching it on
moves intermodal iteration counts by up to ±10 and changes column counts. That is
expected and is not a dropped column. Two channels cause it, neither a
correctness bug — and note the pricing-exhausted `final_round` re-prices every
source regardless of postponement:

- **Stale arc sets.** A cut search does not refresh `_source_arcs[s]`
  (`should_record_arcs`: a partial set would understate the routing and postpone
  a source that a new capacity row does affect), so `filter_for_new_caps` decides
  postponement from the routing that source had at its last *complete* price.
  Different sources priced → different columns that iteration → a different LP →
  a different lazily separated capacity set. Live only when the filter is on
  (`pricer_heavy || pricing_filter`).

  Do **not** "fix" this by treating a cut source as affected — `_source_cut[s]`
  is already that flag. Measured **+31% wall clock** on intermodal, because a
  65–77% fire rate makes nearly every source affected and the filter stops
  filtering; SBT-56295 alone paid **+68%** at an unchanged iteration count. (This
  is the stale-arc experiment noted as unarchived above.)
- **A weaker lower bound.** `salvage_lagr_term` substitutes
  `d_k·(bound_f/SCALE − margin)` for the `sp_k` a truncated search never
  computed. Valid but weaker, so `best_lb` differs and the gap exit fires on a
  different iteration. Live in every configuration; moves the iteration count
  with an identical column set.

## Files

| path | what |
|---|---|
| `pricing_cutoff_runs.csv` | one row per run log: timings, iterations, columns, bound fire counts, `per_price_us` |
| `pricing_cutoff_summary.csv` | one row per cell, off and on arms paired and reduced to medians, with deltas |
| `logs/<sweep>/logs_<solver>_<off\|on>_<rep>/` | the raw run logs both CSVs are derived from |

Regenerate both CSVs from the tracked logs:

    python3 scripts/analyze_pricing_cutoff_ablation.py

Unlike the main benchmark — whose logs live in the gitignored `bench_runs/` and
are regenerable by re-running `benchmark_solvers.py` — these logs are **tracked**.
The ablation is a one-off measurement settling a design question, not a number we
intend to refresh, so the logs are the primary artifact.

## Sweeps

Each sweep ran both arms in the same session on the same build, alternating
`off` and `on` per repetition. When the logs were tracked, the absolute repo
prefix was stripped everywhere it appeared — the `# cmd:` header, the instance
field of the embedded result-CSV row, and transportation's `TNTP: net=/trips=`
line. Nothing else was altered: each tracked log is byte-identical to its
original once that one prefix string is removed.

The "originally" column below names local, gitignored working directories. Like
the `source` column of `results/cg_benchmark.csv` it is a breadcrumb, not a path
you can open from the release.

| sweep | originally | family | formulations | backends | reps |
|---|---|---|---|---|---|
| `intermodal_tree` | `bench_runs/issue41_cutoff_v3` | intermodal (10 instances), `--strategy pricer-heavy` | tree | copt-cpu, copt-gpu | 3, 2 |
| `transportation_tree` | `bench_runs/issue41_transportation` | transportation (6 instances) | tree | copt-gpu | 3 |
| `gridplanar_path_tree` | `bench_runs/issue41_multitarget` | grid (15), planar ≤1000 (9) | path, tree | copt-gpu | 3 |

Reproducing a sweep — the two arms differ only in `--extra-args`:

    for rep in 1 2 3; do
      for arm in off on; do
        [ "$arm" = on ] && extra=--extra-args=--bounded-pricing || extra=
        python3 scripts/benchmark_solvers.py --families intermodal --solvers copt-cpu \
            $extra --out    bench_runs/SWEEP/copt-cpu_${arm}_rep${rep}.csv \
                   --logdir bench_runs/SWEEP/logs_copt-cpu_${arm}_rep${rep}
      done
    done

## Two things not to redo

- **Do not A/B under copt-gpu.** Re-running the *same* config on grid/planar
  differs on 23/48 instances — the same rate as on-vs-off — because the GPU
  barrier's interior point shifts and lazy separation then picks a different
  violated-capacity set. Concretely: grid10 tree diverges at iteration 13 on
  `#row` 1032 vs 1033, with identical `#col` and `LP_obj`. `transportation_tree` is on the same backend and is no
  better: 4 of its 6 cells fail to reproduce their own iteration and column
  counts across same-config reps, which is why 5 of 6 read `traj_moved=1` and
  none is quotable per-price. Read both sweeps' family totals, not their
  per-instance cells. Intermodal under copt-cpu is near-deterministic: 9 of 10
  cells repeat the same iteration *and* column counts across the off-arm reps,
  10 of 10 on the on-arm.
- **Do not compare an arm against `results/cg_benchmark.csv`.** Those cells are a
  valid unbounded configuration — no log behind them carries the flag, and the
  feature postdates them entirely — but they are from other sessions. On cells
  where the bound provably changed nothing (identical iterations *and* columns),
  wall clock still moved by −5.5% to **+15.3%** between sessions across the 39
  such cells. The worst is HiGHS on SBT-43785-0 tree — 6 iterations and 45,733
  columns in both arms, yet `t_LP` up **10.7%** on a byte-identical LP sequence.
  Both arms must come from one session. This is also why the on-arm pass
  reading **+18…+32% on HiGHS** against the committed cells is quoted nowhere as
  a number: the direction is credible — HiGHS is the one backend whose LP share
  is large enough for the trajectory term to dominate — but the magnitude is
  cross-session and therefore not citable.

## Backends

Every sweep here ran on COPT 8.0.1, matching PROVENANCE.txt section 1, with the
pinned barrier regime (presolve off, crossover off, tol 1e-4) that each log's
`[lp-config]` banner records.

The HiGHS row of the LP/pricing table is the one figure the argument leans on
that these sweeps did not produce — it comes from `results/cg_benchmark.csv`. It
is genuinely HiPO and not a silent dual-simplex fallback, which is worth stating
because the `[lp-config]` banner cannot distinguish the two (it echoes the
*requested* method). Re-checked directly on this build:

    mcfcg_cli data/intermodal/BUS-2632-0.txt.gz --solver highs --formulation tree \
        --strategy pricer-heavy --verbose-solver

logs `Running HiPO` for all 17 LP solves, with no `Using dual simplex solver` and
no `features unavailable` line. The benchmark runs themselves were not made with
`--verbose-solver`, so that banner is absent from their logs.
