# Bounded single-source pricing ablation (gh #41, manuscript section 3.3)

Bounded pricing (`mcfcg_cli --bounded-pricing`) stops a source's A* once
the frontier proves no negative-reduced-cost column remains, instead of running
until every sink is settled. It is **exact**: the column set it emits is
identical bit-for-bit — cost, reduced cost, full arc list / arc-flow vector —
pinned by `FeatureTests.BoundedPricingShadow{Tree,Path,IntermodalTree}`. So the
only open question was whether it is *faster*.

The answer is family-dependent. It always saves pricing time, but pricing share
is the ceiling on converting that into wall clock, and on three of the four
families that share is 0.2–4.3% — so the saving disappears into the noise. On
intermodal, where pricing is 75–85% of the clock, it is worth **−2.8%** over 100
paired cells on five backends (round (b)).

**The flag ships off by default**, because the default is global and most
families have nothing to win. That is not the same as the flag being useless, and
this file is the argument for both halves.

The measurement is split into rounds, each a directory named for the axis it
varies. This file carries the argument — mechanism, gain model, traps — and each
round's own README carries what that round measured and which of its numbers are
quotable:

| round | dir | varies | status |
|---|---|---|---|
| (a) | [`families/`](families/README.md) | four families at one backend, 3 reps | landed (gh #43) |
| (b) | [`backends/`](backends/README.md) | five backends on one family, HiGHS at 3 off reps | landed (gh #44) |

Every number below comes from round (a) unless it says otherwise, and every one
of them is re-derivable from that round's tracked logs.

Scope: every number in this file and in the two round READMEs derives from those
rounds' tracked logs, with one exception — the rejected stale-arc fix, whose
measurement is transcribed in full in the appendix below. That one measured a
code change that was then reverted, so no tracked log can carry it and
re-deriving it means re-applying the change; the table is the record.

Nothing from the flag's development survives as an artifact: the pre-rename
passes, the implementation variants that preceded the shipped bound, and the
all-backend on-arm pass whose banner the analyzer refuses by policy were all
deleted rather than archived (gh #45). Rounds (a) and (b) re-measured the shipped
implementation from scratch, and nothing here cites any of it.

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

Pricing share is the ceiling on anything the bound can deliver, and it is known
up front from `results/cg_benchmark.csv`. Under COPT, per family:

| family | pricing share | verdict |
|---|---|---|
| planar | **0.2%** | no gain possible; run as the null |
| grid | **1.4–3.1%** | the decisive test — see below |
| transportation | **4.3%** | settled by grid; not tested on its own merits |
| intermodal | **78–80%** | the only family worth investigating |

What round (a) measured against that:

| family (formulation, backend) | pricing share | Δ pricing time | wall clock |
|---|---|---|---|
| intermodal (tree, copt-cpu) | 85.3% | −5.5% | **−4.8%** |
| intermodal (tree, copt-gpu) | 71.4% | −7.5% | **−6.2%** |
| transportation (tree, copt-gpu) | 22.6% † | −6.5% | **−4.1%** † |
| grid (tree) | 1.9% | **−26.0%** | **−0.33%** |
| grid + planar (path) | 1.0% | −0.5% | **+0.1%** |
| planar (tree) | 1.2% | −6.2% | **+1.1%** |

**grid tree is the argument in one row**: the bound removes a quarter of the
pricing time and the clock does not move, because pricing is 1.9% of it. A
mechanism that cannot pay at 3% share cannot pay at 4.3% either, which is why
transportation is settled by grid rather than by its own cells.

† The transportation row covers 6 of 9 instances — **9% of the family's wall
clock**. The 3 excluded on cost are the other 91% and price for 2.5%. Family-wide
the share is **4.3%**; see `families/README.md` before quoting this row.

Intermodal under copt-cpu is the only family where the saving is mechanistically
attributable: −6.65 s of a −6.79 s wall-clock gain is pricing, with LP flat to
0.01 s. Its conservative estimate is `85.3% × −2.8% ≈ −2.4%` (the cheaper-per-
price term alone); the measured −4.8% also includes a trajectory that shortened
162 → 154 iterations, which is real here but not predictable per instance.

**The wall-clock column is not the effect.** Where the trajectory moves, that
column is ±Δiterations: intermodal's family total covers cells from −27%
(SBT-18765-0, 9 → 7 iterations) to +20% (BUS-23688-0, 13 → 20). The effect is
`pricing_share × per-price`, which on the one family where both are measurable is
85.4% × −2.8% ≈ **−2.4%**. The copt-gpu per-price column is excluded because that
executor's off arm carries an inflated `t_PR` baseline — round (a)'s README works
through the evidence.

A per-price saving is only quotable on a cell where the trajectory held still in
both senses — `traj_moved=0` (the arms' median iteration and column counts agree)
**and** `traj_stable=1` (the repetitions *within* each arm agreed too). Intermodal
has 4 such cells per backend. Transportation has **none**: its only `traj_moved=0`
cell is ChicagoSketch (14 iterations and 2,722 columns in both arms), and its reps
disagree *within* an arm, so the medians agree by coincidence rather than because
nothing moved.

For grid/planar the entry is a range rather than a median because 39 of those 48
cells price for under 0.1 s in total, where the log's 3-digit timing is pure
quantization — the all-cell path median is exactly +0.0%. The range covers the 9
cells whose pricing time is large enough to measure, and those 9 are where the
mechanism shows: **tree** saves 3.6–29.9% per price (median −22.3%) because its
bound tightens on every settle, while **path** ranges −2.9 to +5.6% because its
`max pi` bound waits on the most expensive remaining commodity, which settles
last.

The gain is bounded by

    wall gain  ~  pricing_share x per-price saving   -   LP_share x Delta-iterations

and the two factors of the first term are **anticorrelated across families**.
The bound prunes the tail of a multi-target search, so its per-price saving
grows with commodities per source: grid/planar tree, at 1.2–26 commodities per
source, saves up to 30% on every price. But those families spend **1.0–1.5%** of
wall clock pricing, so none of it reaches the clock. Intermodal is the mirror
image — exactly **one commodity per source**, so −2.8% per price, but **71–85%** of
wall clock spent pricing. The product never exceeds ~2.4% on any family, and it is
that only on instances that finish in under a minute: across the committed
benchmark the pricing share collapses as instances get harder (planar2500 0.1%,
Philadelphia 1.1%, Birmingham 1.6%, Austin 4.3%), while one extra CG iteration
there costs 0.4–1.5% of wall clock. Scaling intermodal up does not open this
up: it has one commodity per source *by construction* — each request is its own
source — so the small per-price saving is structural, not a size effect.

Two caveats that the raw wall-clock column will mislead you about:

- **`t_PR` alone is not a pricing measurement.** The bound shifts the CG
  trajectory (two channels, both under "Trajectory channels" below; neither is
  a correctness bug), so an arm can price fewer sources and post a lower `t_PR`
  with every individual price costing the same. The `per_price_us` column
  (`t_PR / priced_sources`) is the trajectory-immune metric — computable from
  each run's `[bounded-pricing] cut=… priced=…` log line — and the
  `traj_moved` / `traj_stable` pair flags the cells where it can be read.
- **Family totals inside the noise floor are noise.** transportation's −5.2% is
  not a gain: 5 of its 6 cells moved the trajectory, ChicagoRegional's −8.1%
  arrives with 50 → 48 iterations, and Barcelona posts +2.3% wall clock on 0.7%
  pricing share — a number the pricer cannot have caused in either direction. The
  `spread_off_pct` column carries the rep-to-rep noise floor per cell.

Where the model holds still it is accurate. On the four qualifying copt-cpu
intermodal cells (SBT-31275-0, SBT-43785-0, SBT-56295-0, SBT-6255-0), predicted
−3.3/−2.3/−2.8/−0.7% vs measured −3.4/−2.3/−2.9/−1.0% — within 0.26pp everywhere
(compare the `pred_wall_pct` and `d_t_tot_pct` columns). **Quote the copt-cpu
arm.** The same four cells under copt-gpu run identical trajectories and price
identical source counts, yet report more than twice the per-price saving
(−5.59% ± 1.99 pp vs −2.53% ± 1.08 pp) and miss the model by up to 1.59pp: that
executor's *off* arm carries an inflated, poorly repeatable `t_PR` baseline.
[`families/README.md`](families/README.md) works through the evidence.

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
HiGHS ranges wider still. Round (a)'s own sweeps show slightly different
shares — 85.4% for copt-cpu, 71.8% for copt-gpu — because they cover tree only
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
  filtering; SBT-56295 alone paid **+68%** at an unchanged iteration count. The
  full table is in the appendix below — it is the one measurement here with no
  tracked logs behind it.
- **A weaker lower bound.** `salvage_lagr_term` substitutes
  `d_k·(bound_f/SCALE − margin)` for the `sp_k` a truncated search never
  computed. Valid but weaker, so `best_lb` differs and the gap exit fires on a
  different iteration. Live in every configuration; moves the iteration count
  with an identical column set.

## Files

Each round owns its artifacts; nothing sits at this level but this file.

| path | what |
|---|---|
| `families/README.md` | round (a): scope, sweep commands, what it measured, what is quotable |
| `families/runs.csv` | one row per run log: timings, iterations, columns, bound fire counts, `per_price_us` |
| `families/summary.csv` | one row per cell, off and on arms paired and reduced to medians, with deltas |
| `families/logs/<sweep>/logs_<executor>_<off\|on>_<rep>/` | the raw run logs both CSVs are derived from |
| `backends/README.md` | round (b): the five-backend result, and why the reported HiGHS penalty was not real |
| `backends/runs.csv` | as above, for round (b): 240 runs |
| `backends/summary.csv` | as above, for round (b): 100 cells, wall clock decomposed into `t_pr`/`t_lp` |
| `backends/logs/intermodal_path_tree/logs_<solver>_<off\|on>_<rep>/` | round (b)'s raw run logs |

Derive every round's CSVs from its tracked logs — this re-parses and never
re-solves. Add `--round families|backends` to do just one:

    python3 scripts/analyze_bounded_pricing_ablation.py

Unlike the main benchmark — whose logs live in the gitignored `bench_runs/` and
are regenerable by re-running `benchmark_solvers.py` — these logs are **tracked**.
The ablation settles a design question rather than feeding a results table, so
the logs are the primary artifact and the CSVs are derived from them.

Sanitisation is exactly one substitution: the absolute repo prefix, wherever it
appears — the `# cmd:` header, the instance field of the embedded result-CSV row,
and transportation's `TNTP: net=/trips=` line. Nothing else is altered, so each
tracked log is byte-identical to its original once that one string is removed.

See each round's README for its sweep commands. Both arms of a comparison must
run in the same session on the same build, alternating `off` and `on` per
repetition.

## Two things not to redo

- **Do not read per-instance cells off a copt-gpu grid/planar or transportation
  sweep.** Re-running the *same* config differs on **26 of 48** grid/planar cells
  and on **all 6** transportation cells (`traj_stable=0`), because the GPU
  barrier's interior point shifts and lazy separation then picks a different
  violated-capacity set. Concretely: grid10 tree diverges at iteration 13 on
  `#row` 1032 vs 1033, with identical `#col` and `LP_obj`. That is why 5 of 6
  transportation cells read `traj_moved=1` and none is quotable per-price. Read
  those sweeps' family totals, not their per-instance cells. Intermodal is
  near-deterministic on *both* executors — 7 of 10 cells repeat their iteration
  and column counts across reps in both arms, and the same three fail on each —
  so the instability there is the instance, not the backend. But see round (a)'s
  README before quoting a copt-gpu per-price number even on those cells.
- **Do not compare an arm against `results/cg_benchmark.csv`.** Those cells are a
  valid unbounded configuration — no log behind them carries the flag, and the
  feature postdates them entirely — but they are from other sessions. On cells
  where the bound provably changed nothing (identical iterations *and* columns),
  wall clock still moved by −5.5% to **+15.3%** between sessions across the 39
  such cells. The worst is HiGHS on SBT-43785-0 tree — 6 iterations and 45,733
  columns in both arms, yet `t_LP` up **10.7%** on a byte-identical LP sequence.
  Both arms must come from one session. The sharpest case is the **+18…+32%
  HiGHS penalty** an earlier cross-session comparison reported, which is the
  result round (b) was opened to settle: re-running both arms in one session
  found no penalty at all — HiGHS path **−9.8%**, tree **+7.7%**, with `t_PR`
  down on both. The cross-session figure was not merely uncitable in magnitude,
  its sign was wrong on one of the two formulations, and the effect it named
  does not exist.

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

## Appendix: the rejected stale-arc fix

The one measurement cited here with no tracked logs behind it, transcribed
because the code it justifies is still in the tree and the run directory is not
(gh #45). It measured a **reverted** change, so no log of the shipped build could
carry it; reproducing it means re-applying the change below.

A cut source does not refresh `_source_arcs[s]`, so `filter_for_new_caps` decides
postponement from that source's last *complete* price. `_source_cut[s]` is
already a sticky "this arc set is stale" flag, so making the filter respect it is
a one-line change:

```cpp
// in filter_for_new_caps — measured, rejected, NOT in the tree
affected = _source_cut[s] || std::any_of(...);
```

Collected 2026-08-14 on the section 3 host: intermodal, tree, `PricerHeavy`,
COPT 8.0.1 GPUMode 0 (copt-cpu), `--time-limit 7200`, median of 2 reps per arm,
both arms in one session, bounded pricing **on** in both — the arms differ only
in the line above. Both arms hit all 10 reference optima.

| instance | iters before | iters after | t before (s) | t after (s) | Δ |
|---|---|---|---|---|---|
| BUS-13160-0 | 21 | 19 | 4.8 | 5.3 | +12% |
| BUS-18424-0 | 32 | 34 | 8.6 | 9.7 | +12% |
| BUS-23688-0 | 20 | 27 | 9.6 | 15.5 | +61% |
| BUS-2632-0 | 24 | 18 | 1.6 | 1.3 | −22% |
| BUS-7896-0 | 20 | 24 | 2.9 | 3.4 | +17% |
| SBT-18765-0 | 7 | 7 | 8.6 | 8.1 | −5% |
| SBT-31275-0 | 6 | 6 | 24.6 | 25.6 | +4% |
| SBT-43785-0 | 6 | 6 | 35.9 | 36.4 | +1% |
| SBT-56295-0 | 10 | 10 | 54.9 | 92.2 | +68% |
| SBT-6255-0 | 8 | 10 | 2.2 | 3.2 | +47% |
| **total** | | | **153.7** | **200.7** | **+31%** |

Rejected. At a 65–77% fire rate nearly every source reads as affected, so the
filter stops filtering: SBT-56295-0 paid **+68% at an unchanged iteration
count**, which is pure extra pricing and not a trajectory effect. Only the
comments explaining the choice were kept, plus
`FeatureTests.BoundedPricingFilterUsesStaleArcsForCutSources`, which pins the
shipped behaviour.

Two caveats on the numbers. They are same-session wall clocks and are not
comparable to any other sweep in this file. And this run is where intermodal's
near-determinism was first measured across sessions rather than within one: the
off arm moved on BUS-18424-0 (42 → 33 iterations at an identical column count)
and the on arm on BUS-13160-0 (15,008 → 14,990 columns at 21 iterations). The
+31% is far larger than that residual, which is why the rejection stands on 2
reps.
