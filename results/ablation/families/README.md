# Round (a): bounded pricing across four families, one backend (gh #43)

The bounded-pricing ablation is split into rounds, each named for the axis it
varies. **This round varies family at a fixed backend.** Round (b) (`../backends/`,
gh #44) varies backend at a fixed family.

Read `../README.md` first: it carries the mechanism, the gain model, the three
implementation traps and the two comparisons not to repeat. This file is what
round (a) *measured* — scope, how to reproduce it, and which of its numbers are
quotable.

Everything here replaces a mixed-backend, mixed-rep archive that was deleted in
gh #42 rather than carried past the flag's rename. This round is uniform: **one
executor per family cell, 3 reps everywhere, both arms in one session on one
build.**

## What ran

| sweep | family | formulations | executor | instances | reps |
|---|---|---|---|---|---|
| `gridplanar_path_tree` | grid, planar | path, tree | copt-gpu | 15 + 9 (planar ≤ 1000) | 3 |
| `transportation_tree` | transportation | tree | copt-gpu | 6 of 9 | 3 |
| `intermodal_tree` | intermodal, `--strategy pricer-heavy` | tree | copt-gpu **and** copt-cpu | 10 | 3 |

444 logs, 74 paired cells, 1 h 06 min on COPT 8.0.1. Every run passed against its
reference optimum; no cell hit the 1800 s limit.

## Why these families, in this order

The round is a ladder, not a survey. The gain is bounded by `pricing_share x
per-price saving`, so **pricing share is the ceiling** and it is knowable up
front from the committed benchmark. Family pricing shares under COPT:

| family | pricing share | cost of one arm-rep | hypothesis |
|---|---|---|---|
| planar | **0.2%** | 131 s | no gain possible; run it because it is cheap and pins the null |
| grid | **1.4–3.1%** | 148 s | the first share worth a look, and also cheap — test it |
| transportation | **4.3%** | 187 s | only if grid works |
| intermodal | **78–80%** | 162 s | the one family where the ceiling is high — test it |

Each rung decides the next. planar returns the null it should. **grid is the
decisive rung**: the bound works there — pricing time drops **22.4%** on tree —
and the clock does not follow (**+1.95%**, i.e. it got slower), because 3.2% of
nothing is nothing and the LP noise is larger than the whole pricing term. A
mechanism that cannot pay at 3% pricing share cannot pay at 4.3% either, so
transportation is settled by grid rather than by its own numbers.

Only intermodal has a ceiling high enough to matter, and it is the one family
where the measured gain is mechanistically attributable rather than incidental.

### The backends

**COPT, because it has the fastest LP.** Over the 81 cells all five backends
solve to optimality, total LP time is copt-cpu 1.00x, cuopt 1.41x, copt-gpu
1.92x, mosek 2.22x, highs 11.07x. Smallest LP share is what this study wants
twice over: it maximises the pricing signal, and it minimises the model's second
term (`-LP_share x Delta-iterations`), which is the confound that makes the flag
backend-specific. copt-cpu and copt-gpu are one library exercised two ways
(GPUMode 0 vs 2), so the second executor is a control rather than a second
product. Which backends the gain *survives* is round (b)'s question, not this
one's.

### Scope decisions inside the families

- **Path only on grid/planar.** Transportation path costs ~9.5x tree (Austin and
  Philadelphia clamp at the 7200 s limit); intermodal path is identical to tree,
  one commodity per source. grid/planar is the only affordable place the **path**
  bound is observable at all, and section 3.3 claims it separately.
- **Transportation at 6 of 9 instances**, Austin/Birmingham/Philadelphia excluded
  on cost. Read the caveat under the result table before quoting this family:
  those three are 91% of its wall clock, and dropping them changes what its
  pricing share means.

## The result

Per family, medians of 3 reps, off arm as the baseline:

| family (formulation, executor) | pricing share | Δ pricing time | **Δ wall clock** |
|---|---|---|---|
| intermodal (tree, **copt-cpu**) | 85.1% | −4.5% | **−3.7%** |
| intermodal (tree, copt-gpu) | 80.5% | −4.5% | **−3.6%** |
| transportation (tree, copt-gpu) | 26.3% † | −8.7% | **−1.9%** † |
| grid (tree) | 3.2% | **−22.4%** | **+1.95%** |
| grid (path) | 1.7% | +0.3% | +0.10% |
| planar (tree) | 1.6% | −8.9% | −1.50% |
| planar (path) | 1.0% | −0.8% | +0.09% |

**grid tree is the whole argument in one row.** The bound does exactly what it
claims — better than a fifth of the pricing time, gone — and the clock does not
notice; it drifts the *wrong* way, because pricing is 3.2% of it and 0.31 s of
real pricing saving is sitting under 1.13 s of LP noise.

**The path bound is a separate mechanism and it barely fires usefully.** grid
path moves pricing by +0.3% and planar path by −0.8%, against −22.4% and −8.9%
for the same instances under tree. That asymmetry is inherent: the tree bound is
the residual convexity budget over the *sum* of remaining demands and tightens on
every settle, while the path bound is `max π` over unsettled sinks and is hostage
to the single most expensive remaining commodity, which usually settles last.

† **Do not read the transportation row as a family figure.** It covers the 6
instances this round ran, which are **9% of the family's wall clock**. The 3
excluded on cost (Austin, Birmingham, Philadelphia) are the other **91%**, and
they price for **2.5%**. Family-wide, transportation prices for **4.3%** — grid's
range, not intermodal's. The exclusion kept precisely the instances where the
bound looks best, which is why grid, not this row, settles the family.

### Where the wall-clock gain actually comes from

Absolute seconds, so the terms add up:

| group | Δ wall | from pricing | from LP | iters off→on |
|---|---|---|---|---|
| intermodal copt-cpu | −5.06 s | **−5.20 s** | +0.05 s | 159 → 154 |
| intermodal copt-gpu | −5.11 s | **−5.14 s** | −0.09 s | 159 → 154 |
| transportation copt-gpu | −3.03 s | **−3.71 s** | +0.04 s | 194 → 192 |
| grid + planar tree | +0.05 s | −0.39 s | +0.50 s | 514 → 516 |
| grid + planar path | +0.07 s | +0.00 s | +0.05 s | 275 → 275 |

The three families with a pricing share worth anything are all clean rows here:
**the wall saving is the pricing saving**, with LP flat to a twentieth of a
second. That includes both executors — under COPT the LP is a few percent of an
intermodal run either way, so there is nothing for the trajectory term to move.
grid/planar is the mirror image: the pricing saving is real but 0.39 s, and the
LP noise it sits under is larger than it.

**Verdict: intermodal is the only family worth investigating further**, and that
is what round (b) (gh #44) takes up. The conservative, mechanistic estimate of
its gain is `85.1% × −2.4% ≈ −2.1%` — the cheaper-per-price term alone. The
measured −3.7% includes a trajectory that also shortened, 159 → 154 iterations,
which is real here but not predictable per instance, and which round (b) shows
running the other way just as easily on a backend with a larger LP share.

**The flag still ships off.** −2.1% to −3.7% on one family, on instances that
finish in under a minute, is not worth a per-family default — and whether it
survives on a backend with a larger LP share is exactly what round (b) asks.

The two arms agree on the LP optimum everywhere: worst `d_obj_rel` is 6.95e-05
(grid3 path), inside the CG gap tolerance. That is the exactness check this data
can make on its own; the bit-for-bit column identity is a stronger statement and
is pinned in C++ by `FeatureTests.BoundedPricingShadow{Tree,Path,IntermodalTree}`.

## Two executors, and why the round carries both

copt-cpu and copt-gpu are one library exercised two ways, so on a metric that
should be executor-independent they are a control on the measurement itself. On
the four intermodal cells whose trajectory holds still they agree:

| executor | per-price | quotable cells | baseline per-price spread |
|---|---|---|---|
| copt-cpu | **−2.43% ± 0.51 pp** | 4/10 | 1.1% |
| copt-gpu | −2.12% ± 1.10 pp | 4/10 | 1.3% |

They also agree on the trajectory itself — identical median iteration and column
counts on all 10 cells, and identical `priced` counts on 8 of 10 — which is what
makes the comparison meaningful: the same runs, timed twice.

**Quote copt-cpu.** Not because copt-gpu is wrong here, but because its spread is
twice as wide (± 1.10 pp against ± 0.51 pp) on an effect of ~2%, and because a
GPU barrier's timings are the ones exposed to host-side contention during the
long A* sweeps of the off arm. An earlier session of this same round measured
copt-gpu's per-price at more than *twice* copt-cpu's on these four cells, off an
inflated off-arm baseline; that divergence did not reproduce here. Which is
itself the lesson: check the two executors against each other before quoting
either, rather than assuming last session's relationship still holds.

## Quotable is not the same as useful

38 of the 74 cells have `traj_moved=0` and so carry a per-price number; the
median over all of them is **−2.02%**. Do not read that as a family-independent
result — read `spread_per_price_off_pct` next to it:

| group | quotable | per-price median | baseline spread |
|---|---|---|---|
| intermodal copt-cpu tree | 4/10 | −2.37% | **1.1%** |
| intermodal copt-gpu tree | 4/10 | −2.63% | **1.3%** |
| transportation tree | 1/6 | −11.32% | 2.0% |
| grid tree | 6/15 | +0.00% | 7.1% |
| grid path | 9/15 | +2.13% | 6.3% |
| planar tree | 6/9 | −8.00% | **20.8%** |
| planar path | 8/9 | +0.00% | 5.0% |

On grid and planar the per-price estimate is admissible and worthless: the
baseline arm's own repetitions disagree by 5–21% on a quantity whose effect is a
few percent, because those runs price for milliseconds. Only intermodal's spread
is small against the effect, which is why intermodal is the only family this
round quotes per-price. The one transportation cell is a single cell, not a
family figure.

## What reproduces, and what does not

This round has now been measured twice, in two sessions on the same box, and the
comparison is worth recording because it is the best available statement of what
a reader can rely on.

**Reproduces.** The direction and rough size of every pricing number: grid tree
loses a fifth to a quarter of its pricing time and its clock does not follow;
intermodal's per-price saving lands near −2.4% on copt-cpu both times (−2.43% ±
0.51 pp here, −2.53% ± 1.08 pp before); the two arms agree on the LP optimum, with
the worst `d_obj_rel` identical at 6.95e-05 on grid3 path; and 7 of 10 intermodal
cells reproduce their iteration and column counts across all 3 reps, failing on
the same three instances (BUS-13160-0, BUS-18424-0, BUS-23688-0) as before.
Intermodal is near-deterministic, not deterministic.

**Does not reproduce.** Family wall-clock totals to better than a couple of
percentage points — intermodal read −4.8%/−6.2% before and −3.7%/−3.6% here — and
the sign on the families where pricing share is small: grid tree was −0.33% and
is now +1.95%. Both are the same statement: below a few percent of pricing share,
the wall-clock delta is LP noise with a pricing signal buried in it, and its sign
is not a property of the flag. Also gone is the earlier session's copt-gpu
baseline inflation, discussed above.

The conclusion did not move either time.

## Reproducing

The build must have COPT: `cmake -B build -DMCFCG_USE_COPT=ON && cmake --build build -j`.

Both arms run in one session on one build — cross-session wall clock is not
comparable (see `../README.md`, "Two things not to redo"). Off and on alternate
within each repetition:

```sh
SWEEP=bench_runs/ablation/families
for rep in 1 2 3; do
  for arm in off on; do
    extra=(); [ "$arm" = on ] && extra=(--extra-args=--bounded-pricing)

    python3 scripts/benchmark_solvers.py \
      --families grid,planar --max-planar 1000 --formulations path,tree \
      --solvers copt-gpu --time-limit 1800 "${extra[@]}" \
      --out    "$SWEEP/gridplanar_path_tree/copt-gpu_${arm}_rep${rep}.csv" \
      --logdir "$SWEEP/gridplanar_path_tree/logs_copt-gpu_${arm}_rep${rep}"

    python3 scripts/benchmark_solvers.py \
      --families transportation \
      --instances Barcelona,BerlinCenter,ChicagoRegional,ChicagoSketch,Sydney,Winnipeg \
      --solvers copt-gpu --time-limit 1800 "${extra[@]}" \
      --out    "$SWEEP/transportation_tree/copt-gpu_${arm}_rep${rep}.csv" \
      --logdir "$SWEEP/transportation_tree/logs_copt-gpu_${arm}_rep${rep}"

    for ex in copt-gpu copt-cpu; do
      python3 scripts/benchmark_solvers.py \
        --families intermodal \
        --solvers "$ex" --time-limit 1800 "${extra[@]}" \
        --out    "$SWEEP/intermodal_tree/${ex}_${arm}_rep${rep}.csv" \
        --logdir "$SWEEP/intermodal_tree/logs_${ex}_${arm}_rep${rep}"
    done
  done
done
```

Then derive both CSVs — this re-parses the logs and never re-solves:

```sh
python3 scripts/analyze_bounded_pricing_ablation.py
```

## Files

| path | what |
|---|---|
| `runs.csv` | one row per run log (444): timings, iterations, columns, bound fire counts, `per_price_us` |
| `summary.csv` | one row per cell (74): the arms paired and reduced to medians, with deltas, `traj_moved` / `traj_stable`, both baseline spread columns, and the cost model's `pred_wall_pct` |
| `logs/<sweep>/logs_<executor>_<off\|on>_<rep>/` | the 444 raw run logs both CSVs are derived from |

The logs are **tracked**, unlike everything else a benchmark writes. This round
settles a design question rather than feeding a results table, so the logs are
the primary artifact and the CSVs are derived from them.

Sanitisation is exactly one substitution: the absolute repo prefix, wherever it
appears — the `# cmd:` header, the embedded result-CSV `instance` field, and
transportation's `TNTP: net=/trips=` line. Nothing else is altered, so each
tracked log is byte-identical to its original once that one string is removed.

```sh
sed "s|$PWD/||g" bench_runs/.../run.log > results/ablation/families/logs/.../run.log
```

`CommittedAblationTest` (`test/python/analyze_bounded_pricing_ablation_test.py`)
pins the chain: 444 logs, 74 cells, 3 reps per arm on every cell, off arms that
never fired the bound, and both CSVs field-for-field against what the logs say.
