# Round (a): bounded pricing across four families, one backend (gh #43)

The bounded-pricing ablation is split into rounds, each named for the axis it
varies. **This round varies family at a fixed backend.** Round (b) (`../backends/`,
gh #44) varies backend at a fixed family.

Read `../README.md` first: it carries the mechanism, the gain model, the three
implementation traps and the two comparisons not to repeat. This file is what
round (a) *measured* — scope, how to reproduce it, and which of its numbers are
quotable.

Everything here replaces a mixed-backend, mixed-rep archive that was deleted in
gh #42 rather than carried past the `--pricing-cutoff` → `--bounded-pricing`
rename. This round is uniform: **one executor per family cell, 3 reps
everywhere, both arms in one session on one build.**

## What ran

| sweep | family | formulations | executor | instances | reps |
|---|---|---|---|---|---|
| `gridplanar_path_tree` | grid, planar | path, tree | copt-gpu | 15 + 9 (planar ≤ 1000) | 3 |
| `transportation_tree` | transportation | tree | copt-gpu | 6 of 9 | 3 |
| `intermodal_tree` | intermodal, `--strategy pricer-heavy` | tree | copt-gpu **and** copt-cpu | 10 | 3 |

444 logs, 74 paired cells, 1 h 46 min on COPT 8.0.1. Every run passed against its
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
decisive rung**: the bound works there — pricing time drops **26.0%** on tree —
and the clock does not move (**−0.33%**), because 1.9% of nothing is nothing. A
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
| intermodal (tree, **copt-cpu**) | 85.3% | −5.5% | **−4.8%** |
| intermodal (tree, copt-gpu) | 71.4% | −7.5% | **−6.2%** |
| transportation (tree, copt-gpu) | 22.6% † | −6.5% | **−4.1%** † |
| grid (tree) | 1.9% | **−26.0%** | **−0.33%** |
| grid (path) | 1.2% | −4.1% | −1.30% |
| planar (tree) | 1.2% | −6.2% | +1.06% |
| planar (path) | 0.8% | +2.8% | −0.03% |

**grid tree is the whole argument in one row.** The bound does exactly what it
claims — a quarter of the pricing time, gone — and the clock does not notice,
because pricing is 1.9% of it.

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
| intermodal copt-cpu | −6.79 s | **−6.65 s** | +0.01 s | 162 → 154 |
| intermodal copt-gpu | −10.66 s | −9.32 s | −1.24 s | 162 → 154 |
| transportation copt-gpu | −7.93 s | −2.80 s | **−3.35 s** | 190.7 → 190.3 |
| grid + planar tree | +0.80 s | −0.45 s | +1.44 s | 521.7 → 525.0 |
| grid + planar path | −0.05 s | −0.02 s | +0.03 s | 276.3 → 274.3 |

Intermodal under copt-cpu is the only clean row: **essentially the entire saving
is pricing**, with LP flat to 0.01 s. Transportation's gain is majority-LP at an
unchanged iteration count — a different separated capacity set, not a pricing
effect, and copt-gpu-only (the same trajectories move LP by −3.86% on copt-gpu
and +0.34% on copt-cpu, so that executor's LP timings drift between arms).
grid/planar's real pricing saving is 0.45 s against 1.44 s of LP noise.

The family wall-clock deltas are robust in sign — across all nine off×on rep
pairings, intermodal spans [−3.9%, −8.8%] and never touches zero, while
grid/planar spans [−1.1%, +2.7%] and straddles it.

**Verdict: intermodal is the only family worth investigating further**, and that
is what round (b) (gh #44) takes up. The conservative, mechanistic estimate of
its gain is `85.3% × −2.8% ≈ −2.4%` — the cheaper-per-price term alone. The
measured −4.8% includes a trajectory that also shortened, 162 → 154 iterations,
which is real here but not predictable per instance.

**The flag still ships off.** −2.4% to −4.8% on one family, on instances that
finish in under a minute, is not worth a per-family default — and whether it
survives on a backend with a larger LP share is exactly what round (b) asks.

The two arms agree on the LP optimum everywhere: worst `d_obj_rel` is 6.95e-05
(grid3 path), inside the CG gap tolerance. That is the exactness check this data
can make on its own; the bit-for-bit column identity is a stronger statement and
is pinned in C++ by `FeatureTests.BoundedPricingShadow{Tree,Path,IntermodalTree}`.

## Quote copt-cpu, not copt-gpu

copt-gpu reports more than twice copt-cpu's per-price saving on the same four
cells — −5.59% ± 1.99 pp against −2.53% ± 1.08 pp — on a metric that should be
executor-independent. It is not a bigger saving. It is a noisier baseline:

- `priced` is **identical** between the two executors on all 10 cells, in both
  arms. The trajectory is executor-independent — the same medians for iterations
  and columns, cell for cell — so the whole divergence is in measured `t_PR`.
- The two executors' **on** arms agree to within 1%. It is the **off** arms that
  differ, copt-gpu's running 0–7% higher.
- copt-gpu's off arm is also far less repeatable than its own on arm: on
  SBT-43785-0 the three off reps span 9.2% (28.32/28.78/30.93 s) against 0.4% for
  the on reps (27.78/27.84/27.89 s). Under copt-cpu both arms sit near 5%.

An inflated, noisy off arm inflates the saving. The cost model says so
independently: on those four cells it predicts copt-cpu's wall delta to within
**0.26 pp** (−3.3/−2.3/−2.8/−0.7 predicted vs −3.4/−2.3/−2.9/−1.0 measured) and
misses copt-gpu's by up to **1.59 pp**. A model that tracks one executor and not
the other, on cells where the trajectory is identical, is pointing at the
measurement rather than at the mechanism.

The likely cause is host-side contention while the GPU barrier is resident —
the off arm's full A* sweeps are the longest, most memory-hungry pricing windows
in the round, so they absorb the most of it — but this data cannot prove that,
and it does not need to: the copt-cpu arm is clean, agrees with the model, and is
the one the round quotes.

Its **−2.53% ± 1.08 pp reproduces the deleted round's −2.45% ± 0.56 pp**, which
is the strongest thing this re-run says: the number survived a flag rename, a
different session and a different rep count.

## Against the issue's acceptance criteria

Two of the five predictions held and three did not; recording which, because they
were written to be falsifiable and two of them were falsified.

| criterion | outcome |
|---|---|
| all 10 intermodal cells reproduce iterations and columns across all 3 reps, both arms | **7/10**, on *both* executors, failing on the same three (BUS-13160-0, BUS-18424-0, BUS-23688-0) |
| SBT-6255-0's `d_per_price_pct` turns negative | **yes** — −7.99% copt-gpu, −1.04% copt-cpu (it read +0.20% on the deleted 2-rep arm) |
| …and its \|pred − measured\| drops toward ≤0.16 pp | **no** — 1.59 pp on copt-gpu, 0.26 pp on copt-cpu |
| pooled copt-gpu per-price lands near copt-cpu's −2.45% ± 0.56 pp | **no** — −5.59% ± 1.99 pp |
| the analyzer re-derives both CSVs from the tracked logs, rerun byte-identical | yes |

The three unstable BUS cells are intrinsic to those instances, not to the
executor: copt-gpu and copt-cpu produce the *same* median iteration and column
counts on all 10 cells and are unstable on the same three. That matches what the
gh #41 handoff recorded — intermodal is near-deterministic, not deterministic —
so the criterion was simply written a notch stronger than the family supports.

The last two failures are one finding, not two: they are the copt-gpu baseline
above, and the issue's own instruction for that case was that "the copt-gpu arm
should not be quoted per-price". It is not.

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
| `summary.csv` | one row per cell (74): the arms paired and reduced to medians, with deltas, `traj_moved` / `traj_stable`, and the cost model's `pred_wall_pct` |
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
