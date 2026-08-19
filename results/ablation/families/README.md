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

Three scope decisions, recorded with their reasons so they are not re-litigated:

- **Path only on grid/planar.** Transportation path costs ~9.5× tree (Austin and
  Philadelphia clamp at the 7200 s limit) and prices for 0.1–0.9% of wall clock.
  Intermodal path ≡ tree — one commodity per source. grid/planar is the only
  affordable place the **path** bound is observable at all, and §3.3 claims it
  separately from the tree analogue.
- **Transportation at 6 of 9 instances.** Austin, Birmingham and Philadelphia
  excluded on cost (~4 h added). The family has no quotable cell either way, so
  more instances were unlikely to produce one. Cost, not a finding.
- **copt-gpu is the axis; copt-cpu corroborates on intermodal only.** copt-gpu is
  the one executor that already covered all four families, and its LP share on
  the other three is within 0.2 pp of copt-cpu's, so copt-cpu cannot expose signal
  there that copt-gpu missed. Extending it to those families costs ~61 h, ~60 of
  which measures families that price for under 2.3% of wall clock. Intermodal is
  the only family where a per-price number is quotable at all, so that is where
  the second executor earns its 14 minutes — and it turned out to be the arm to
  quote (below).

## The result

| family (formulation, executor) | pricing share | per-price, quotable cells | total `t_PR` | wall clock |
|---|---|---|---|---|
| intermodal (tree, **copt-cpu**) | 85.4% | **−2.8%** median, 4 cells | −5.5% | −4.8% |
| intermodal (tree, copt-gpu) | 71.8% | −5.6% median, 4 cells — *not quotable, see below* | −7.4% | −5.9% |
| transportation (tree, copt-gpu) | 22.5% | *no qualifying cell* | −6.7% | −5.2% |
| grid + planar (tree, copt-gpu) | 1.5% | −1.8% median, 8 cells (−20.0 … +50.0%) | −17.7% | +0.9% |
| grid + planar (path, copt-gpu) | 1.0% | +0.0% median, 13 cells (−33.3 … +7.1%) | −0.5% | +0.1% |

Those grid/planar ranges are mostly quantization: 39 of the 48 cells price for
under 0.1 s in total, against a 3-digit timing field. Restricted to the 9 cells
whose pricing time is large enough to measure, the mechanism is clean and
one-directional — **tree** −3.6 … −29.9% per price (median −22.3%), because its
bound tightens on every settle; **path** −2.9 … +5.6%, because its `max π` bound
waits on the most expensive remaining commodity, which settles last.

**The conclusion is unchanged: the flag stays off.** Where the trajectory holds
still — the only place a wall-clock number means "the pricer got faster" — the
gain is 85.4% × −2.8% ≈ **−2.4%**, on a family whose instances finish in under a
minute. Everywhere else the pricing share is too small to carry any per-price
saving to the clock: grid/planar tree saves 17.7% of its total pricing time and
*loses* 0.9% of wall clock, because pricing is 1.5% of it.

The family wall-clock column is **not** the effect. On intermodal, 6 of 10 cells
have `traj_moved=1`, and their deltas run from −27% (SBT-18765-0, 9 → 7
iterations) to +20% (BUS-23688-0, 13 → 20). The family total is which way that
coin landed, not what the pricer did. Likewise transportation's −5.2%: 5 of its 6
cells moved, ChicagoRegional's −8.1% comes with 50 → 48 iterations, and its only
`traj_moved=0` cell (ChicagoSketch) has reps that disagree within an arm. **No
transportation cell is quotable per-price**, which is the same verdict the
deleted round reached.

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
