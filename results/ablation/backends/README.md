# Round (b): bounded pricing across five backends, one family (gh #44)

**One result reproduces on every backend, and one does not reproduce at all.**

`t_PR` falls in all ten solver×formulation groups, −1.3% to −6.1%. The bound does
what it claims, everywhere, without exception.

Wall clock follows only where pricing dominates. COPT and MOSEK — 78–85% of the
clock in the pricer — post **−5.4%**, **−6.0%** and **−2.2%**. HiGHS and cuOpt do
not, and the reason is not that they price differently: it is that the bound
shifts the CG trajectory, and where the LP is a large share of the clock (HiGHS,
42%) or the barrier is GPU-nondeterministic (cuOpt), ±Δiterations swamps a 2–4%
pricing saving in both directions.

Round (a) established that intermodal is the only family with pricing share high
enough for the bound to pay at all. This round takes that one family and varies
the axis round (a) held fixed: **the LP backend**. Both arms ran in a single
session on a single build, which is the whole point.

## What each backend measured

| solver | form | wall | t_PR | t_LP | PR share |
|---|---|---|---|---|---|
| copt-cpu | path | **−7.5%** | −6.1% | −17.0% | 83.2% |
| copt-cpu | tree | −3.4% | −4.1% | +1.6% | 85.1% |
| copt-gpu | path | **−7.5%** | −5.7% | −14.6% | 77.5% |
| copt-gpu | tree | −4.4% | −4.2% | −5.6% | 79.6% |
| cuopt | path | +5.0% | −1.3% | +24.6% | 76.3% |
| cuopt | tree | −0.5% | −4.0% | +11.5% | 76.0% |
| **highs** | **path** | **+39.0%** | −2.0% | +70.6% | 41.2% |
| **highs** | **tree** | **−9.3%** | −3.1% | −14.7% | 43.1% |
| mosek | path | −2.0% | −3.8% | +9.1% | 82.9% |
| mosek | tree | −2.3% | −3.7% | +6.3% | 82.8% |

Rolled up per backend over its 20 cells (10 instances × path and tree):

| backend | off (s) | on (s) | wall | cells faster |
|---|---|---|---|---|
| copt-gpu | 288.5 | 271.2 | **−6.0%** | 16/20 |
| copt-cpu | 269.4 | 254.7 | **−5.4%** | 16/20 |
| mosek | 290.0 | 283.7 | **−2.2%** | 16/20 |
| cuopt | 297.1 | 303.8 | +2.2% | 14/20 |
| highs | 527.1 | 610.1 | +15.7% | 14/20 |
| **all** | **1672.1** | **1723.6** | **+3.1%** | **76/100** |

Both halves of that last row are true and neither is the whole story: the bound is
faster on **76 of 100 cells**, median **−1.7%**, and slower in time-weighted total
because a handful of large HiGHS cells swing by tens of seconds. Quote the
per-backend rows, not the total.

## The wall-clock sign is ±Δiterations, and HiGHS shows it uncut

HiGHS is where the mechanism is legible, because its LP share is the outlier and
its per-iteration LP cost is large. Its 20 cells split cleanly by whether the
iteration count moved:

| instance | form | iters off→on | wall | t_LP |
|---|---|---|---|---|
| BUS-23688-0 | path | 20→44 | **+135.5%** | +160.8% |
| BUS-18424-0 | path | 20→43 | **+121.6%** | +142.0% |
| BUS-13160-0 | path | 20→34 | **+63.2%** | +69.7% |
| BUS-18424-0 | tree | 19→26 | +30.8% | +37.6% |
| BUS-23688-0 | tree | 38→20 | **−48.7%** | −52.1% |
| BUS-7896-0 | tree | 15→14 | −5.0% | −5.9% |
| BUS-13160-0 | tree | 28→27 | −3.0% | −3.3% |
| *the 13 cells at an unchanged iteration count* | | | **−2.4% … +0.3%** | |

Every large move is an iteration-count move, in both directions, and the cells
where the trajectory held still are flat to slightly faster with `t_PR` down. The
column set is identical throughout — pinned bit-for-bit in C++, see the exactness
note below — so this is *when* columns arrive, not *which*.

## The HiGHS wall-clock aggregate does not reproduce across sessions

This round exists because a cross-session comparison reported a +18–32% HiGHS
penalty. The first same-session measurement (committed at 7fc24e1, earlier the
same day, same box, same 100 cells) found no penalty: HiGHS path **−9.8%**, tree
**+7.7%**. This measurement, also same-session, finds path **+39.0%**, tree
**−9.3%**.

Two properly-conducted measurements of the same quantity, disagreeing in sign on
both formulations. That is the finding, and it is stronger than either number:
**on HiGHS, intermodal wall clock is not a stable quantity to two significant
figures.** It is decided by which of three BUS instances happens to take 20
iterations and which takes 44, and the flag perturbs that lottery without biasing
it. Nothing in that range — neither the +18–32% the issue reported, nor −9.8%,
nor +39.0% — should be quoted as the flag's effect on HiGHS.

What *is* stable across both sessions: `t_PR` falls on every backend in every
group, and the cells whose trajectory does not move are flat in wall clock with
their pricing time down.

## Per-price: about −2%, and that part is solid

Restricting to the 42 cells whose trajectory did not move: wall **−1.6%**, `t_PR`
**−1.9%**, `t_LP` **+0.4%** — the LP term drops out exactly as it should when the
iteration count holds, leaving a pricing effect that pays through to the clock.
The per-price medians, by group:

| solver | path | tree |
|---|---|---|
| copt-cpu | −2.43% | −2.58% |
| copt-gpu | −1.45% | −1.54% |
| cuopt | −1.21% | −2.37% |
| highs | −0.54% | −1.93% |
| mosek | −2.64% | −2.36% |

Four to five cells per group, median **−2.0%** over all 42. Round (a) measured the
same family at 3 reps and got −2.4% (copt-cpu) / −2.6% (copt-gpu), so this agrees
with the better-replicated round.

**These cells carry no error bar.** Eight of the ten groups ran one repetition per
arm, where `spread_off_pct` and `spread_per_price_off_pct` are both blank because
there is no spread to measure, and `traj_stable` is blank because a single run
agrees with itself. HiGHS, the exception at 3 off reps, measures a baseline
per-price spread of 1.4–2.1% — the same order as the effect. Read these as
corroborating round (a), not as independent evidence.

## Disposition: off globally, and backend-conditional on this family

The flag stays **off as the library default**. That verdict is about the global
default: pricing share is the ceiling on anything the bound can deliver, and round
(a) measured 1.0–3.2% on grid/planar against 78–85% here. It is turned on for
intermodal alone in `scripts/benchmark_solvers.py`, on the strength of the numbers
below.

On intermodal the case is real but narrower than the pricing numbers alone
suggest. Where the LP is not the bottleneck — COPT either executor, MOSEK — it is
a reliable 2–6% and 16 of 20 cells faster. Where the LP is a large share of the
clock, the trajectory term is larger than the pricing term and the sign is not
predictable, so enabling the flag there buys a coin flip. `PricerHeavy` is the
natural home if you act on this, but it is not intermodal-only and it does not
know which backend it is running under, so wiring it there needs its own
before/after check.

## The exactness claim is not measured here

The column set is identical bit-for-bit with the bound on or off, and this round
does not test that — it is pinned in C++ by
`FeatureTests.BoundedPricingShadow{Tree,Path,IntermodalTree}`, which shadow every
dual vector with a second bounded-on pricer and compare every emitted column field
by field. What this round checks is weaker and independent: both arms reach the
same LP optimum on every cell (`d_obj_rel < 1e-3`, asserted by
`CommittedBackendsAblationTest`).

## Reproducing

The CSVs re-derive from the logs with no re-solve, byte-identically:

```
python3 scripts/analyze_bounded_pricing_ablation.py --round backends
```

An argument-less invocation regenerates every round in `ROUNDS`.

The sweep was 240 runs, 0 failures, ~80 min. **Solver is the outer loop**, so a
cell's two arms sit one solver-pass apart (~5 min) rather than half an hour —
round (a) could afford arms-outer because 3 reps average the drift away, and at
1 rep nothing does. The log-dir tag must be `logs_<solver>_<arm>_<rep>`: the
analyzer's `collect()` splits it with `rsplit("_", 2)` for the arm and rep, and
takes the real solver from the log *filename*. The sweep basename must also
differ from round (a)'s `intermodal_tree`, since `collect()` hard-errors on two
sweep dirs sharing one.

```sh
SWEEP=bench_runs/ablation/backends/intermodal_path_tree
TL=7200

# A HiGHS-only build does not stop the sweep — benchmark_solvers.py always exits
# 0, so it would fill 80 cells with error rows and surface an hour later as
# "unpaired cell" warnings. A push resets this build to HiGHS-only.
for lib in cuopt copt mosek; do
  ldd build/mcfcg_cli | grep -q "$lib" || { echo "error: not linked against $lib" >&2; exit 1; }
done

# Refuse rather than append: a half-sweep resumed in a later session silently
# reconstitutes the cross-session confound this round exists to remove.
[ -e "$SWEEP" ] && { echo "error: $SWEEP exists; both arms must come from ONE session" >&2; exit 1; }

run_cell() {  # arm solver rep
  local extra=(); [ "$1" = on ] && extra=(--extra-args=--bounded-pricing)
  python3 scripts/benchmark_solvers.py \
    --families intermodal --solvers "$2" --formulations path,tree \
    --time-limit $TL "${extra[@]}" \
    --out    "$SWEEP/$2_$1_$3.csv" \
    --logdir "$SWEEP/logs_$2_$1_$3"
}

for solver in highs copt-cpu copt-gpu cuopt mosek; do
  # HiGHS gets three OFF reps — it is the backend whose OFF arm wobbles most on
  # this family, and three reps buy `spread_off_pct`, a measured noise floor one
  # rep cannot produce.
  if [ "$solver" = highs ]; then off_reps="rep1 rep2 rep3"; else off_reps="rep1"; fi
  for rep in $off_reps; do run_cell off "$solver" "$rep"; done
  run_cell on "$solver" rep1
done
```

Two things that invalidate a re-run if you get them wrong, one guarded above and
one not: the build must carry all five backends, and HiGHS must genuinely be on
HiPO rather than silently falling back to dual simplex (`--verbose-solver` must
print `Running HiPO`). One thing the guards do *not* fix: OFF always precedes ON,
so a monotone warm-up across a pair biases every cell the same way. Shrinking the
gap shrinks that drift; only alternating the order removes it.

And one thing this round now demonstrates rather than warns about: a re-run will
not reproduce these wall-clock aggregates on HiGHS or cuOpt, and that is a
property of the measurement, not a mistake in it.

## Files

| file | what |
|---|---|
| `README.md` | this file: what round (b) measured and which of its numbers are quotable |
| `runs.csv` | one row per run log (240): timings, iterations, columns, bound fire counts |
| `summary.csv` | one row per cell (100): off and on arms paired and reduced to medians, with deltas |
| `logs/intermodal_path_tree/logs_<solver>_<off\|on>_<rep>/` | the raw run logs both CSVs derive from |
