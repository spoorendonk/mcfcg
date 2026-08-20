# Round (b): bounded pricing across five backends, one family (gh #44)

Round (a) established that intermodal is the only family where the bound has room
to pay — pricing share is the ceiling on anything it can deliver, and only
intermodal's is high (78–80% under COPT) rather than 0.2–4.3%. This round takes
that one family and varies the axis round (a) held fixed: **the LP backend**.

Both arms ran in a single session on a single build, which is the whole point.
The measurement this round replaces was cross-session, and the effect it appeared
to show did not survive being measured properly.

## Headline: the HiGHS penalty is not real

The issue this round was opened to settle reported a **+18–32%** HiGHS penalty
(rising to +28…+125% after a two-stage offset correction), derived by comparing an
archived on-arm pass against `results/cg_benchmark.csv` from a different session.
Measured with both arms in one session, it does not exist:

| solver | form | wall | t_PR | t_LP | PR share |
|---|---|---|---|---|---|
| copt-cpu | path | **−8.6%** | −7.4% | −18.0% | 83.7% |
| copt-cpu | tree | −6.4% | −6.5% | −6.4% | 84.9% |
| copt-gpu | path | −6.8% | −4.5% | −16.3% | 77.7% |
| copt-gpu | tree | −2.5% | −3.2% | −0.6% | 80.7% |
| cuopt | path | +1.6% | −5.9% | +25.2% | 75.7% |
| cuopt | tree | −4.8% | −9.2% | +9.2% | 75.2% |
| **highs** | **path** | **−9.8%** | −5.6% | −12.4% | 34.3% |
| **highs** | **tree** | **+7.7%** | −3.4% | +17.1% | 44.6% |
| mosek | path | +0.1% | −3.4% | +24.4% | 83.0% |
| mosek | tree | +2.2% | −2.0% | +27.1% | 83.4% |

HiGHS path is a −9.8% **gain**; HiGHS tree is +7.7%. Nothing here is +18%, let
alone +125%. The prior figure was the session offset it was corrected for, plus a
handful of trajectory moves read as a pricing effect.

## What the round actually shows

**`t_PR` falls in all ten groups, −2.0% to −9.2%.** The bound does what it claims,
on every backend, in both formulations, without exception. That is the one
unambiguous result here.

**The wall-clock sign is decided by `t_LP`, which swings −18.0% to +27.1%.** The
bound emits an identical column set (see the exactness check below), but it changes
*when* columns arrive, and that shifts the CG trajectory — 59 of 100 cells moved.
Where the shift shortens the run the LP time falls with it; where it lengthens the
run the LP time rises and swamps the pricing saving. This is the cost model's
second term, `−LP_share × Δiterations`, and round (a) already flagged it as not
predictable per instance.

So the flag's effect is **not backend-specific because backends price differently**
— they don't, `t_PR` moves the same way everywhere. It is backend-specific because
the trajectory shift lands on an LP that costs a different fraction of the clock.
HiGHS is the clearest case precisely because its LP share is the outlier (34–45%
against 75–85% for the others), so the same trajectory noise moves its wall clock
furthest in both directions.

None of this changes the disposition. The flag stays **off by default**: a
mechanism whose sign depends on which way an unpredictable trajectory shift lands
is not one to enable globally, even though its direct effect is a consistent
pricing win.

## Nothing here is quotable per-price

`traj_stable` is blank on all 100 cells, and the analyzer reports
`quotable per-price (moved=0 and stable=1): none of 10 cells` for every group.
That is correct and deliberate, not a gap in the data.

At one rep per arm a stability check is vacuous — a single run agrees with itself
whatever it holds — so `traj_stable` is left blank below two reps rather than
reading a free 1. Without that, this round would have printed 100 confidently
quotable per-price numbers off evidence it never had. Round (a) is where per-price
numbers come from; this round is wall clock, decomposed.

## Rep structure, and why it is mixed

HiGHS carries **3 off reps**; the other four backends run 1+1. That is 80 cells at
`(1,1)` and 20 at `(3,1)`, pinned exactly by `CommittedBackendsAblationTest`.

The reason is that round (a) measured this same family and found `BUS-18424-0`'s
off arm non-deterministic within its own reps — 42/33/33 iterations against a
32/32/32 on arm — and that is the cell the old +132% headline rested on. HiGHS is
the backend this round was opened to adjudicate, so it is the one that needed a
measured noise floor rather than a single sample.

It was worth it. HiGHS's own off-arm rep-to-rep spread reaches **30–46%** on
several cells (`spread_off_pct` in `summary.csv`), far above the 1.3–6% the issue
assumed when it argued one rep would suffice. Read any single non-HiGHS cell here
with that in mind: those four backends are **directional only**.

## Cross-round check, for free

Twenty of this round's cells (`copt-cpu/tree`, `copt-gpu/tree`, same ten
instances) were already measured in round (a) at 3 reps. Wall clock across rounds
is cross-session and not comparable, but **iterations and columns are exact and
session-independent**, so the overlap calibrates the thing that actually makes one
rep risky.

**19 of 20 off-arm iteration counts reproduce round (a) exactly**, across a
different session and a different build. The single miss is `BUS-18424-0` under
copt-cpu, where this round landed on 42 — one of the values round (a) itself
observed. The trajectory is reproducible; that instance's instability is a property
of the instance, not of the session.

## Exactness

The two arms agree on the LP optimum on every cell: `d_obj_rel` peaks at
**2.4e-05**, within the CG gap tolerance. This is the check the ablation data can
make on its own; the stronger bit-for-bit column identity claim is pinned in C++ by
`FeatureTests.BoundedPricingShadow*`.

## Reproducing

The CSVs re-derive from the tracked logs with no re-solve, byte-identically:

```
python3 scripts/analyze_bounded_pricing_ablation.py --round backends
```

An argument-less invocation regenerates every round in `ROUNDS`.

The sweep itself was `bench_runs/run_round_b.sh` — intermodal, five backends, path
and tree, `--time-limit 7200`, both arms back to back per solver so a cell's two
arms sit minutes apart rather than half an hour. 240 runs, 0 failures, ~87 min.

Two things that invalidate a re-run if you get them wrong. The build must carry all
five backends (`ldd build/mcfcg_cli` must show cuopt, copt and mosek — a push
resets it to HiGHS-only), and HiGHS must genuinely be on HiPO rather than silently
falling back to dual simplex (`--verbose-solver` must print `Running HiPO`). The
driver guards the first and refuses to resume into an existing sweep dir, since a
half-sweep finished in a later session would silently reconstitute the cross-session
confound this round exists to remove.

## Files

| file | what |
|---|---|
| `README.md` | this file: what round (b) measured and which of its numbers are quotable |
| `runs.csv` | one row per run log (240): timings, iterations, columns, bound fire counts |
| `summary.csv` | one row per cell (100): off and on arms paired and reduced to medians, with deltas |
| `logs/intermodal_path_tree/logs_<solver>_<off\|on>_<rep>/` | the raw run logs both CSVs derive from |
