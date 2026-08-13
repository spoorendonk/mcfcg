# Scripts

## Instance Preparation

### `download_commalab.sh`

Downloads CommaLab/UniPi benchmark instances into `data/commalab/`.
These are the Grid and Planar instance families used in the paper.

### `prepare_intermodal.sh`

End-to-end pipeline for intermodal instances (SUBWAY, BUS, SBT families):

1. Clones the [tumBAIS intermodal repo](https://github.com/tumBAIS/intermodalTransportationNetworksCG)
2. Fetches LFS data files (network XML, schedule XML)
3. Generates raw instances via `generate_instances.py`
4. Cleans instances with `mcfcg_clean` (removes unreachable commodities)

**Prerequisites:** Python 3 with networkx, pandas, numpy, geopandas, shapely, lxml.
Build `mcfcg_clean` first: `cmake --build build -j$(nproc)`

### `generate_instances.py`

Generates intermodal MMCF instances from the tumBAIS repo data.
Called by `prepare_intermodal.sh`; can also be run standalone.

```
python3 generate_instances.py --repo data/intermodal-repo --output data/intermodal/raw \
    --seeds 0 --modes subway bus sbt
```

## Instance Families

| Family | Format | Instances | Source |
|--------|--------|-----------|--------|
| Grid | CommaLab | 15 (grid1-grid15) | [CommaLab](https://commalab.di.unipi.it/) |
| Planar | CommaLab | 10 (planar30-planar2500) | [CommaLab](https://commalab.di.unipi.it/) |
| SUBWAY | CommaLab | 4 (4 passenger counts, seed 0) | tumBAIS intermodal repo |
| BUS | CommaLab | 5 (5 passenger counts, seed 0) | tumBAIS intermodal repo |
| SBT | CommaLab | 5 (5 passenger counts, seed 0) | tumBAIS intermodal repo |
| Transportation | TNTP | 9 cities | [TransportationNetworks](https://github.com/bstabler/TransportationNetworks) |

### Transportation Cities

TNTP instances are committed as gzipped files in `data/transportation/`.
Each city has a demand coefficient used to scale raw OD demands:

| City | Coefficient |
|------|------------|
| Austin | 6.0 |
| Barcelona | 5050.0 |
| BerlinCenter | 0.5 |
| Birmingham | 0.9 |
| ChicagoRegional | 4.1 |
| ChicagoSketch | 2.4 |
| Philadelphia | 7.0 |
| Sydney | 1.9 |
| Winnipeg | 2000.0 |

Pass TNTP files to the CLI (plain or gzipped):
```
./build/mcfcg_cli data/transportation/Winnipeg_net.tntp.gz
./build/mcfcg_cli path/to/CityName_net.tntp
```
The CLI auto-detects the format, derives the trips path, and looks up the coefficient.

## Reproducing the Benchmark

`benchmark_solvers.py` drives the CLI over the whole instance suite and checks
each reported objective against the family's `optimal.csv` reference. Run bare,
it reproduces the full benchmark — its defaults already encode the canonical
configuration:

```
python3 scripts/benchmark_solvers.py
```

| default | value |
|---------|-------|
| `--solvers` | `highs,mosek,cuopt,copt-cpu,copt-gpu` (the full {CPU,GPU}×{OSS,commercial} matrix + COPT GPU-off control) |
| `--families` | `grid,planar,transportation,intermodal` |
| `--time-limit` | `7200` (2 h CG wall-clock per run, enforced at iteration boundaries) |
| formulation | per-family default: **tree** everywhere (intermodal additionally uses `--strategy pricer-heavy`) |

Each run emits one CSV row (`instance,formulation,iterations,columns,objective,
lower_bound,optimal,time,…`) to `--out` (default `bench_runs/cg/results.csv`) and
saves its full per-iteration CG log to `--logdir` (default `bench_runs/cg/logs/`),
one file per run named `<family>__<instance>__<formulation>__<solver>.log`. Both
defaults live under the gitignored `bench_runs/` tree, never repo root.

Common narrowing:

```
# one family, both formulations
python3 scripts/benchmark_solvers.py --families grid --formulations path,tree

# a single backend on an instance glob
python3 scripts/benchmark_solvers.py --solvers copt-cpu --instances 'BUS-*'

# skip the largest planar instances
python3 scripts/benchmark_solvers.py --families planar --max-planar 1000
```

### Prerequisite: all backends compiled in

The default `--solvers` lists all five configs, but a label whose backend was
not compiled into `build/mcfcg_cli` reports as an `error` row rather than
silently dropping. The repo's standard build (and the pre-push clean build)
produces a **HiGHS-only** binary, so before a multi-backend run reconfigure with
the optional backends enabled and confirm they linked:

```
export MOSEK_HOME=/opt/mosek/<ver>/tools/platform/linux64x86 \
       COPT_HOME=/opt/copt80 CUOPT_ROOT=/path/to/cuopt
cmake -B build -DCMAKE_INSTALL_MESSAGE=LAZY \
      -DMCFCG_USE_MOSEK=ON -DMCFCG_USE_COPT=ON -DMCFCG_USE_CUOPT=ON
cmake --build build -j$(nproc)
ldd build/mcfcg_cli | grep -E 'cuopt|copt|mosek'   # all three should resolve
```

(cuOpt needs the `spoorendonk/cuopt` fork checkout as `CUOPT_ROOT`; stock cuOpt
requires `-DMCFCG_CUOPT_DELTA_API=OFF`. See the top-level CLAUDE.md for the
delta-API rationale.)

Each backend reports the version of the library it actually loaded in the
`[lp-config]` banner, captured in every run log and in the sweep CSV's `config`
column:

```
[lp-config] backend=mosek version=11.0.30 method=barrier exec=CPU presolve=off crossover=off tol=0.0001 threads=auto(32)
```

That version is read back from the loaded library, not from the vendor header's
compile-time macros, so a stale `LD_LIBRARY_PATH` pointing at a second install
shows up in the log instead of being silently misreported as the version you
built against. Two things it cannot tell you, both recorded in `PROVENANCE.txt`
instead: whether the HiGHS HiPO patch is applied (`highsGithash()` reports the
upstream tag either way), and whether cuOpt is the delta-API fork
(`cuOptGetVersion` reports the upstream RAPIDS version).

### Verifying a reproduction

`check_reproduction.py` joins a fresh sweep CSV against the committed
`results/cg_benchmark.csv` on `(family, instance, formulation, solver)` and
compares objectives:

```
python3 scripts/benchmark_solvers.py --families grid --solvers highs \
    --formulations path,tree --out /tmp/grid_highs.csv
python3 scripts/check_reproduction.py /tmp/grid_highs.csv
```

It exits non-zero on any mismatch, so it can gate a release checklist.

**Objectives are the only thing worth comparing.** Time and peak RSS are
properties of the host, the GPU and the solver build — a faithful reproduction
on other hardware differs on both, and comparing them would report failures that
are not failures. Iteration and column counts are excluded for a subtler reason:
they are deterministic for a fixed backend version but shift when it changes,
because they depend on where the barrier's interior point lands.

**And only objectives that were certified optimal gate the result.** A run
stopped by `--time-limit` reports whatever bound it had reached when the clock
ran out — a measure of host speed as much as of the formulation. Twenty
committed cells are not certified: 14 hit the time limit, 5 aborted after 0–5
iterations when an LP solve returned a non-optimal status (the four cuOpt
`transportation/*/path` cells and `planar2500/path/highs`), and one was
SIGKILLed (`optimal` blank; the rest are `optimal=0`). Two of the timed-out ones
differ by over 1% between backends on the reference machine itself:
`planar2500/tree` spans 1.2481e10 to 1.2661e10, `Philadelphia/path` spans
2.5095e7 to 2.5893e7. Gating on those would fail every host that is not exactly
as fast as this one, so they print as `note` lines and are counted under
`advisory`.

**Their `objective` is usually a lower bound.** For 17 of the 19 that produced a
number, `objective` equals `lower_bound`: CG falls back to `best_lb` when it
never recorded a slack-free incumbent, so the column holds the Lagrangian bound
rather than a feasible cost. Only `Philadelphia/path/mosek` reports a true UB.
Quote these rows accordingly.

A **lost certification** — the reference proved optimality and your run did not —
is called out on its own line in the summary. It cannot gate (on slower hardware
it is expected), but it is also the shape of a backend regression, so it must not
hide inside the advisory total.

Two more cases are advisory because the reference cell is not a target worth
hitting:

- **This host produced something the reference did not.** `Sydney/path/mosek` was
  SIGKILLed at a 95.8 GB peak and has no objective; a machine with more headroom
  may well solve it.
- **The reference value is known-bad.** `Sydney/path/cuopt` recorded `-inf` — its
  first LP solve failed and CG broke out after 0 iterations, so that is the "no
  objective established" sentinel, not a computed value. A correct rerun is
  *expected* to disagree.

A self-compare is a different case: those two cells agree with themselves, so it
reports 420 matched, 18 advisory and 2 agreed by absence or non-finite value —
440 cells, nothing gated that should not be.

The default tolerance is `1e-3`, matching `benchmark_solvers.py`'s pass
criterion. It is deliberately *looser* than `BARRIER_TOL` / `RELATIVE_FEAS_TOL`
(1e-4): 1e-4 is the gap at which CG stops, so two faithful runs may legitimately
differ by nearly that much. Reruns of the same grid/highs cells on the reference
host itself land 5e-5 apart — half the budget — so gating at 1e-4 would risk
failing correct builds, most likely on a host with a different core count. The per-cell `rel=`
printout stays, so drift toward the limit is still visible.

Cells the sweep did not run are counted as `not run`, never as passes; a narrowed
sweep must not read as having reproduced the whole matrix. A cell present in the
fresh run but absent from the reference is reported as `unknown` and fails the
check — it means the two tables disagree about what the matrix *is*, which is a
real discrepancy even though no objective differs.

### What to commit

Everything a benchmark writes lands under the gitignored `bench_runs/` tree
(per-run CG/solver logs, `*.stdout`, incremental/partial CSVs) — bulky and fully
regenerable, so it is never tracked. The **committed "one truth" is a single
consolidated results CSV per experiment under the tracked `results/` folder**
(compact, diff-friendly), rebuilt from the logs by a consolidator that re-parses,
never re-solves:

```
# CG suite: per-run logs -> results/cg_benchmark.csv
python3 scripts/consolidate_cg_logs.py                       # default logdir bench_runs/cg/logs
# CG per-iteration trace -> results/cg_iterations.csv
python3 scripts/extract_iterations.py                        # same --logdir set
# MPS baseline: per-cell logs -> results/mps_compact_baseline.csv
python3 scripts/consolidate_mps_logs.py
```

### Counting columns: three different numbers

`results/cg_benchmark.csv` carries `columns` (final master size, from the CLI's
result row) alongside `columns_generated`, `columns_seeded` and `columns_purged`
(summed from the iteration trace). They are genuinely different quantities and
the paper must say which it quotes:

- **`columns`** — `master.num_columns()` at termination. Shrinks with
  `--col-age-limit` purging, and **excludes slack columns**.
- **`columns_generated`** — Σ `+col`, what the pricer actually produced. A `+col`
  printed as `*N` was priced but never added (the loop hit the gap and returned
  without `add_columns`, `cg_loop.h`) and is excluded.
- **`columns_seeded`** — the warm-start pool, added *before* the loop and so
  never counted by `+col`. On path masters it dominates: Austin path starts at
  1,082,300 columns for 1,081,717 commodities — one per commodity.

That last point matters. Quoting the final master size as "columns generated"
credits the path formulation's warm start to its pricer:

| Austin, copt-cpu | iters | seeded | generated | final master | converged |
|---|---|---|---|---|---|
| tree | 223 | 2,234 | 228,464 | 23,262 | yes |
| path | 8 | 1,082,300 | 350,000 | 1,414,271 | no (timed out) |

These figures do **not** close into an identity — the trace's `#col` grows by
more than `+col` reports on most runs. Quote each for what it names.

`consolidate_cg_logs.py` takes one or more `--logdir`s in priority order (a later
dir overrides an earlier one for the same cell), so a multi-pass historical log set
— e.g. an authoritative HiGHS-1.15.1 HiPO ablation layered over an earlier run —
consolidates correctly; a single fresh `benchmark_solvers.py` run needs just the
one default dir.

### Peak memory is the one metric that must be written down

Every other column can be re-derived by re-parsing a log, because the CLI prints
it. Peak RSS cannot: it is measured *outside* the child by GNU `time -f %M`, so if
it is not persisted at measurement time it is gone. `write_log` therefore records
it in the log header, and the consolidator reads it back:

```
# peak_rss_kb: 14704
# peak_rss_source: measured
```

`mem_source` records where the number came from:

| value | meaning |
|---|---|
| `measured` | GNU time read it during the run that wrote this log. **The only tag a run can write** — everything below is historical. |
| `backfilled:<csv>` | the same run's number, relocated into its log from a pre-header sweep CSV; the CSV row's `time` and `outcome` matched the log's. |
| `backfilled-untimed:<csv>` | relocated, but the row had no `time` to check (the run errored), so only `outcome` was matched. One cell: `transportation/Sydney/path/mosek`, SIGKILLed at 95.8 GB — see `exit_status` below. |

The two `backfilled` tags come from a one-shot performed once over the log tree
before archiving; the script is not in the release, since its input (pre-header
sweep CSVs under the gitignored `bench_runs/`) is not either. The readers keep
the vocabulary because the logs and `results/cg_benchmark.csv` still carry it —
see `PROVENANCE.txt` section 1.1 for the full account, and
`test/python/log_headers_test.py` for the round-trip test that keeps
`parse_peak_rss` able to read those tags back.

Its match rule is worth remembering if a comparable relocation is ever needed
again: an agreement on `time` and `outcome` is a **tripwire, not a proof**. Times
print to 3 decimals, so two runs of a fast deterministic instance can agree
exactly while being different executions — and peak RSS depends on machine, build
and solver version, not just the instance. The pairing rested on knowing which
sweep produced the logs in a given dir, with the gate catching mistakes.

### How a run died is the other thing only the header knows

A run killed by a signal prints nothing about it — the child is gone before it can
report anything, so the exit disposition survives only in `write_log`'s header:

```
# returncode: 137
```

`parse_returncode` reads it back and `format_exit_status` decodes it into the
`exit_status` column of both CSVs. It is the single place a return code is interpreted, so the
sweep CSV, the consolidated CSV and the console verdict cannot drift apart:

| value | meaning |
|---|---|
| `ok` | clean exit (rc 0). **Reports the exit, not the answer** — `transportation/Sydney/path/cuopt` exited 0 with `objective=-inf` after 0 iterations: the cuOpt barrier failed, the first LP solve returned non-optimal and CG broke out, so `-inf` is the "no objective established" sentinel, not a computed value. A backend can also swallow such a failure and report a plausible *wrong optimum* instead (gh #33, fixed in the required fork) — that one shows `optimal=1`. A clean exit that printed nothing parseable also lands here, with a blank `objective` and `(no result row)` on `source`. Read it with `optimal`/`rel_err`, never alone. |
| `error rc=N` | non-zero exit that is not a signal death. |
| `killed SIGKILL` | died on a signal, name resolved. `killed sig=N` when Python's enum has no name for the number. |
| *(empty)* | no `# returncode:` header — a pre-header or hand-assembled log. **Unknown, not ok**; never infer success from a missing header. |

`exit_status` is not redundant with the sweep CSV's `outcome`: every way a run can die
collapses into `outcome=error`, and only `exit_status` says which. It also qualifies
a blank `optimal`. A time-limited run still prints its result row, so "ran but did
not certify" is `optimal=0`; `optimal` is blank only when there is no result row at
all, and `exit_status` is what says why there is none.

**A signal names itself, never its cause.** `killed SIGKILL` is consistent with
the kernel OOM killer, but equally with a manual `kill -9` or a cgroup limit, and
the harness keeps no kernel-log evidence to separate them. Read it next to
`mem_gb` and judge — the one such cell, `transportation/Sydney/path/mosek`, peaked
at 95.8 GB on a 125 GB box with the harness documented as never killing its child
(`run_one`), which makes OOM the strong reading but still a reading. The column
deliberately does not say `oom`: that would record an inference as a measurement.

Because the logs also live under the gitignored `bench_runs/`, **deleting that
tree makes `results/cg_benchmark.csv` unreproducible outright — not just its
memory column**, and peak RSS specifically would need a full rerun. Audit the
tracked CSV with:

```
python3 - <<'PY'
import collections, csv
rows = list(csv.DictReader(open('results/cg_benchmark.csv')))
print(sum(1 for r in rows if not r['mem_gb']), 'cells missing mem_gb')
print(collections.Counter(r['mem_source'] for r in rows if r['mem_gb']))
print(collections.Counter(r['exit_status'] for r in rows))
PY
```

Coverage is complete: 440/440 cells carry `mem_gb` (gh #37, closed). 439 cells
report `exit_status=ok`; the one exception is `transportation/Sydney/path/mosek`.
A cell reappearing in the "missing" count means a log lost its header — most
likely a sweep rerun into a `--logdir` whose logs already carried memory
(`write_log` truncates), which is unrecoverable without a re-solve.

## Compact-Model Baseline (`benchmark_mps.py`)

The column-generation numbers above are compared against a **direct** solve of the
compact *source* LP — one variable `f^s_e` per source/edge, `|S|·|V|` conservation
rows plus one capacity row per arc — handed to each vendor's native barrier in a
single shot (no CG). This compact LP corresponds to the tree formulation's LP
(same optimum), so the head-to-head is `direct-X` here vs `tree-CG-X` in
`benchmark_solvers.py` for the same backend `X`.

`benchmark_mps.py` drives the vendors' **native** command-line barriers directly
(the mcfcg CLI does not read external MPS) over `data/mps/*.mps.gz`:

```
python3 scripts/benchmark_mps.py                 # full sweep, all 5 backends, 2 h/solve
python3 scripts/benchmark_mps.py --max-gz-bytes 4e8   # skip the 6 giant instances
python3 scripts/benchmark_mps.py --min-gz-bytes 4e8   # ONLY the giants (separate pass)
```

Prerequisites:

- The `.mps.gz` dumps must exist — regenerable via `scripts/write_mps.sh`; they are
  gitignored (`*.mps.gz`, ~5.6 GB) as derived/downloadable data, not committed.
- Native solver binaries on the expected paths, overridable via env: **HiGHS 1.15.1
  built with working HiPO** (`-DBUILD_SHARED_EXTRAS_LIB=OFF`; the stock 1.15.1 CLI
  aborts HiPO with `features unavailable: amd, blas, metis, rcm`) via `HIGHS_BIN`;
  `$MOSEK_HOME/bin/mosek`; `$COPT_HOME/bin/copt_cmd`; `cuopt_cli` via `CUOPT_BIN`.

| default | value |
|---------|-------|
| `--solvers` | `highs,mosek,cuopt,copt-cpu,copt-gpu` (COPT twice: GPUMode 0 vs 2) |
| `--time-limit` | `7200` (barrier wall-clock per solve) |
| `--mem-max` | `105G` — per-solve RSS cap via `systemd-run --user --scope` (`MemoryMax`, `MemorySwapMax=0`, `RuntimeMaxSec`); a runaway is OOM-killed cleanly instead of swap-thrashing the box. A preflight aborts if the cap isn't actually enforced; `--mem-max off` to disable. |
| barrier regime | crossover off, tol `1e-4`, presolve at **each solver's default** (a direct monolith benefits from presolve; forcing it off would handicap the baseline) |

Instances run size-ordered (small→large) so a memory-cautious sweep hits the risky
giants last; `--min/--max-gz-bytes` select a size band. Each solve writes one raw
per-cell log `<instance>__<solver>.log` and one incremental CSV row — both default
under the gitignored `bench_runs/mps/` tree (never repo root).

`consolidate_mps_logs.py` rebuilds the results CSV from those per-cell logs — it
**re-parses, never re-solves**, so partial / interrupted / multi-pass sweeps merge
into one clean table (and parser fixes apply retroactively):

```
python3 scripts/consolidate_mps_logs.py    # bench_runs/mps/logs -> results/mps_compact_baseline.csv
```

Its default `--out` writes straight to the tracked `results/mps_compact_baseline.csv`
— the committed one truth. The bulky per-cell logs under `bench_runs/mps/` stay
local/regenerable.

### Memory (`mem_gb`, `vram_gb`) and the `--probe-iters` shortcut

Every solve is wrapped in GNU `time -f %M` **inside** the cgroup guard, and its
peak RSS is written into the log header as `# peak_rss_kb:` — the same
serialization `benchmark_solvers.py` uses for the CG runs, so the two tables'
memory columns mean the same thing and one parser reads both. GPU configs also
get `# peak_vram_mib:`, sampled from `nvidia-smi` per process group (host RSS
says nothing about the device-side factorization, which is what OOMs on the
giants). The guard passes `OOMPolicy=continue` so that on a cap hit the kernel
kills only the solver and `time` survives to report the peak it reached — under
systemd's default policy the whole scope is torn down and the measurement of the
most interesting runs is lost.

Memory is the one metric that **cannot** be recovered by re-parsing a log, so
a cell solved before these headers existed has no peak of its own and needs a
rerun — or the probe's number relocated into it, which is what the committed
table does for 159 of its 175 cells (see the `mem_source` section below).
`--probe-iters N` is the cheap rerun: it caps the barrier at N iterations
(`ipm_iteration_limit` / `MSK_IPAR_INTPNT_MAX_ITERATIONS` / `BarIterLimit` /
`--iteration-limit`) instead of solving. A barrier's peak is the symbolic +
numeric ADAT factorization, allocated on the first iteration and reused, so the
iteration-1 high-water mark is a tight lower bound on the full-solve peak at a
fraction of the runtime.

```
python3 scripts/benchmark_mps.py --probe-iters 3        # -> bench_runs/mps_probe/
python3 scripts/consolidate_mps_logs.py --probe         # -> results/mps_compact_memory.csv
```

Measured against full solves on grid7 / planar300 / grid10 (all 5 backends), the
probe recovers **0.88–1.00** of the full-solve peak RSS at 1.6–12× less runtime,
and peak VRAM to within 2%. Three iterations is the better default: the only cell
below 0.93 was HiPO, which goes 0.88 → 0.95 for ~1.6× the probe time (still 7×
cheaper than its full solve); every other backend moves <1%. Report probe memory
as the lower bound it is, never as the full solve's footprint.

Probe runs default to their **own** logdir/CSV, record **no objective**, and
carry a `# probe_iters:` header; the consolidator keeps the two populations apart
in either direction. A capped barrier's iterate is not a solution, and the point
of the separation is that it can never be scored as one.

### The baseline's `mem_source` column

The 2026-08-03..05 baseline sweep predates the memory headers, and re-running it
for memory alone would cost days. Each probe cell's peak was instead copied into
the matching baseline log, keyed on `(instance, solver)`, by a one-shot performed
once before archiving; `consolidate_mps_logs.py` surfaces the provenance as a
`mem_source` column in `results/mps_compact_baseline.csv`:

| value | meaning |
|---|---|
| `measured` | the solve in that row reported its own peak (the five grid1 re-runs plus the ten Sydney / BUS-2632-0 cells). **The only tag a run writes for itself.** |
| `probeN:<log>` | **relocated** from the named probe log: the model-setup peak (read + presolve + N barrier iterations), a **lower bound** on this row's full-solve peak, from a different execution with a different `time_wall` and `outcome`. |
| `probeN-partial:<log>` | relocated from a probe that stopped short of its cap (cgroup OOM, backend error, or a clean exit that never reached the barrier) — a lower bound on a lower bound. |
| *(empty)* | unmeasured in both sweeps. One cell: `ChicagoRegional x highs`. |

Never quote a relocated figure as the peak of the solve whose `time_wall` sits
next to it. There was deliberately **no** time/outcome gate on this relocation
(unlike the CG one above): the two runs are different executions and are
*expected* to disagree (Austin x copt-cpu, 1106 s probed vs 7267 s solved). The
only pairing evidence is `(instance, solver)` identity, which is why the tag
names the source log. See PROVENANCE.txt section 2.2.

The relocating script is not in the release — its input, the probe log tree, is
gitignored and outside the archive, and a fresh sweep never needs it: every cell
`benchmark_mps.py` completes writes its own `measured` peak. Rebuilding the
committed table from the logs is unaffected:

```bash
python3 scripts/consolidate_mps_logs.py            # rebuild the results CSV
```
