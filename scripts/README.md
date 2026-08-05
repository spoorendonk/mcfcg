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
# MPS baseline: per-cell logs -> results/mps_compact_baseline.csv
python3 scripts/consolidate_mps_logs.py
```

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

`mem_source` is `measured` for a live run, or `backfilled:<csv>` when relocated
from a pre-header sweep CSV by `backfill_log_memory.py` (a one-shot recovery tool;
it refuses to write unless the CSV row's `time` matches the log's, proving both
came from the same execution). **Never delete `bench_runs/` while any cell still
reads `backfilled:` from a CSV that is the only copy** — check with:

```
python3 -c "import csv;rows=list(csv.DictReader(open('results/cg_benchmark.csv')));\
print(sum(1 for r in rows if not r['mem_gb']),'cells missing mem_gb')"
```

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
