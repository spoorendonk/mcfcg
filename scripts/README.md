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
lower_bound,optimal,time,…`) to `--out` (default `bench-results.csv`) and saves
its full per-iteration CG log to `--logdir` (default `bench-logs/`), one file per
run named `<family>__<instance>__<formulation>__<solver>.log`.

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

The **result CSV is the reproducibility artifact** — compact and diff-friendly;
commit it. The per-run CG logs under `bench-logs/` are bulky and fully
regenerable, so they are not tracked by default; keep a specific log only when
it documents a notable run.
