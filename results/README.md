# Results

The committed result tables behind the accompanying paper (arXiv:2509.24656).
Each file is derived from per-run solver logs by a consolidator script — the
consolidators **re-parse logs, they never re-solve**, so regenerating a table
from the same logs is deterministic.

`PROVENANCE.txt` at the repo root is the citable record: it pins the exact solver
versions, the host, and the two non-stock builds behind these numbers. Read it
before quoting anything here.

| file | rows | what it is | provenance | regenerate with |
|---|---|---|---|---|
| `cg_benchmark.csv` | 440 | The headline column-generation table: one row per (instance × formulation × backend) cell. | `PROVENANCE.txt` §1, §1.1 | `scripts/consolidate_cg_logs.py` |
| `cg_iterations.csv` | 9601 | Per-iteration CG trace for 439 of those 440 cells (the one `killed SIGKILL` row has no trace). | `PROVENANCE.txt` §1 | `scripts/extract_iterations.py` |
| `mps_compact_baseline.csv` | 175 | Direct barrier solves of the compact arc-flow LP, for comparison against CG. | `PROVENANCE.txt` §2, §2.2, §2.3 | `scripts/consolidate_mps_logs.py` |
| `mps_compact_memory.csv` | 165 | Iteration-capped memory probe feeding the baseline table's `mem_gb`. | `PROVENANCE.txt` §2.1 | `scripts/consolidate_mps_logs.py` |
| `ablation/families/` | 444 logs + 2 CSVs | Round (a) of the gh#41 bounded-pricing A/B: four families at one backend, 3 reps. Settled why the flag stays **off** by default — pricing share is 0.2–4.3% outside intermodal. | `ablation/README.md`, `ablation/families/README.md`, `PROVENANCE.txt` §5.1 | `scripts/analyze_bounded_pricing_ablation.py --round families` |
| `ablation/backends/` | 240 logs + 2 CSVs | Round (b) of the same A/B: one family (intermodal) across five backends, mixed reps. Settled that the reported HiGHS penalty was cross-session drift, and that the bound wins **−2.8%** on this family. | `ablation/README.md`, `ablation/backends/README.md`, `PROVENANCE.txt` §5.1 | `scripts/analyze_bounded_pricing_ablation.py --round backends` |

`ablation/` is the one place raw logs are tracked — it settles a design question
rather than feeding a results table, so the logs are the primary artifact and the
CSVs are derived from them. It is split into rounds named for the axis each
varies: `families/` is round (a), `backends/` is round (b). Read
`ablation/README.md` for the argument the two rounds jointly make, and the round
README before quoting any number from it. Everywhere else, per-run logs are
bulky, regenerable, and gitignored.

`scripts/README.md` documents each consolidator's `--logdir` conventions and the
full-sweep drivers (`benchmark_solvers.py`, `benchmark_mps.py`) that produce the
logs in the first place.

## Two caveats a reader needs

**The `source` and `mem_source` columns name directories that are not in this
archive.** They record *which local run* produced each row — a provenance
breadcrumb, not a path you can open. `PROVENANCE.txt` §5 states this explicitly
and lists the six log directories the table is consolidated from.

**Objectives, not timings, are the reproducible quantity.** Wall-clock numbers
measure the host pinned in `PROVENANCE.txt` §3. Use `scripts/check_reproduction.py`
to gate a fresh sweep against `cg_benchmark.csv`; it compares objectives, and only
for cells that proved optimality — a time-limited run's objective measures host
speed, not the algorithm. See `PROVENANCE.txt` §6 for what each backend
configuration costs (HiGHS needs no licence and no GPU; only MOSEK and COPT need a
commercial licence).
