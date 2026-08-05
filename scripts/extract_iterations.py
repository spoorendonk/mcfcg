#!/usr/bin/env python3
"""Extract the per-iteration CG trace from the run logs into a tidy CSV.

`benchmark_solvers.py` captures the CLI's per-iteration table (stderr, at
Verbosity::Iteration) verbatim in every log, but the consolidated results CSV
carries only end-of-run summaries. This walks the same priority-ordered
`--logdir` set as `consolidate_cg_logs.py` and emits one row per
(family, instance, formulation, solver, iteration) — read-only over existing
logs, no re-solving (gh #38).

What it answers that the summary row cannot:

- **columns generated** = sum of `+col`, which is NOT the final master size:
  aging purges columns (`--col-age-limit`), so the master ends smaller than the
  total generated.
- **lazy capacity rows added** = sum of `+cut`, quantifying the active-set
  mechanism.
- **LB/UB convergence traces** for per-instance deep dives.

Two semantics that a table caption must state, both discovered from the logs:

1. `+col` EXCLUDES warm-start seed columns. Those are added before the loop, so
   iteration 1 typically shows a large `#col` with `+col = 0`. On path masters
   the seed dominates: Austin path starts at 1,082,300 columns for 1,081,717
   commodities — one per commodity — so `columns_generated` (350,000) is far
   below the final master size (1,414,271). Quoting the final master size as
   "columns generated" therefore credits the path formulation's warm start to
   its pricer. `columns_seeded` in results/cg_benchmark.csv separates them.
   The trace's `#col` also counts slack columns while the result row's
   `columns` does not; `slack_columns` accounts for that gap at termination.
   These figures do NOT close into a per-iteration identity — `#col` grows by
   more than `+col` reports on most runs. Quote each for what it names.
2. A `+col` value printed as `*N` was priced but never added — the loop hit the
   optimality gap and returned without calling `add_columns` (cg_loop.h). Those
   are excluded from `columns_generated`.

Note the trace's UB/LB/LP_obj are printed to 4 significant figures, which is
plenty for convergence plots but NOT for reporting objectives — take those from
`results/cg_benchmark.csv`, which parses the full-precision result row.

Usage:
  python3 scripts/extract_iterations.py                       # bench_runs/cg/logs
  python3 scripts/extract_iterations.py --logdir \
      bench_runs/logs bench_runs/intermodal_logs bench_runs/transportation_logs \
      bench_runs/highs_hipo_ablation/logs
"""

import argparse
import csv
import os
import sys

import benchmark_solvers as bs  # sibling module; run from scripts/ or repo root
import consolidate_cg_logs as cc

FIELDS = ["family", "instance", "formulation", "solver", "iteration",
          "ub", "lb", "lp_obj", "n_col", "n_row", "n_slk",
          "added_col", "removed_col", "added_cut", "removed_cut",
          "col_committed", "t_lp", "t_pricing", "t_separation", "t_iter", "t_acc"]


def iter_logs(logdirs):
    """Yield (cell_key, log_text) with later logdirs overriding earlier ones."""
    cells = {}
    for logdir in logdirs:
        if not os.path.isdir(logdir):
            sys.stderr.write(f"skip missing logdir: {logdir}\n")
            continue
        for fn in sorted(os.listdir(logdir)):
            if not fn.endswith(".log"):
                continue
            key = cc.split_stem(fn[:-4])
            if not key or key[1].startswith("SUBWAY"):
                continue  # same exclusions as the results consolidator
            cells[key] = os.path.join(logdir, fn)
    for key in sorted(cells):
        yield key, open(cells[key], errors="replace").read()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", nargs="+", default=["bench_runs/cg/logs"],
                    help="one or more CG log dirs, priority-ordered (a LATER dir "
                         "overrides an earlier one for the same cell).")
    ap.add_argument("--out", default="results/cg_iterations.csv",
                    help="tracked tidy per-iteration CSV.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_rows, n_cells, no_table = [], 0, []
    for (family, inst, form, solver), text in iter_logs(args.logdir):
        rows = bs.parse_iteration_table(text)
        if not rows:
            no_table.append(f"{family}/{inst}/{form}/{solver}")
            continue
        n_cells += 1
        for r in rows:
            rec = {"family": family, "instance": inst, "formulation": form,
                   "solver": solver}
            rec.update({k: r.get(k, "") for k in FIELDS[4:]})
            out_rows.append(rec)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} iteration rows from {n_cells} runs to {args.out}")
    if no_table:
        print(f"no iteration table ({len(no_table)}): "
              + ", ".join(no_table[:5]) + (" ..." if len(no_table) > 5 else ""))


if __name__ == "__main__":
    main()
