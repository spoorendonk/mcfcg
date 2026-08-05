#!/usr/bin/env python3
"""Rebuild the CG benchmark results CSV from the per-run CG logs.

`benchmark_solvers.py` writes one `<family>__<instance>__<formulation>__<solver>.log`
per run (via write_log): a `# === STDOUT (result CSV) ===` section carrying the
CLI's 2-line result CSV, plus the full CG iteration log on stderr. This tool walks
one or more log directories, re-parses each log's result row with
`benchmark_solvers.parse_csv_row`, re-checks the objective against the CURRENT
`optimal.csv` references, and writes one consolidated CSV — the log-based CG analog
of `consolidate_mps_logs.py`. Because it derives everything from the logs it makes
the committed `results/cg_benchmark.csv` reproducible: run the full suite with
`benchmark_solvers.py` (logs -> bench_runs/cg/logs), then run this.

Peak RSS (`mem_gb`) is read from each log's `# peak_rss_kb:` header, with
`mem_source` recording whether it was measured live or relocated from an older
sweep CSV by backfill_log_memory.py. Memory is the only metric measured outside
the child process, so it must be in the log for this tool to see it; logs
predating that header report an empty mem_gb until backfilled.

Multiple `--logdir`s are applied in PRIORITY ORDER: a later dir overrides an earlier
one for the same (family,instance,formulation,solver) cell. That is how the
authoritative HiPO ablation (HiGHS 1.15.1) supersedes a deprecated earlier `highs`
run when consolidating historical, multi-pass log sets; a single fresh run needs
just one dir.

Usage:
  # fresh run: one dir
  python3 scripts/consolidate_cg_logs.py                       # bench_runs/cg/logs -> results/cg_benchmark.csv
  # historical multi-pass sets, ablation last (authoritative highs):
  python3 scripts/consolidate_cg_logs.py --logdir \
      bench_runs/logs bench_runs/intermodal_logs bench_runs/transportation_logs \
      bench_runs/highs_hipo_ablation/logs
"""

import argparse
import csv
import os
import sys

import benchmark_solvers as bs  # sibling module; run from scripts/ or repo root

STDOUT_MARKER = "# === STDOUT (result CSV) ==="


def load_all_refs():
    refs = {}
    for fam, rel in bs.FAMILY_OPTIMAL.items():
        p = os.path.join(bs.REPO, "data", rel)
        if os.path.exists(p):
            refs.update(bs.load_optimal(p))
    return refs


def split_stem(stem):
    # "<family>__<instance>__<formulation>__<solver>"; instance keys carry no "__"
    # (e.g. BUS-2632-0), so a plain split yields exactly four fields.
    parts = stem.split("__")
    return tuple(parts) if len(parts) == 4 else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", nargs="+", default=["bench_runs/cg/logs"],
                    help="one or more CG log dirs, priority-ordered (a LATER dir "
                         "overrides an earlier one for the same cell).")
    ap.add_argument("--out", default="results/cg_benchmark.csv",
                    help="the committed 'one truth' CG results CSV (tracked in results/).")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    refs = load_all_refs()

    # cell key -> record; later logdirs overwrite earlier ones (priority order).
    cells = {}
    for logdir in args.logdir:
        if not os.path.isdir(logdir):
            sys.stderr.write(f"skip missing logdir: {logdir}\n")
            continue
        for fn in sorted(os.listdir(logdir)):
            if not fn.endswith(".log"):
                continue
            keyparts = split_stem(fn[:-4])
            if not keyparts:
                sys.stderr.write(f"skip {fn}: not a family__inst__form__solver log\n")
                continue
            family, inst, form, solver = keyparts
            if inst.startswith("SUBWAY"):
                continue  # unit-test-only family, no optimal.csv reference (excluded)
            text = open(os.path.join(logdir, fn), errors="replace").read()
            body = text.split(STDOUT_MARKER, 1)[1] if STDOUT_MARKER in text else text
            row = bs.parse_csv_row(body)
            # Peak RSS comes from the log HEADER, not the child's output: it is
            # measured externally by GNU time. Read it here so an errored run
            # (no result row) still contributes its memory — the OOM cells are
            # precisely the ones the memory comparison turns on.
            kb, mem_source = bs.parse_peak_rss(text)
            cells[keyparts] = (row, os.path.basename(logdir) + "/" + fn, kb, mem_source)

    fields = ["family", "instance", "formulation", "solver", "objective", "ref",
              "rel_err", "pass", "optimal", "time", "mem_gb", "mem_source", "source"]
    out_rows = []
    for (family, inst, form, solver), (row, source, kb, mem_source) in sorted(cells.items()):
        rec = {"family": family, "instance": inst, "formulation": form,
               "solver": solver, "objective": "", "ref": "", "rel_err": "",
               "pass": "", "optimal": "", "time": "",
               "mem_gb": bs.mem_gb_from_kb(kb), "mem_source": mem_source,
               "source": source}
        ref = refs.get(inst)
        if ref is not None:
            rec["ref"] = ref
        if row is None:
            # No result row in the log -> the run errored / timed out / crashed.
            rec["source"] = source + " (no result row)"
            out_rows.append(rec)
            continue
        try:
            obj = float(row["objective"])
        except (KeyError, ValueError):
            out_rows.append(rec)
            continue
        rec["objective"] = obj
        rec["optimal"] = row.get("optimal", "")
        rec["time"] = row.get("time", "")
        if ref is not None:
            rel = abs(obj - ref) / max(1.0, abs(ref))
            rec["rel_err"] = rel
            rec["pass"] = rel < args.tol
        out_rows.append(rec)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    # Coverage report vs the expected family x formulation x solver grid.
    SOLVERS = ["highs", "mosek", "cuopt", "copt-cpu", "copt-gpu"]
    have = set(cells)
    npass = sum(1 for r in out_rows if r["pass"] is True)
    nrow = len(out_rows)
    nmem = sum(1 for r in out_rows if r["mem_gb"] != "")
    print(f"Wrote {nrow} rows to {args.out}  (pass={npass}, "
          f"non-pass/err={nrow - npass})")
    print(f"peak RSS present: {nmem}/{nrow}"
          + (f"  MISSING {nrow - nmem}" if nmem < nrow else ""))
    by_fam = {}
    for (fam, inst, form, sol) in have:
        by_fam.setdefault(fam, set()).add(inst)
    print("=== coverage (present cells / instances x 5 solvers x {tree,path}) ===")
    for fam in ("grid", "planar", "transportation", "intermodal"):
        insts = sorted(by_fam.get(fam, ()))
        for form in ("tree", "path"):
            present = sum(1 for i in insts for s in SOLVERS
                          if (fam, i, form, s) in have)
            print(f"  {fam:<15} {form:<5}: {present:>3}/{len(insts) * len(SOLVERS)}")


if __name__ == "__main__":
    main()
