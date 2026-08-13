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
`mem_source` recording whether the run wrote it live (`measured`) or it was
relocated into the log from that same run's row in an older sweep CSV, back when
the header did not yet exist (`backfilled:<csv>`, a one-shot performed before
archiving — PROVENANCE.txt section 1.1). Memory is the only metric measured
outside the child process, so it must be in the log for this tool to see it; a
log with no header reports an empty mem_gb and can only be re-measured by
re-running the cell.

Multiple `--logdir`s are applied in PRIORITY ORDER: a later dir overrides an earlier
one for the same (family,instance,formulation,solver) cell. That is how the
authoritative HiPO ablation (HiGHS 1.15.1) supersedes a deprecated earlier `highs`
run when consolidating historical, multi-pass log sets; a single fresh run needs
just one dir.

Usage:
  # fresh run: one dir
  python3 scripts/consolidate_cg_logs.py                       # bench_runs/cg/logs -> results/cg_benchmark.csv
  # the committed results/cg_benchmark.csv, exactly: six historical passes in
  # priority order, each later dir superseding the cells it re-ran. Verified
  # byte-identical to the tracked CSV when that CSV was regenerated; treat this
  # list as the canonical incantation rather than reconstructing it from the
  # `source` column again, and keep it in sync with extract_iterations.py --
  # the two tracked CSVs must be built from the SAME six dirs or they describe
  # different runs.
  #
  # It is NOT a pure replay: ref/rel_err/pass are recomputed against the CURRENT
  # data/*/optimal.csv and --tol, so updating a reference changes those three
  # columns with every log untouched. A diff there is a reference change; a diff
  # elsewhere is a log-set or parser change.
  python3 scripts/consolidate_cg_logs.py --logdir \
      bench_runs/logs bench_runs/intermodal_logs bench_runs/transportation_logs \
      bench_runs/transportation_logs_v2 bench_runs/highs_hipo_ablation/logs \
      bench_runs/issue40_rerun/logs
"""

import argparse
import csv
import os
import sys

import benchmark_solvers as bs  # sibling module; run from scripts/ or repo root

STDOUT_MARKER = bs.STDOUT_MARKER  # format owned by benchmark_solvers.write_log


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
            # (no result row) still contributes its memory — a cell killed at its
            # high-water mark is precisely the one the memory comparison turns on.
            kb, mem_source = bs.parse_peak_rss(text)
            # Exit disposition, likewise from the header — the child cannot report
            # how it died, so a run killed by a signal prints nothing at all and
            # the ONLY trace is `# returncode:`. Without this a killed cell landed
            # here as a row of blanks annotated "(no result row)", identical to a
            # crash or a licence failure.
            exit_status = bs.format_exit_status(bs.parse_returncode(text))
            # Per-iteration aggregates: `columns` from the result row is the
            # FINAL master size, which reconciles to neither the generated count
            # (aging purges) nor the seed. Carry all three so the paper's table
            # can say which is which. extract_iterations.py owns the trace.
            iters = bs.parse_iteration_table(text)
            slack_mode = bs.parse_slack_mode(text)
            # Full dir, not basename: bench_runs/logs and
            # bench_runs/highs_hipo_ablation/logs both basename to "logs", which
            # left `source` unable to say which logdir actually won a cell — it
            # named the superseded dir for the 88 ablation rows.
            cells[keyparts] = (row, os.path.join(logdir, fn), kb, mem_source, iters,
                               slack_mode, exit_status)

    # Columns carried straight through from the CLI's result row. `columns` is
    # master.num_columns() at TERMINATION (cg_loop.h) — the final master size,
    # not the number of columns generated, which differs whenever aging purges
    # (--col-age-limit). The generated count is a per-iteration sum of `+col`;
    # extract_iterations.py produces it. Both quantities are wanted (gh #38).
    PASSTHROUGH = ["iterations", "columns", "lower_bound", "time_lp",
                   "time_pricing", "time_separation"]

    # Derived from the iteration trace, not the result row. `columns_seeded` is
    # the warm-start pool that `+col` never counts. These do NOT reconcile into
    # an identity with `columns` — see summarize_iterations for the measurements
    # and the open question about where the extra columns come from.
    DERIVED = ["columns_generated", "columns_seeded", "columns_purged",
               "slack_columns", "cuts_added", "cuts_removed"]

    # `exit_status` sits next to `optimal` because it qualifies it. A time-limited
    # run still prints its result row, so "ran but did not certify" is
    # `optimal=0`; `optimal` is blank only when there is NO result row at all, and
    # `exit_status` is what says why there is none -- killed by a signal, nonzero
    # exit, or a clean exit whose output would not parse. Empty `exit_status` =
    # the log predates the returncode header, which is unknown rather than ok
    # (bs.format_exit_status).
    # NOT named `status`: results/mps_compact_baseline.csv already uses that for
    # the solver's SOLUTION status, and the two CSVs get compared side by side.
    fields = (["family", "instance", "formulation", "solver", "objective", "ref",
               "rel_err", "pass", "optimal", "exit_status", "time"] + PASSTHROUGH + DERIVED
              + ["mem_gb", "mem_source", "source"])
    out_rows = []
    for (family, inst, form, solver), (row, source, kb, mem_source, iters, smode,
                                       exit_status) in sorted(cells.items()):
        rec = {"family": family, "instance": inst, "formulation": form,
               "solver": solver, "objective": "", "ref": "", "rel_err": "",
               "pass": "", "optimal": "", "exit_status": exit_status, "time": "",
               "mem_gb": bs.mem_gb_from_kb(kb), "mem_source": mem_source,
               "source": source}
        rec.update({f: "" for f in PASSTHROUGH})
        # The trace exists even when the run errored before printing a result
        # row, so these are filled before the row is None early-out below.
        rec.update({f: "" for f in DERIVED})
        rec.update(bs.summarize_iterations(iters, smode))
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
        rec.update({f: row.get(f, "") for f in PASSTHROUGH})
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
