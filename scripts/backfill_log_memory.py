#!/usr/bin/env python3
"""One-shot: relocate measured peak RSS from old sweep CSVs into the run logs.

Context. `benchmark_solvers.py` has always measured peak RSS (GNU `time -f %M`)
but, until the `# peak_rss_kb:` header was added, wrote it ONLY into the sweep's
result CSV. Those CSVs live under the gitignored `bench_runs/`. Once
`results/cg_benchmark.csv` became log-derived (consolidate_cg_logs.py), memory
stopped reaching the committed results entirely — the logs never carried it.

This script closes that gap for the historical runs WITHOUT re-solving: for each
old sweep CSV row it finds the matching log and injects the header. The numbers
are real measurements from the very same executions, not estimates.

Provenance gate. A CSV row and a log file are only accepted as the same
execution when their `time` fields agree to within --time-tol (relative). On any
mismatch the script refuses to write that cell and reports it — a mismatch would
mean the memory number came from a different run than the log it is being
attached to, which is exactly the kind of quiet corruption this is meant to
avoid. Backfilled headers are tagged `# peak_rss_source: backfilled:<csv>` so a
reader can always tell them from `measured`.

Rows whose run errored carry no time and therefore cannot be time-matched; they
are skipped unless --allow-untimed is given (see --help). The known case is
transportation/Sydney/path/mosek, which errored with no result row but whose
peak RSS (~95.8 GB) is itself a headline datum.

Idempotent: a log that already has a `# peak_rss_kb:` header is left alone
unless --force is given.

Usage:
  # dry run first — prints exactly what would change
  python3 scripts/backfill_log_memory.py --dry-run
  # then apply
  python3 scripts/backfill_log_memory.py --allow-untimed
"""

import argparse
import csv
import os
import sys

import benchmark_solvers as bs  # sibling module; run from scripts/ or repo root

# Sweep CSVs that carry a mem_gb column, mapped to the log dir holding the runs
# they came from. Priority order matches consolidate_cg_logs.py: a later entry
# wins for the same cell (the HiPO ablation supersedes earlier `highs` runs).
DEFAULT_SOURCES = [
    ("bench_runs/grid_planar_5cfg.csv", "bench_runs/logs"),
    ("bench_runs/intermodal_5cfg_pathtree.csv", "bench_runs/intermodal_logs"),
    ("bench_runs/transportation_5cfg_pathtree.csv", "bench_runs/transportation_logs"),
    ("bench_runs/transportation_path_sydney_winnipeg.csv", "bench_runs/transportation_logs"),
    ("bench_runs/highs_hipo_ablation/highs_hipo_tree.csv",
     "bench_runs/highs_hipo_ablation/logs"),
    ("bench_runs/highs_hipo_ablation/highs_hipo_path.csv",
     "bench_runs/highs_hipo_ablation/logs"),
]


def log_name(row):
    return (f"{row['family']}__{row['instance']}__{row['formulation']}"
            f"__{row['solver']}.log")


def times_agree(csv_time, log_time, tol):
    """True when both times are present and match to `tol` relative."""
    if not csv_time or not log_time:
        return None  # untimed: cannot prove same-run either way
    try:
        a, b = float(csv_time), float(log_time)
    except ValueError:
        return None
    return abs(a - b) <= tol * max(1.0, abs(a))


def inject_header(text, kb, source):
    """Insert (or replace) the peak-RSS headers in a saved log's header block."""
    lines = text.splitlines(keepends=True)
    kept, i = [], 0
    while i < len(lines) and lines[i].startswith("#") and not lines[i].startswith("# ==="):
        if not (lines[i].startswith("# peak_rss_kb:")
                or lines[i].startswith("# peak_rss_source:")):
            kept.append(lines[i])
        i += 1
    kept.append(f"# peak_rss_kb: {kb}\n")
    kept.append(f"# peak_rss_source: {source}\n")
    return "".join(kept) + "".join(lines[i:])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change; write nothing.")
    ap.add_argument("--force", action="store_true",
                    help="overwrite a peak_rss header that is already present.")
    ap.add_argument("--allow-untimed", action="store_true",
                    help="also backfill cells whose CSV row has no time (errored "
                         "runs). Same-run provenance cannot be proven for these, so "
                         "they are tagged backfilled-untimed:<csv> in the log.")
    ap.add_argument("--time-tol", type=float, default=1e-6,
                    help="relative tolerance for the CSV-vs-log time match "
                         "(default 1e-6).")
    args = ap.parse_args()

    os.chdir(bs.REPO)
    applied, skipped_have, mismatched, untimed, nolog, nomem = 0, 0, [], [], [], 0

    for csv_path, logdir in DEFAULT_SOURCES:
        if not os.path.exists(csv_path):
            sys.stderr.write(f"skip missing sweep CSV: {csv_path}\n")
            continue
        for row in csv.DictReader(open(csv_path)):
            mem_gb = (row.get("mem_gb") or "").strip()
            if not mem_gb:
                nomem += 1
                continue
            path = os.path.join(logdir, log_name(row))
            if not os.path.exists(path):
                nolog.append((log_name(row), csv_path))
                continue
            text = open(path, errors="replace").read()
            have_kb, _ = bs.parse_peak_rss(text)
            if have_kb is not None and not args.force:
                skipped_have += 1
                continue

            body = text.split("# === STDOUT", 1)
            log_row = bs.parse_csv_row(body[1]) if len(body) > 1 else None
            log_time = (log_row or {}).get("time", "")
            agree = times_agree(row.get("time", ""), log_time, args.time_tol)
            if agree is False:
                mismatched.append((log_name(row), row.get("time"), log_time, csv_path))
                continue
            if agree is None:
                if not args.allow_untimed:
                    untimed.append((log_name(row), csv_path))
                    continue
                source = f"backfilled-untimed:{csv_path}"
            else:
                source = f"backfilled:{csv_path}"

            # mem_gb was rounded to 3 decimals on the way into the CSV, so the
            # original KB integer is not exactly recoverable. Convert back and
            # mark it backfilled: the value is the real measurement to ~1 MB.
            kb = int(round(float(mem_gb) * 1024 * 1024))
            if not args.dry_run:
                with open(path, "w") as f:
                    f.write(inject_header(text, kb, source))
            applied += 1

    verb = "would backfill" if args.dry_run else "backfilled"
    print(f"{verb}: {applied} logs")
    print(f"already had a header (skipped): {skipped_have}")
    print(f"CSV rows with no mem_gb: {nomem}")
    if untimed:
        print(f"\nuntimed (errored runs; pass --allow-untimed to include): {len(untimed)}")
        for name, src in untimed:
            print(f"  {name}  ({src})")
    if nolog:
        print(f"\nno matching log file: {len(nolog)}")
        for name, src in nolog[:20]:
            print(f"  {name}  ({src})")
        if len(nolog) > 20:
            print(f"  ... and {len(nolog) - 20} more")
    if mismatched:
        print(f"\n*** REFUSED — time mismatch, CSV row is from a different execution "
              f"than the log: {len(mismatched)} ***")
        for name, ct, lt, src in mismatched:
            print(f"  {name}  csv_t={ct}  log_t={lt}  ({src})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
