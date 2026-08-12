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

Provenance gate. A CSV row and a log file are accepted as the same execution
only when their `time` and `outcome` both agree (time to within --time-tol,
relative). On any mismatch the script refuses that cell and reports it — a
mismatch means the memory number came from a different run than the log it is
being attached to, which is exactly the kind of quiet corruption this exists to
avoid.

The gate is NECESSARY, NOT SUFFICIENT. Times print to 3 decimals, so two runs of
a fast deterministic instance can agree exactly while being different executions
on different hardware — and peak RSS depends on machine, build and solver
version, not just the instance. Empirically, two legacy-sweep rows pass the time
gate against unrelated logs with memory off by ~13x. So only add a CSV to
DEFAULT_SOURCES when you already know from provenance notes that it produced the
logs in the paired logdir; the gate is a tripwire against mistakes, not evidence
of pairing.

Backfilled headers are tagged `# peak_rss_source: backfilled:<csv>` so a reader
can always tell them from `measured`.

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
# they came from. No two pairs target the same log file today; if a future pair
# did, the FIRST entry to write a header would win, because a log that already
# carries one is skipped (see --force). Supersession between runs is resolved by
# consolidate_cg_logs.py's --logdir priority, not here — the HiPO ablation has
# its own log dir, so it never collides with the earlier `highs` runs.
#
# bench_runs/legacy-root/*.csv are deliberately NOT listed. They do carry mem_gb
# for 14 of the 28 still-missing transportation/path cells, but from a different
# execution than the logs those cells consolidate from (different timings, their
# own logs under legacy-root/bench-logs/). Their memory belongs to those runs,
# not these; the fix for the 28 is the rerun in gh #37 ask (c), not a cross-sweep
# graft. This is the trap the "necessary, not sufficient" note above is about.
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
    """Tri-state match on the CSV-vs-log `time` field.

    True/False when both times are present and parseable (relative comparison
    against `tol`); None when either is simply absent, meaning "cannot check
    either way" — an errored run records no time. The caller treats the two
    negative cases differently: None is skipped unless --allow-untimed, False is
    always refused, because False is positive evidence of a different execution.
    An unparseable time is likewise refused: garbage is not the same as absent.
    """
    if not csv_time or not log_time:
        return None  # untimed: nothing to compare
    try:
        a, b = float(csv_time), float(log_time)
    except ValueError:
        return False  # corrupt: cannot trust this pairing
    return abs(a - b) <= tol * max(1.0, abs(a))


def parse_outcome(log_text):
    """Read write_log's `# outcome:` header ("ok"/"error"), or "" if absent."""
    for line in bs.iter_header_lines(log_text):
        if line.startswith("# outcome:"):
            return line.split(":", 1)[1].strip()
    return ""


def inject_header(text, kb, source):
    """Insert (or replace) the peak-RSS headers in a saved log's header block.

    Rewriting is benchmark_solvers.rewrite_header_block and the lines themselves
    are format_peak_rss_headers, so a backfilled log carries exactly what write_log
    produces live — and the sibling injector (inject_probe_memory.py) shares both.
    """
    return bs.rewrite_header_block(
        text, bs.PEAK_RSS_HEADER_PREFIXES,
        bs.format_peak_rss_headers(kb, source))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change; write nothing.")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing BACKFILLED peak_rss header. A "
                         "`measured` header (a live GNU-time reading) is never "
                         "overwritten — it is strictly more precise.")
    ap.add_argument("--allow-untimed", action="store_true",
                    help="also backfill cells whose CSV row has no time (errored "
                         "runs). The time check cannot run for these, so they are "
                         "tagged backfilled-untimed:<csv> in the log. The outcome "
                         "check still applies.")
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
            have_kb, have_source = bs.parse_peak_rss(text)
            # A live `measured` header always wins over a backfill: it is the
            # original integer from GNU time, whereas a backfilled value is
            # reconstructed from a 3-decimal GB round-trip and carries ~0.5 MB of
            # fabricated precision. --force may refresh an earlier backfill; it
            # must never clobber a real measurement.
            if have_kb is not None and (have_source == "measured" or not args.force):
                skipped_have += 1
                continue

            # Same split AND same fallback as consolidate_cg_logs.py: a log
            # missing the marker still gets its result row read, so the gate uses
            # every scrap of evidence rather than degrading to "untimed".
            body = text.split(bs.STDOUT_MARKER, 1)
            log_row = bs.parse_csv_row(body[1] if len(body) > 1 else text)
            # Outcome is a second, independent signal that costs nothing and
            # catches cross-sweep mixups the time check cannot: a log recording
            # `ok` cannot be the execution behind an errored CSV row. It is the
            # ONLY check available for untimed (errored) rows.
            log_outcome = parse_outcome(text)
            csv_outcome = (row.get("outcome") or "").strip()
            if csv_outcome and log_outcome and csv_outcome != log_outcome:
                mismatched.append((log_name(row), f"outcome={csv_outcome}",
                                   f"outcome={log_outcome}", csv_path))
                continue
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
                # Write-then-rename: this log is the only surviving record of the
                # run (regenerating one costs hours), so never truncate in place.
                tmp = path + ".tmp"
                with open(tmp, "w") as f:
                    f.write(inject_header(text, kb, source))
                os.replace(tmp, path)
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
        print(f"\n*** REFUSED — CSV row disagrees with the log, so it is from a "
              f"different execution: {len(mismatched)} ***")
        for name, ct, lt, src in mismatched:
            print(f"  {name}  csv={ct}  log={lt}  ({src})")
        return 1
    if nolog and not applied:
        # Recovered nothing and found no logs: almost certainly pointed at the
        # wrong tree. Don't report success — this is a data-recovery tool.
        print("\n*** nothing backfilled and no logs matched — wrong --logdir tree? ***")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
