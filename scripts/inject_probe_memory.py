#!/usr/bin/env python3
"""Inject the memory-probe sweep's peaks into the full-solve baseline logs.

Context. The compact-model baseline (`benchmark_mps.py`, bench_runs/mps/logs,
2026-08-03..05) predates the memory instrumentation, so its logs carry no peak
RSS and results/mps_compact_baseline.csv has a memory figure for only the five
grid1 cells that were re-run afterwards. The iteration-capped probe sweep
(`benchmark_mps.py --probe-iters 3`, bench_runs/mps_probe/logs, 2026-08-11..12)
measured all 165 cells cheaply. Re-running the full sweep purely for memory costs
days, so the probe's numbers are relocated into the baseline logs instead, and
results/mps_compact_baseline.csv gains a memory column from them.

    THE INJECTED NUMBER IS NOT THE FULL SOLVE'S PEAK.

It is the high-water mark of model read + presolve + the first N barrier
iterations — the regime that contains the symbolic and numeric factorization of
the normal equations, which dominates a barrier's footprint. Call it the
model-setup / initial peak. Treat it as a LOWER BOUND on the peak of the full
solve whose wall-clock sits beside it in the same CSV row. Calibration on grid7,
planar300 and grid10 across all five backends put three iterations at 0.95-1.00
of the full-solve peak, but that is a measured range on small instances, not a
guarantee for the giants.

Every layer keeps the two regimes apart:
    log      `# peak_rss_source: probe3:<probe log path>` — never the bare
             `backfilled:` tag, which means "same execution, different file".
    CSV      a `mem_source` column, surfaced by consolidate_mps_logs.py.
    docs     PROVENANCE.txt section 2.

NO time/outcome gate here — deliberately, and do not "fix" it.
backfill_log_memory.py checks that the CSV row and the log agree on time and
outcome because it relocates the SAME execution's measurement and a disagreement
would prove a mixup. This script relocates a DIFFERENT execution's measurement on
purpose: a 3-iteration probe and a full solve are expected to disagree on both
(Austin x copt-cpu: 1106 s probe vs 7267 s full; and 24 probe cells stopped at
the iteration cap or died at the cgroup cap while their full solves ran to a
timeout or an optimum). The only pairing evidence that exists — and the only one
that means anything — is the (instance, solver) identity of the two log files.

What is refused rather than guessed:
    * a probe log with no `# probe_iters:` header (not from a probe sweep);
    * a probe log whose peak is itself injected (no chains);
    * a baseline log that carries `# probe_iters:` (the trees are swapped);
    * a cell present in one tree and not the other.
A probe cell with no peak stays blank in the baseline — the known one is
ChicagoRegional x highs, killed by the harness timeout before GNU `time` could
write (a hole since closed: benchmark_mps.kill_preserving_mem).

Idempotent: a baseline log that already carries a peak is left alone unless
--force, and a live `measured` header (the five grid1 cells) is never overwritten
even then.

Usage:
  python3 scripts/inject_probe_memory.py --dry-run     # report, write nothing
  python3 scripts/inject_probe_memory.py
  python3 scripts/consolidate_mps_logs.py              # rebuild the results CSV
"""

import argparse
import os
import sys

import benchmark_mps as bm  # sibling modules; run from scripts/ or repo root
import benchmark_solvers as bs
import consolidate_mps_logs as cm  # same (instance, solver) key as the CSV rows

BASELINE_LOGS = "bench_runs/mps/logs"
PROBE_LOGS = "bench_runs/mps_probe/logs"

# Every header line the injected block owns. Listed once: rewrite_header_block
# drops exactly these and re-adds whichever of them the probe supplies, so a
# re-injection can never leave a stale VRAM figure beside a fresh RSS one.
MEM_HEADERS = ("# peak_rss_kb:", "# peak_rss_source:", "# peak_vram_mib:")


def source_tag(probe_iters, reached_cap, probe_path):
    """Provenance string for an injected peak.

    `probeN:` — the probe reached its iteration cap, so the model-setup regime
    (read, presolve, symbolic + numeric factorization, N iterations) completed.
    `probeN-partial:` — it stopped short: OOM at the cgroup cap, a backend
    error, or a clean rc=0 exit that never reached the barrier at all (cuOpt
    does this on a VRAM exhaustion it fails to report — see the #33 note in
    CLAUDE.md). The peak is wherever it stopped: still a lower bound on the full
    solve, but a weaker one, and the reader is told which kind they have.

    Keyed on the backend's own iteration-limit marker rather than on the exit
    status, because a clean exit is NOT evidence the barrier ran.
    """
    suffix = "" if reached_cap else "-partial"
    return f"probe{probe_iters}{suffix}:{probe_path}"


def collect(logdir):
    """{(instance, solver): filename} for the .log files in `logdir`.

    Keyed by consolidate_mps_logs.split_key_solver, so a pair here is exactly a
    pair of rows there — the join the CSV will show is the join we injected on.
    """
    out = {}
    for fn in sorted(os.listdir(logdir)):
        if fn.endswith(".log"):
            out[cm.split_key_solver(fn[:-4])] = fn
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-logs", default=BASELINE_LOGS,
                    help=f"full-solve logs to inject INTO (default {BASELINE_LOGS}).")
    ap.add_argument("--probe-logs", default=PROBE_LOGS,
                    help=f"memory-probe logs to read FROM; never written "
                         f"(default {PROBE_LOGS}).")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change; write nothing.")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing INJECTED peak. A full-solve "
                         "reading (`measured` or `backfilled:`) is never "
                         "overwritten: it is the stronger number.")
    args = ap.parse_args()

    os.chdir(bs.REPO)
    for d in (args.baseline_logs, args.probe_logs):
        if not os.path.isdir(d):
            sys.exit(f"[inject] ABORT: not a log directory: {d}")

    base_logs, probe_logs = collect(args.baseline_logs), collect(args.probe_logs)
    applied, skipped_measured, skipped_have = 0, 0, 0
    no_peak, refused = [], []

    for cell in sorted(probe_logs):
        probe_path = os.path.join(args.probe_logs, probe_logs[cell])
        with open(probe_path, errors="replace") as f:
            probe_text = f.read()
        iters = bm.parse_probe_iters(probe_text)
        if iters is None:
            refused.append((cell, "probe log has no `# probe_iters:` header"))
            continue
        kb, probe_source = bs.parse_peak_rss(probe_text)
        if kb is None:
            no_peak.append(cell)
            continue
        if probe_source != "measured":
            refused.append((cell, f"probe peak is itself {probe_source!r}, "
                                  "not a live measurement"))
            continue
        if cell not in base_logs:
            refused.append((cell, "no matching baseline log"))
            continue

        base_path = os.path.join(args.baseline_logs, base_logs[cell])
        with open(base_path, errors="replace") as f:
            base_text = f.read()
        if bm.parse_probe_iters(base_text) is not None:
            refused.append((cell, "baseline log is itself a probe -- trees swapped?"))
            continue
        have_kb, have_source = bs.parse_peak_rss(base_text)
        if have_kb is not None:
            # Only a peak this script injected is refreshable. `measured` and
            # `backfilled:` are both the FULL SOLVE's own reading -- backfill
            # relocates the same execution's number under a time/outcome gate --
            # so a probe lower bound must never replace either, --force or not.
            # Memory cannot be re-derived: a downgrade here is unrecoverable.
            if not have_source.startswith("probe"):
                skipped_measured += 1
                continue
            if not args.force:
                skipped_have += 1
                continue

        # `outcome == "ok"` is not enough: cuOpt exits 0 after a swallowed VRAM
        # failure having never run an iteration (ChicagoRegional, Philadelphia,
        # planar2500 -- their bodies stop after presolve, at ~15.07 of 15.47 GiB).
        # Require the backend's own iteration-limit marker, which is what "the
        # probe reached its cap" actually means. Unknown solvers cannot be
        # judged, so they take the weaker tag rather than a KeyError.
        _, _, probe_outcome = bm.parse_run_header(probe_text)
        reached_cap = (probe_outcome == "ok"
                       and cell[1] in cm.KNOWN_SOLVERS
                       and bm.probe_hit_iter_limit(
                           cell[1], probe_text.split(cm.MARKER, 1)[-1]))
        tag = source_tag(iters, reached_cap, probe_path)
        # format_extra_headers(..., 0): the VRAM line travels with the RSS one,
        # but the `# probe_iters:` line MUST NOT -- consolidate_mps_logs.py keys
        # the solve/probe populations off it, so a baseline log carrying it would
        # silently drop out of the results CSV. The cap is recorded in the tag.
        block = (bs.format_peak_rss_headers(kb, tag)
                 + bm.format_extra_headers(bm.parse_peak_vram_mib(probe_text), 0))
        if not args.dry_run:
            # Write-then-rename: these logs are the only surviving record of runs
            # that cost days to reproduce.
            tmp = base_path + ".tmp"
            with open(tmp, "w") as f:
                f.write(bs.rewrite_header_block(base_text, MEM_HEADERS, block))
            os.replace(tmp, base_path)
        applied += 1

    # A baseline cell with no probe counterpart is only a finding when it has no
    # memory of its OWN. Cells solved after the instrumentation landed (Sydney,
    # BUS-2632-0) are measured directly, which beats an injected lower bound --
    # flagging those would cry wolf on every future run and, since unpaired cells
    # exit non-zero, would leave the injector permanently "failing".
    unpaired_measured, unpaired_base = [], []
    for cell in sorted(set(base_logs) - set(probe_logs)):
        with open(os.path.join(args.baseline_logs, base_logs[cell]),
                  errors="replace") as f:
            kb, _ = bs.parse_peak_rss(f.read())
        (unpaired_measured if kb is not None else unpaired_base).append(cell)
    verb = "would inject" if args.dry_run else "injected"
    print(f"{verb}: {applied} baseline logs")
    print(f"kept the full solve's own measurement (skipped): {skipped_measured}")
    print(f"already injected (skipped; --force to refresh): {skipped_have}")
    print(f"baseline cells: {len(base_logs)}   probe cells: {len(probe_logs)}")
    if no_peak:
        print(f"\nprobe cell with no peak -- left BLANK, never invented: {len(no_peak)}")
        for inst, solver in no_peak:
            print(f"  {inst} x {solver}")
    if unpaired_measured:
        print(f"\nno probe cell, but measured in the full solve -- nothing to "
              f"inject: {len(unpaired_measured)}")
        for inst, solver in unpaired_measured:
            print(f"  {inst} x {solver}")
    if unpaired_base:
        print(f"\n*** baseline cells with no probe cell AND no memory: "
              f"{len(unpaired_base)} ***")
        for inst, solver in unpaired_base:
            print(f"  {inst} x {solver}")
    if refused:
        print(f"\n*** REFUSED: {len(refused)} ***")
        for cell, why in refused:
            print(f"  {cell[0]} x {cell[1]}: {why}")
    if refused or unpaired_base:
        # The two sweeps are meant to cover the same matrix. Them disagreeing
        # about what the matrix IS is a finding, not a detail to print quietly.
        return 1
    # "Did the operator point at the wrong directories?" is a question about
    # PAIRING, not about how many writes happened. A tree whose every probe cell
    # legitimately has no peak pairs fine and must not be reported as misaimed.
    if not (applied or skipped_measured or skipped_have or no_peak):
        print("\n*** no cell paired between the two trees -- "
              "wrong log directories? ***")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
