#!/usr/bin/env python3
"""Compare a fresh benchmark_solvers.py sweep against the committed results.

Answers one question: did this machine reproduce the objective values in
`results/cg_benchmark.csv` for the cells it ran?

Only the objective is compared. Wall-clock time and peak RSS are properties of
the host, the GPU and the solver build, so a faithful reproduction on different
hardware will differ on both -- comparing them would report failures that are
not failures. Iteration and column counts are likewise excluded: they are
deterministic for a given backend version but shift with it, since they depend
on the barrier's interior point.

For the same reason the gate covers only cells that PROVED optimality on both
sides. A run stopped by the time limit reports whatever bound it had reached
when the clock ran out, which is a measure of host speed as much as of the
formulation; those cells are compared and printed but never fail the gate. So
is the case where this host solved a cell the reference host could not -- doing
better is not a failure to reproduce.

Cells the sweep did not run are reported as skipped, never as passes: a
narrowed sweep must not read as having reproduced the whole matrix.

    python3 scripts/check_reproduction.py bench_runs/cg/results.csv
    python3 scripts/check_reproduction.py fresh.csv --tol 1e-5
    python3 scripts/check_reproduction.py fresh.csv --reference results/other.csv

Exit status is 0 when every gated cell matches and every fresh cell was found in
the reference, 1 otherwise, so this can gate a CI job or a release checklist. A
cell present in the fresh run but absent from the reference fails too: it was
never compared, and the two tables disagreeing about what the matrix *is* is a
real discrepancy even when no objective differs.
"""

import argparse
import csv
import math
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REFERENCE = os.path.join(REPO, "results", "cg_benchmark.csv")

# Join key. A cell in either table is uniquely identified by these four.
KEY = ("family", "instance", "formulation", "solver")


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    table = {}
    for row in rows:
        missing = [k for k in KEY if k not in row]
        if missing:
            sys.exit(f"{path}: not a benchmark CSV -- missing column(s) "
                     f"{', '.join(missing)}")
        table[tuple(row[k] for k in KEY)] = row
    if not table:
        sys.exit(f"{path}: no rows")
    return table


def objective(row):
    """The row's objective as a float, or None when the run produced none."""
    raw = (row.get("objective") or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def certified(row):
    """True when the run proved optimality.

    A run stopped by the time limit reports the best bound it had reached when
    the clock ran out, so its objective is a function of how fast the host is --
    the very property this script excludes time and iteration counts for. Twenty
    committed cells are not certified: 14 hit the time limit, 5 aborted after an
    LP solve returned a non-optimal status, and one was killed. For 17 of the 19
    that produced a number, that number is the Lagrangian LOWER bound -- CG
    reports best_lb when no slack-free incumbent was ever recorded -- so it is
    not an objective at all. Two of the timed-out ones vary by more than 1%
    between backends on the reference machine alone.
    """
    return (row.get("optimal") or "").strip() == "1"


def compare(fresh_row, ref_row, tol):
    """Compare one cell. Returns (verdict, relative_error, note).

    verdict is one of:
      "ok"       -- reproduced; counts toward the gate passing
      "advisory" -- differs, but for a reason that is not a reproduction
                    failure; reported, never gating
      "diff"     -- failed to reproduce

    `relative_error` is None whenever the cells agree by something other than a
    numeric comparison, which the caller reports separately -- an agreement on
    "this run produced nothing" is a much weaker statement than an agreement on
    a value, and the summary should not blur the two.
    """
    got, want = objective(fresh_row), objective(ref_row)
    if got is None and want is None:
        # Reproducing a reference failure IS a reproduction. The committed
        # matrix contains a cell that was SIGKILLed with no objective; scoring
        # it as a mismatch would make a byte-perfect rerun of the full matrix
        # fail the gate.
        return "ok", None, "no objective on either side"
    if want is None:
        # This host produced something the reference host did not -- the one such
        # cell was SIGKILLed at a 95.8 GB peak (OOM the strong reading, but a
        # signal names itself, never its cause). Not a failure to reproduce, but
        # a divergence worth showing.
        return "advisory", None, "reference has no objective; this run produced one"
    if got is None:
        return "diff", None, "no objective in the fresh run"
    # NaN never equals itself, and inf - inf is NaN, so every non-finite case
    # has to be settled before the relative-error formula runs. NaN needs the
    # explicit test precisely because `==` cannot express agreement on it.
    if (math.isnan(got) and math.isnan(want)) or (got == want and not math.isfinite(want)):
        # Both runs failed the same way: the same kind of agreement as both
        # producing no objective at all.
        return "ok", None, f"both {got}"
    if not math.isfinite(want):
        # A non-finite reference objective is not a reproducible target. The one
        # such cell is -inf: cuOpt's barrier failed, CG's first LP solve came back
        # non-optimal and the loop broke after 0 iterations, so that -inf is the
        # "no objective established" sentinel rather than a computed value. A
        # correct rerun is EXPECTED to disagree. Same reasoning as a missing
        # reference objective above: not a reproduction failure.
        return "advisory", None, "reference objective is non-finite; not a reproducible target"
    if not math.isfinite(got):
        # Reference has a real value and this run produced inf/NaN. That is a
        # failure, and the only remaining non-finite case.
        return "diff", None, "non-finite objective in the fresh run"
    rel = abs(got - want) / max(1.0, abs(want))
    if certified(ref_row) and not certified(fresh_row):
        # The reference proved optimality here and this run did not. Expected on
        # slower hardware -- the time limit is wall-clock -- so it cannot gate.
        # But it is also what a backend regression looks like, so name it rather
        # than filing it under the same word as "this host did better".
        return "advisory", rel, f"rel={rel:.3e}, LOST CERTIFICATION (reference certified this cell)"
    if not certified(fresh_row) or not certified(ref_row):
        return "advisory", rel, f"rel={rel:.3e}, not certified optimal"
    return ("ok" if rel < tol else "diff"), rel, f"rel={rel:.3e}"


def describe(key):
    family, instance, formulation, solver = key
    return f"{family}/{instance}/{formulation}/{solver}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fresh", help="CSV written by a benchmark_solvers.py run")
    ap.add_argument("--reference", default=DEFAULT_REFERENCE,
                    help="committed results CSV to compare against "
                         "(default: results/cg_benchmark.csv)")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="relative objective tolerance (default 1e-3, matching "
                         "benchmark_solvers.py's pass criterion). Deliberately "
                         "NOT 1e-4: that is RELATIVE_FEAS_TOL, the gap at which "
                         "CG stops, so two faithful runs may legitimately differ "
                         "by nearly that much -- reruns of the same grid cells on "
                         "the reference host itself land 5e-5 apart. Gating at the "
                         "stopping tolerance would fail correct builds.")
    ap.add_argument("--quiet", action="store_true",
                    help="print only mismatches and the verdict")
    args = ap.parse_args()

    fresh = load(args.fresh)
    reference = load(args.reference)

    matched, non_numeric, advisory, mismatched, unmatched = [], [], [], [], []
    for key, row in sorted(fresh.items()):
        ref_row = reference.get(key)
        if ref_row is None:
            unmatched.append(key)
            continue
        verdict, rel, why = compare(row, ref_row, args.tol)
        got, want = objective(row), objective(ref_row)
        if verdict == "diff":
            mismatched.append((key, got, want, why))
        elif verdict == "advisory":
            advisory.append((key, got, want, why))
        elif rel is None:
            non_numeric.append((key, why))
        else:
            matched.append((key, rel))

    skipped = sorted(set(reference) - set(fresh))

    if not args.quiet:
        for key, rel in matched:
            print(f"  ok    {describe(key)}  rel={rel:.2e}")
        for key, why in non_numeric:
            print(f"  ok*   {describe(key)}  {why}")
    for key, got, want, why in advisory:
        got_s = "-" if got is None else f"{got:.6g}"
        want_s = "-" if want is None else f"{want:.6g}"
        print(f"  note  {describe(key)}  got={got_s} want={want_s}  ({why})")
    for key, got, want, why in mismatched:
        got_s = "-" if got is None else f"{got:.6g}"
        want_s = "-" if want is None else f"{want:.6g}"
        print(f"  DIFF  {describe(key)}  got={got_s} want={want_s}  ({why})")
    for key in unmatched:
        print(f"  ?     {describe(key)}  not present in {os.path.basename(args.reference)}")

    print()
    print(f"reference : {args.reference} ({len(reference)} cells)")
    print(f"fresh     : {args.fresh} ({len(fresh)} cells)")
    print(f"tolerance : {args.tol:g} relative on objective")
    print(f"matched   : {len(matched)}")
    if non_numeric:
        # Agreeing that a cell produced no usable objective is a reproduction,
        # but a much weaker one than agreeing on a value. Counted apart so the
        # summary cannot overstate what was checked.
        print(f"agreed    : {len(non_numeric)} by absence or non-finite value, not by objective")
    if advisory:
        print(f"advisory  : {len(advisory)} not gated (time-limited, or solved here "
              f"but not in the reference)")
        lost = [a for a in advisory if "LOST CERTIFICATION" in a[3]]
        if lost:
            # Called out separately: expected on slower hardware, but also the
            # shape of a backend regression, and it must not hide inside the
            # advisory total.
            print(f"  of which: {len(lost)} lost certification the reference had")
    print(f"mismatched: {len(mismatched)}")
    if unmatched:
        print(f"unknown   : {len(unmatched)} (in the fresh run, absent from the reference)")
    if skipped:
        print(f"not run   : {len(skipped)} reference cells this sweep did not cover")

    ok = not mismatched and not unmatched
    print()
    print("REPRODUCED" if ok else "NOT REPRODUCED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
