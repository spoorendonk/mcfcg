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

Cells the sweep did not run are reported as skipped, never as passes: a
narrowed sweep must not read as having reproduced the whole matrix.

    python3 scripts/check_reproduction.py bench_runs/cg/results.csv
    python3 scripts/check_reproduction.py fresh.csv --tol 1e-5
    python3 scripts/check_reproduction.py fresh.csv --reference results/other.csv

Exit status is 0 when every compared cell matches, 1 otherwise, so this can gate
a CI job or a release checklist.
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


def compare(got, want, tol):
    """Compare one cell's objectives. Returns (ok, relative_error, note).

    `relative_error` is None whenever the cells agree by something other than a
    numeric comparison, which the caller reports separately -- an agreement on
    "this run produced nothing" is a much weaker statement than an agreement on
    a value, and the summary should not blur the two.
    """
    if got is None and want is None:
        # Reproducing a reference failure IS a reproduction. The committed
        # matrix contains a cell that was SIGKILLed with no objective; scoring
        # it as a mismatch would make a byte-perfect rerun of the full matrix
        # fail the gate.
        return True, None, "no objective on either side"
    if got is None or want is None:
        return False, None, ("no objective in the fresh run" if got is None
                             else "reference cell has no objective")
    # NaN never equals itself, and inf - inf is NaN, so both cases have to be
    # settled before the relative-error formula runs. The reference does contain
    # an infinite objective (a swallowed barrier failure recorded as -inf).
    if math.isnan(got) or math.isnan(want):
        return got is want, None, "NaN objective"
    if math.isinf(got) or math.isinf(want):
        return got == want, None, ("both " + repr(got) if got == want
                                   else "one side is infinite")
    rel = abs(got - want) / max(1.0, abs(want))
    return rel < tol, rel, f"rel={rel:.3e}"


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
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="relative objective tolerance (default 1e-4, the "
                         "barrier convergence tolerance every backend is pinned to)")
    ap.add_argument("--quiet", action="store_true",
                    help="print only mismatches and the verdict")
    args = ap.parse_args()

    fresh = load(args.fresh)
    reference = load(args.reference)

    matched, non_numeric, mismatched, unmatched = [], [], [], []
    for key, row in sorted(fresh.items()):
        ref_row = reference.get(key)
        if ref_row is None:
            unmatched.append(key)
            continue
        got, want = objective(row), objective(ref_row)
        ok, rel, why = compare(got, want, args.tol)
        if not ok:
            mismatched.append((key, got, want, why))
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
