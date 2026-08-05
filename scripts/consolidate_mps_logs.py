#!/usr/bin/env python3
"""Rebuild the benchmark_mps results CSV from the per-cell solver logs.

benchmark_mps.py writes one `<instance>__<solver>.log` per solve (with a
`# wall=..s rc=.. outcome=..` header + the raw solver output). This tool scans a
log directory and reconstructs the results CSV from those logs, reusing
benchmark_mps's own PARSERS/verdict logic. Because it reads logs, not a single
run's in-memory rows, it merges results produced across SEPARATE runs -- e.g. the
small instances solved in one pass and the large tail solved in another, or a
giant pass that was interrupted/OOM'd partway. That's exactly what we need to
avoid re-solving already-converged instances just to get one clean CSV.

Usage:
  python3 scripts/consolidate_mps_logs.py --logdir bench_runs/mps/logs \
      --out bench_runs/mps/results_batchA.csv [--tol 1e-3]
"""

import argparse
import csv
import os
import re
import sys

import benchmark_mps as bm  # sibling module; run from scripts/ or repo root

HDR = re.compile(r"# wall=([\d.]+)s rc=(-?\d+) outcome=(\w+)")
MARKER = "# === solver output ==="
KNOWN_SOLVERS = list(bm.CONFIGS)  # highs, mosek, cuopt, copt-cpu, copt-gpu


def parse_log(path):
    with open(path, errors="replace") as f:
        text = f.read()
    m = HDR.search(text)
    wall = float(m.group(1)) if m else None
    rc = int(m.group(2)) if m else None
    outcome = m.group(3) if m else "error"
    warn = "# WARN " in text
    body = text.split(MARKER, 1)[1] if MARKER in text else text
    return wall, rc, outcome, warn, body


def split_key_solver(stem):
    # "<instance>__<solver>"; solver may contain a hyphen (copt-cpu) but no "__".
    inst, _, solver = stem.partition("__")
    return inst, solver


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", default="bench_runs/mps/logs")
    ap.add_argument("--out", default="bench_runs/mps/results_all.csv")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    refs = bm.load_refs()
    fields = ["instance", "solver", "gpu", "outcome", "objective", "ref",
              "rel_err", "pass", "time_wall", "time_solve", "status", "detail"]
    rows = []
    logs = sorted(f for f in os.listdir(args.logdir) if f.endswith(".log"))
    for fn in logs:
        stem = fn[:-4]
        inst, solver = split_key_solver(stem)
        if solver not in KNOWN_SOLVERS:
            sys.stderr.write(f"skip {fn}: unknown solver '{solver}'\n")
            continue
        wall, rc, outcome, warn, body = parse_log(os.path.join(args.logdir, fn))
        obj, tsolve, status = bm.parse_output(solver, body)
        # Swallowed-failure guards: a GPU backend can exit rc=0 after a VRAM OOM /
        # "fail to solve" and (cuOpt #33) may even hand back a garbage incumbent.
        # Force an error and drop the objective so it can never be scored as an
        # optimum -- and this catches logs written before the harness carried the
        # guards, too.
        guard_detail = None
        if solver == "cuopt" and bm.cuopt_solve_failed(body):
            outcome, obj = "error", None
            guard_detail = "cuOpt #33 VRAM-OOM / numerical error"
        elif solver.startswith("copt") and bm.copt_solve_failed(body):
            outcome, obj = "error", None
            guard_detail = "COPT failed to solve (GPU memory issue / infeasible)"
        elif solver == "highs" and (
                "features are unavailable" in body or "Running HiPO" not in body):
            # Re-derive the HiPO-fallback guard from the body (mirrors
            # benchmark_mps.run_one) so a log from an older/unguarded run where HiPO
            # SILENTLY fell back to dual simplex -- no `# WARN` line, header
            # outcome=ok -- is never consolidated as a valid HiPO baseline result.
            outcome, obj = "error", None
            guard_detail = "HiPO did not run (extras unavailable / fallback)"
        elif warn:
            guard_detail = "HiPO fallback flagged"
        gpu = bm.CONFIGS[solver][1]
        ref = refs.get(inst)
        rec = {"instance": inst, "solver": solver, "gpu": gpu,
               "outcome": outcome, "objective": obj,
               "ref": "" if ref is None else ref, "rel_err": "", "pass": "",
               "time_wall": "" if wall is None else f"{wall:.3f}",
               "time_solve": tsolve, "status": status,
               "detail": guard_detail or ""}
        if outcome != "ok" or obj is None:
            rec["detail"] = (rec["detail"] + "; " if rec["detail"] else "") + \
                f"rc={rc} outcome={outcome}"
        elif ref is None:
            rec["detail"] = "no reference in optimal.csv"
        else:
            rel = abs(obj - ref) / max(1.0, abs(ref))
            rec["rel_err"] = rel
            rec["pass"] = rel < args.tol
        rows.append(rec)

    # Stable order: by family/instance then solver, for readable diffs.
    rows.sort(key=lambda r: (r["instance"], r["solver"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    insts = sorted({r["instance"] for r in rows})
    npass = sum(1 for r in rows if r["pass"] is True)
    nerr = sum(1 for r in rows if r["outcome"] != "ok" or r["objective"] is None)
    print(f"Consolidated {len(rows)} cells across {len(insts)} instances -> {args.out}")
    print(f"  pass={npass}  error/timeout={nerr}  "
          f"other={len(rows) - npass - nerr}")
    # Coverage matrix: which (instance x solver) cells are missing.
    have = {(r["instance"], r["solver"]) for r in rows}
    missing = [(i, s) for i in insts for s in KNOWN_SOLVERS if (i, s) not in have]
    if missing:
        print(f"  MISSING {len(missing)} cells:")
        for i in insts:
            miss = [s for s in KNOWN_SOLVERS if (i, s) not in have]
            if miss:
                print(f"    {i}: {', '.join(miss)}")


if __name__ == "__main__":
    main()
