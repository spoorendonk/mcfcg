#!/usr/bin/env python3
"""Benchmark mcfcg LP backends (COPT, cuOpt) over the instance suite.

Drives the mcfcg CLI (which already exposes --solver / --formulation) over the
instances in data/, one (instance, solver) run at a time with a per-instance
timeout, and checks the reported objective against the paper reference in each
family's optimal.csv. No library or test changes — pure CLI driver.

The CLI prints a 2-line CSV to stdout (header + data row); the preamble goes to
stderr. See src/main.cpp. Columns:
  instance,formulation,iterations,columns,objective,lower_bound,optimal,
  time,time_lp,time_pricing,time_separation

Outcomes per run: ok (parsed), timeout (exceeded --timeout), error (nonzero
exit / crash / unparseable output).

Examples:
  # smoke one instance, both backends
  python3 scripts/benchmark_solvers.py --families grid --instances grid1 \
      --solvers copt,cuopt --timeout 120

  # full cuOpt pass, 2h/instance
  python3 scripts/benchmark_solvers.py --solvers cuopt --out bench-cuopt.csv
"""

import argparse
import csv
import fnmatch
import glob
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PLANAR_SIZES = [30, 50, 80, 100, 150, 300, 500, 800, 1000, 2500]


def load_optimal(path):
    """Read an optimal.csv (instance,optimal) into {name: float}."""
    refs = {}
    if not os.path.exists(path):
        return refs
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 2 and row[0]:
                refs[row[0]] = float(row[1])
    return refs


def enumerate_family(family):
    """Yield (instance_path, ref_key, formulation, extra_args) for a family."""
    d = os.path.join(REPO, "data")
    if family == "grid":
        for i in range(1, 16):
            p = os.path.join(d, "commalab/grid", f"grid{i}")
            if os.path.exists(p):
                yield p, f"grid{i}", "path", []
        return
    if family == "planar":
        for n in PLANAR_SIZES:
            p = os.path.join(d, "commalab/planar", f"planar{n}")
            if os.path.exists(p):
                yield p, f"planar{n}", "path", []
        return
    if family == "transportation":
        for net in sorted(glob.glob(os.path.join(d, "transportation", "*_net.tntp.gz"))):
            key = os.path.basename(net)[: -len("_net.tntp.gz")]
            yield net, key, "path", []
        return
    if family == "intermodal":
        for inst in sorted(glob.glob(os.path.join(d, "intermodal", "*.txt.gz"))):
            key = os.path.basename(inst)[: -len(".txt.gz")]
            # CLAUDE.md: intermodal needs tree formulation + PricerHeavy.
            yield inst, key, "tree", ["--strategy", "pricer-heavy"]
        return
    raise ValueError(f"unknown family '{family}'")


FAMILY_OPTIMAL = {
    "grid": "commalab/grid/optimal.csv",
    "planar": "commalab/planar/optimal.csv",
    "transportation": "transportation/optimal.csv",
    "intermodal": "intermodal/optimal.csv",
}


def parse_csv_row(stdout):
    """Return the data dict from the CLI's 2-line CSV stdout, or None."""
    header = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("instance,formulation,"):
            header = line.split(",")
            continue
        if header and "," in line:
            vals = line.split(",")
            if len(vals) == len(header):
                return dict(zip(header, vals))
    return None


def run_one(binary, instance, solver, formulation, extra, timeout, max_iters):
    cmd = [binary, instance, "--solver", solver, "--formulation", formulation,
           "--max-iters", str(max_iters)] + extra
    # The binary inherits the caller's environment. A cuOpt build embeds an
    # RPATH to libcuopt, so no LD_LIBRARY_PATH is normally needed; set it in your
    # shell only if the cuOpt fork build was moved after configuring.
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"outcome": "timeout"}
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-3:])
        return {"outcome": "error", "returncode": proc.returncode, "stderr_tail": tail}
    row = parse_csv_row(proc.stdout)
    if row is None:
        tail = "\n".join(proc.stderr.strip().splitlines()[-3:])
        return {"outcome": "error", "returncode": 0, "stderr_tail": "no CSV row; " + tail}
    return {
        "outcome": "ok",
        "objective": float(row["objective"]),
        "lower_bound": row.get("lower_bound", "") or "",
        "optimal": int(row["optimal"]),
        "iterations": int(row["iterations"]),
        "columns": int(row["columns"]),
        "time": float(row["time"]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default=os.path.join(REPO, "build/mcfcg_cli"))
    ap.add_argument("--solvers", default="copt,cuopt")
    ap.add_argument("--families", default="grid,planar,transportation,intermodal")
    ap.add_argument("--instances", default=None,
                    help="fnmatch glob on the ref key to filter (e.g. 'grid1', 'BUS-*').")
    ap.add_argument("--max-planar", type=int, default=None,
                    help="skip planar instances larger than this vertex count.")
    ap.add_argument("--timeout", type=float, default=7200.0, help="seconds per instance (default 2h).")
    ap.add_argument("--max-iters", type=int, default=10000)
    ap.add_argument("--tol", type=float, default=1e-3, help="relative objective tolerance for pass.")
    ap.add_argument("--out", default="bench-results.csv")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        sys.exit(f"binary not found: {args.binary} (build it first)")

    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    families = [f.strip() for f in args.families.split(",") if f.strip()]

    fields = ["family", "instance", "solver", "formulation", "outcome", "objective",
              "ref", "rel_err", "pass", "optimal", "iterations", "columns", "time", "detail"]
    rows = []
    summary = {s: {"pass": 0, "fail": 0, "timeout": 0, "error": 0, "noref": 0} for s in solvers}

    for family in families:
        refs = load_optimal(os.path.join(REPO, "data", FAMILY_OPTIMAL[family]))
        for instance, key, formulation, extra in enumerate_family(family):
            if args.instances and not fnmatch.fnmatch(key, args.instances):
                continue
            if family == "planar" and args.max_planar is not None:
                n = int(re.sub(r"\D", "", key))
                if n > args.max_planar:
                    continue
            for solver in solvers:
                sys.stderr.write(f"[{family}] {key} :: {solver} ... ")
                sys.stderr.flush()
                r = run_one(args.binary, instance, solver, formulation, extra,
                            args.timeout, args.max_iters)
                ref = refs.get(key)
                rec = {"family": family, "instance": key, "solver": solver,
                       "formulation": formulation, "outcome": r["outcome"],
                       "objective": "", "ref": "" if ref is None else ref,
                       "rel_err": "", "pass": "", "optimal": "", "iterations": "",
                       "columns": "", "time": "", "detail": ""}
                if r["outcome"] == "ok":
                    rec.update(objective=r["objective"], optimal=r["optimal"],
                               iterations=r["iterations"], columns=r["columns"], time=r["time"])
                    if ref is None:
                        rec["detail"] = "no reference in optimal.csv"
                        summary[solver]["noref"] += 1
                        verdict = "NOREF"
                    else:
                        rel = abs(r["objective"] - ref) / max(1.0, abs(ref))
                        ok = rel < args.tol
                        rec["rel_err"] = rel
                        rec["pass"] = ok
                        summary[solver]["pass" if ok else "fail"] += 1
                        verdict = "PASS" if ok else f"FAIL rel={rel:.2e}"
                    sys.stderr.write(f"{verdict} obj={r['objective']:.3f} t={r['time']:.1f}s\n")
                else:
                    rec["detail"] = r.get("stderr_tail", "") or f"rc={r.get('returncode','')}"
                    summary[solver][r["outcome"]] += 1
                    sys.stderr.write(f"{r['outcome'].upper()}\n")
                rows.append(rec)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out}\n")
    print(f"{'solver':<8} {'pass':>5} {'fail':>5} {'timeout':>8} {'error':>6} {'noref':>6}")
    for s in solvers:
        c = summary[s]
        print(f"{s:<8} {c['pass']:>5} {c['fail']:>5} {c['timeout']:>8} {c['error']:>6} {c['noref']:>6}")
    nonpass = [r for r in rows if r["pass"] is not True]
    if nonpass:
        print("\nNon-pass runs:")
        for r in nonpass:
            d = r["detail"] or (f"rel={r['rel_err']:.2e}" if r["rel_err"] != "" else "")
            print(f"  {r['solver']:<6} {r['family']:<14} {r['instance']:<16} {r['outcome']:<8} {d}")


if __name__ == "__main__":
    main()
