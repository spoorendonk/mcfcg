#!/usr/bin/env python3
"""Benchmark mcfcg LP backends over the instance suite.

Drives the mcfcg CLI over the instances in data/, one (instance, solver-config)
run at a time with a per-instance CLI-side time budget (--time-limit), and checks
the reported objective against the paper reference in each family's optimal.csv.
Pure CLI driver. Every run's full CG log is saved under --logdir.

Backends are selected by config label (see SOLVER_CONFIGS), forming a
{CPU,GPU} x {OSS,commercial} matrix plus a same-solver GPU-off control:

                 CPU            GPU
    OSS          highs          cuopt
    commercial   mosek          copt-gpu     (+ copt-cpu = COPT GPUMODE 0 control)

All backends run the same pinned barrier regime (presolve off, crossover off,
tol 1e-4); each run's [lp-config] provenance banner is captured into the CSV
`config` column. A run that reports optimal=1 but whose objective is off the
reference is flagged WRONG-OPTIMAL (a correctness bug, counted separately from
honest time-limited fails), so a fast-but-wrong result is never credited.

The CLI prints a 2-line CSV to stdout (header + data row); the preamble goes to
stderr. See src/main.cpp. Columns:
  instance,formulation,iterations,columns,objective,lower_bound,optimal,
  time,time_lp,time_pricing,time_separation

Outcomes per run: ok (parsed) or error (nonzero exit / crash / unparseable
output). The subprocess is never killed — the CLI's --time-limit is the only
stopping mechanism. It is enforced at CG iteration boundaries, so a run
self-terminates and reports its best UB/LB once it reaches one; a backend stuck
inside a single LP solve (e.g. a barrier pathology) is not interrupted.

Examples:
  # smoke one instance across the full backend matrix
  python3 scripts/benchmark_solvers.py --families grid --instances grid1 \
      --time-limit 120

  # COPT GPU-on vs GPU-off control on planar
  python3 scripts/benchmark_solvers.py --families planar --solvers copt-cpu,copt-gpu

  # both formulations on every family (override per-family default)
  python3 scripts/benchmark_solvers.py --formulations path,tree
"""

import argparse
import csv
import fnmatch
import glob
import os
import re
import subprocess
import sys
import tempfile

# GNU time (not the bash builtin) reports a process's peak RSS via `-f %M` in
# kilobytes — the same whole-process high-water mark the paper's mem_gb column
# uses. We wrap each run in it when present so the comparison carries memory.
GNU_TIME = "/usr/bin/time"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PLANAR_SIZES = [30, 50, 80, 100, 150, 300, 500, 800, 1000, 2500]

# Named backend configs. The label is the comparison unit (the CSV `solver`
# column); the value is the extra CLI args that select that backend/mode. COPT
# is split into a CPU and a GPU config so GPUMODE 0 vs 2 is a clean same-solver
# control for the GPU-speedup question. Every backend runs the same pinned
# barrier regime (presolve off, crossover off, tol 1e-4) — see the C++ side;
# the [lp-config] banner each run prints is captured into the `config` column.
#
#                CPU            GPU
#   OSS          highs          cuopt
#   commercial   mosek          copt-gpu     (+ copt-cpu = GPU-off control)
SOLVER_CONFIGS = {
    "highs": ["--solver", "highs"],
    "mosek": ["--solver", "mosek"],
    "cuopt": ["--solver", "cuopt"],
    "copt": ["--solver", "copt"],  # COPT default (GPU mode 2)
    "copt-cpu": ["--solver", "copt", "--copt-gpu-mode", "0"],
    "copt-gpu": ["--solver", "copt", "--copt-gpu-mode", "2"],
}


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
                yield p, f"grid{i}", "tree", []
        return
    if family == "planar":
        for n in PLANAR_SIZES:
            p = os.path.join(d, "commalab/planar", f"planar{n}")
            if os.path.exists(p):
                yield p, f"planar{n}", "tree", []
        return
    if family == "transportation":
        for net in sorted(glob.glob(os.path.join(d, "transportation", "*_net.tntp.gz"))):
            key = os.path.basename(net)[: -len("_net.tntp.gz")]
            yield net, key, "tree", []
        return
    if family == "intermodal":
        # Only BUS and SBT are benchmark instances. SUBWAY instances exist for
        # unit tests only (no paper reference in optimal.csv) and are excluded.
        insts = sorted(glob.glob(os.path.join(d, "intermodal", "BUS-*.txt.gz")) +
                       glob.glob(os.path.join(d, "intermodal", "SBT-*.txt.gz")))
        for inst in insts:
            key = os.path.basename(inst)[: -len(".txt.gz")]
            # Tree is the default everywhere; intermodal additionally needs PricerHeavy.
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


def write_log(log_path, cmd, stdout, stderr, returncode, outcome):
    """Persist a run's full console output for later forensic reconstruction.

    The CLI runs at Verbosity::Iteration (src/main.cpp), so stderr carries the
    per-iteration CG log (It/UB/LB/LP_obj/#col/#row/#slk/+col/-col/+cut/-cut,
    timings, t_acc) plus the summary line — enough to reconstruct cut growth,
    slack dynamics, and bound history. stdout carries the final CSV row. We dump
    both verbatim regardless of outcome (success or error) so a partial run is
    never silently lost.
    """
    with open(log_path, "w") as f:
        f.write("# cmd: " + " ".join(cmd) + "\n")
        f.write(f"# outcome: {outcome}\n")
        if returncode is not None:
            f.write(f"# returncode: {returncode}\n")
        f.write("# === STDERR (CG iteration log + preamble) ===\n")
        f.write(stderr or "")
        f.write("\n# === STDOUT (result CSV) ===\n")
        f.write(stdout or "")


def read_peak_mem_gb(mem_path):
    """Parse GNU time's `-f %M` output (peak RSS in KB) into GB, or '' on miss."""
    if not mem_path or not os.path.exists(mem_path):
        return ""
    try:
        with open(mem_path) as f:
            kb = int(f.read().strip().split()[-1])
        return round(kb / 1024.0 / 1024.0, 3)
    except (ValueError, IndexError):
        return ""
    finally:
        try:
            os.remove(mem_path)
        except OSError:
            pass


def parse_lp_config(stderr):
    """Return the backend's [lp-config] provenance line (sans prefix), or ''."""
    for line in stderr.splitlines():
        line = line.strip()
        if line.startswith("[lp-config]"):
            return line[len("[lp-config]"):].strip()
    return ""


def run_one(binary, instance, solver, formulation, extra, max_iters, log_path=None,
            time_limit=None):
    # `solver` is a config label (see SOLVER_CONFIGS): maps to the --solver flag
    # plus any mode args (e.g. copt-cpu -> --solver copt --copt-gpu-mode 0).
    solver_args = SOLVER_CONFIGS.get(solver, ["--solver", solver])
    cmd = [binary, instance] + solver_args + ["--formulation", formulation,
           "--max-iters", str(max_iters)] + extra
    # Rely entirely on the CLI's own --time-limit: it stops the CG loop at an
    # iteration boundary and still prints its result, so the run self-terminates
    # cleanly. We never kill the subprocess — no wall-clock safety net.
    if time_limit:
        cmd = cmd + ["--time-limit", str(time_limit)]
    # Wrap in GNU time to capture whole-process peak RSS (paper's mem_gb metric).
    # `-o` keeps the measurement out of the child's stderr so the saved log stays
    # the clean CG iteration log. Falls back to the bare command if unavailable.
    mem_path = None
    run_cmd = cmd
    if os.path.exists(GNU_TIME):
        fd, mem_path = tempfile.mkstemp(prefix="mcfcg_mem_", suffix=".txt")
        os.close(fd)
        run_cmd = [GNU_TIME, "-o", mem_path, "-f", "%M"] + cmd
    # Stream stdout/stderr straight to temp files rather than buffering in memory,
    # so a partial run's output survives — the saved log is assembled from these
    # files regardless of outcome.
    out_fd, out_path = tempfile.mkstemp(prefix="mcfcg_out_", suffix=".txt")
    err_fd, err_path = tempfile.mkstemp(prefix="mcfcg_err_", suffix=".txt")
    os.close(out_fd)
    os.close(err_fd)
    # The binary inherits the caller's environment. A cuOpt build embeds an
    # RPATH to libcuopt, so no LD_LIBRARY_PATH is normally needed; set it in your
    # shell only if the cuOpt fork build was moved after configuring.
    returncode = None
    try:
        with open(out_path, "w") as out_f, open(err_path, "w") as err_f:
            proc = subprocess.Popen(run_cmd, stdout=out_f, stderr=err_f, text=True)
            # Never kill the subprocess: wait unconditionally. The CLI's
            # --time-limit is the only stopping mechanism.
            returncode = proc.wait()
        with open(out_path) as f:
            stdout = f.read()
        with open(err_path) as f:
            stderr = f.read()
    finally:
        for p in (out_path, err_path):
            try:
                os.remove(p)
            except OSError:
                pass
    mem_gb = read_peak_mem_gb(mem_path)
    config = parse_lp_config(stderr)
    outcome = "ok" if returncode == 0 else "error"
    if log_path:
        write_log(log_path, cmd, stdout, stderr, returncode, outcome)
    if returncode != 0:
        tail = "\n".join(stderr.strip().splitlines()[-3:])
        return {"outcome": "error", "returncode": returncode, "stderr_tail": tail,
                "mem_gb": mem_gb, "config": config}
    row = parse_csv_row(stdout)
    if row is None:
        tail = "\n".join(stderr.strip().splitlines()[-3:])
        return {"outcome": "error", "returncode": 0, "stderr_tail": "no CSV row; " + tail,
                "mem_gb": mem_gb, "config": config}
    return {
        "outcome": "ok",
        "objective": float(row["objective"]),
        "lower_bound": row.get("lower_bound", "") or "",
        "optimal": int(row["optimal"]),
        "iterations": int(row["iterations"]),
        "columns": int(row["columns"]),
        "time": float(row["time"]),
        "mem_gb": mem_gb,
        "config": config,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--binary", default=os.path.join(REPO, "build/mcfcg_cli"))
    ap.add_argument("--solvers", default="highs,mosek,cuopt,copt-cpu,copt-gpu",
                    help="comma-separated backend config labels (see SOLVER_CONFIGS): "
                         "highs, mosek, cuopt, copt (GPU default), copt-cpu (GPUMODE 0), "
                         "copt-gpu (GPUMODE 2). A label whose backend was not compiled in "
                         "reports as an error row (informative, not silent).")
    ap.add_argument("--families", default="grid,planar,transportation,intermodal")
    ap.add_argument("--formulations", default=None,
                    help="comma-separated formulations to run for every instance "
                         "(e.g. 'path,tree'). Overrides each family's default. "
                         "Omit to use the per-family default (tree for every family; "
                         "intermodal additionally uses --strategy pricer-heavy).")
    ap.add_argument("--instances", default=None,
                    help="fnmatch glob on the ref key to filter (e.g. 'grid1', 'BUS-*').")
    ap.add_argument("--max-planar", type=int, default=None,
                    help="skip planar instances larger than this vertex count.")
    ap.add_argument("--time-limit", type=float, default=7200.0,
                    help="CLI-side CG wall-clock budget in seconds, passed as --time-limit "
                         "(default 2h). The solver stops at the next iteration and still reports "
                         "its best UB/LB (result marked non-optimal). The subprocess is never "
                         "killed; this is the only stopping mechanism, enforced at iteration "
                         "boundaries (a barrier stuck mid-solve is not interrupted).")
    ap.add_argument("--max-iters", type=int, default=10000)
    ap.add_argument("--tol", type=float, default=1e-3, help="relative objective tolerance for pass.")
    ap.add_argument("--out", default="bench-results.csv")
    ap.add_argument("--logdir", default="bench-logs",
                    help="directory to save each run's full stdout+stderr (the per-iteration "
                         "CG log: cut growth, slack/bound history, timings). One file per run, "
                         "named <family>__<instance>__<formulation>__<solver>.log. "
                         "Always written (default 'bench-logs').")
    args = ap.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    if not os.path.exists(args.binary):
        sys.exit(f"binary not found: {args.binary} (build it first)")

    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    override_forms = ([f.strip() for f in args.formulations.split(",") if f.strip()]
                      if args.formulations else None)

    fields = ["family", "instance", "solver", "formulation", "outcome", "objective",
              "ref", "rel_err", "pass", "optimal", "iterations", "columns", "time", "mem_gb",
              "config", "detail"]
    rows = []
    summary = {s: {"pass": 0, "fail": 0, "wrong": 0, "error": 0, "noref": 0} for s in solvers}

    for family in families:
        refs = load_optimal(os.path.join(REPO, "data", FAMILY_OPTIMAL[family]))
        for instance, key, default_form, extra in enumerate_family(family):
            if args.instances and not fnmatch.fnmatch(key, args.instances):
                continue
            if family == "planar" and args.max_planar is not None:
                n = int(re.sub(r"\D", "", key))
                if n > args.max_planar:
                    continue
            forms = override_forms if override_forms is not None else [default_form]
            for formulation in forms:
              for solver in solvers:
                sys.stderr.write(f"[{family}] {key} :: {formulation}/{solver} ... ")
                sys.stderr.flush()
                log_path = os.path.join(args.logdir,
                                        f"{family}__{key}__{formulation}__{solver}.log")
                r = run_one(args.binary, instance, solver, formulation, extra,
                            args.max_iters, log_path=log_path,
                            time_limit=args.time_limit)
                ref = refs.get(key)
                rec = {"family": family, "instance": key, "solver": solver,
                       "formulation": formulation, "outcome": r["outcome"],
                       "objective": "", "ref": "" if ref is None else ref,
                       "rel_err": "", "pass": "", "optimal": "", "iterations": "",
                       "columns": "", "time": "", "mem_gb": r.get("mem_gb", ""),
                       "config": r.get("config", ""), "detail": ""}
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
                        if ok:
                            summary[solver]["pass"] += 1
                            verdict = "PASS"
                        elif r["optimal"] == 1:
                            # Claimed optimality but the objective is wrong: a
                            # correctness bug (e.g. a swallowed barrier failure),
                            # not a time-limited near-miss. Flag it loudly and
                            # count it separately from honest non-optimal fails.
                            summary[solver]["wrong"] += 1
                            rec["detail"] = f"WRONG-OPTIMAL: claimed optimal, rel={rel:.2e}"
                            verdict = f"WRONG-OPTIMAL rel={rel:.2e}"
                        else:
                            summary[solver]["fail"] += 1
                            verdict = f"FAIL rel={rel:.2e}"
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
    print(f"{'solver':<10} {'pass':>5} {'fail':>5} {'wrong':>6} {'error':>6} {'noref':>6}")
    for s in solvers:
        c = summary[s]
        print(f"{s:<10} {c['pass']:>5} {c['fail']:>5} {c['wrong']:>6} {c['error']:>6} "
              f"{c['noref']:>6}")
    total_wrong = sum(c["wrong"] for c in summary.values())
    if total_wrong:
        print(f"\n*** {total_wrong} WRONG-OPTIMAL run(s): a backend reported optimal with a "
              f"wrong objective. Treat as a correctness bug, not a benchmark miss. ***")
    nonpass = [r for r in rows if r["pass"] is not True]
    if nonpass:
        print("\nNon-pass runs:")
        for r in nonpass:
            d = r["detail"] or (f"rel={r['rel_err']:.2e}" if r["rel_err"] != "" else "")
            print(f"  {r['solver']:<6} {r['formulation']:<5} {r['family']:<14} "
                  f"{r['instance']:<16} {r['outcome']:<8} {d}")


if __name__ == "__main__":
    main()
