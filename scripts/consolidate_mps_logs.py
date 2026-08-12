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

Memory (`mem_gb`, `vram_gb`) comes from the log's `# peak_rss_kb:` /
`# peak_vram_mib:` headers, so it survives re-consolidation like everything else.
Logs from before those headers existed simply leave the columns empty -- unlike
the objective, memory CANNOT be re-derived from the body, so a blank there means
that cell was never measured and needs a rerun (see --probe for the cheap one).

`mem_source` says which regime that memory came from, and the distinction is not
cosmetic:
    measured    the full solve's own peak, from GNU `time` on this very run.
    probeN:...  INJECTED by inject_probe_memory.py from the iteration-capped
                probe sweep (N barrier iterations). It is the model-setup /
                initial peak -- read, presolve, symbolic+numeric factorization,
                N iterations -- and therefore a LOWER BOUND on this row's
                full-solve peak, measured in a different execution with a
                different wall time and outcome. Never quote it as the peak of
                the solve whose time_wall sits next to it.
    probeN-partial:...  the same, from a probe that died (cgroup OOM at the cap,
                or a backend error) before finishing its N iterations: a lower
                bound on a lower bound.

--probe consolidates the memory-probe sweep (benchmark_mps.py --probe-iters)
instead: iteration-capped runs, written to their own tree, whose peak RSS stands
in for the full solve's. It emits a memory-only CSV with the cap recorded per
row. The two modes never mix -- a probe log encountered in normal mode is
skipped, because its objective is a mid-barrier iterate that would score as a
wildly failing solve.

Usage:
  python3 scripts/consolidate_mps_logs.py   # bench_runs/mps/logs -> results/mps_compact_baseline.csv
  python3 scripts/consolidate_mps_logs.py --probe
      # bench_runs/mps_probe/logs -> results/mps_compact_memory.csv
"""

import argparse
import csv
import os
import sys

import benchmark_mps as bm  # sibling module; run from scripts/ or repo root
import benchmark_solvers as bs  # peak-RSS header parser, shared with the CG logs

MARKER = "# === solver output ==="
KNOWN_SOLVERS = list(bm.CONFIGS)  # highs, mosek, cuopt, copt-cpu, copt-gpu


# Return codes that mean "the process was killed", so the absence of any marker
# in its output tells us nothing about how it was configured. 137 = 128+SIGKILL
# (the cgroup OOM killer, or the harness), 9 = the bare signal number GNU `time`
# reports for the same death, -1 = the harness's own timeout verdict.
KILL_RCS = (137, 9, -1)


def killed_before_solving(rc):
    return rc is None or rc in KILL_RCS


def parse_log(path):
    with open(path, errors="replace") as f:
        text = f.read()
    wall, rc, outcome = bm.parse_run_header(text)
    warn = "# WARN " in text
    body = text.split(MARKER, 1)[1] if MARKER in text else text
    rss_kb, rss_source = bs.parse_peak_rss(text)
    return {
        "wall": wall, "rc": rc, "outcome": outcome, "warn": warn, "body": body,
        "mem_gb": bs.mem_gb_from_kb(rss_kb),
        # Provenance of BOTH memory columns: the peak-RSS and peak-VRAM headers are
        # written, and rewritten, as one block (see inject_probe_memory.py).
        "mem_source": rss_source,
        "vram_gb": bm.gb_from_mib(bm.parse_peak_vram_mib(text)),
        "probe_iters": bm.parse_probe_iters(text),
    }


def split_key_solver(stem):
    # "<instance>__<solver>"; solver may contain a hyphen (copt-cpu) but no "__".
    inst, _, solver = stem.partition("__")
    return inst, solver


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", default=None,
                    help="default bench_runs/mps/logs, or bench_runs/mps_probe/logs "
                         "under --probe.")
    ap.add_argument("--out", default=None,
                    help="the committed 'one truth' results CSV (tracked in "
                         "results/); rebuilt from the gitignored per-cell logs. "
                         "Default results/mps_compact_baseline.csv, or "
                         "results/mps_compact_memory.csv under --probe.")
    ap.add_argument("--probe", action="store_true",
                    help="consolidate memory-probe logs (benchmark_mps.py "
                         "--probe-iters) into a memory-only CSV instead.")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    args.logdir = args.logdir or (
        "bench_runs/mps_probe/logs" if args.probe else "bench_runs/mps/logs")
    args.out = args.out or (
        "results/mps_compact_memory.csv" if args.probe
        else "results/mps_compact_baseline.csv")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    refs = bm.load_refs()
    if args.probe:
        fields = ["instance", "solver", "gpu", "outcome", "probe_iters",
                  "mem_gb", "vram_gb", "time_wall", "status", "detail"]
    else:
        # `mem_source` covers mem_gb AND vram_gb (one header block, one
        # provenance). It exists only in this mode: probe logs are live readings
        # by construction -- the probe sweep postdates the headers and nothing
        # injects into it -- so the column would be the constant "measured" there,
        # and mps_compact_memory.csv is pinned in PROVENANCE section 2.1.
        # `probe_iters` is deliberately NOT carried across: on a full-solve row it
        # would read as "this solve was capped", which is the opposite of true.
        # The cap lives inside the mem_source tag (`probe3:`) instead.
        fields = ["instance", "solver", "gpu", "outcome", "objective", "ref",
                  "rel_err", "pass", "time_wall", "time_solve", "mem_gb",
                  "vram_gb", "mem_source", "status", "detail"]
    rows = []
    wrong_mode = 0
    logs = sorted(f for f in os.listdir(args.logdir) if f.endswith(".log"))
    for fn in logs:
        stem = fn[:-4]
        inst, solver = split_key_solver(stem)
        if solver not in KNOWN_SOLVERS:
            sys.stderr.write(f"skip {fn}: unknown solver '{solver}'\n")
            continue
        p = parse_log(os.path.join(args.logdir, fn))
        wall, rc, outcome, warn, body = (p["wall"], p["rc"], p["outcome"],
                                         p["warn"], p["body"])
        # Keep the two populations apart no matter which directory they were found
        # in: a probe row has no solution to score, a solve row has no cap to
        # report, and silently mixing them is how a mid-barrier iterate ends up in
        # a results table.
        if bool(p["probe_iters"]) != args.probe:
            wrong_mode += 1
            continue
        if args.probe:
            if p["mem_gb"] == "":
                detail = f"unmeasured (rc={rc})"
            elif outcome != "ok":
                # e.g. OOM-killed at the cgroup cap: the number is where the run
                # DIED, a lower bound on what the solve needs -- not its footprint.
                detail = f"peak at failure, not at completion (rc={rc})"
            else:
                detail = ""
            rows.append({
                "instance": inst, "solver": solver, "gpu": bm.CONFIGS[solver][1],
                "outcome": outcome, "probe_iters": p["probe_iters"],
                "mem_gb": p["mem_gb"], "vram_gb": p["vram_gb"],
                "time_wall": "" if wall is None else f"{wall:.3f}",
                "status": bm.parse_output(solver, body)[2], "detail": detail,
            })
            continue
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
            guard_detail = ("COPT never read the model (readmps failed); peak RSS "
                            "is the reader's, not a solve's"
                            if bm.copt_read_failed(body) else
                            "COPT failed to solve (GPU memory issue / infeasible)")
        elif solver == "highs" and not killed_before_solving(rc) and (
                "features are unavailable" in body or "Running HiPO" not in body):
            # Re-derive the HiPO-fallback guard from the body (mirrors
            # benchmark_mps.run_one) so a log from an older/unguarded run where HiPO
            # SILENTLY fell back to dual simplex -- no `# WARN` line, header
            # outcome=ok -- is never consolidated as a valid HiPO baseline result.
            #
            # Gated on HOW the run died, not on whether it succeeded. The marker
            # is equally absent from a run killed before HiPO could start, so
            # ungated an OOM-killed cell (Sydney/BUS x highs: model read, then
            # SIGKILL at the memory cap) gets relabelled "HiPO did not run" --
            # blaming a config fallback for what is actually a memory failure.
            # But gating on `outcome == "ok"` would be too strong in the other
            # direction: HiGHS can silently fall back to dual simplex AND THEN
            # blow its own time limit (rc=1, outcome=error), and that log must
            # still have its objective discarded rather than consolidated as a
            # HiPO result. Only a signal death is genuinely uninformative.
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
               "time_solve": tsolve, "mem_gb": p["mem_gb"],
               "vram_gb": p["vram_gb"], "mem_source": p["mem_source"],
               "status": status, "detail": guard_detail or ""}
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
    print(f"Consolidated {len(rows)} cells across {len(insts)} instances -> {args.out}")
    if args.probe:
        nmem = sum(1 for r in rows if r["mem_gb"] != "")
        print(f"  measured={nmem}  unmeasured={len(rows) - nmem}")
    else:
        npass = sum(1 for r in rows if r["pass"] is True)
        nerr = sum(1 for r in rows
                   if r["outcome"] != "ok" or r["objective"] is None)
        nmem = sum(1 for r in rows if r["mem_gb"] != "")
        nfull = sum(1 for r in rows if r["mem_source"] == "measured")
        print(f"  pass={npass}  error/timeout={nerr}  "
              f"other={len(rows) - npass - nerr}")
        # Split the memory count by regime: lumping the two together would report
        # an injected lower bound as if the full solve had been measured.
        print(f"  peak RSS on {nmem}/{len(rows)} cells: {nfull} from the full solve, "
              f"{nmem - nfull} injected from the probe sweep (lower bounds)")
        if nmem < len(rows):
            print(f"  {len(rows) - nmem} cell(s) unmeasured in both sweeps "
                  f"(`--probe-iters` re-measures cheaply)")
    if wrong_mode:
        kind = "solve" if args.probe else "probe"
        print(f"  skipped {wrong_mode} {kind} log(s) found in this logdir "
              f"(wrong population for --probe={args.probe})")
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
