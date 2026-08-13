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
import signal
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


# The saved log is a real serialization format: consolidate_cg_logs.py and
# consolidate_mps_logs.py parse it back. Keep the delimiters and the peak-RSS
# header builder here, in the module that writes them, so no reader can drift.
HEADER_END_PREFIX = "# ==="
STDERR_MARKER = "# === STDERR (CG iteration log + preamble) ==="
STDOUT_MARKER = "# === STDOUT (result CSV) ==="


def format_peak_rss_headers(peak_rss_kb, peak_rss_source):
    """Render the two peak-RSS header lines, or "" when there is nothing to record.

    Sole producer of this format — write_log and benchmark_mps.run_one both use
    it, so every log carries the pair in one spelling.
    """
    if peak_rss_kb is None:
        return ""
    return (f"# peak_rss_kb: {peak_rss_kb}\n"
            f"# peak_rss_source: {peak_rss_source}\n")


def write_log(log_path, cmd, stdout, stderr, returncode, outcome, peak_rss_kb=None,
              peak_rss_source="measured"):
    """Persist a run's full console output for later forensic reconstruction.

    The CLI runs at Verbosity::Iteration (src/main.cpp), so stderr carries the
    per-iteration CG log (It/UB/LB/LP_obj/#col/#row/#slk/+col/-col/+cut/-cut,
    timings, t_acc) plus the summary line — enough to reconstruct cut growth,
    slack dynamics, and bound history. stdout carries the final CSV row. We dump
    both verbatim regardless of outcome (success or error) so a partial run is
    never silently lost.

    Peak RSS goes in the header because it is the ONE metric not present in the
    child's own output: everything else can be re-derived by re-parsing the log,
    but memory is measured externally (GNU time) and would otherwise survive only
    in the sweep's result CSV. `peak_rss_source` records how the number got here
    ("measured" for a live run; a relocation tag otherwise — see parse_peak_rss
    for the full vocabulary) so provenance stays legible in the log.

    `cmd` is the command as ACTUALLY spawned, GNU `time` wrapper included (the
    sibling benchmark_mps.py logs its wrapped argv the same way). That matters for
    reading `# returncode:` back: a wrapped run encodes a signal death as
    128+signum, an unwrapped one as Python's negative convention, and without the
    wrapper in the header a bare number does not say which vocabulary it belongs
    to. Nothing parses this line — it is forensic only.
    """
    with open(log_path, "w") as f:
        f.write("# cmd: " + " ".join(cmd) + "\n")
        f.write(f"# outcome: {outcome}\n")
        if returncode is not None:
            f.write(f"# returncode: {returncode}\n")
        f.write(format_peak_rss_headers(peak_rss_kb, peak_rss_source))
        f.write(STDERR_MARKER + "\n")
        f.write(stderr or "")
        f.write("\n" + STDOUT_MARKER + "\n")
        f.write(stdout or "")


def read_peak_mem_kb(mem_path):
    """Parse GNU time's `-f %M` output (peak RSS in KB) into an int, or None.

    Consumes the temp file — this is the only copy of the measurement, so the
    caller must persist it (write_log's `# peak_rss_kb:` header) rather than
    keeping it in memory alone. Memory is the one metric that cannot be
    recovered by re-parsing a log afterwards, so losing it means a rerun.
    """
    if not mem_path or not os.path.exists(mem_path):
        return None
    try:
        with open(mem_path) as f:
            return int(f.read().strip().split()[-1])
    except (ValueError, IndexError):
        return None
    finally:
        try:
            os.remove(mem_path)
        except OSError:
            pass


def mem_gb_from_kb(kb):
    """Peak RSS in KB -> GB rounded to 3 decimals, or '' when unmeasured."""
    return "" if kb is None else round(kb / 1024.0 / 1024.0, 3)


def iter_header_lines(log_text):
    """Yield write_log's leading `#` header lines, stopping at the STDERR marker.

    Reader-side counterpart to format_peak_rss_headers: every header parser shares
    this one scan, so none of them can drift on where the header block ends.
    """
    for line in log_text.splitlines():
        if line.startswith(HEADER_END_PREFIX) or not line.startswith("#"):
            return  # headers are the leading block, ending at the STDERR marker
        yield line


def parse_peak_rss(log_text):
    """Read write_log's peak-RSS header back: (kb, source) or (None, "").

    `kb` is an int; `source` says where the number came from. A live run always
    writes "measured" (a GNU-time reading of the very solve the log describes),
    and that is the only tag any tool in this tree produces today. The historical
    logs behind the committed results also carry two RELOCATION tags, written
    once each by one-shot scripts that are no longer in the tree (PROVENANCE.txt
    sections 1.1 and 2.2) and still parsed here because the logs and
    results/*.csv keep them:

        backfilled[-untimed]:<csv>  the SAME execution's peak, moved out of an
                                    old sweep CSV into its log after the header
                                    was introduced. As good as `measured`.
        probeN[-partial]:<log>      a DIFFERENT, iteration-capped execution's
                                    peak — a LOWER BOUND on this row's solve,
                                    never its peak.

    Logs written before the header existed yield (None, "").
    """
    kb, source = None, ""
    for line in iter_header_lines(log_text):
        if line.startswith("# peak_rss_kb:"):
            try:
                kb = int(line.split(":", 1)[1].strip())
            except ValueError:
                kb = None
        elif line.startswith("# peak_rss_source:"):
            source = line.split(":", 1)[1].strip()
    # Keep the pair atomic: a source with no usable kb would write a row claiming
    # provenance for a measurement that isn't there.
    return (kb, source) if kb is not None else (None, "")


def parse_returncode(log_text):
    """Read write_log's `# returncode:` header back as an int, or None.

    None means no usable header — absent (a log written before write_log recorded
    it, or a hand-assembled one) or present with a value that will not parse.
    Either way that is genuinely "unknown", not "ok": do not infer success from a
    missing header (format_exit_status keeps it blank for that reason).
    """
    for line in iter_header_lines(log_text):
        if line.startswith("# returncode:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def format_exit_status(returncode):
    """Decode a return code into the results CSVs' `exit_status` column.

    Named `exit_status`, not `status`: results/mps_compact_baseline.csv already
    has a `status` column carrying the SOLVER's reported solution status ("Optimal",
    "Model status : Not Set", ...), parsed from solver output. That CSV and
    results/cg_benchmark.csv are compared head-to-head, so one name meaning two
    orthogonal things across them would mislead in both directions. This column is
    the PROCESS exit disposition and says so.

    The one place a return code is interpreted, so the sweep CSV, the consolidated
    CSV and the console verdict all agree. Vocabulary:

        ""                no returncode recorded -> unknown, NOT assumed ok
        "ok"              clean exit
        "killed SIGKILL"  died on a signal, name resolved
        "killed sig=N"    died on a signal Python's enum has no name for
        "error rc=N"      non-zero exit that is not a signal death

    "ok" reports the EXIT, not the answer. transportation/Sydney/path/cuopt exits
    0 with objective=-inf: the cuOpt barrier failed, CG's first LP solve came back
    non-optimal, and the loop broke after 0 iterations, so that -inf is CGResult's
    "no objective established" sentinel rather than a computed value. A backend
    can also swallow its own failure and report a plausible wrong optimum instead,
    which looks quite different — optimal=1 (gh #33, fixed in the required fork).
    Either way this column is only ever read alongside `optimal`/`rel_err`, never
    alone.

    Signal encoding is ambiguous by construction: subprocess returns a NEGATIVE
    code on signal death, but we normally run the child under GNU `time`, which
    exits 128+signum instead. Both are decoded, which means a genuine exit status
    above 128 would be misreported as a signal — no benchmarked run exits that
    way, and losing "killed" is the worse failure mode.

    A signal death names the signal, never a cause. SIGKILL is *consistent* with
    the kernel OOM killer but equally with a manual `kill -9` or a cgroup limit,
    and the harness keeps no kernel-log evidence to tell them apart — read it
    alongside `mem_gb` (which does survive, via GNU time) and judge. Recording
    "oom" here would be recording an inference as a measurement.
    """
    if returncode is None:
        return ""
    if returncode == 0:
        return "ok"
    signum = -returncode if returncode < 0 else (
        returncode - 128 if 128 < returncode <= 128 + 64 else None)  # Linux: sig 1..64
    if signum is not None:
        try:
            return f"killed {signal.Signals(signum).name}"
        except ValueError:
            return f"killed sig={signum}"
    return f"error rc={returncode}"


# Iteration-table header -> tidy column name. The CLI prints this table on stderr
# at Verbosity::Iteration; write_log captures it verbatim, so it is the only
# record of per-iteration behaviour (gh #38).
ITER_COLUMNS = {
    "It": "iteration", "UB": "ub", "LB": "lb", "LP_obj": "lp_obj",
    "#col": "n_col", "#row": "n_row", "#slk": "n_slk",
    "+col": "added_col", "-col": "removed_col",
    "+cut": "added_cut", "-cut": "removed_cut",
    "t_LP": "t_lp", "t_PR": "t_pricing", "t_SP": "t_separation",
    "t_Tot": "t_iter", "t_acc": "t_acc",
}


def parse_iteration_table(log_text):
    """Parse the per-iteration CG table into a list of dicts, [] if absent.

    Numeric fields become float/int; `inf` stays a float inf. The `+col` count
    may carry a leading '*' meaning the columns were priced but NOT added — the
    loop hit the optimality gap and returned without calling add_columns
    (cg_loop.h). Those are reported as `added_col` with `col_committed=False`,
    so a "columns generated" total can exclude them instead of overcounting.
    """
    lines = log_text.splitlines()
    header_idx, names = None, None
    for i, line in enumerate(lines):
        toks = line.split()
        if toks[:1] == ["It"] and "LP_obj" in toks:
            header_idx, names = i, [ITER_COLUMNS.get(t, t) for t in toks]
            break
    if header_idx is None:
        return []

    rows = []
    for line in lines[header_idx + 1:]:
        toks = line.split()
        if len(toks) != len(names) or not toks[0].isdigit():
            break  # end of table (summary line, blank, or the STDOUT marker)
        rec, committed = {}, True
        for name, tok in zip(names, toks):
            if name == "added_col" and tok.startswith("*"):
                committed, tok = False, tok[1:]
            try:
                rec[name] = int(tok) if name.startswith(("n_", "iteration")) \
                    or name.startswith(("added_", "removed_")) else float(tok)
            except ValueError:
                rec[name] = tok
        rec["col_committed"] = committed
        rows.append(rec)
    return rows


def parse_slack_mode(log_text):
    """Read the preamble's `Slack mode:` line -> (mode, struct_rows), or (None, 0).

    MasterBase::init picks the placement (master_base.h): CommodityRows puts one
    slack on every structural row up front; EdgeRows pairs one with each lazily
    added capacity row. That choice determines how many of the master's columns
    are slacks, which the iteration trace counts and the result row does not.
    """
    for line in log_text.splitlines():
        m = re.search(r"Slack mode: (\w+) \(struct=(\d+), capped_arcs=(\d+)\)", line)
        if m:
            return m.group(1), int(m.group(2))
    return None, 0


def summarize_iterations(rows, slack_mode=(None, 0)):
    """Per-run aggregates derivable only from the iteration trace.

    `columns_seeded` is the warm-start pool the pricer never reports: added
    before the loop, so iteration 1 shows them in `#col` with `+col = 0`.
    Uncommitted (`*N`) counts are excluded from `columns_generated`.

    `slack_columns` is how many of the master's columns are slacks, implied by
    the slack mode rather than counted: CommodityRows places one per structural
    row at init, EdgeRows one per lazily added capacity row. At termination it
    accounts for the gap between the trace's `#col` (which counts slacks) and
    the result row's `columns` (which does not) in 436 of the 438 runs having
    both; the 2 exceptions report `columns=0`.

    CAUTION: these do NOT close into a per-iteration identity. `#col` grows by
    more than `+col` reports on many runs (e.g. grid11/path/highs iter 3: `+col`
    2282, `#col` +2406), and adding the `+cut` slack term does not fix it — 292
    of 437 runs still violate it. Treat each figure as the quantity it names;
    do not publish a derived reconciliation without first establishing where the
    extra columns come from.
    """
    if not rows:
        return {}
    mode, struct = slack_mode
    if mode == "CommodityRows":
        slack_cols = struct              # one per structural row, placed at init
    elif mode == "EdgeRows":
        slack_cols = rows[-1]["n_row"] - struct   # one per lazily added cap row
    else:
        slack_cols = ""
    return {
        "columns_generated": sum(r["added_col"] for r in rows if r["col_committed"]),
        "columns_seeded": rows[0]["n_col"] - rows[0]["added_col"],
        "columns_purged": sum(r["removed_col"] for r in rows),
        "slack_columns": slack_cols,
        "cuts_added": sum(r["added_cut"] for r in rows),
        "cuts_removed": sum(r["removed_cut"] for r in rows),
    }


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
    peak_rss_kb = read_peak_mem_kb(mem_path)
    mem_gb = mem_gb_from_kb(peak_rss_kb)
    config = parse_lp_config(stderr)
    # Parse BEFORE the log write so the header's `# outcome:` is the same verdict
    # this function returns. It used to be rc-derived only, so a clean exit that
    # printed nothing parseable wrote `# outcome: ok` into a log whose CSV row said
    # `error` -- one run describing itself two ways, which is exactly the
    # disagreement the mem_gb relocation gate (PROVENANCE 1.1) reads as a mixup.
    # parse_csv_row is total (returns None, never raises), so moving it above the
    # write keeps write_log's "dump both verbatim regardless of outcome" promise.
    row = parse_csv_row(stdout)
    outcome = "ok" if returncode == 0 and row is not None else "error"
    if log_path:
        write_log(log_path, run_cmd, stdout, stderr, returncode, outcome,
                  peak_rss_kb=peak_rss_kb)
    if returncode != 0:
        tail = "\n".join(stderr.strip().splitlines()[-3:])
        return {"outcome": "error", "returncode": returncode, "stderr_tail": tail,
                "mem_gb": mem_gb, "config": config}
    if row is None:
        # rc really was 0 -- the child exited cleanly and printed nothing
        # parseable. Reporting anything else here would put an inference where a
        # measurement belongs; the caller names the condition in `detail`, so the
        # tail no longer carries its own "no CSV row" prefix.
        tail = "\n".join(stderr.strip().splitlines()[-3:])
        return {"outcome": "error", "returncode": 0, "stderr_tail": tail,
                "mem_gb": mem_gb, "config": config}
    return {
        "outcome": "ok",
        "returncode": returncode,
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
    ap.add_argument("--out", default="bench_runs/cg/results.csv",
                    help="results CSV; default under the gitignored bench_runs/ "
                         "tree, not repo root. Promote a finalized full run to "
                         "results/ to commit it.")
    ap.add_argument("--logdir", default="bench_runs/cg/logs",
                    help="directory to save each run's full stdout+stderr (the per-iteration "
                         "CG log: cut growth, slack/bound history, timings). One file per run, "
                         "named <family>__<instance>__<formulation>__<solver>.log. "
                         "Default under gitignored bench_runs/, not repo root.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    if not os.path.exists(args.binary):
        sys.exit(f"binary not found: {args.binary} (build it first)")

    # Peak RSS is the only metric measured outside the child, so a missing GNU
    # time costs the whole sweep its memory column with nothing to re-parse
    # afterwards (gh #37). Say so up front, not after a 20-hour run.
    if not os.path.exists(GNU_TIME):
        sys.stderr.write(
            f"WARNING: {GNU_TIME} not found — peak RSS will NOT be measured, so the\n"
            "  `# peak_rss_kb:` header is omitted from every log this sweep writes.\n"
            "  write_log truncates, so rerunning into a --logdir whose logs already\n"
            "  carry memory REPLACES them with headerless ones and loses those\n"
            "  measurements permanently — memory cannot be re-derived from a log.\n"
            "  Install GNU time (`apt install time`) before a sweep whose mem_gb\n"
            "  column matters, and prefer a fresh --logdir for reruns.\n")

    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    override_forms = ([f.strip() for f in args.formulations.split(",") if f.strip()]
                      if args.formulations else None)

    # `exit_status` decodes the child's exit disposition (format_exit_status);
    # `outcome` is the coarse ok/error bucket the summary counts. They are not
    # redundant: every way a run can die -- signal, crash, bad args -- collapses
    # into "error", and only `exit_status` says which, so a cell killed by a signal
    # is distinguishable from a licence failure without opening the log.
    fields = ["family", "instance", "solver", "formulation", "outcome", "exit_status",
              "objective", "ref", "rel_err", "pass", "optimal", "iterations", "columns",
              "time", "mem_gb", "config", "detail"]
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
                       "exit_status": format_exit_status(r.get("returncode")),
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
                    # The exit disposition LEADS the detail. It used to be a
                    # fallback for an empty stderr, so a run that died mid-log --
                    # exactly the interesting kind -- had its stderr tail win and
                    # the return code silently dropped: a signal-killed cell was
                    # indistinguishable from any other error in this CSV.
                    #
                    # An errored run whose exit_status is "ok" is the clean-exit-
                    # but-nothing-parseable case; leading with a bare "ok" on an
                    # error row reads as a contradiction, so name that condition
                    # instead of restating the exit code.
                    lead = rec["exit_status"] or "rc=?"
                    if lead == "ok":
                        lead = "exited 0 without a result row"
                    tail = r.get("stderr_tail", "")
                    rec["detail"] = f"{lead}; {tail}" if tail else lead
                    summary[solver][r["outcome"]] += 1
                    sys.stderr.write(f"{r['outcome'].upper()} ({lead})\n")
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
