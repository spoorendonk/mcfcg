#!/usr/bin/env python3
"""Benchmark the monolithic compact-source LP (MPS files) across native solver binaries.

This is the *direct-solve baseline* that the column-generation method
(scripts/benchmark_solvers.py) is compared against: each `.mps.gz` is the full
compact source LP (one variable f^s_e per source/edge, |S|*|V| conservation rows
+ one capacity row per arc) handed to a barrier in a single shot — no CG. The
compact source LP corresponds to the TREE formulation's LP (same optimum), so
the natural head-to-head is `direct-X` here vs `tree-CG-X` in benchmark_solvers.py
for the same backend X.

Unlike benchmark_solvers.py (which drives the mcfcg CLI), this driver invokes the
vendors' *native* command-line solvers directly, because the mcfcg CLI does not
read external MPS. Barrier regime: interior point, **presolve left at each solver's
default** (let the solver decide), crossover OFF, relative tol = 1e-4 (tolerances.h
BARRIER_TOL). Presolve is deliberately NOT forced (unlike the CG masters, which pin
it OFF): a direct monolith solve normally benefits a lot from presolve, and forcing
it off would handicap the baseline and flatter CG, so we give the monolith each
vendor's natural setting. (Caveat: most solvers default to presolve on/auto, but
cuOpt's LP default is presolve OFF — "let the solver decide" means it stays off
there unless you pass --presolve 1.) Only presolve is left free; crossover-off +
tol 1e-4 keep the solution type and stopping tolerance identical for a fair
objective comparison. Each run's full log is saved.

Intermodal has no row here on purpose: its compact source LP is |S|*|E| with
|S| ~ |K| (unique-source), which is intractable to even build/store — the mcfcg
MPS exporter throws by design (see src/source/source_lp.cpp). That asymmetry is a
result, not a gap.

Solver binaries (paths overridable via env or the BINARIES table below):
  highs     build/bin/highs                              (built by this repo)
  mosek     $MOSEK_HOME/bin/mosek
  copt-cpu  $COPT_HOME/bin/copt_cmd  (GPUMode 0)
  copt-gpu  $COPT_HOME/bin/copt_cmd  (GPUMode 2)
  cuopt     $CUOPT_BIN or /usr/local/cuopt/bin/cuopt_cli (GPU)

NOTE: the per-solver parameter names below are the first-cut mapping from each
CLI's --help. Validate objective/time parsing on one small instance (e.g. grid1)
when the machine is free before trusting a full sweep; parsing regexes are
centralized in PARSERS for easy adjustment.

Memory. Every solve is wrapped in GNU `time -f %M` (inside the cgroup guard, so
it measures the solver and not `systemd-run`) and the peak RSS is written into
the log header as `# peak_rss_kb:` — the same serialization benchmark_solvers.py
uses for the CG runs, so one reader handles both. GPU configs additionally get
`# peak_vram_mib:` sampled from `nvidia-smi` (see VramSampler): host RSS says
nothing about the device-side factorization, which is precisely what OOMs on the
giants. Memory is the one metric that cannot be recovered by re-parsing a log
afterwards, so an unmeasured run means a rerun — which is why both ways a solve
can be killed preserve the reading: a cgroup OOM through the guard's
OOMPolicy=continue, and a harness timeout through kill_preserving_mem.

--probe-iters N caps the barrier at N iterations (each cmd_* builder injects its
backend's knob: ipm_iteration_limit / MSK_IPAR_INTPNT_MAX_ITERATIONS /
BarIterLimit / --iteration-limit) instead of solving. That is the cheap way to get memory for cells
whose full solve costs hours: a barrier's peak RSS is dominated by the symbolic +
numeric ADAT factorization, which is allocated on the first iteration and reused,
so a few iterations reach a tight lower bound on the full-solve peak: measured
0.88-1.00 of it (0.95-1.00 at N=3) across all 5 backends on grid7/planar300/grid10,
for 1.6-12x less runtime. Report it as the lower bound it is.
Probe runs write to their own logdir/CSV, carry a `# probe_iters:` header, and
report NO objective — an iterate one step off the central path is not a solution
and must never reach a results table.

CSV columns:
  instance,solver,gpu,outcome,objective,ref,rel_err,pass,time_wall,time_solve,
  mem_gb,vram_gb,status,detail
"""

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time

import benchmark_solvers as bs  # sibling module: peak-RSS log header + GNU time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default(path_env, fallback):
    v = os.environ.get(path_env)
    return v if v else fallback


MOSEK_HOME = _default("MOSEK_HOME", "/opt/mosek/11.0/tools/platform/linux64x86")
COPT_HOME = _default("COPT_HOME", "/opt/copt80")


def _first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return paths[-1]  # report the preferred default even if missing (warns later)


# HiPO-capable standalone HiGHS v1.15.1 (matches CMakeLists' GIT_TAG). The stock
# system /usr/local/bin/highs at v1.15.1 CANNOT run HiPO from the CLI — it aborts
#   "The HiPO solver ... features are unavailable: amd, blas, metis, rcm" (rc 255)
# because the 1.15 standalone resolves its numerical extras by dlopen'ing
# libhighs_extras.so at runtime and that .so is not shipped/linked (even placing it
# on LD_LIBRARY_PATH does not fix the standalone). We build a fixed 1.15.1 `highs`
# from the SAME patched source the repo fetches (build/_deps/highs-src) with
# BUILD_SHARED_EXTRAS_LIB=OFF + BUILD_SHARED_LIBS=OFF, so the extras (AMD/METIS/
# RCM/BLAS) are compiled into a static libhighs and baked into a self-contained
# exe (only dynamic dep is system BLAS). Validated: "Running HiPO" -> Optimal,
# objectives match MOSEK/COPT/cuOpt. Install it over /usr/local/bin/highs (sudo) to
# fix the box globally; until then the harness prefers the known-good build output.
# Override with HIGHS_BIN. run_one asserts HiPO actually ran (guards a silent
# dual-simplex/other fallback slipping into the results).
HIGHS_BIN = _first_existing(
    os.environ.get("HIGHS_BIN"),
    "/home/simon/opt/highs-1.15.1/bin/highs",         # our fixed 1.15.1 (HiPO works)
    "/home/simon/opt/highs-1.15.1-build/bin/highs",   # build-tree copy
    "/usr/local/bin/highs",                           # after sudo reinstall, this too
)

BINARIES = {
    "highs": HIGHS_BIN,
    "mosek": os.path.join(MOSEK_HOME, "bin/mosek"),
    "copt": os.path.join(COPT_HOME, "bin/copt_cmd"),
    "cuopt": _default("CUOPT_BIN", "/usr/local/cuopt/bin/cuopt_cli"),
}

# Pinned barrier regime shared with the CG experiments (tolerances.h BARRIER_TOL).
BARRIER_TOL = 1e-4

# The solver's own --time-limit is the real stopping mechanism; the harness gives
# it this much extra wall-clock before declaring a hang and killing the tree.
HARNESS_TIMEOUT_GRACE_SEC = 1800
# After that kill, how long the GNU `time` wrapper gets to reap the solver and
# write its report before we escalate to a blunt killpg (see kill_preserving_mem).
KILL_FLUSH_SEC = 120.0
# Extra slack on the scope's RuntimeMaxSec so systemd's own teardown -- which
# SIGTERMs every process in the scope, `time` included -- can never fire while
# that flush window is still open.
SCOPE_TEARDOWN_MARGIN_SEC = 60

# Optional per-solve memory cgroup guard (set from CLI in main()). Wrapping each
# solver in a transient systemd scope with MemoryMax + MemorySwapMax=0 means a
# runaway barrier (the giants can need >100 GB to form the normal equations) is
# OOM-killed cleanly at the cap instead of dragging the whole box into a swap
# death-spiral. Empty list => no guard.
MEM_GUARD = []

# Reference optima per family (same files benchmark_solvers.py checks against).
FAMILY_OPTIMAL = {
    "grid": "data/commalab/grid/optimal.csv",
    "planar": "data/commalab/planar/optimal.csv",
    "transportation": "data/transportation/optimal.csv",
}


def load_refs():
    refs = {}
    for fam, rel in FAMILY_OPTIMAL.items():
        p = os.path.join(REPO, rel)
        if not os.path.exists(p):
            continue
        with open(p, newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) >= 2 and row[0]:
                    refs[row[0]] = float(row[1])
    return refs


# --- Per-config command builders --------------------------------------------
# Each returns argv (list) plus an optional dict of extra files to write first
# (path -> contents), e.g. a HiGHS options file or a MOSEK parameter file.


def cmd_highs(mps, time_limit, logdir, tag, probe_iters=0):
    opts = os.path.join(logdir, f"{tag}.highs-opts")
    # Match the CG HiGHS backend exactly (src/lp/highs_solver.cpp): the HiPO
    # interior-point solver, crossover off, primal/dual feasibility tol = 1e-4.
    # Presolve is left at HiGHS' default (not forced) per the monolith regime.
    contents = (
        "solver = hipo\n"
        "run_crossover = off\n"
        f"primal_feasibility_tolerance = {BARRIER_TOL}\n"
        f"dual_feasibility_tolerance = {BARRIER_TOL}\n"
        f"time_limit = {time_limit}\n"
    )
    if probe_iters:
        contents += f"ipm_iteration_limit = {probe_iters}\n"
    argv = [BINARIES["highs"], "--model_file", mps, "--options_file", opts]
    return argv, {opts: contents}


def cmd_mosek(mps, time_limit, logdir, tag, probe_iters=0):
    # -itro /dev/null: MOSEK writes the interior solution to <input>.sol next to
    # the input by default (would land in data/mps and be multi-GB on the giants);
    # redirect it to the bit bucket. Barrier tol pinned on all three components to
    # match the CG backend (src/lp/mosek_solver.cpp). Presolve left at MOSEK's
    # default (MSK_PRESOLVE_MODE_FREE) per the monolith regime.
    argv = [
        BINARIES["mosek"],
        "-d", "MSK_IPAR_OPTIMIZER", "MSK_OPTIMIZER_INTPNT",
        "-d", "MSK_IPAR_INTPNT_BASIS", "MSK_BI_NEVER",  # crossover/basis off
        "-d", "MSK_DPAR_INTPNT_TOL_PFEAS", str(BARRIER_TOL),
        "-d", "MSK_DPAR_INTPNT_TOL_DFEAS", str(BARRIER_TOL),
        "-d", "MSK_DPAR_INTPNT_TOL_REL_GAP", str(BARRIER_TOL),
        "-d", "MSK_DPAR_OPTIMIZER_MAX_TIME", str(time_limit),
        "-itro", "/dev/null",
    ]
    if probe_iters:
        argv += ["-d", "MSK_IPAR_INTPNT_MAX_ITERATIONS", str(probe_iters)]
    argv.append(mps)
    return argv, {}


def cmd_copt(mps, time_limit, logdir, tag, gpu_mode, probe_iters=0):
    # readmps reads .mps and .mps.gz. FeasTol+DualTol are the barrier stopping
    # tolerances used by the CG backend (src/lp/copt_solver.cpp); COPT 8.0 has NO
    # "BarConvTol" parameter (it errors "Unknown COPT parameter"). Presolve left at
    # COPT's default (auto) per the monolith regime. The trailing "quit" is
    # REQUIRED: `copt_cmd -c <script>` drops into an interactive prompt after the
    # script and would otherwise block on stdin until the subprocess timeout
    # (run_one also passes stdin=DEVNULL as a belt-and-suspenders).
    script = (
        f"readmps {mps}; "
        "set LpMethod 2; "        # barrier
        "set Crossover 0; "
        f"set GPUMode {gpu_mode}; "
        f"set FeasTol {BARRIER_TOL}; "
        f"set DualTol {BARRIER_TOL}; "
        f"set TimeLimit {time_limit}; "
        + (f"set BarIterLimit {probe_iters}; " if probe_iters else "")
        + "optimize; "
        "quit"
    )
    return [BINARIES["copt"], "-c", script], {}


def cmd_cuopt(mps, time_limit, logdir, tag, probe_iters=0):
    # Discard the primal solution: on the giant instances a converged solution has
    # tens of millions of entries (multi-GB per file), and we only ever read the
    # objective/time from stdout. Mirrors MOSEK's `-itro /dev/null`.
    sol = "/dev/null"
    # --method 3 = CUOPT_METHOD_BARRIER (0=concurrent, 1=PDLP, 2=dual-simplex,
    # 3=barrier). WITHOUT it cuOpt runs the default concurrent/PDLP path, NOT the
    # GPU barrier the CG backend uses (src/lp/cuopt_solver.cpp pins
    # CUOPT_METHOD_BARRIER). Presolve left at cuOpt's LP default (off).
    argv = [
        BINARIES["cuopt"], mps,
        "--method", "3",
        "--crossover", "0",
        "--relative-gap-tolerance", str(BARRIER_TOL),
        "--relative-primal-tolerance", str(BARRIER_TOL),
        "--relative-dual-tolerance", str(BARRIER_TOL),
        "--time-limit", str(time_limit),
        "--solution-file", sol,
    ]
    if probe_iters:
        argv += ["--iteration-limit", str(probe_iters)]
    return argv, {}


# Config label -> (command builder, gpu annotation).
CONFIGS = {
    "highs": (lambda m, t, d, g, p: cmd_highs(m, t, d, g, p), "cpu"),
    "mosek": (lambda m, t, d, g, p: cmd_mosek(m, t, d, g, p), "cpu"),
    "copt-cpu": (lambda m, t, d, g, p: cmd_copt(m, t, d, g, 0, p), "cpu"),
    "copt-gpu": (lambda m, t, d, g, p: cmd_copt(m, t, d, g, 2, p), "gpu"),
    "cuopt": (lambda m, t, d, g, p: cmd_cuopt(m, t, d, g, p), "gpu"),
}

# Objective + solver-reported solve-time parsers (best-effort; centralized so a
# single validation run can correct any that drift from the installed versions).
PARSERS = {
    # HiGHS CLI: "Objective value     :  1.2345678900e+04"
    "highs": {
        "obj": re.compile(r"Objective value\s*:\s*([-\d.eE+]+)"),
        "time": re.compile(r"(?:Solve|HiGHS run) time\s*:?\s*([\d.]+)"),
        "status": re.compile(r"Model status\s*:\s*(.+)"),
    },
    # MOSEK: "Primal.  obj: 1.23e4" / "Optimizer terminated. Time: 65.32"
    "mosek": {
        "obj": re.compile(r"Primal\.\s*obj:\s*([-\d.eE+]+)"),
        "time": re.compile(r"Optimizer terminated\. Time:\s*([\d.]+)"),
        "status": re.compile(r"(Interior-point solution summary|Optimizer terminated)"),
    },
    # COPT. `obj` is an ordered list: the FIRST regex that matches anywhere wins
    # (its last match is taken). We prefer the barrier's "Primal objective:" line
    # because on a TIME_LIMIT stop COPT's summary line prints "Objective:
    # 0.0000000000e+00" (no certified solution) even though the barrier reached a
    # near-optimal primal iterate -- taking that 0.0 would misreport a near-solved
    # giant as a 100% FAIL. On a clean Optimal solve both lines agree. The
    # Status-line "Objective:" is kept only as a fallback (e.g. a solve that
    # printed no barrier block). "Time:\s*([\d.]+)s" matches the summary line, not
    # the per-iteration time column.
    "copt": {
        "obj": [
            re.compile(r"Primal objective:\s*([-\d.eE+]+)"),
            re.compile(r"Objective:\s*([-\d.eE+]+)"),
        ],
        "time": re.compile(r"Time:\s*([\d.]+)s"),
        "status": re.compile(r"Status:\s*(\w+)"),
    },
    # cuOpt CLI. On convergence it prints "Objective +8.27e5"; on a
    # "Barrier time limit exceeded" stop there is NO such line, so we fall back to
    # the primal objective of the last barrier iteration row
    # ("  609   +1.80e+09  +1.80e+09  ..."). timing from "... iterations and 0.21s"
    # and "Barrier finished in 0.21 seconds".
    "cuopt": {
        "obj": [
            # [^\S\n]* (not \s*) keeps the match on the same line as "Objective",
            # so a bare "Objective" header can't grab a number off the next line.
            re.compile(r"Objective[^\S\n]*[:=]?[^\S\n]*([-\d.eE+]+)"),
            # last barrier-iteration primal; "-" is in the class so a negative
            # exponent (e.g. +1.80e-05) parses rather than truncating at the 'e'.
            re.compile(r"(?m)^\s*\d+\s+([-+][\d.][\d.eE+-]*)\s+[-+]"),
        ],
        "time": re.compile(r"(?:finished in|iterations and)\s+([\d.]+)"),
        "status": re.compile(r"(Optimal|TimeLimit|time limit|Infeasible|termination)", re.I),
    },
}


def parser_key(solver):
    return "copt" if solver.startswith("copt") else solver


# cuOpt (#33): the GPU barrier can hit a cuDSS device-alloc / numerical error and
# still exit rc=0, returning a non-optimal incumbent as if solved. On the giants
# this VRAM OOM is expected. Any of these markers in the log means the barrier did
# NOT produce a trustworthy solution -> we force an error and discard the objective
# so a swallowed failure never masquerades as a (possibly garbage) optimum.
CUOPT_FAIL_MARKERS = (
    "numerical error", "Out of memory in barrier", "bad_alloc",
    "cudaErrorMemoryAllocation", "out_of_memory",
)


def cuopt_solve_failed(text):
    return any(m in text for m in CUOPT_FAIL_MARKERS)


# COPT can hit "GPU memory issue" and print "[ERROR] Fail to solve" yet still exit
# rc=0. If COPT actually recovered (e.g. GPU->CPU fallback) it prints a valid
# "Primal objective:" and no "Fail to solve", so keying on this marker only trips
# on a genuine non-solve.
#
# `copt_cmd` is a SCRIPTED SHELL, which gives failure a second shape: when
# `readmps` fails, the script does not abort -- the subsequent `optimize` runs
# against an empty model, complains "Must read problem first", and `quit` exits
# rc=0. Observed on BUS-2632-0, whose 2.64e9 nonzeros exceed the int32 range;
# COPT bailed at 60 GB, well under the memory cap, so nothing else about the run
# looked like a failure either. Without these markers that cell scores as `ok`
# with no objective, i.e. a solver that read NOTHING is recorded as a clean run.
COPT_READ_FAIL_MARKERS = (
    "Reading failed",           # readmps gave up (size limit / malformed / OOM)
    "Must read problem first",  # optimize ran with no model loaded
)
COPT_FAIL_MARKERS = ("Fail to solve",) + COPT_READ_FAIL_MARKERS


def copt_solve_failed(text):
    return any(m in text for m in COPT_FAIL_MARKERS)


def copt_read_failed(text):
    """True when COPT never loaded the model, as opposed to failing to solve it.

    Worth separating in the record: "the barrier gave up" and "the file was never
    read" are different findings, and only the first says anything about the
    solver's ability to handle the problem.
    """
    return any(m in text for m in COPT_READ_FAIL_MARKERS)


# --probe-iters: how each backend reports "I stopped because you capped the
# barrier iterations". A probe that hits its cap did exactly what was asked --
# read the model, presolve, factor, iterate -- so its peak RSS is the datum we
# came for, even though highs (rc 1) and mosek (rc 160) call that a nonzero exit.
# Without these markers every probe would be recorded as an `error` and the
# memory would look untrustworthy. A probe missing its marker really did fail
# (OOM, crash, time limit) and stays an error.
PROBE_LIMIT_MARKERS = {
    "highs": ("Iteration limit reached", "Reached maximum iterations"),
    "mosek": ("MSK_RES_TRM_MAX_ITERATIONS",),
    "copt": ("ITER_LIMIT", "Status: IterLimit"),
    "cuopt": ("Iteration Limit",),
}


def probe_hit_iter_limit(solver, text):
    return any(m in text for m in PROBE_LIMIT_MARKERS[parser_key(solver)])


NVIDIA_SMI = "nvidia-smi"


def gpu_pid_memory():
    """{pid: MiB} for every current CUDA compute app, or None if the query failed.

    Empty dict and None are deliberately different: {} means nvidia-smi answered
    and nothing is on the GPU, None means we could not measure at all. Collapsing
    them would let a host with no nvidia-smi report every GPU solve as using 0 MiB
    of VRAM, which reads as a measurement rather than the absence of one.
    """
    try:
        r = subprocess.run(
            [NVIDIA_SMI, "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    out = {}
    for line in r.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            out[int(parts[0])] = int(parts[1])
    return out


class VramSampler(threading.Thread):
    """Poll peak device memory of one solve, in MiB, for the GPU backends.

    GNU time's %M is host RSS only, so for cuopt/copt-gpu it misses the thing
    that actually decides whether a giant fits: the device-side normal-equations
    factorization. nvidia-smi reports per-PID compute memory, so we sample it and
    keep the high-water mark.

    Attribution is by process group, not by name or by whole-GPU usage: run_one
    spawns the solver with start_new_session=True, so every process in the solve
    (systemd-run's scope child, the solver, any worker it forks) shares pgid ==
    proc.pid, while the user's desktop/browser do not. Summing whole-GPU usage
    instead would silently fold a compositor's VRAM into the measurement.

    Sampling is a floor, not a bound: an allocation that comes and goes entirely
    within one interval is invisible. That is acceptable here because the barrier
    holds its pool for the duration of the solve, which is what we are measuring.

    The interval backs off from `interval` to MAX_INTERVAL. Each sample forks an
    nvidia-smi, so a fixed 0.1s would spend ~72k process spawns on a 2h solve and
    perturb the wall-clock this harness also records; the pool is allocated during
    setup and the first factorization, so sample that densely and then coast.
    """

    MAX_INTERVAL = 2.0

    def __init__(self, pgid, interval=0.1):
        super().__init__(daemon=True)
        self.pgid = pgid
        self.interval = interval
        self.peak_mib = 0
        self._measured = False
        self._done = threading.Event()

    def run(self):
        wait = self.interval
        while not self._done.is_set():
            sample = gpu_pid_memory()
            if sample is not None:
                self._measured = True
                total = 0
                for pid, mib in sample.items():
                    try:
                        if os.getpgid(pid) == self.pgid:
                            total += mib
                    except OSError:
                        continue  # exited between the query and the getpgid
                self.peak_mib = max(self.peak_mib, total)
            self._done.wait(wait)
            wait = min(wait * 1.5, self.MAX_INTERVAL)

    def stop(self):
        """Peak MiB, or None when nvidia-smi never answered (never a bare 0)."""
        self._done.set()
        self.join(timeout=30)
        return self.peak_mib if self._measured else None


# The per-run summary header. Writer and reader live together so the consolidator
# cannot drift from what run_one emits.
RUN_HEADER = re.compile(r"# wall=([\d.]+)s rc=(-?\d+) outcome=(\w+)")


def format_run_header(wall, rc, outcome):
    return f"# wall={wall:.3f}s rc={rc} outcome={outcome}\n"


def parse_run_header(log_text):
    """(wall, rc, outcome) from the run header; (None, None, "error") if absent.

    A log with no readable summary header is treated as an errored run, not as an
    unknown one: every complete log has this line, so its absence means the run
    died before the writer got to it.
    """
    m = RUN_HEADER.search(log_text)
    if not m:
        return None, None, "error"
    return float(m.group(1)), int(m.group(2)), m.group(3)


# Log-header extras beyond benchmark_solvers' peak-RSS pair. Writer and readers
# live together so consolidate_mps_logs.py cannot drift from what is emitted.
def format_extra_headers(peak_vram_mib, probe_iters):
    s = ""
    if peak_vram_mib is not None:
        s += f"# peak_vram_mib: {peak_vram_mib}\n"
    if probe_iters:
        s += f"# probe_iters: {probe_iters}\n"
    return s


def _parse_int_header(log_text, prefix):
    val = None
    for line in bs.iter_header_lines(log_text):
        if line.startswith(prefix):
            try:
                val = int(line.split(":", 1)[1].strip())
            except ValueError:
                val = None
    return val


def parse_peak_vram_mib(log_text):
    """Peak device memory in MiB from the log header, or None (CPU config / no smi)."""
    return _parse_int_header(log_text, "# peak_vram_mib:")


def parse_probe_iters(log_text):
    """Barrier iteration cap this log was produced under, or None for a real solve.

    Any non-None value means the run was STOPPED EARLY on purpose: its objective
    is a mid-barrier iterate and must not be scored against a reference.
    """
    return _parse_int_header(log_text, "# probe_iters:")


def gb_from_mib(mib):
    return "" if mib is None else round(mib / 1024.0, 3)


def parse_output(solver, text):
    pk = parser_key(solver)
    pats = PARSERS[pk]
    obj = tlast = status = None
    # obj may be a single regex or an ordered priority list. Try each pattern in
    # order and take the LAST match that parses as a float; a pattern whose match
    # is a non-numeric fragment (the permissive char classes can capture a bare
    # sign/exponent) falls THROUGH to the next pattern instead of poisoning the
    # result to None -- so e.g. cuOpt's final-objective regex can't suppress the
    # iteration-primal fallback.
    obj_pats = pats["obj"]
    if not isinstance(obj_pats, (list, tuple)):
        obj_pats = [obj_pats]
    for pat in obj_pats:
        cand = None
        for m in pat.finditer(text):
            try:
                cand = float(m.group(1))
            except ValueError:
                continue
        if cand is not None:
            obj = cand
            break
    for m in pats["time"].finditer(text):
        tlast = m.group(1)
    for m in pats["status"].finditer(text):  # last match = terminal status
        status = m.group(0)
    try:
        tlast = float(tlast) if tlast is not None else None
    except ValueError:
        tlast = None
    return obj, tlast, status


def scope_runtime_max_sec(time_limit):
    """RuntimeMaxSec for a solve's systemd scope.

    Must land STRICTLY AFTER the harness has finished its own teardown, because
    systemd's expiry SIGTERMs every process in the scope -- the GNU `time`
    wrapper included -- which destroys the measurement kill_preserving_mem
    exists to save. The harness needs `time_limit + HARNESS_TIMEOUT_GRACE_SEC`
    to declare the hang and up to KILL_FLUSH_SEC more to let the wrapper flush;
    SCOPE_TEARDOWN_MARGIN_SEC is the headroom on top. At the historical value of
    exactly the subprocess timeout the two teardowns fired in the same instant
    and systemd won often enough to matter.

    Defined here rather than inline in main() so the ordering invariant is
    testable without reimplementing it (test/python/timeout_memory_test.py).
    """
    return int(time_limit + HARNESS_TIMEOUT_GRACE_SEC
               + KILL_FLUSH_SEC + SCOPE_TEARDOWN_MARGIN_SEC)


def pgid_members(pgid, exclude=()):
    """Live pids whose process group is `pgid`, minus `exclude`. [] without /proc.

    Read from /proc rather than tracked in Python because the solve's tree is
    opaque to us: systemd-run's scope child, the solver, and any worker it forks
    all inherit the group (start_new_session in run_one), and only the kernel
    knows the current membership.
    """
    out = []
    try:
        entries = os.listdir("/proc")
    except OSError:
        return out
    for d in entries:
        if not d.isdigit() or int(d) in exclude:
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                stat = f.read()
            # Fields after the comm field, which is parenthesized and may itself
            # contain spaces/parens -- split after the LAST ')': [state, ppid, pgrp].
            if int(stat.rsplit(")", 1)[1].split()[2]) == pgid:
                out.append(int(d))
        except (OSError, ValueError, IndexError):
            continue  # exited, or unreadable: nothing to kill
    return out


def is_time_wrapper(pid, mem_path):
    """True when `pid` is the GNU `time` process writing to `mem_path`.

    Identified by its argv, not by name: mem_path is a per-solve mkstemp name, so
    a match cannot be anything but this run's wrapper. Under the systemd guard
    `systemd-run --scope` execs the wrapper in place (verified: the direct child's
    cmdline IS `/usr/bin/time -o <mem_path> ...`), so the same check covers the
    guarded and unguarded spawns. It is False in the window before that exec, and
    False when GNU time is missing entirely -- both fall back to the blunt kill.
    """
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv = f.read().split(b"\0")
    except OSError:
        return False
    return (len(argv) > 2 and os.path.basename(argv[0].decode(errors="replace"))
            == os.path.basename(bs.GNU_TIME)
            and mem_path.encode() in argv)


def kill_preserving_mem(proc, mem_path):
    """Kill a hung solve on harness timeout, keeping its peak-RSS measurement.

    Returns (stdout, stderr), having reaped `proc`.

    A plain `killpg` loses the number: GNU `time` sits in the same process group
    as the solver, so it dies alongside it and never writes its `-f %M` report --
    and memory is the one metric that cannot be recovered by re-parsing a log
    afterwards. The historical casualty is the compact-baseline probe cell
    ChicagoRegional x highs, unmeasured for exactly this reason.

    So kill every OTHER member of the group and let the wrapper -- our direct
    child -- reap the corpse and flush. GNU time reports a signalled child
    normally ("Command terminated by signal 9" then the peak), and read_peak_mem_kb
    takes the last token, so the reading survives the kill intact.

    Chosen over reading the cgroup's `memory.peak` (systemd-run --unit=<name> to
    fix the scope path) for two reasons: `memory.peak` counts page cache and
    kernel memory charged to the cgroup, so a timeout cell would carry a number
    that is not comparable with the `time -f %M` RSS every other cell reports;
    and it exists only under `--mem-max`, leaving `--mem-max off` runs unmeasured.
    This path yields the same measurement, from the same instrument, as a cell
    that finished.

    Escalates to `killpg` whenever the wrapper cannot be identified or does not
    exit within KILL_FLUSH_SEC: preserving a measurement never outranks making
    sure a runaway solver is dead.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        # Nothing left to signal -- but still drain with a bound rather than
        # bare: the wrapper being gone does not prove every pipe holder is.
        return _drain(proc)
    if mem_path and is_time_wrapper(proc.pid, mem_path):
        for pid in pgid_members(pgid, exclude=(proc.pid,)):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        # Second sweep, BEFORE we block: anything forked between the scan above
        # and its kills would otherwise hold the inherited stdout/stderr pipe and
        # keep communicate() waiting out the whole flush window. Doing it here
        # also means we never signal after reaping -- once communicate() reaps
        # the wrapper, its pid (which IS the pgid) is free, and so is every pid
        # we might have snapshotted, so a late SIGKILL could land on an unrelated
        # process. Anything forked during the flush window is not swept up: its
        # parent is already dead, and under the memory guard the scope's
        # RuntimeMaxSec is the backstop.
        for pid in pgid_members(pgid, exclude=(proc.pid,)):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            return proc.communicate(timeout=KILL_FLUSH_SEC)
        except subprocess.TimeoutExpired:
            pass  # wrapper wedged too -- fall through and take the group down
    if mem_path:
        # The blunt kill takes the `time` wrapper with it. Say so: an unmeasured
        # cell that announces itself can be re-run, which is the whole difference
        # between this and the silent hole that motivated the function.
        sys.stderr.write(
            f"[mps] WARNING: falling back to killpg (pid {proc.pid} is not the "
            "`time` wrapper, or it wedged); peak RSS for this cell may be LOST\n")
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return _drain(proc)


def _drain(proc):
    """communicate() with a bound, so teardown can never hang the sweep.

    An unbounded communicate() waits for EOF on the inherited stdout/stderr
    pipes, not for the child -- so any process still holding a write end blocks
    it forever. SIGKILL to the process group normally covers that, but a solver
    that setsid'd (or a helper systemd keeps in the scope) is outside the group
    and keeps the pipe open. This is the one path whose entire purpose is
    guaranteeing forward progress through a multi-hour sweep, so it must not be
    the path that wedges it. Give up on the output rather than on the sweep: the
    cell still gets its log, its outcome and -- the metric that cannot be
    re-derived -- its peak RSS from the `time` wrapper's file.
    """
    try:
        return proc.communicate(timeout=KILL_FLUSH_SEC)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            return proc.communicate(timeout=KILL_FLUSH_SEC)
        except subprocess.TimeoutExpired:
            return "", "[harness] output pipes never closed; solver output lost\n"


def run_one(solver, mps, key, time_limit, logdir, probe_iters=0):
    builder, gpu = CONFIGS[solver]
    tag = f"{key}__{solver}"
    argv, extra_files = builder(mps, time_limit, logdir, tag, probe_iters)
    for path, contents in extra_files.items():
        with open(path, "w") as f:
            f.write(contents)
    log_path = os.path.join(logdir, f"{tag}.log")
    # Prepend the memory-cgroup guard (if configured). systemd-run --scope runs the
    # child synchronously and propagates its exit status; a MemoryMax kill surfaces
    # as rc 137 (128+SIGKILL, via the `time` wrapper below) which we record as an
    # OOM error rather than a crash.
    #
    # GNU time goes INSIDE the guard: `systemd-run --scope` execs the solver as its
    # own child, so wrapping the guard would time systemd-run (and, worse, measure
    # nothing when the scope is torn down wholesale). Inside, `time` is a ~2 MB
    # bystander in the same cgroup -- the OOM killer picks the multi-GB solver, so
    # `time` survives to report the RSS the solve reached AT the cap, which is the
    # whole point of measuring the runs that die. Note the return-code vocabulary
    # changes: GNU time reports a signal death as the bare signal number.
    mem_path = None
    run_argv = MEM_GUARD + argv
    if os.path.exists(bs.GNU_TIME):
        fd, mem_path = tempfile.mkstemp(prefix="mcfcg_mps_mem_", suffix=".txt")
        os.close(fd)
        run_argv = MEM_GUARD + [bs.GNU_TIME, "-o", mem_path, "-f", "%M"] + argv
    else:
        sys.stderr.write(f"[mps] WARNING: {bs.GNU_TIME} missing; no memory for {tag}\n")
    t0 = time.monotonic()
    wall_override = None  # set only on the timeout path; see below
    # start_new_session=True puts the child in its own process group so a timeout
    # can killpg the WHOLE tree -- subprocess.run would SIGKILL only its immediate
    # child, orphaning the actual solver (under the systemd guard the solver is a
    # reparented grandchild; the guard's RuntimeMaxSec is the primary teardown, and
    # this is belt-and-suspenders for the unguarded path). Popen+communicate also
    # returns decodable str on timeout, sidestepping subprocess.run's
    # TimeoutExpired.stdout-as-bytes quirk that would TypeError and crash the whole
    # sweep on the exact hang path it is meant to survive. stdin=/dev/null feeds
    # copt_cmd's post-`quit` interactive prompt an EOF so it exits.
    try:
        proc = subprocess.Popen(
            run_argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, text=True, start_new_session=True)
    except OSError as e:
        # Missing/unexecutable binary (or absent systemd-run guard): record this
        # cell as an error and let the rest of the multi-hour sweep continue rather
        # than aborting the whole run.
        out = f"[harness] failed to launch: {e}\n"
        rc, outcome = 127, "error"
        vram_mib = None
    else:
        # start_new_session put the child in a new process group == its pid; the
        # sampler attributes device memory by that pgid (see VramSampler).
        sampler = VramSampler(proc.pid) if gpu == "gpu" else None
        if sampler:
            sampler.start()
        try:
            # The solver's own --time-limit is the real stopping mechanism; this
            # generous subprocess timeout only guards a true hang.
            stdout, stderr = proc.communicate(
                timeout=time_limit + HARNESS_TIMEOUT_GRACE_SEC)
            rc = proc.returncode
            outcome = "ok" if rc == 0 else "error"
        except subprocess.TimeoutExpired:
            # Stamp the wall BEFORE the kill: teardown can take several times
            # KILL_FLUSH_SEC in the worst case (the flush window, then both of
            # _drain's bounded waits), and folding that into time_wall would
            # report a timed-out solve as having run minutes longer than it did.
            wall_override = time.monotonic() - t0
            # Kills the solver but spares the `time` wrapper, so a hang still
            # yields peak RSS -- see kill_preserving_mem.
            stdout, stderr = kill_preserving_mem(proc, mem_path)
            rc, outcome = -1, "timeout"
        vram_mib = sampler.stop() if sampler else None
        out = (stdout or "") + "\n" + (stderr or "")
    wall = time.monotonic() - t0 if wall_override is None else wall_override
    peak_rss_kb = bs.read_peak_mem_kb(mem_path)
    # A probe stopped by its own iteration cap is a SUCCESS: it read, factored and
    # iterated, which is all the memory measurement needs. Recognize that before
    # the failure guards below, which are about solves that produced nothing.
    if probe_iters and outcome == "error" and probe_hit_iter_limit(solver, out):
        outcome = "ok"
    # Guard against a silent HiGHS solver fallback: if we asked for HiPO but the
    # log shows the extras-unavailable abort (or never reports "Running HiPO"),
    # flag it so a dual-simplex/other result never masquerades as the HiPO baseline.
    note = ""
    if solver == "highs" and outcome == "ok":
        if "features are unavailable" in out or "Running HiPO" not in out:
            outcome = "error"
            note = "HiPO did not run (extras unavailable / fallback)"
    if solver == "cuopt" and outcome == "ok" and cuopt_solve_failed(out):
        outcome = "error"
        note = "cuOpt barrier numerical error / VRAM OOM (#33) -- objective discarded"
    if solver.startswith("copt") and outcome == "ok" and copt_solve_failed(out):
        outcome = "error"
        note = ("COPT never read the model (readmps failed) -- peak RSS is the "
                "READER's, not a solve's"
                if copt_read_failed(out) else
                "COPT failed to solve (GPU memory issue / infeasible) -- "
                "objective discarded")
    # Write atomically (temp + os.replace) so an interrupted driver can never leave
    # a truncated log carrying an intact "outcome=ok" header -- the consolidator
    # would otherwise re-parse the partial body and score a mid-barrier iterate as
    # the final objective. A log is now either absent or complete.
    tmp_path = log_path + ".partial"
    with open(tmp_path, "w") as f:
        f.write("# cmd: " + " ".join(run_argv) + "\n")
        f.write(format_run_header(wall, rc, outcome))
        f.write(bs.format_peak_rss_headers(peak_rss_kb, "measured"))
        f.write(format_extra_headers(vram_mib, probe_iters))
        if note:
            f.write(f"# WARN {note}\n")
        f.write("# === solver output ===\n")
        f.write(out)
    os.replace(tmp_path, log_path)
    obj, tsolve, status = parse_output(solver, out)
    if outcome == "error" and note:
        obj = None      # never surface an objective from a guarded failure
        status = note
    if probe_iters:
        obj = None      # a capped barrier's iterate is not a solution
    return {
        "outcome": outcome, "objective": obj, "time_wall": wall,
        "time_solve": tsolve, "status": status, "gpu": gpu, "rc": rc,
        "mem_gb": bs.mem_gb_from_kb(peak_rss_kb), "vram_gb": gb_from_mib(vram_mib),
    }


def enumerate_mps(mps_dir, glob_filter, max_gz_bytes=None, min_gz_bytes=None):
    # Yield (path, key) ordered by compressed file SIZE ascending (cheapest first),
    # NOT alphabetically -- an alpha sort would front-load the giant transportation
    # cities (Austin, BerlinCenter, Birmingham, ChicagoRegional, Philadelphia)
    # before the tiny grids, exactly the wrong order for a memory-cautious sweep.
    # max_gz_bytes, if set, SKIPS (and logs) instances whose .mps.gz exceeds it, so
    # the risky monsters can be deferred to a dedicated pass -- no silent drop.
    import glob as _glob
    import fnmatch
    paths = _glob.glob(os.path.join(mps_dir, "*.mps.gz")) + \
        _glob.glob(os.path.join(mps_dir, "*.mps"))
    for p in sorted(paths, key=lambda q: os.path.getsize(q)):
        key = os.path.basename(p)
        key = key[:-7] if key.endswith(".mps.gz") else key[:-4]
        if glob_filter and not fnmatch.fnmatch(key, glob_filter):
            continue
        sz = os.path.getsize(p)
        if max_gz_bytes is not None and sz > max_gz_bytes:
            sys.stderr.write(
                f"[mps] SKIP {key}: {sz/1e6:.0f} MB gz exceeds "
                f"--max-gz-bytes {max_gz_bytes/1e6:.0f} MB\n")
            continue
        if min_gz_bytes is not None and sz < min_gz_bytes:
            sys.stderr.write(
                f"[mps] SKIP {key}: {sz/1e6:.0f} MB gz below "
                f"--min-gz-bytes {min_gz_bytes/1e6:.0f} MB\n")
            continue
        yield p, key


def verify_mem_guard(guard_args):
    """Probe that the memory cap is actually ENFORCED, not just requested.

    If the cgroup-v2 memory controller isn't delegated to the user manager, a
    `systemd-run --user --scope -p MemoryMax=...` still succeeds (rc 0) but the
    limit is silently ignored (memory.max == "max") -- i.e. solvers run UNGUARDED
    and a 100+ GB barrier can swap-kill the box. Create a throwaway scope with the
    real guard params and read back its effective memory.max. Returns (ok, detail).
    """
    probe = guard_args + [
        "bash", "-c",
        "cat /sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup)/memory.max 2>/dev/null",
    ]
    try:
        r = subprocess.run(probe, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"probe failed to launch ({e})"
    if r.returncode != 0:
        return False, f"probe rc={r.returncode}: {r.stderr.strip()[:160]}"
    val = r.stdout.strip()
    if not val.isdigit():
        return False, f"memory.max not enforced (got {val!r}); controller not delegated?"
    return True, f"memory.max={int(val)} bytes"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mps-dir", default=os.path.join(REPO, "data/mps"))
    ap.add_argument("--solvers", default="highs,mosek,cuopt,copt-cpu,copt-gpu")
    ap.add_argument("--instances", default=None,
                    help="fnmatch glob on the instance key (e.g. 'grid1', 'planar*').")
    ap.add_argument("--time-limit", type=float, default=7200.0,
                    help="solver-side wall-clock budget in seconds (default 2h).")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="relative objective tolerance for pass.")
    ap.add_argument("--out", default=None,
                    help="results CSV; default bench_runs/mps/results.csv "
                         "(bench_runs/mps_probe/results.csv under --probe-iters), "
                         "under the gitignored bench_runs/ tree, not repo root.")
    ap.add_argument("--logdir", default=None,
                    help="per-cell solver logs; default bench_runs/mps/logs "
                         "(bench_runs/mps_probe/logs under --probe-iters), under "
                         "gitignored bench_runs/, not repo root.")
    ap.add_argument("--probe-iters", type=int, default=0, metavar="N",
                    help="MEMORY PROBE: cap the barrier at N iterations instead of "
                         "solving, to measure peak memory at a fraction of the cost "
                         "(a barrier's peak is the ADAT factorization, allocated on "
                         "iteration 1 and reused). Probe runs report no objective, "
                         "carry a `# probe_iters:` log header, and default to their "
                         "own logdir so they can never be consolidated as solves. "
                         "0 (default) = solve normally.")
    ap.add_argument("--mem-max", default="105G",
                    help="per-solve RSS cap enforced via `systemd-run --user "
                         "--scope -p MemoryMax=... -p MemorySwapMax=0` so a runaway "
                         "is OOM-killed cleanly. Empty string or 'off' disables it.")
    ap.add_argument("--max-gz-bytes", type=float, default=None,
                    help="skip (and log) instances whose .mps.gz exceeds this many "
                         "bytes; use e.g. 4e8 to defer the giant cities to a "
                         "separate pass. Suffix-free integer bytes.")
    ap.add_argument("--min-gz-bytes", type=float, default=None,
                    help="skip (and log) instances whose .mps.gz is below this many "
                         "bytes; pair with --max-gz-bytes to run only a size band "
                         "(e.g. re-run just the large tail without redoing the "
                         "already-converged small instances).")
    args = ap.parse_args()

    if not os.path.isdir(args.mps_dir):
        sys.exit(f"[mps] ABORT: --mps-dir is not a directory: {args.mps_dir}")
    if args.probe_iters < 0:
        sys.exit("[mps] ABORT: --probe-iters must be >= 0")
    # Probe output defaults to a SEPARATE tree. Sharing bench_runs/mps/logs would
    # overwrite a real solve's log with an iteration-capped one -- destroying an
    # hours-long measurement and leaving a log whose objective is a mid-barrier
    # iterate. The consolidator also refuses to mix the two, but the strongest
    # protection is not writing them to the same place.
    base = "bench_runs/mps_probe" if args.probe_iters else "bench_runs/mps"
    args.out = args.out or f"{base}/results.csv"
    args.logdir = args.logdir or f"{base}/logs"
    if args.probe_iters:
        sys.stderr.write(
            f"[mps] MEMORY PROBE: barrier capped at {args.probe_iters} iteration(s); "
            "objectives are NOT recorded\n")

    global MEM_GUARD
    if args.mem_max and args.mem_max.lower() != "off":
        # RuntimeMaxSec makes systemd tear down the ENTIRE scope (every process in
        # its cgroup) if a solve hangs past the subprocess timeout, so a runaway
        # barrier can't be orphaned and accumulate RSS against the box across the
        # sweep. Set BEYOND the subprocess timeout (solver's own limit + 30 min
        # grace) by the flush window plus a margin: at the old value of exactly the
        # subprocess timeout the two teardowns fired in the same instant, and
        # systemd's SIGTERMs the `time` wrapper -- destroying the very measurement
        # kill_preserving_mem exists to save. The harness kill must get there first.
        runtime_max = scope_runtime_max_sec(args.time_limit)
        # OOMPolicy=continue is what makes an OOM MEASURABLE. Under the default
        # policy systemd reacts to the cgroup OOM event by stopping the whole
        # scope -- SIGTERM to every process in it, including the GNU `time`
        # wrapper, which then dies before writing its report and the run's peak
        # RSS is lost exactly where it matters most. With `continue`, systemd
        # stays out of it: the kernel OOM killer takes the multi-GB solver (the
        # cap still binds), `time` survives as a ~2 MB bystander and records the
        # high-water mark the solve reached at the cap. Runaways are still bounded
        # by MemoryMax and RuntimeMaxSec.
        MEM_GUARD = [
            "systemd-run", "--user", "--scope", "--quiet",
            "-p", f"MemoryMax={args.mem_max}", "-p", "MemorySwapMax=0",
            "-p", "OOMPolicy=continue",
            "-p", f"RuntimeMaxSec={runtime_max}",
        ]
        ok, detail = verify_mem_guard(MEM_GUARD)
        if not ok:
            sys.exit(
                f"[mps] ABORT: memory guard is NOT effective ({detail}). The "
                "cgroup memory controller may not be delegated to the user "
                "manager -- solvers would run UNGUARDED and a big barrier could "
                "swap-kill the machine. Fix delegation, or re-run with "
                "`--mem-max off` to proceed deliberately unguarded.")
        sys.stderr.write(
            f"[mps] memory guard: MemoryMax={args.mem_max}, MemorySwapMax=0, "
            f"RuntimeMaxSec={runtime_max}s per solve; verified ({detail})\n")
    else:
        sys.stderr.write("[mps] memory guard DISABLED (--mem-max off)\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    unknown = [s for s in solvers if s not in CONFIGS]
    if unknown:
        sys.exit(f"unknown solver config(s): {unknown}; known: {list(CONFIGS)}")
    for s in solvers:
        binkey = parser_key(s)
        b = BINARIES[binkey]
        if not os.path.exists(b):
            sys.stderr.write(f"WARNING: binary for {s} not found at {b}\n")

    refs = load_refs()
    rows = []
    fields = ["instance", "solver", "gpu", "outcome", "objective", "ref",
              "rel_err", "pass", "time_wall", "time_solve", "mem_gb", "vram_gb",
              "status", "detail"]
    summary = {s: {"pass": 0, "fail": 0, "error": 0, "noref": 0} for s in solvers}

    # Write the CSV INCREMENTALLY (header first, then flush after every cell) so a
    # kill / OOM / power loss mid-sweep never discards completed rows -- essential
    # for the multi-hour giant pass where a single HiPO solve can run for 2h. Each
    # per-cell solver log is also already persisted by run_one, so results survive
    # independently of this CSV too.
    out_f = open(args.out, "w", newline="")
    writer = csv.DictWriter(out_f, fieldnames=fields)
    writer.writeheader()
    out_f.flush()

    for mps, key in enumerate_mps(args.mps_dir, args.instances, args.max_gz_bytes,
                                  args.min_gz_bytes):
        ref = refs.get(key)
        for solver in solvers:
            sys.stderr.write(f"[mps] {key} :: {solver} ... ")
            sys.stderr.flush()
            r = run_one(solver, mps, key, args.time_limit, args.logdir,
                        args.probe_iters)
            rec = {"instance": key, "solver": solver, "gpu": r["gpu"],
                   "outcome": r["outcome"], "objective": r["objective"],
                   "ref": "" if ref is None else ref, "rel_err": "", "pass": "",
                   "time_wall": f"{r['time_wall']:.3f}", "time_solve": r["time_solve"],
                   "mem_gb": r["mem_gb"], "vram_gb": r["vram_gb"],
                   "status": r["status"], "detail": ""}
            if args.probe_iters:
                # Nothing to score: the verdict is whether the probe got far enough
                # to yield a memory number.
                rec["detail"] = f"memory probe, barrier capped at {args.probe_iters}"
                ok = r["outcome"] == "ok" and r["mem_gb"] != ""
                summary[solver]["pass" if ok else "error"] += 1
                verdict = f"MEM {r['mem_gb'] or 'NA'} GB" if ok else \
                    (r["outcome"] or "error").upper()
                if r["vram_gb"] not in ("", 0.0):
                    verdict += f" vram={r['vram_gb']} GB"
            elif r["outcome"] != "ok" or r["objective"] is None:
                summary[solver]["error"] += 1
                rec["detail"] = f"rc={r['rc']} status={r['status']}"
                verdict = (r["outcome"] or "error").upper()
            elif ref is None:
                summary[solver]["noref"] += 1
                rec["detail"] = "no reference in optimal.csv"
                verdict = "NOREF"
            else:
                rel = abs(r["objective"] - ref) / max(1.0, abs(ref))
                rec["rel_err"] = rel
                ok = rel < args.tol
                rec["pass"] = ok
                summary[solver]["pass" if ok else "fail"] += 1
                verdict = "PASS" if ok else f"FAIL rel={rel:.2e}"
            ow = r["objective"]
            obj_part = "" if args.probe_iters else \
                f"obj={ow if ow is not None else 'NA'} "
            mem_part = "" if args.probe_iters else f"mem={r['mem_gb'] or 'NA'}GB "
            sys.stderr.write(
                f"{verdict} {obj_part}{mem_part}t={r['time_wall']:.1f}s\n")
            writer.writerow(rec)
            out_f.flush()
            rows.append(rec)
    out_f.close()

    print(f"\nWrote {len(rows)} rows to {args.out} (flushed incrementally)\n")
    first = "measured" if args.probe_iters else "pass"
    print(f"{'solver':<10}{first:>9}{'fail':>5}{'error':>6}{'noref':>6}")
    for s in solvers:
        c = summary[s]
        print(f"{s:<10}{c['pass']:>9}{c['fail']:>5}{c['error']:>6}{c['noref']:>6}")


if __name__ == "__main__":
    main()
