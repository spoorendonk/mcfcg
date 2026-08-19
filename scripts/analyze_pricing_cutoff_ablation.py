#!/usr/bin/env python3
"""Summarize the bounded-pricing A/B ablation (gh #41, manuscript section 3.3).

Bounded pricing (`mcfcg_cli --bounded-pricing`) stops a source's A* once the
frontier proves no negative-reduced-cost column remains. It is exact — the
emitted column set is identical bit-for-bit, pinned by
`FeatureTests.BoundedPricingShadow*` — so
the only question is whether it is FASTER, and the answer decided that it ships
off by default. This tool is the measurement behind that decision, and it stays
in the tree so the numbers in the manuscript can be regenerated from the logs.

An ablation sweep is a directory of `logs_<solver>_<off|on>_<repN>/` produced by
running `benchmark_solvers.py` twice per repetition, identically except for
`--extra-args=--bounded-pricing`:

    for rep in 1 2 3; do
      for arm in off on; do
        [ "$arm" = on ] && extra=--extra-args=--bounded-pricing || extra=
        python3 scripts/benchmark_solvers.py --families intermodal --solvers copt-cpu \
            $extra --out   bench_runs/SWEEP/copt-cpu_${arm}_rep${rep}.csv \
                   --logdir bench_runs/SWEEP/logs_copt-cpu_${arm}_rep${rep}
      done
    done

Both arms must run in the SAME session on the SAME build: cross-session wall
clock is not comparable (drift was observed between sessions on cells the bound
provably did not touch -- see results/ablation/README.md for the magnitude), so
an "off" arm taken from an archived sweep or from results/cg_benchmark.csv
silently confounds the flag with that drift.

## The metric that matters

`t_PR` alone is NOT a pricing measurement. The bound shifts the CG trajectory
(see the CLAUDE.md bounded-pricing notes for the two channels), so an arm can price fewer
sources and post a lower `t_PR` while every individual price cost the same. The
first intermodal result reported this way was wrong for exactly that reason.

`per_price_us` = `t_PR / priced_sources` is the trajectory-immune metric, and
`priced` comes from the `[bounded-pricing] cut=... priced=...` line every run
emits (with `enabled=0`, so off-arm logs carry it too). Two flags say whether a
cell's wall delta can be read as a pricing effect at all:

  * `traj_moved` -- the arms' MEDIAN iteration or column count differs, so the
    delta is dominated by +/-Delta-iterations rather than by pricing;
  * `traj_stable` -- every repetition *within* each arm agreed on (iterations,
    columns). A cell can have `traj_moved=0` while individual reps disagreed and
    the medians happened to coincide, which is not the same thing. Quote per-price
    numbers over `traj_moved=0 AND traj_stable=1`.

`pred_wall_pct` is the cost model's first term, `pricing_share x per-price
saving` -- the gain the bound can deliver when the trajectory holds still. On
the four copt-cpu intermodal cells that qualify it predicted
-2.3/-2.5/-1.6/-2.1% against measured -2.3/-2.4/-1.7/-2.3%, i.e. within 0.13pp.
The model's second term, `-LP_share x Delta-iterations`, is what makes the flag
backend-specific and is not predictable per instance; it is why a single-backend
measurement of this flag is worthless.

`d_obj_rel` is the exactness check the ablation data can make on its own: the two
arms must agree on the LP optimum. It stays under 7e-5 on all 74 cells, within
the CG gap tolerance. (The bit-for-bit column identity claim is a stronger
statement and is pinned in C++, by FeatureTests.BoundedPricingShadow*.)

Usage:
  # regenerate the committed ablation CSVs from the three paired sweeps
  python3 scripts/analyze_pricing_cutoff_ablation.py

  # a fresh sweep, print only
  python3 scripts/analyze_pricing_cutoff_ablation.py bench_runs/mysweep --no-write
"""

import argparse
import csv
import glob
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_solvers import REPO, parse_csv_row, parse_iteration_table  # noqa: E402

# The three paired same-session sweeps behind the manuscript's section 3.3
# numbers. Unlike the main benchmark, whose logs live in bench_runs/ (gitignored,
# regenerable by re-running benchmark_solvers.py), these are TRACKED: the ablation is
# a one-off measurement we do not intend to repeat, so its logs are the primary
# evidence and ship with the repo. Order is presentation order -- nothing
# supersedes anything here, unlike consolidate_cg_logs.py's --logdir list.
DEFAULT_SWEEPS = [
    # intermodal, tree + PricerHeavy, copt-cpu (3 reps) and copt-gpu (2 reps)
    "results/ablation/logs/intermodal_tree",
    # transportation, tree, copt-gpu, 3 reps
    "results/ablation/logs/transportation_tree",
    # grid + planar<=1000, path and tree, copt-gpu, 3 reps
    "results/ablation/logs/gridplanar_path_tree",
]

RUNS_CSV = "results/ablation/pricing_cutoff_runs.csv"
SUMMARY_CSV = "results/ablation/pricing_cutoff_summary.csv"

SUMMARY_LINE = re.compile(
    r"CG optimal after (\d+) iterations\. UB=(\S+) LB=(\S+) gap=(\S+) tol=(\S+)\s+"
    r"t_LP=(\S+)\s+t_PR=(\S+)\s+t_SP=(\S+)\s+t_Tot=(\S+)")
# `enabled=` is absent in logs from before the banner carried it; those runs are
# all bounded-on arms, so a missing flag is not ambiguous, but prefer the field.
# Both tag spellings are accepted: the flag was renamed --pricing-cutoff ->
# --bounded-pricing in gh #42, and every tracked log predating that carries the
# old tag. Neither spelling may be dropped while both log sets are evidence.
CUTOFF_LINE = re.compile(
    r"\[(?:pricing-cutoff|bounded-pricing)\] (?:enabled=(\d+) )?cut=(\d+) priced=(\d+)")
# The exit type comes from the last iteration row's '+col' count, which carries a
# '*' when the loop returned via the gap test rather than by exhausting pricing;
# benchmark_solvers.parse_iteration_table reports that as col_committed=False.
# Reuse its header-driven parser rather than a positional regex -- a column added
# to the CG log would silently make a positional match fail and then label every
# run "priced-out".
INSTANCE_LINE = re.compile(
    r"^Instance: (\d+) vertices, (\d+) arcs, (\d+) commodities, (\d+) sources", re.M)

RUN_FIELDS = [
    "sweep", "family", "instance", "formulation", "solver", "arm", "rep",
    "commodities", "sources", "per_source",
    "iterations", "columns", "objective", "lower_bound", "optimal", "exit",
    "t_tot", "t_lp", "t_pr", "t_sp",
    "cutoff_enabled", "cut", "priced", "cut_rate_pct", "per_price_us",
]

SUMMARY_FIELDS = [
    "sweep", "family", "instance", "formulation", "solver",
    "per_source", "reps_off", "reps_on",
    "t_tot_off", "t_tot_on", "d_t_tot_pct",
    "t_pr_off", "t_pr_on", "d_t_pr_pct",
    "per_price_us_off", "per_price_us_on", "d_per_price_pct",
    "iters_off", "iters_on", "cols_off", "cols_on",
    "priced_off", "priced_on", "traj_moved", "traj_stable",
    "pricing_share_pct", "cut_rate_pct", "pred_wall_pct",
    "obj_off", "obj_on", "d_obj_rel",
    "lb_delta", "exit_off", "exit_on", "spread_off_pct",
]

# Median counts are written as ints when they land on one, so a consumer can
# int() the column. An even rep count makes statistics.median average the two
# middles, which is how a 2-rep cell can otherwise emit "26.0".
INT_FIELDS = {"iters_off", "iters_on", "cols_off", "cols_on",
              "priced_off", "priced_on"}


def parse_log(path):
    """Extract one ablation record from a benchmark_solvers.py run log."""
    with open(path, errors="ignore") as fh:
        text = fh.read()

    rec = {}
    inst = INSTANCE_LINE.search(text)
    if inst:
        rec["commodities"] = int(inst.group(3))
        rec["sources"] = int(inst.group(4))
        rec["per_source"] = rec["commodities"] / max(1, rec["sources"])

    row = parse_csv_row(text) or {}
    for key in ("iterations", "columns"):
        if row.get(key):
            rec[key] = int(row[key])
    for key in ("objective", "lower_bound"):
        if row.get(key):
            rec[key] = float(row[key])
    if row.get("optimal"):
        rec["optimal"] = int(row["optimal"])

    m = SUMMARY_LINE.search(text)
    if not m:
        # Timed out, was killed, or hit --max-iters: no usable timing breakdown.
        rec["exit"] = "non-optimal"
        return rec
    rec["iterations"] = int(m.group(1))
    rec["lower_bound"] = float(m.group(3))
    rec["t_lp"], rec["t_pr"], rec["t_sp"], rec["t_tot"] = (
        float(m.group(i)) for i in range(6, 10))

    iters = parse_iteration_table(text)
    rec["exit"] = "gap" if iters and not iters[-1]["col_committed"] else "priced-out"

    c = CUTOFF_LINE.search(text)
    if c:
        rec["cutoff_enabled"] = int(c.group(1)) if c.group(1) is not None else 1
        rec["cut"], rec["priced"] = int(c.group(2)), int(c.group(3))
        # Leave both rates absent rather than 0.0 when nothing was priced: a fake
        # zero is exactly the silent failure this whole tier guards against.
        if rec["priced"]:
            rec["cut_rate_pct"] = 100.0 * rec["cut"] / rec["priced"]
            rec["per_price_us"] = 1e6 * rec["t_pr"] / rec["priced"]
    return rec


def collect(sweeps):
    """Parse every log under every sweep dir into flat run records."""
    runs = []
    seen_names = {}
    for sweep in sweeps:
        # Cells are keyed by the sweep's basename, so two sweep dirs sharing one
        # would merge silently (consolidate_cg_logs.py hit exactly this).
        name = os.path.basename(os.path.normpath(sweep))
        if seen_names.setdefault(name, sweep) != sweep:
            sys.exit(f"error: sweep basename '{name}' is used by both "
                     f"{seen_names[name]} and {sweep}; their cells would merge.")
        logdirs = sorted(glob.glob(os.path.join(sweep, "logs_*")))
        if not logdirs:
            print(f"warning: no logs_<solver>_<arm>_<rep>/ dirs under {sweep}",
                  file=sys.stderr)
        for logdir in logdirs:
            tag = os.path.basename(logdir)[len("logs_"):]
            try:
                _dir_solver, arm, rep = tag.rsplit("_", 2)
            except ValueError:
                print(f"warning: skipping unparseable log dir {logdir}", file=sys.stderr)
                continue
            if arm not in ("off", "on"):
                print(f"warning: skipping {logdir}: arm is '{arm}', not off/on",
                      file=sys.stderr)
                continue
            for log in sorted(glob.glob(os.path.join(logdir, "*.log"))):
                parts = os.path.basename(log).removesuffix(".log").split("__")
                if len(parts) != 4:
                    print(f"warning: skipping unparseable log name {log}", file=sys.stderr)
                    continue
                family, instance, formulation, log_solver = parts
                rec = parse_log(log)
                # The arm label lives in the directory name; the log itself
                # records what actually ran. A mislabelled dir pairs an arm
                # against itself and reports a 0% effect, which looks like a
                # result rather than like a mistake.
                want = 1 if arm == "on" else 0
                if rec.get("cutoff_enabled") not in (None, want):
                    print(f"warning: {log} sits in an '{arm}' dir but reports "
                          f"enabled={rec['cutoff_enabled']}", file=sys.stderr)
                rec.update(sweep=name, family=family,
                           instance=instance, formulation=formulation,
                           solver=log_solver, arm=arm, rep=rep)
                runs.append(rec)
    return runs


def med(vals):
    return statistics.median(vals) if vals else None


def pct(new, old):
    """Relative change new-vs-old in percent, or None when it is undefined."""
    if old in (None, 0) or new is None:
        return None
    return 100.0 * (new - old) / old


def summarize(runs):
    """Pair the off and on arms per cell and reduce each to medians."""
    cells = {}
    for r in runs:
        key = (r["sweep"], r["family"], r["instance"], r["formulation"], r["solver"])
        cells.setdefault(key, {"off": [], "on": []})[r["arm"]].append(r)

    rows = []
    for key, arms in sorted(cells.items()):
        # A run with no timing breakdown cannot be paired; keep it out of the
        # medians rather than let a partial record skew them.
        off = [r for r in arms["off"] if "t_pr" in r]
        on = [r for r in arms["on"] if "t_pr" in r]
        if not off or not on:
            print(f"warning: unpaired cell {key} (off={len(off)} on={len(on)}), skipped",
                  file=sys.stderr)
            continue

        def m(recs, field):
            return med([r[field] for r in recs if r.get(field) is not None])

        row = dict(zip(("sweep", "family", "instance", "formulation", "solver"), key))
        row["per_source"] = off[0].get("per_source")
        row["reps_off"], row["reps_on"] = len(off), len(on)
        for field, name in (("t_tot", "t_tot"), ("t_pr", "t_pr"),
                            ("per_price_us", "per_price_us"), ("iterations", "iters"),
                            ("columns", "cols"), ("priced", "priced")):
            row[f"{name}_off"], row[f"{name}_on"] = m(off, field), m(on, field)
        row["d_t_tot_pct"] = pct(row["t_tot_on"], row["t_tot_off"])
        row["d_t_pr_pct"] = pct(row["t_pr_on"], row["t_pr_off"])
        row["d_per_price_pct"] = pct(row["per_price_us_on"], row["per_price_us_off"])
        row["traj_moved"] = int(row["iters_off"] != row["iters_on"]
                                or row["cols_off"] != row["cols_on"])
        # traj_moved compares MEDIANS, so it reads 0 for a cell whose reps
        # disagreed but whose medians happened to coincide. traj_stable says the
        # reps within each arm actually agreed; per-price numbers are only worth
        # quoting when both hold.
        def shape(recs):
            return {(r.get("iterations"), r.get("columns")) for r in recs}
        row["traj_stable"] = int(len(shape(off)) == 1 and len(shape(on)) == 1)
        row["pricing_share_pct"] = (100.0 * row["t_pr_off"] / row["t_tot_off"]
                                    if row["t_tot_off"] else None)
        row["cut_rate_pct"] = med([r.get("cut_rate_pct") for r in on
                                   if r.get("cut_rate_pct") is not None])
        # Cost model, first term only: a saving on each price, scaled by how much
        # of the wall clock pricing actually is. Second term (-LP_share x
        # Delta-iterations) is not predictable per instance -- see traj_moved.
        if row["pricing_share_pct"] is not None and row["d_per_price_pct"] is not None:
            row["pred_wall_pct"] = row["pricing_share_pct"] * row["d_per_price_pct"] / 100.0
        else:
            row["pred_wall_pct"] = None
        # The two arms must land on the same LP optimum -- the ablation's own
        # check on the exactness claim (the bit-for-bit column identity is a
        # stronger statement, pinned in C++ by the shadow tests).
        row["obj_off"], row["obj_on"] = m(off, "objective"), m(on, "objective")
        if row["obj_off"] not in (None, 0) and row["obj_on"] is not None:
            row["d_obj_rel"] = abs(row["obj_on"] - row["obj_off"]) / abs(row["obj_off"])
        else:
            row["d_obj_rel"] = None
        lb_off, lb_on = m(off, "lower_bound"), m(on, "lower_bound")
        row["lb_delta"] = (lb_on - lb_off) if (lb_off is not None and lb_on is not None) else None
        row["exit_off"] = "/".join(sorted({r["exit"] for r in off}))
        row["exit_on"] = "/".join(sorted({r["exit"] for r in on}))
        # Rep-to-rep spread of the BASELINE arm: the noise floor any on-vs-off
        # delta has to clear before it means anything.
        base = [r["t_pr"] for r in off]
        row["spread_off_pct"] = (100.0 * (max(base) - min(base)) / med(base)
                                 if len(base) > 1 and med(base) else None)
        rows.append(row)
    return rows


def fmt(val, spec, dash="-"):
    return format(val, spec) if val is not None else f"{dash:>{len(format(0, spec))}}"


def print_tables(rows):
    """One table per (sweep, solver, formulation), sorted by commodities/source.

    That sort order is the point: the bound prunes the tail of a multi-target
    search, so its per-price saving tracks how many sinks a source carries --
    while the pricing share that turns that saving into wall clock does not.
    """
    groups = sorted({(r["sweep"], r["solver"], r["formulation"]) for r in rows})
    for sweep, solver, formulation in groups:
        sel = [r for r in rows
               if (r["sweep"], r["solver"], r["formulation"]) == (sweep, solver, formulation)]
        nrep = max(r["reps_off"] for r in sel)
        print(f"\n=== {sweep} :: {solver} / {formulation} (median of {nrep} reps) ===")
        print(f"{'instance':<16} {'k/src':>7} {'per-price off':>13} {'on':>9} {'d%':>7} "
              f"{'PR share':>9} {'t_tot off':>10} {'on':>9} {'d%':>7} {'pred%':>7} "
              f"{'iters':>11} {'cut%':>6} {'moved':>6} {'stable':>7}")
        for r in sorted(sel, key=lambda r: (r["per_source"] or 0.0, r["instance"])):
            print(f"{r['instance']:<16} {fmt(r['per_source'], '7.1f')} "
                  f"{fmt(r['per_price_us_off'], '13.4f')} {fmt(r['per_price_us_on'], '9.4f')} "
                  f"{fmt(r['d_per_price_pct'], '+6.1f')}% {fmt(r['pricing_share_pct'], '8.1f')}% "
                  f"{fmt(r['t_tot_off'], '10.3f')} {fmt(r['t_tot_on'], '9.3f')} "
                  f"{fmt(r['d_t_tot_pct'], '+6.1f')}% {fmt(r['pred_wall_pct'], '+6.2f')}% "
                  f"{fmt(r['iters_off'], '5.0f')}->{fmt(r['iters_on'], '<4.0f')} "
                  f"{fmt(r['cut_rate_pct'], '5.1f')}% {r['traj_moved']:>6} {r['traj_stable']:>7}")

        tt_off = sum(r["t_tot_off"] for r in sel)
        tt_on = sum(r["t_tot_on"] for r in sel)
        pr_off = sum(r["t_pr_off"] for r in sel)
        pr_on = sum(r["t_pr_on"] for r in sel)
        # Widths track the header above: 7 pads the d% slot (6 + '%'), and the
        # t_PR row's 60 lands its total under `t_tot off`, not under `per-price on`.
        print(f"{'TOTAL':<16} {'':>7} {'':>13} {'':>9} {'':>7} "
              f"{100 * pr_off / tt_off:8.1f}% {tt_off:10.3f} {tt_on:9.3f} "
              f"{pct(tt_on, tt_off):+6.1f}%")
        print(f"{'  of which t_PR':<16} {pr_off:60.3f} {pr_on:9.3f} "
              f"{pct(pr_on, pr_off):+6.1f}%")

        spreads = [r["spread_off_pct"] for r in sel if r["spread_off_pct"] is not None]
        if spreads:
            print(f"  baseline t_PR rep-to-rep spread (max-min)/median: "
                  f"median {med(spreads):.1f}%, worst {max(spreads):.1f}%")
        moved = sum(r["traj_moved"] for r in sel)
        # A cell needs a per-price delta to be quotable at all: a run whose log
        # carried no bounded-pricing banner has no `priced`, hence no metric.
        quotable = [r for r in sel if not r["traj_moved"] and r["traj_stable"]
                    and r["d_per_price_pct"] is not None]
        print(f"  trajectory moved on {moved}/{len(sel)} cells "
              f"(wall delta there is +/-Delta-iterations, not pricing)")
        if quotable:
            print(f"  quotable per-price (moved=0 and stable=1): {len(quotable)}/{len(sel)} "
                  f"cells, median {med([r['d_per_price_pct'] for r in quotable]):+.1f}%")
        else:
            print(f"  quotable per-price (moved=0 and stable=1): none of {len(sel)} cells")


def cell(row, key):
    val = row.get(key)
    if val is None:
        return ""
    if key in INT_FIELDS and isinstance(val, float) and val.is_integer():
        return int(val)
    return val


def write_csv(path, fields, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in sorted(rows, key=lambda r: [str(r.get(f, "")) for f in fields[:7]]):
            w.writerow({k: cell(row, k) for k in fields})
    print(f"\nWrote {len(rows)} rows to {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweeps", nargs="*", default=None,
                    help="ablation sweep directories, each holding "
                         "logs_<solver>_<off|on>_<rep>/ subdirs. Defaults to the three "
                         "paired sweeps behind the committed CSVs.")
    ap.add_argument("--runs-out", default=RUNS_CSV,
                    help=f"per-run CSV, one row per log (default: {RUNS_CSV})")
    ap.add_argument("--summary-out", default=SUMMARY_CSV,
                    help=f"paired per-cell CSV, off vs on (default: {SUMMARY_CSV})")
    ap.add_argument("--no-write", action="store_true",
                    help="print the tables only; write no CSVs")
    args = ap.parse_args()

    # The default output paths ARE the committed ablation record, so they are
    # only valid for the default sweeps. A custom sweep must name its own outputs
    # (or pass --no-write) rather than silently clobber the tracked evidence.
    if args.sweeps and not args.no_write and (args.runs_out, args.summary_out) == (
            RUNS_CSV, SUMMARY_CSV):
        sys.exit("error: custom sweep dirs with the default --runs-out/--summary-out "
                 "would overwrite the committed ablation CSVs. Pass --no-write, or "
                 "give your own --runs-out/--summary-out.")

    sweeps = args.sweeps or [os.path.join(REPO, s) for s in DEFAULT_SWEEPS]
    missing = [s for s in sweeps if not os.path.isdir(s)]
    if missing:
        sys.exit(f"error: sweep dir(s) not found: {', '.join(missing)}\n"
                 f"See results/ablation/README.md for the sweep layout.")

    runs = collect(sweeps)
    if not runs:
        sys.exit("error: no run logs found")
    rows = summarize(runs)
    print_tables(rows)
    if not args.no_write:
        write_csv(os.path.join(REPO, args.runs_out), RUN_FIELDS, runs)
        write_csv(os.path.join(REPO, args.summary_out), SUMMARY_FIELDS, rows)


if __name__ == "__main__":
    main()
