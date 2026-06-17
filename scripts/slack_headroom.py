#!/usr/bin/env python3
"""Slack-cost-ceiling vs cost-scale headroom diagnostic for the mcfcg suite.

The CG master makes infeasibility feasible with SLACK columns whose cost is
bumped up until they leave the basis (see MasterBase in include/mcfcg/cg/).
For that to work — and for the solver not to certify a too-low optimum — the
slack-cost ceiling must dominate the real column costs. The ceiling is
`clamp(10 * slack_cost_upper_bound, 1e6, 1e7)`, where slack_cost_upper_bound is
a WORST-CASE bound (path: |V|*max_arc_cost; tree: *max_src_demand_sum). That
worst case is far looser than typical column costs, so the right risk metric is
the ceiling against a REALISTIC per-source column-cost proxy (optimum / #sources
from the reference), not against slack_cost_upper_bound itself.

This driver runs `mcfcg_cli --stats-only` (no solve) over every instance, joins
the reference optimum, and reports:
  - slack_cost_ceiling vs per-source proxy (optimum / sources)  -> RISKY if <1
  - slack_cost_upper_bound (the loose worst case)               -> CLAMPED flag
  - Flowty's penalty for cross-reference (Sum(costs)+1, per-source *demand,
    tree-capped at order 1e8)

RISKY  = ceiling < per-source proxy: slacks may not out-price real columns, so
         they can stay basic (no early UB) or the LP could certify too low.
CLAMPED= 10*slack_cost_upper_bound > 1e7: the formula ceiling was clamped down;
         usually benign (the bound is loose) but worth noting on high-cost
         instances where it may bind below real costs.

Example:
  python3 scripts/slack_headroom.py --out slack-headroom.csv
  python3 scripts/slack_headroom.py --families planar,intermodal
"""

import argparse
import csv
import fnmatch
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_solvers import (  # noqa: E402
    REPO,
    FAMILY_OPTIMAL,
    enumerate_family,
    load_optimal,
    parse_csv_row,
)

CEILING_CLAMP_HI = 1e7  # MasterBase::init upper clamp


def run_stats(binary, instance, formulation, extra, timeout, solver="highs"):
    """Run mcfcg_cli --stats-only and return the parsed stats dict, or None.

    The reported slack_cost_ceiling reflects the chosen backend's
    LPSolver::max_slack_cost (1e7 for HiGHS/cuOpt, 1e9 for MOSEK/COPT), so pass
    the solver you actually intend to run.  --stats-only never optimizes, but a
    non-HiGHS backend still needs its license/GPU just to be constructed.
    """
    cmd = ([binary, instance, "--formulation", formulation, "--solver", solver, "--stats-only"]
           + extra)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    return parse_csv_row(proc.stdout)


def flowty_penalty(sum_arc_costs, max_src_demand_sum, formulation):
    """Flowty's slack penalty for cross-reference (mcf_model.cpp doPenalty)."""
    penalty = sum_arc_costs + 1.0
    origin_penalty = penalty * max_src_demand_sum
    if formulation == "tree" and max_src_demand_sum > 0.0:
        # Tree cap: scale so log10(maxSumDemands * penalty) <= 8.
        order = math.log10(max(max_src_demand_sum * penalty, 1.0))
        if order > 8.0:
            origin_penalty = origin_penalty / (10.0 ** (order - 8.0))
    return origin_penalty


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--binary", default=os.path.join(REPO, "build/mcfcg_cli"))
    ap.add_argument("--solver", default="highs",
                    help="backend whose slack ceiling to report (highs/copt/cuopt/mosek). "
                         "Default highs needs no license; others must be built in + licensed.")
    ap.add_argument("--families", default="grid,planar,transportation,intermodal")
    ap.add_argument("--instances", default=None, help="fnmatch glob on the ref key to filter.")
    ap.add_argument("--timeout", type=float, default=600.0, help="seconds per --stats-only run.")
    ap.add_argument("--out", default="slack-headroom.csv")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        sys.exit(f"binary not found: {args.binary} (build it first)")

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    fields = ["family", "instance", "formulation", "vertices", "sources", "max_arc_cost",
              "optimum", "per_row_proxy", "slack_cost_upper_bound", "slack_cost_ceiling",
              "ceiling_over_proxy", "flowty_penalty", "flag"]
    rows = []

    for family in families:
        refs = load_optimal(os.path.join(REPO, "data", FAMILY_OPTIMAL[family]))
        for instance, key, formulation, extra in enumerate_family(family):
            if args.instances and not fnmatch.fnmatch(key, args.instances):
                continue
            sys.stderr.write(f"[{family}] {key} :: {formulation} ... ")
            sys.stderr.flush()
            s = run_stats(args.binary, instance, formulation, extra, args.timeout, args.solver)
            if s is None:
                sys.stderr.write("FAILED/timeout\n")
                rows.append({"family": family, "instance": key, "formulation": formulation,
                             "flag": "ERROR"})
                continue

            sources = int(s["sources"])
            ceiling = float(s["slack_cost_ceiling"])
            ub = float(s["slack_cost_upper_bound"])
            total_demand = float(s["total_demand"])
            opt = refs.get(key)
            # Realistic per-row column-cost proxy, matched to the formulation's
            # slack placement: path slacks sit on per-commodity demand rows and
            # compete PER UNIT OF DEMAND (proxy = opt/total_demand); tree slacks
            # sit on per-source convexity rows and compete against the whole
            # demand-weighted tree (proxy = opt/sources).  Using the wrong
            # denominator (e.g. opt/sources for path) over-flags by orders of
            # magnitude.
            if opt is None:
                per_row = None
            elif formulation == "tree":
                per_row = (opt / sources) if sources > 0 else None
            else:
                per_row = (opt / total_demand) if total_demand > 0 else None
            ratio = (ceiling / per_row) if per_row else None
            flowty = flowty_penalty(float(s["sum_arc_costs"]), float(s["max_src_demand_sum"]),
                                    formulation)

            flag = "OK"
            if per_row is not None and ceiling < per_row:
                flag = "RISKY"  # ceiling below typical per-row column cost
            elif 10.0 * ub > CEILING_CLAMP_HI:
                flag = "CLAMPED"  # formula ceiling clamped down (usually benign)

            rec = {"family": family, "instance": key, "formulation": formulation,
                   "vertices": s["vertices"], "sources": sources,
                   "max_arc_cost": float(s["max_arc_cost"]),
                   "optimum": opt if opt is not None else "",
                   "per_row_proxy": per_row if per_row is not None else "",
                   "slack_cost_upper_bound": ub, "slack_cost_ceiling": ceiling,
                   "ceiling_over_proxy": ratio if ratio is not None else "",
                   "flowty_penalty": flowty, "flag": flag}
            rows.append(rec)
            sys.stderr.write(
                f"{flag} ceiling={ceiling:.2g} per_row={per_row:.2g}\n"
                if per_row else f"{flag} ceiling={ceiling:.2g}\n"
            )

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out}\n")
    # Summary: risky first, then clamped, then ok.
    order = {"RISKY": 0, "ERROR": 1, "CLAMPED": 2, "OK": 3}
    rows.sort(key=lambda r: (order.get(r.get("flag", "OK"), 9), r["family"], r["instance"]))
    print(f"{'flag':<8} {'instance':<16} {'form':<5} {'ceiling':>10} {'per_row':>10} "
          f"{'ratio':>8}")
    for r in rows:
        if r.get("flag") in ("ERROR",):
            print(f"{r['flag']:<8} {r['instance']:<16} {r.get('formulation',''):<5}")
            continue
        ps = r["per_row_proxy"]
        ratio = r["ceiling_over_proxy"]
        print(f"{r['flag']:<8} {r['instance']:<16} {r['formulation']:<5} "
              f"{r['slack_cost_ceiling']:>10.2g} "
              f"{(ps if ps != '' else float('nan')):>10.2g} "
              f"{(ratio if ratio != '' else float('nan')):>8.2g}")


if __name__ == "__main__":
    main()
