#!/usr/bin/env python3
"""The bounded-pricing ablation's failure modes are all sign errors and silent zeros.

Fourth file in this tier (see test/python/log_headers_test.py for the template).
This analyzer is the only consumer of the one tracked log set in the repo, and
the numbers it derives are the sole evidence for shipping bounded pricing
disabled. Every way it can be wrong is quiet:

  * `per_price_us` built from the wrong counter -- `cut` instead of `priced`, say
    -- still produces a plausible column, but it is no longer the
    trajectory-immune metric the whole argument rests on;
  * an off-arm log read as an on-arm one (the `enabled=` field is optional in
    older logs) pairs an arm against itself and reports a 0% effect;
  * off and on swapped anywhere in the pairing flips the sign of every delta,
    turning "rejected" into "adopt it".

Run:  python3 -m unittest discover -s test/python -p '*_test.py'
"""

import contextlib
import csv
import io
import os
import shutil
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import analyze_bounded_pricing_ablation as abl  # noqa: E402

# A minimal but real-shaped run log: the CLI's 2-line result CSV, the summary
# line, two iteration rows, and the bounded-pricing banner.
LOG_TEMPLATE = """\
# cmd: build/mcfcg_cli data/intermodal/BUS-2632-0.txt.gz --formulation tree
Instance: 119865 vertices, 397362 arcs, 5256 commodities, 2628 sources
   It           UB           LB       LP_obj   #col   #row  #slk   +col   -col   +cut   -cut    t_LP    t_PR    t_SP   t_Tot     t_acc
    1   1.0000e+00   0.0000e+00   1.0000e+00    100     10     0    100      0      0      0   0.100   0.200   0.000   0.300     0.300
    2   1.0000e+00   1.0000e+00   1.0000e+00    120     10     0    {last_col}      0      0      0   0.100   0.200   0.000   0.300     0.600
CG optimal after 2 iterations. UB=71026.500000 LB=71021.870000 gap=1.0e+00 tol=1.0e+00  \
t_LP=0.200  t_PR={t_pr}  t_SP=0.000  t_Tot={t_tot}
[bounded-pricing] enabled={enabled} cut={cut} priced={priced} rate=0.0%
instance,formulation,iterations,columns,objective,lower_bound,optimal,time,time_lp,time_pricing,time_separation
data/intermodal/BUS-2632-0.txt.gz,tree,2,{columns},71026.500000,71021.870000,1,{t_tot},0.200,{t_pr},0.000
"""


def write_log(path, *, t_pr="1.000", t_tot="2.000", enabled=0, cut=0, priced=500,
              columns=120, last_col="*20"):
    with open(path, "w") as fh:
        fh.write(LOG_TEMPLATE.format(t_pr=t_pr, t_tot=t_tot, enabled=enabled, cut=cut,
                                     priced=priced, columns=columns, last_col=last_col))


class ParseLogTest(unittest.TestCase):
    """One log in, one record out — with the fields the argument depends on."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="mcfcg_abl_test_")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def parse(self, **kw):
        p = os.path.join(self.dir, "intermodal__BUS-2632-0__tree__copt-cpu.log")
        write_log(p, **kw)
        return abl.parse_log(p)

    def test_per_price_divides_by_priced_not_by_cut(self):
        """The metric is t_PR per source PRICED. Dividing by `cut` would track the
        fire rate instead and silently reward a bound that fires more."""
        rec = self.parse(t_pr="1.000", priced=500, cut=250)
        self.assertAlmostEqual(rec["per_price_us"], 1e6 * 1.000 / 500)
        self.assertAlmostEqual(rec["cut_rate_pct"], 50.0)

    def test_enabled_field_is_read(self):
        self.assertEqual(self.parse(enabled=0)["bounded_enabled"], 0)
        self.assertEqual(self.parse(enabled=1)["bounded_enabled"], 1)

    def test_a_stale_banner_yields_no_priced_rather_than_a_wrong_one(self):
        """Forward-only (gh #42): the old `[pricing-cutoff]` tag and the
        pre-`enabled=` banner format are both rejected outright. The failure has
        to be absence, not a plausible default -- a record silently carrying
        `priced` from an unrecognised line would feed per_price_us, the metric
        the whole argument rests on."""
        current = "[bounded-pricing] enabled=1 cut=250 priced=500 rate=0.0%"
        for stale in ("[pricing-cutoff] enabled=1 cut=250 priced=500 rate=0.0%",
                      "[bounded-pricing] cut=250 priced=500 rate=0.0%"):
            with self.subTest(stale=stale):
                path = os.path.join(self.dir, "intermodal__BUS-2632-0__tree__copt-cpu.log")
                write_log(path, enabled=1, cut=250, priced=500)
                with open(path) as fh:
                    text = fh.read()
                self.assertIn(current, text, "fixture banner drifted from the parser")
                with open(path, "w") as fh:
                    fh.write(text.replace(current, stale))
                rec = abl.parse_log(path)
                self.assertNotIn("priced", rec)
                self.assertNotIn("bounded_enabled", rec)

    def test_exit_type_from_the_iteration_table(self):
        # '*' on the last row's +col marks the gap exit; a bare count means the
        # loop ran out of columns to price.
        self.assertEqual(self.parse(last_col="*20")["exit"], "gap")
        self.assertEqual(self.parse(last_col="0")["exit"], "priced-out")

    def test_timings_and_instance_shape(self):
        rec = self.parse(t_pr="1.500", t_tot="4.000")
        self.assertAlmostEqual(rec["t_pr"], 1.5)
        self.assertAlmostEqual(rec["t_tot"], 4.0)
        self.assertEqual(rec["commodities"], 5256)
        self.assertEqual(rec["sources"], 2628)
        self.assertAlmostEqual(rec["per_source"], 2.0)

    def test_run_without_a_summary_line_is_not_silently_zero(self):
        """A timed-out run has no timing breakdown; it must be unusable, not 0.0,
        or it would drag a median toward zero and manufacture a speedup."""
        p = os.path.join(self.dir, "intermodal__BUS-2632-0__tree__copt-cpu.log")
        write_log(p)
        with open(p) as fh:
            text = "\n".join(ln for ln in fh.read().splitlines()
                             if not ln.startswith("CG optimal"))
        with open(p, "w") as fh:
            fh.write(text)
        rec = abl.parse_log(p)
        self.assertEqual(rec["exit"], "non-optimal")
        self.assertNotIn("t_pr", rec)


def run(arm, **kw):
    base = dict(sweep="s", family="intermodal", instance="BUS-2632-0",
                formulation="tree", solver="copt-cpu", arm=arm, rep="rep1",
                per_source=1.0, exit="gap", lower_bound=100.0,
                t_tot=2.0, t_pr=1.0, per_price_us=100.0, iterations=10,
                columns=1000, priced=500)
    base.update(kw)
    return base


class CollectTest(unittest.TestCase):
    """The arm label comes from a directory NAME, which is the weakest link.

    Nothing downstream can tell that `logs_copt-cpu_on_rep1` actually held
    bounded-off runs; it would just pair an arm against itself and report a 0%
    effect. These run on a synthetic tree so the coverage survives in a checkout
    without the tracked ablation logs.
    """

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="mcfcg_abl_collect_")
        self.sweep = os.path.join(self.root, "mysweep")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def add(self, logdir, name="intermodal__BUS-2632-0__tree__copt-cpu.log", **kw):
        d = os.path.join(self.sweep, logdir)
        os.makedirs(d, exist_ok=True)
        write_log(os.path.join(d, name), **kw)

    def test_arm_solver_and_rep_come_off_the_directory_name(self):
        self.add("logs_copt-cpu_off_rep1", enabled=0)
        self.add("logs_copt-cpu_on_rep2", enabled=1)
        runs = abl.collect([self.sweep])
        self.assertEqual({(r["arm"], r["rep"]) for r in runs},
                         {("off", "rep1"), ("on", "rep2")})
        self.assertEqual({r["sweep"] for r in runs}, {"mysweep"})
        # solver/family/formulation come off the FILE name, not the dir.
        self.assertEqual({r["solver"] for r in runs}, {"copt-cpu"})
        self.assertEqual({r["formulation"] for r in runs}, {"tree"})
        self.assertEqual({r["family"] for r in runs}, {"intermodal"})

    def test_a_log_contradicting_its_arm_directory_warns(self):
        self.add("logs_copt-cpu_off_rep1", enabled=1)  # bounded-on run in an off dir
        with contextlib.redirect_stderr(io.StringIO()) as err:
            abl.collect([self.sweep])
        self.assertIn("reports enabled=1", err.getvalue())

    def test_unusable_directory_and_file_names_are_skipped_loudly(self):
        self.add("logs_copt-cpu_off_rep1")
        self.add("logs_copt-cpu_sideways_rep1")           # arm is not off/on
        self.add("logs_copt-cpu_off_rep2", name="not__enough.log")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            runs = abl.collect([self.sweep])
        self.assertEqual(len(runs), 1)
        self.assertIn("arm is 'sideways'", err.getvalue())
        self.assertIn("unparseable log name", err.getvalue())

    def test_colliding_sweep_basenames_are_refused(self):
        """Cells are keyed by basename, so two sweeps named alike would merge."""
        self.add("logs_copt-cpu_off_rep1")
        other = os.path.join(self.root, "elsewhere", "mysweep")
        os.makedirs(os.path.join(other, "logs_copt-cpu_on_rep1"))
        write_log(os.path.join(other, "logs_copt-cpu_on_rep1",
                               "intermodal__BUS-2632-0__tree__copt-cpu.log"), enabled=1)
        with self.assertRaises(SystemExit):
            abl.collect([self.sweep, other])


class SummarizePairingTest(unittest.TestCase):
    """off is the baseline and on is the treatment; every delta is on-vs-off."""

    def test_deltas_are_signed_on_minus_off(self):
        rows = abl.summarize([run("off", t_tot=2.0, t_pr=1.0, per_price_us=100.0),
                              run("on", t_tot=1.8, t_pr=0.9, per_price_us=90.0)])
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertAlmostEqual(r["d_t_tot_pct"], -10.0)
        self.assertAlmostEqual(r["d_t_pr_pct"], -10.0)
        self.assertAlmostEqual(r["d_per_price_pct"], -10.0)
        # A faster `on` arm must read negative -- the whole table is read by sign.
        self.assertLess(r["d_t_tot_pct"], 0.0)

    def test_pricing_share_and_prediction(self):
        rows = abl.summarize([run("off", t_tot=4.0, t_pr=2.0, per_price_us=100.0),
                              run("on", t_tot=4.0, t_pr=2.0, per_price_us=90.0)])
        r = rows[0]
        self.assertAlmostEqual(r["pricing_share_pct"], 50.0)
        # 50% of the clock, 10% cheaper per price -> 5% of the clock.
        self.assertAlmostEqual(r["pred_wall_pct"], -5.0)

    def test_traj_moved_flags_iteration_or_column_changes(self):
        same = abl.summarize([run("off"), run("on")])[0]
        self.assertEqual(same["traj_moved"], 0)
        iters = abl.summarize([run("off", iterations=10), run("on", iterations=12)])[0]
        self.assertEqual(iters["traj_moved"], 1)
        cols = abl.summarize([run("off", columns=1000), run("on", columns=1001)])[0]
        self.assertEqual(cols["traj_moved"], 1)

    def test_unpaired_cell_is_dropped_not_half_reported(self):
        with contextlib.redirect_stderr(io.StringIO()) as err:
            rows = abl.summarize([run("off"), run("off", rep="rep2")])
        self.assertEqual(rows, [])
        self.assertIn("unpaired cell", err.getvalue())  # dropped loudly, not silently

    def test_medians_over_reps(self):
        rows = abl.summarize([run("off", rep="rep1", t_tot=1.0),
                              run("off", rep="rep2", t_tot=2.0),
                              run("off", rep="rep3", t_tot=9.0),
                              run("on", t_tot=2.0)])
        self.assertAlmostEqual(rows[0]["t_tot_off"], 2.0)  # median, not mean
        self.assertEqual(rows[0]["reps_off"], 3)


class ReportingTest(unittest.TestCase):
    """print_tables/write_csv sum and divide raw fields, so a None or a zero
    there is a TypeError at report time -- after the parse work is already done."""

    def test_tables_render_with_missing_optional_fields(self):
        rows = abl.summarize([run("off", per_price_us=None, priced=0),
                              run("on", per_price_us=None, priced=0)])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            abl.print_tables(rows)
        text = out.getvalue()
        self.assertIn("TOTAL", text)
        self.assertIn("none of 1 cells", text)  # nothing quotable without per-price

    def test_write_csv_emits_blanks_not_the_string_none(self):
        rows = abl.summarize([run("off", per_price_us=None), run("on", per_price_us=None)])
        path = os.path.join(tempfile.mkdtemp(prefix="mcfcg_abl_csv_"), "s.csv")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                abl.write_csv(path, abl.SUMMARY_FIELDS, rows)
            with open(path, newline="") as fh:
                row = next(iter(csv.DictReader(fh)))
            self.assertEqual(row["per_price_us_off"], "")
            self.assertNotIn("None", ",".join(row.values()))
            # Counts must stay int-typed so a consumer can int() them.
            self.assertEqual(row["iters_off"], "10")
        finally:
            shutil.rmtree(os.path.dirname(path), ignore_errors=True)


class CommittedAblationTest(unittest.TestCase):
    """The tracked CSVs must still be what the tracked logs say.

    This is the provenance chain the manuscript cites: logs -> analyzer -> CSV.
    A parser change that silently alters a column would otherwise leave the
    committed CSV describing runs it no longer matches.
    """

    @classmethod
    def setUpClass(cls):
        cls.sweeps = [os.path.join(REPO, s) for s in abl.DEFAULT_SWEEPS]
        if not all(os.path.isdir(s) for s in cls.sweeps):
            raise unittest.SkipTest("results/ablation/families logs not present")
        cls.rows = abl.summarize(abl.collect(cls.sweeps))

    def test_every_cell_is_paired_at_the_round_s_uniform_three_reps(self):
        # 3 reps everywhere is the point of round (a): the archived sweep it
        # replaced was mixed-rep, and its 2-rep cells misread the sign of the
        # per-price effect (gh #43). An exact count, not a floor, so a cell
        # silently dropping to 2 fails here rather than being quoted.
        #
        # Pin the totals too: a whole logs_*/ dir going missing would still leave
        # a full rep count on the cells that survive.
        #   288 = 48 grid+planar cells x 6 dirs (3 reps x 2 arms)
        #    36 =  6 transportation x 6
        #   120 = 10 intermodal x 6, on each of copt-gpu and copt-cpu
        self.assertEqual(len(abl.collect(self.sweeps)), 444, "log count changed")
        self.assertEqual(len(self.rows), 74, "cell count changed")
        for r in self.rows:
            self.assertEqual(r["reps_off"], 3, f"{r['sweep']}/{r['instance']} off reps")
            self.assertEqual(r["reps_on"], 3, f"{r['sweep']}/{r['instance']} on reps")

    def test_the_two_arms_agree_on_the_optimum(self):
        """The bound is a proof, not a heuristic: switching it on must not move
        the LP optimum. This is the exactness check the ablation data can make on
        its own (the bit-for-bit column identity is pinned in C++)."""
        for r in self.rows:
            self.assertIsNotNone(r["d_obj_rel"])
            self.assertLess(r["d_obj_rel"], 1e-3,
                            f"{r['sweep']}/{r['instance']} objectives diverged")

    def test_off_arms_never_fired_the_bound(self):
        """The control has to be a control: an off log reporting cuts would mean
        the arms were mislabelled and every delta in the table is meaningless."""
        for rec in abl.collect(self.sweeps):
            # No .get() defaults: a log MISSING the banner would otherwise pass
            # silently, and per_price_us -- the metric the argument rests on --
            # needs `priced` from that same line.
            self.assertIn("priced", rec, f"{rec['sweep']}/{rec['instance']} has no banner")
            if rec["arm"] == "off":
                self.assertEqual(rec["cut"], 0,
                                 f"{rec['sweep']}/{rec['instance']} off arm fired the bound")
                self.assertEqual(rec["bounded_enabled"], 0)
            else:
                self.assertEqual(rec["bounded_enabled"], 1)

    def test_runs_csv_matches_the_logs(self):
        """The per-run CSV carries the fields the summary is built from, so a
        parser change touching only those would slip past a summary-only check."""
        with open(os.path.join(REPO, abl.RUNS_CSV), newline="") as fh:
            committed = list(csv.DictReader(fh))
        derived = {(r["sweep"], r["solver"], r["formulation"], r["instance"],
                    r["arm"], r["rep"]): r for r in abl.collect(self.sweeps)}
        self.assertEqual(len(committed), len(derived))
        for row in committed:
            key = (row["sweep"], row["solver"], row["formulation"], row["instance"],
                   row["arm"], row["rep"])
            self.assertIn(key, derived)
            for field in abl.RUN_FIELDS:
                want = abl.cell(derived[key], field)
                self.assertEqual(row[field], "" if want == "" else str(want),
                                 f"{key} {field} drifted from the logs")

    def test_summary_csv_matches_the_logs(self):
        path = os.path.join(REPO, abl.SUMMARY_CSV)
        with open(path, newline="") as fh:
            committed = list(csv.DictReader(fh))
        self.assertEqual(len(committed), len(self.rows))
        derived = {(r["sweep"], r["solver"], r["formulation"], r["instance"]): r
                   for r in self.rows}
        for row in committed:
            key = (row["sweep"], row["solver"], row["formulation"], row["instance"])
            self.assertIn(key, derived)
            # Every field, not a sample: the numbers the docs quote are the
            # DERIVED columns (pred_wall_pct, d_*_pct), which a sampled check
            # leaves free to drift away from the prose citing them.
            for field in abl.SUMMARY_FIELDS:
                want = abl.cell(derived[key], field)
                self.assertEqual(row[field], "" if want == "" else str(want),
                                 f"{key} {field} drifted from the logs")


if __name__ == "__main__":
    unittest.main()
