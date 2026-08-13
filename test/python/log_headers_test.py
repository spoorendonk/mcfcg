#!/usr/bin/env python3
"""The log header is a serialization format; these are its round-trip tests.

Second file in this tier — the template for adding a third. Nothing here spawns
a process, so it is the cheap half of `Python.Scripts`: pure functions that
decide what reaches a committed results table. A drift between a writer and its
reader is silent (a blank or mislabelled CSV column), which is the same failure
mode test/python/timeout_memory_test.py exists to prevent, one layer down.

Several of these pin claims that are currently only asserted in comments — that
the run-header writer and reader "cannot drift", that a `#` line in the body is
never mistaken for a header, that a provenance tag never survives the peak it
describes.

Run:  python3 -m unittest discover -s test/python -p '*_test.py'
"""

import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import benchmark_mps as bm  # noqa: E402
import benchmark_solvers as bs  # noqa: E402
import consolidate_mps_logs as cm  # noqa: E402

# A benchmark_mps log, as run_one writes one.
LOG = ("# cmd: /usr/bin/time -o /tmp/x -f %M solver model.mps\n"
       "# wall=12.500s rc=0 outcome=ok\n"
       "# peak_rss_kb: 1234\n"
       "# peak_rss_source: measured\n"
       "# === solver output ===\n"
       "# this hash is BODY, not header\n"
       "Objective value : 1.0\n")


class RunHeaderRoundTripTest(unittest.TestCase):
    """format_run_header and RUN_HEADER must not drift; nothing else pairs them."""

    def test_round_trip(self):
        for wall, rc, outcome in [(0.0, 0, "ok"), (12.5, 1, "error"),
                                  (7267.274, -1, "timeout"), (1.0, 137, "error")]:
            with self.subTest(rc=rc):
                got = bm.parse_run_header(bm.format_run_header(wall, rc, outcome))
                self.assertEqual(got, (round(wall, 3), rc, outcome))

    def test_missing_header_reads_as_error(self):
        self.assertEqual(bm.parse_run_header("no header here\n"),
                         (None, None, "error"))


class PeakRssHeaderRoundTripTest(unittest.TestCase):
    """format_peak_rss_headers writes it, parse_peak_rss reads it, nothing else."""

    def _log_with(self, block):
        """A run log carrying `block` as its only memory header."""
        return ("# cmd: solver model.mps\n"
                "# wall=12.500s rc=0 outcome=ok\n"
                + block + cm.MARKER + "\nObjective value : 1.0\n")

    def test_round_trip(self):
        """Including the two relocation tags: the committed logs still carry them.

        A live run only ever writes `measured`, but consolidate_*_logs.py has to
        keep reading `backfilled[-untimed]:` and `probeN[-partial]:` back —
        results/cg_benchmark.csv and results/mps_compact_baseline.csv report them
        in `mem_source` (PROVENANCE.txt sections 1.1 and 2.2), so a reader that
        dropped the vocabulary would silently blank those columns.
        """
        for source in ("measured", "backfilled:bench_runs/a.csv",
                       "backfilled-untimed:bench_runs/a.csv", "probe3:p.log",
                       "probe3-partial:p.log"):
            with self.subTest(source=source):
                text = self._log_with(bs.format_peak_rss_headers(999, source))
                self.assertEqual(bs.parse_peak_rss(text), (999, source))

    def test_nothing_to_record_writes_nothing(self):
        self.assertEqual(bs.format_peak_rss_headers(None, "measured"), "")

    def test_a_source_without_a_peak_is_not_reported(self):
        """The pair is atomic: never a provenance claim for a missing number."""
        text = self._log_with("# peak_rss_source: measured\n")
        self.assertEqual(bs.parse_peak_rss(text), (None, ""))

    def test_a_hash_line_in_the_body_is_not_treated_as_a_header(self):
        self.assertEqual(bs.parse_peak_rss(LOG), (1234, "measured"))
        body_header = LOG + "# peak_rss_kb: 2\n# peak_rss_source: measured\n"
        self.assertEqual(bs.parse_peak_rss(body_header), (1234, "measured"))


class OutcomeGateTest(unittest.TestCase):
    """Guards that decide whether an objective reaches results/."""

    def test_killed_before_solving(self):
        # -9 is the pre-OOMPolicy sweep's convention, 137 the current one; both
        # are the same death and both must suppress the marker-absence guards.
        for rc in (137, 9, -9, -1, None):
            self.assertTrue(cm.killed_before_solving(rc), rc)
        for rc in (0, 1, 160, 255):
            self.assertFalse(cm.killed_before_solving(rc), rc)

    def test_copt_read_failure_is_distinguished_from_a_solve_failure(self):
        self.assertTrue(bm.copt_solve_failed("[ERROR] Reading failed"))
        self.assertTrue(bm.copt_read_failed("[ERROR] Reading failed"))
        self.assertTrue(bm.copt_read_failed("Must read problem first"))
        # A GPU non-solve is a failure, but the model WAS read.
        self.assertTrue(bm.copt_solve_failed("[ERROR] Fail to solve"))
        self.assertFalse(bm.copt_read_failed("[ERROR] Fail to solve"))
        self.assertFalse(bm.copt_solve_failed("Primal objective: 1.0"))


class ScopeOrderingTest(unittest.TestCase):
    """systemd must not tear the scope down while the harness is still flushing."""

    def test_scope_outlives_harness_teardown(self):
        for time_limit in (0.0, 1.0, 7200.0):
            with self.subTest(time_limit=time_limit):
                self.assertGreater(
                    bm.scope_runtime_max_sec(time_limit),
                    time_limit + bm.HARNESS_TIMEOUT_GRACE_SEC + bm.KILL_FLUSH_SEC)


if __name__ == "__main__":
    unittest.main()
