#!/usr/bin/env python3
"""The log header is a serialization format; these are its round-trip tests.

Second file in this tier — the template for adding a third. Nothing here spawns
a process, so it is the cheap half of `Python.Scripts`: pure functions that
decide what reaches a committed results table. A drift between a writer and its
reader is silent (a blank or mislabelled CSV column), which is the same failure
mode test/python/timeout_memory_test.py exists to prevent, one layer down.

Several of these pin claims that are currently only asserted in comments — that
the run-header writer and reader "cannot drift", that a truncated log is never
fused onto, that a fresh RSS peak never ends up beside a stale VRAM figure.

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
import inject_probe_memory as ip  # noqa: E402

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


class RewriteHeaderBlockTest(unittest.TestCase):
    """The one rewriter of logs that cost hours to days to reproduce."""

    def test_replaces_in_the_block_and_leaves_the_body_alone(self):
        out = bs.rewrite_header_block(
            LOG, ip.MEM_HEADERS, bs.format_peak_rss_headers(999, "probe3:p.log"))
        self.assertEqual(bs.parse_peak_rss(out), (999, "probe3:p.log"))
        self.assertEqual(out.count("# peak_rss_kb:"), 1)
        self.assertEqual(out.split(cm.MARKER, 1)[1], LOG.split(cm.MARKER, 1)[1],
                         "the body was modified")
        # Still parses as the same run.
        self.assertEqual(bm.parse_run_header(out), (12.5, 0, "ok"))

    def test_is_idempotent(self):
        block = bs.format_peak_rss_headers(999, "probe3:p.log")
        once = bs.rewrite_header_block(LOG, ip.MEM_HEADERS, block)
        self.assertEqual(bs.rewrite_header_block(once, ip.MEM_HEADERS, block), once)

    def test_truncated_header_is_not_fused_onto(self):
        """A log cut off mid-header must not gain `...# peak_rss_kb:` on one line."""
        out = bs.rewrite_header_block("# cmd: solver model.mps",
                                      ip.MEM_HEADERS,
                                      bs.format_peak_rss_headers(7, "measured"))
        self.assertEqual(out.splitlines()[0], "# cmd: solver model.mps")
        self.assertEqual(bs.parse_peak_rss(out), (7, "measured"))

    def test_drops_a_stale_vram_line_with_the_rss_pair(self):
        """MEM_HEADERS travel as one block: no fresh RSS beside a stale VRAM."""
        stale = LOG.replace(cm.MARKER, "# peak_vram_mib: 5000\n" + cm.MARKER)
        out = bs.rewrite_header_block(stale, ip.MEM_HEADERS,
                                      bs.format_peak_rss_headers(999, "probe3:p"))
        self.assertIsNone(bm.parse_peak_vram_mib(out))

    def test_a_hash_line_in_the_body_is_not_treated_as_a_header(self):
        out = bs.rewrite_header_block(LOG, (), "# peak_rss_kb: 2\n")
        self.assertIn(cm.MARKER + "\n# this hash is BODY, not header\n", out)


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

    def test_source_tag_marks_a_probe_that_stopped_short(self):
        self.assertEqual(ip.source_tag(3, True, "p.log"), "probe3:p.log")
        self.assertEqual(ip.source_tag(3, False, "p.log"), "probe3-partial:p.log")


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
