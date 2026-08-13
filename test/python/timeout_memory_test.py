#!/usr/bin/env python3
"""Peak RSS must survive a harness-timeout kill (scripts/benchmark_mps.py).

The measurement is taken by a GNU `time` wrapper living in the same process
group as the solver, so the obvious teardown -- `killpg` the group -- destroys
the wrapper before it writes its report. Memory is the one benchmark metric that
cannot be recovered by re-parsing a log, so that loss is permanent: the
compact-baseline probe cell ChicagoRegional x highs is unmeasured for exactly
this reason. `benchmark_mps.kill_preserving_mem` kills every other member of the
group and lets the wrapper reap and flush.

This is verified by EXECUTION, not by inspection: a synthetic child allocates a
known ~1.5 GiB and hangs, the harness's real timeout path kills it, and the
saved log must carry a `# peak_rss_kb:` header of that magnitude. The paired
negative test forces the old blunt kill and asserts the header is GONE -- without
it the positive test would still pass if the wrapper were never killed at all.

Run:  python3 -m unittest discover -s test/python -p '*_test.py'
      (or via ctest as Python.Scripts).  The pattern is required: unittest
      defaults to `test*.py` and would silently discover NOTHING here, since the
      file is named for the repo's `<module>_test` convention.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import benchmark_mps as bm  # noqa: E402
import benchmark_solvers as bs  # noqa: E402

# The child allocates (and touches) this much, then sleeps forever. Large enough
# that no plausible interpreter/harness overhead could be mistaken for it.
HOG_BYTES = 1500 * 1024 * 1024
HOG_KB_MIN = 1300 * 1024   # ~1.27 GiB: the allocation, minus nothing much
HOG_KB_MAX = 2600 * 1024   # ~2.54 GiB: generous ceiling on interpreter overhead

HOG_SRC = """
import sys, time
n = int(sys.argv[1])
buf = bytearray(n)
for i in range(0, n, 4096):
    buf[i] = 1          # fault every page in so it is resident, not just mapped
sys.stderr.write("allocated\\n")
sys.stderr.flush()
time.sleep(3600)        # hang: only the harness timeout can end this run
"""

# Wall-clock budget for one harness-timeout cell: run_one waits
# time_limit + HARNESS_TIMEOUT_GRACE_SEC before declaring a hang. The real grace
# is 30 minutes, so the test patches it; 6 s is ample for a ~1 s allocation.
TIME_LIMIT = 1.0
GRACE = 5.0
# Production gives the wrapper 120 s to flush; a synthetic hog flushes in
# milliseconds. Keeping the real value here would buy nothing on the passing path
# and cost minutes per BROKEN run -- every way this fix can regress ends in this
# window expiring, so a red suite would take ~6 minutes to say so.
FLUSH = 15.0


class HarnessTimeoutMemoryTest(unittest.TestCase):
    """Drive benchmark_mps.run_one down its real TimeoutExpired path."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(bs.GNU_TIME):
            raise unittest.SkipTest(f"{bs.GNU_TIME} missing: nothing measures RSS")
        if not os.path.exists("/proc/self/stat"):
            raise unittest.SkipTest("no /proc: process-group scan unavailable")

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mcfcg_timeout_test_")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.hog = os.path.join(self.tmp, "hog.py")
        with open(self.hog, "w") as f:
            f.write(HOG_SRC)

    def _run_cell(self, guard=()):
        """One run_one call that is guaranteed to hit the timeout; returns the log.

        NO SOLVER IS INVOLVED — not HiGHS, and none of the licensed backends. The
        stand-in is a registered `synthetic` config whose command builder returns
        the hog's argv; its name matches none of run_one's solver-specific output
        guards (highs / cuopt / copt*), so nothing rewrites the outcome behind the
        test's back. Everything downstream of the builder -- GNU time wrapping,
        the cgroup guard, the timeout, the kill, the log writer -- is the
        production code path, unmocked.
        """
        argv = [sys.executable, self.hog, str(HOG_BYTES)]
        cfg = dict(bm.CONFIGS, synthetic=(lambda m, t, d, g, p: (argv, {}), "cpu"))
        # parse_output looks the solver up in PARSERS; the hog prints nothing any
        # of them would match, so which patterns it gets is immaterial.
        parsers = dict(bm.PARSERS, synthetic=bm.PARSERS["mosek"])
        with mock.patch.object(bm, "CONFIGS", cfg), \
                mock.patch.object(bm, "PARSERS", parsers), \
                mock.patch.object(bm, "HARNESS_TIMEOUT_GRACE_SEC", GRACE), \
                mock.patch.object(bm, "KILL_FLUSH_SEC", FLUSH), \
                mock.patch.object(bm, "MEM_GUARD", list(guard)):
            res = bm.run_one("synthetic", "unused.mps", "hog", TIME_LIMIT, self.tmp)
        self.assertEqual(res["outcome"], "timeout",
                         "the cell did not take the timeout path")
        with open(os.path.join(self.tmp, "hog__synthetic.log")) as f:
            return res, f.read()

    def _assert_hog_sized(self, kb, source):
        self.assertIsNotNone(kb, "no peak_rss_kb header: the measurement was lost")
        self.assertEqual(source, "measured")
        self.assertGreater(kb, HOG_KB_MIN, f"peak {kb} KB is far below the ~1.5 GiB "
                                           "the child allocated")
        self.assertLess(kb, HOG_KB_MAX, f"peak {kb} KB is implausibly above it")

    def test_timeout_preserves_peak_rss(self):
        res, log = self._run_cell()
        kb, source = bs.parse_peak_rss(log)
        self._assert_hog_sized(kb, source)
        self.assertNotEqual(res["mem_gb"], "", "mem_gb missing from the result row")

    def test_blunt_killpg_loses_it(self):
        """Control: the pre-fix teardown. If this ever starts passing a peak
        through, the positive test above has stopped proving anything."""
        with mock.patch.object(bm, "is_time_wrapper", return_value=False):
            res, log = self._run_cell()
        kb, _ = bs.parse_peak_rss(log)
        self.assertIsNone(kb, "killpg somehow left a measurement behind")
        self.assertEqual(res["mem_gb"], "")

    def test_timeout_preserves_peak_rss_under_cgroup_guard(self):
        """Same, inside the systemd scope the real sweeps use.

        Separate from the unguarded case because the guard adds its own killer:
        RuntimeMaxSec teardown SIGTERMs the whole scope, `time` included. It must
        not be able to fire inside the flush window.
        """
        # RuntimeMaxSec comes from the PRODUCTION formula so this case exercises
        # it end to end -- but note what it does NOT prove: with GRACE patched
        # down, the scope outlives the harness kill by minutes under EITHER the
        # new formula or the old `time_limit + 1800`, so a revert keeps this case
        # green. ScopeRuntimeMaxTest below is what pins the ordering invariant.
        with mock.patch.object(bm, "HARNESS_TIMEOUT_GRACE_SEC", GRACE):
            runtime_max = bm.scope_runtime_max_sec(TIME_LIMIT)
        guard = ["systemd-run", "--user", "--scope", "--quiet",
                 "-p", "MemoryMax=8G", "-p", "MemorySwapMax=0",
                 "-p", "OOMPolicy=continue", "-p", f"RuntimeMaxSec={runtime_max}"]
        if not shutil.which("systemd-run"):
            self.skipTest("systemd-run not available")
        ok, detail = bm.verify_mem_guard(guard)
        if not ok:
            self.skipTest(f"cgroup guard not effective here: {detail}")
        _, log = self._run_cell(guard=guard)
        self._assert_hog_sized(*bs.parse_peak_rss(log))


class ScopeRuntimeMaxTest(unittest.TestCase):
    """The scope must outlive the harness's own teardown, or systemd wins.

    Cheap, non-flaky coverage of the ordering invariant itself, complementing the
    end-to-end guarded test above: systemd's RuntimeMaxSec expiry SIGTERMs the
    whole scope including the `time` wrapper, so if it can fire while the harness
    is still flushing, the measurement is lost exactly as it was before the fix.
    """

    def test_outlives_harness_teardown(self):
        for time_limit in (0.0, 1.0, 7200.0):
            with self.subTest(time_limit=time_limit):
                harness_done = (time_limit + bm.HARNESS_TIMEOUT_GRACE_SEC
                                + bm.KILL_FLUSH_SEC)
                self.assertGreater(
                    bm.scope_runtime_max_sec(time_limit), harness_done,
                    "systemd would tear the scope down while the harness is "
                    "still waiting for `time` to write its report")

    def test_margin_is_actually_positive(self):
        """Guards the degenerate reading of the above: equal is not greater."""
        self.assertGreater(bm.SCOPE_TEARDOWN_MARGIN_SEC, 0)


class PgidMembersTest(unittest.TestCase):
    """The /proc scan kill_preserving_mem targets its SIGKILLs with."""

    @classmethod
    def setUpClass(cls):
        # Same guard as HarnessTimeoutMemoryTest: without /proc, pgid_members
        # returns [] BY DESIGN, which is a skip, not a failure. (On macOS the
        # split bites: BSD /usr/bin/time exists, /proc does not.)
        if not os.path.exists("/proc/self/stat"):
            raise unittest.SkipTest("no /proc: process-group scan unavailable")

    def test_finds_a_known_child_and_excludes(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                start_new_session=True)
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        pgid = os.getpgid(proc.pid)
        self.assertIn(proc.pid, bm.pgid_members(pgid))
        self.assertNotIn(proc.pid, bm.pgid_members(pgid, exclude=(proc.pid,)))

    def test_is_time_wrapper_rejects_a_plain_process(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(proc.wait)
        self.addCleanup(proc.kill)
        self.assertFalse(bm.is_time_wrapper(proc.pid, "/tmp/nonexistent_mem.txt"))


if __name__ == "__main__":
    unittest.main()
