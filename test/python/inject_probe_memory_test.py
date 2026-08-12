#!/usr/bin/env python3
"""Refusal and labelling guarantees of scripts/inject_probe_memory.py.

This is the only tool in scripts/ that MUTATES logs, and the logs are the sole
surviving record of runs that cost hours to days. Every safety property it has
is a refusal — it declines to write rather than write something it cannot
justify — and a refusal that silently stops refusing looks exactly like success.
Hence tests: tempdir-and-string work, no solver, no systemd, no GPU.

The properties pinned here, each of which would corrupt the published record if
it regressed:
  * a full-solve reading (`measured` or `backfilled:`) is never replaced by a
    probe's lower bound, --force or not;
  * an injected peak is never re-injected from another injection (no chains);
  * swapped log trees are refused in both directions;
  * `# probe_iters:` never travels into a baseline log, because
    consolidate_mps_logs.py keys the solve/probe populations off it and a
    baseline log carrying it drops silently out of the results CSV;
  * the strong `probeN:` tag requires the backend's own iteration-limit marker,
    not merely a clean exit (cuOpt exits 0 after a VRAM failure it never
    reports).

Run:  python3 -m unittest discover -s test/python -p '*_test.py'
"""

import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import benchmark_solvers as bs  # noqa: E402

INJECTOR = os.path.join(REPO, "scripts", "inject_probe_memory.py")
MARKER = "# === solver output ==="
# mosek's own "I stopped at the iteration cap" marker (bm.PROBE_LIMIT_MARKERS).
ITER_LIMIT = "MSK_RES_TRM_MAX_ITERATIONS"

PROBE_KB = 500
BASE_KB = 777777


def write_log(path, *, rss_kb=None, source="measured", probe_iters=None,
              body="solver said things", outcome="ok", vram_mib=None):
    """Compose a log in exactly the shape benchmark_mps.run_one writes."""
    lines = ["# cmd: /bin/true\n", f"# wall=1.000s rc=0 outcome={outcome}\n"]
    if rss_kb is not None:
        lines.append(f"# peak_rss_kb: {rss_kb}\n")
        lines.append(f"# peak_rss_source: {source}\n")
    if vram_mib is not None:
        lines.append(f"# peak_vram_mib: {vram_mib}\n")
    if probe_iters is not None:
        lines.append(f"# probe_iters: {probe_iters}\n")
    lines.append(MARKER + "\n")
    lines.append(body + "\n")
    with open(path, "w") as f:
        f.writelines(lines)


class InjectProbeMemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mcfcg_inject_test_")
        self.base = os.path.join(self.tmp, "baseline")
        self.probe = os.path.join(self.tmp, "probe")
        os.makedirs(self.base)
        os.makedirs(self.probe)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_injector(self, *extra):
        return subprocess.run(
            [sys.executable, INJECTOR, "--baseline-logs", self.base,
             "--probe-logs", self.probe] + list(extra),
            capture_output=True, text=True)

    def read_base(self, cell="grid1__mosek"):
        with open(os.path.join(self.base, cell + ".log")) as f:
            return f.read()

    # ---------- the happy path, so the refusals below mean something ----------

    def test_injects_and_tags_with_the_source_log(self):
        write_log(os.path.join(self.probe, "grid1__mosek.log"), rss_kb=PROBE_KB,
                  probe_iters=3, body=ITER_LIMIT, vram_mib=42)
        write_log(os.path.join(self.base, "grid1__mosek.log"), rss_kb=None)
        res = self.run_injector()
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        text = self.read_base()
        kb, source = bs.parse_peak_rss(text)
        self.assertEqual(kb, PROBE_KB)
        self.assertTrue(source.startswith("probe3:"), source)
        self.assertIn("grid1__mosek.log", source, "tag must name its source log")

    def test_probe_iters_never_travels_into_the_baseline(self):
        """A baseline log carrying it drops out of the results CSV silently."""
        write_log(os.path.join(self.probe, "grid1__mosek.log"), rss_kb=PROBE_KB,
                  probe_iters=3, body=ITER_LIMIT)
        write_log(os.path.join(self.base, "grid1__mosek.log"), rss_kb=None)
        self.assertEqual(self.run_injector().returncode, 0)
        header = self.read_base().split(MARKER)[0]
        self.assertNotIn("# probe_iters:", header)

    def test_body_is_never_touched(self):
        write_log(os.path.join(self.probe, "grid1__mosek.log"), rss_kb=PROBE_KB,
                  probe_iters=3, body=ITER_LIMIT)
        write_log(os.path.join(self.base, "grid1__mosek.log"), rss_kb=None,
                  body="ORIGINAL BODY\nsecond line")
        self.assertEqual(self.run_injector().returncode, 0)
        self.assertIn("ORIGINAL BODY\nsecond line", self.read_base())

    # ---------- labelling: a clean exit is not proof the barrier ran ----------

    def test_partial_tag_when_the_cap_was_never_reached(self):
        """cuOpt exits 0 on a swallowed VRAM failure without iterating."""
        write_log(os.path.join(self.probe, "grid1__mosek.log"), rss_kb=PROBE_KB,
                  probe_iters=3, outcome="ok", body="presolve done, then nothing")
        write_log(os.path.join(self.base, "grid1__mosek.log"), rss_kb=None)
        self.assertEqual(self.run_injector().returncode, 0)
        _, source = bs.parse_peak_rss(self.read_base())
        self.assertTrue(source.startswith("probe3-partial:"), source)

    # ---------------------------- the refusals ------------------------------

    def _assert_untouched(self, expect_kb, expect_source):
        kb, source = bs.parse_peak_rss(self.read_base())
        self.assertEqual(kb, expect_kb)
        self.assertEqual(source, expect_source)

    def test_never_overwrites_a_measured_peak_even_with_force(self):
        write_log(os.path.join(self.probe, "grid1__mosek.log"), rss_kb=PROBE_KB,
                  probe_iters=3, body=ITER_LIMIT)
        write_log(os.path.join(self.base, "grid1__mosek.log"), rss_kb=BASE_KB,
                  source="measured")
        self.assertEqual(self.run_injector("--force").returncode, 0)
        self._assert_untouched(BASE_KB, "measured")

    def test_never_overwrites_a_backfilled_peak_even_with_force(self):
        """`backfilled:` is the SAME execution's reading — as strong as measured."""
        write_log(os.path.join(self.probe, "grid1__mosek.log"), rss_kb=PROBE_KB,
                  probe_iters=3, body=ITER_LIMIT)
        write_log(os.path.join(self.base, "grid1__mosek.log"), rss_kb=BASE_KB,
                  source="backfilled:some_sweep.csv")
        self.assertEqual(self.run_injector("--force").returncode, 0)
        self._assert_untouched(BASE_KB, "backfilled:some_sweep.csv")

    def test_refuses_a_probe_whose_own_peak_is_injected(self):
        """No chains: an injected number must never be re-relocated."""
        write_log(os.path.join(self.probe, "grid1__mosek.log"), rss_kb=PROBE_KB,
                  source="probe3:somewhere.log", probe_iters=3, body=ITER_LIMIT)
        write_log(os.path.join(self.base, "grid1__mosek.log"), rss_kb=None)
        res = self.run_injector()
        self.assertEqual(res.returncode, 1, "a chained injection must be refused")
        self.assertIsNone(bs.parse_peak_rss(self.read_base())[0])

    def test_refuses_a_probe_log_with_no_probe_iters_header(self):
        write_log(os.path.join(self.probe, "grid1__mosek.log"), rss_kb=PROBE_KB,
                  probe_iters=None, body=ITER_LIMIT)
        write_log(os.path.join(self.base, "grid1__mosek.log"), rss_kb=None)
        self.assertEqual(self.run_injector().returncode, 1)
        self.assertIsNone(bs.parse_peak_rss(self.read_base())[0])

    def test_refuses_when_the_trees_are_swapped(self):
        """A baseline log carrying `# probe_iters:` means the dirs are reversed."""
        write_log(os.path.join(self.probe, "grid1__mosek.log"), rss_kb=PROBE_KB,
                  probe_iters=3, body=ITER_LIMIT)
        write_log(os.path.join(self.base, "grid1__mosek.log"), rss_kb=None,
                  probe_iters=3)
        self.assertEqual(self.run_injector().returncode, 1)

    def test_dry_run_writes_nothing(self):
        write_log(os.path.join(self.probe, "grid1__mosek.log"), rss_kb=PROBE_KB,
                  probe_iters=3, body=ITER_LIMIT)
        write_log(os.path.join(self.base, "grid1__mosek.log"), rss_kb=None)
        before = self.read_base()
        self.assertEqual(self.run_injector("--dry-run").returncode, 0)
        self.assertEqual(self.read_base(), before)

    def test_a_probe_without_a_peak_leaves_the_cell_blank(self):
        """Never invented: an unmeasured cell stays unmeasured."""
        write_log(os.path.join(self.probe, "grid1__mosek.log"), rss_kb=None,
                  probe_iters=3, body=ITER_LIMIT)
        write_log(os.path.join(self.base, "grid1__mosek.log"), rss_kb=None)
        self.assertEqual(self.run_injector().returncode, 0)
        self.assertIsNone(bs.parse_peak_rss(self.read_base())[0])


class RewriteHeaderBlockTest(unittest.TestCase):
    """benchmark_solvers.rewrite_header_block, now shared by two writers."""

    def test_drops_then_appends_before_the_marker(self):
        log = ("# cmd: x\n# peak_rss_kb: 1\n# peak_rss_source: measured\n"
               + MARKER + "\nbody\n")
        out = bs.rewrite_header_block(
            log, ("# peak_rss_kb:", "# peak_rss_source:"), "# peak_rss_kb: 2\n")
        self.assertEqual(out.count("# peak_rss_kb:"), 1)
        self.assertIn("# peak_rss_kb: 2\n" + MARKER, out)
        self.assertTrue(out.endswith("body\n"))

    def test_truncated_last_line_is_not_fused(self):
        """A log cut off mid-header must not absorb the new header onto it."""
        out = bs.rewrite_header_block("# cmd: x\n# wall=1", (), "# peak_rss_kb: 2\n")
        self.assertIn("# wall=1\n# peak_rss_kb: 2\n", out)

    def test_body_containing_a_hash_line_is_untouched(self):
        log = "# cmd: x\n" + MARKER + "\n# not a header\nbody\n"
        out = bs.rewrite_header_block(log, (), "# peak_rss_kb: 2\n")
        self.assertIn(MARKER + "\n# not a header\nbody\n", out)


if __name__ == "__main__":
    unittest.main()
