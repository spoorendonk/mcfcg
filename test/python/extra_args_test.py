#!/usr/bin/env python3
"""`--extra-args` changes what a benchmark row MEANS, so it needs pinning.

Third file in this tier (see test/python/log_headers_test.py for the template).
Two failure modes, both silent:

  * the flags never reach the child, so an "A/B" is two identical arms whose
    difference is a filename convention;
  * the flags reach the child but not the CSV, so a promoted results file cannot
    say which arm it came from.

Neither shows up as a crash -- the sweep completes and the numbers look
plausible -- which is precisely why this tier exists.

Run:  python3 -m unittest discover -s test/python -p '*_test.py'
"""

import csv
import io
import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import benchmark_solvers as bs  # noqa: E402


class ExtraArgsReachChildTest(unittest.TestCase):
    """The flags must land in the command run_one builds, after the per-family ones."""

    def build_cmd(self, extra):
        seen = {}

        def fake_popen(cmd, **kwargs):
            seen["cmd"] = cmd
            raise RuntimeError("stop before exec")

        real_popen = bs.subprocess.Popen
        real_time = bs.GNU_TIME
        bs.subprocess.Popen = fake_popen
        # Drop the GNU time wrapper so the assertion reads the bare command.
        bs.GNU_TIME = "/nonexistent"
        try:
            bs.run_one("/bin/true", "inst.txt", "copt-cpu", "tree", extra, 10)
        except RuntimeError:
            pass
        finally:
            bs.subprocess.Popen = real_popen
            bs.GNU_TIME = real_time
        return seen.get("cmd", [])

    def test_family_defaults_and_extra_args_both_present(self):
        cmd = self.build_cmd(["--strategy", "pricer-heavy", "--pricing-cutoff"])
        self.assertIn("--pricing-cutoff", cmd)
        # The per-family default must survive alongside it, not be replaced.
        self.assertIn("--strategy", cmd)
        self.assertIn("pricer-heavy", cmd)
        # copt-cpu is a config label, not a --solver value.
        self.assertIn("--copt-gpu-mode", cmd)

    def test_no_extra_args_leaves_command_unchanged(self):
        self.assertNotIn("--pricing-cutoff", self.build_cmd([]))


class ExtraArgsQuotingTest(unittest.TestCase):
    """argparse hands --extra-args over as one string; shlex.split is what splits it."""

    def test_multiple_flags_split_into_separate_argv_entries(self):
        self.assertEqual(bs.shlex.split("--pricing-cutoff --threads 4"),
                         ["--pricing-cutoff", "--threads", "4"])

    def test_empty_value_yields_no_flags(self):
        self.assertEqual(bs.shlex.split(""), [])


class ExtraArgsRecordedInCsvTest(unittest.TestCase):
    """The results CSV must carry the arm, not just the --out path."""

    def test_extra_args_is_a_declared_column(self):
        # Mirrors the fields list in main(); a rename there without one here is
        # the drift this asserts against.
        fields = ["family", "instance", "solver", "formulation", "extra_args", "outcome",
                  "exit_status", "objective", "ref", "rel_err", "pass", "optimal",
                  "iterations", "columns", "time", "mem_gb", "config", "detail"]
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        writer.writerow({f: "" for f in fields} | {"extra_args": "--pricing-cutoff"})
        row = next(iter(csv.DictReader(io.StringIO(buf.getvalue()))))
        self.assertEqual(row["extra_args"], "--pricing-cutoff")

    def test_source_fields_list_still_contains_extra_args(self):
        with open(os.path.join(REPO, "scripts", "benchmark_solvers.py")) as fh:
            src = fh.read()
        self.assertIn('"extra_args"', src,
                      "extra_args dropped from benchmark_solvers.py")


if __name__ == "__main__":
    unittest.main()
