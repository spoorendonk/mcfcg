#!/usr/bin/env python3
"""`--instances` decides WHICH cells a sweep contains, so it needs pinning.

Fifth file in this tier (see test/python/log_headers_test.py for the template).
The failure mode is silent in both directions: a filter that is too loose adds
instances nobody costed, and one that is too tight drops instances the README
claims are in the round -- and either way the sweep completes, the CSV parses,
and only the row count says anything is wrong.

The gh #43 ablation needs 6 of transportation's 9 instances (Austin, Birmingham
and Philadelphia are excluded on cost), which is why the filter takes a LIST of
globs rather than one: fnmatch has no negation, and the character-class forms
that come close match by accident as instance names change.

Run:  python3 -m unittest discover -s test/python -p '*_test.py'
"""

import os
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import benchmark_solvers as bs  # noqa: E402


class InstanceGlobParsingTest(unittest.TestCase):
    def test_no_filter_is_an_empty_list_not_a_one_element_one(self):
        # [""] would match nothing and silently empty the sweep.
        self.assertEqual(bs.parse_instance_globs(None), [])
        self.assertEqual(bs.parse_instance_globs(""), [])

    def test_a_single_name_still_behaves_as_before_the_list_form(self):
        self.assertEqual(bs.parse_instance_globs("grid1"), ["grid1"])

    def test_whitespace_and_empty_fields_are_dropped(self):
        self.assertEqual(bs.parse_instance_globs("grid1, grid2 ,,"), ["grid1", "grid2"])


class InstanceMatchingTest(unittest.TestCase):
    def test_an_absent_filter_admits_everything(self):
        self.assertTrue(bs.instance_matches("Austin", []))

    def test_a_name_matches_only_itself_not_its_prefixes(self):
        # 'grid1' must not pull in grid10..grid15: the ablation costs its cells
        # per instance, and a prefix match would quietly multiply them.
        globs = bs.parse_instance_globs("grid1")
        self.assertTrue(bs.instance_matches("grid1", globs))
        self.assertFalse(bs.instance_matches("grid10", globs))

    def test_globs_and_literal_names_mix_in_one_spec(self):
        globs = bs.parse_instance_globs("Barcelona,Chicago*")
        self.assertTrue(bs.instance_matches("Barcelona", globs))
        self.assertTrue(bs.instance_matches("ChicagoSketch", globs))
        self.assertFalse(bs.instance_matches("Sydney", globs))

    def test_the_ablation_transportation_subset_is_exactly_six(self):
        """The round (a) sweep as results/ablation/families/README.md records it.

        Pinned against the real family listing rather than a hand-written list,
        so adding a TNTP instance to data/ shows up here as a failure to decide
        whether it belongs in the round, not as a silently larger sweep.
        """
        keys = [key for _p, key, _f, _e in bs.enumerate_family("transportation")]
        globs = bs.parse_instance_globs(
            "Barcelona,BerlinCenter,ChicagoRegional,ChicagoSketch,Sydney,Winnipeg")
        selected = [k for k in keys if bs.instance_matches(k, globs)]
        self.assertEqual(selected, ["Barcelona", "BerlinCenter", "ChicagoRegional",
                                    "ChicagoSketch", "Sydney", "Winnipeg"])
        self.assertEqual(sorted(set(keys) - set(selected)),
                         ["Austin", "Birmingham", "Philadelphia"])


if __name__ == "__main__":
    unittest.main()
