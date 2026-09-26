"""Behavioural tests for the claims the docs make.  Run:  PYTHONPATH=src python -m unittest -v"""
import unittest

import numpy as np

from ber.decide.assign import exclusive
from ber.decide.expected_f import choose_set, evpi
from ber.eval.metric import f_beta_set, score
from ber.features.pairwise import legal_match, soft_tfidf
from ber.normalize.abbrev import is_abbrev, mine_abbreviations
from ber.normalize.address import parse_address
from ber.normalize.name import mine_legal_tokens, strip_tail
from ber.normalize.text import fold


class Metric(unittest.TestCase):
    def test_worked_example(self):
        f = f_beta_set({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
        self.assertAlmostEqual(f, 0.7143, places=4)

    def test_singletons(self):
        self.assertEqual(f_beta_set(set(), set()), 1.0)
        self.assertEqual(f_beta_set({"S2-1"}, set()), 0.0)
        self.assertEqual(score({"a": []}, {"a": set(), "b": {"x"}}).macro_f, 0.5)


class ExpectedF(unittest.TestCase):
    def pick(self, p):
        ids, _ = choose_set([f"c{i}" for i in range(len(p))], np.array(p))
        return len(ids)

    def test_documented_decisions(self):
        self.assertEqual(self.pick([0.97, 0.10, 0.05]), 1)
        self.assertEqual(self.pick([0.95, 0.91, 0.08]), 2)
        self.assertEqual(self.pick([0.22, 0.15, 0.09]), 0)   # abstain: singleton won
        self.assertEqual(self.pick([0.55, 0.12]), 1)

    def test_evpi_prefers_uncertain(self):
        g_sure, _ = evpi(np.array([0.99]))
        g_unsure, _ = evpi(np.array([0.5]))
        self.assertGreater(g_unsure, g_sure)


class OneOwner(unittest.TestCase):
    def test_uncontested_unchanged_contested_shrinks(self):
        q = exclusive(np.array([1, 2, 2]), np.array([0.8, 0.8, 0.8]))
        self.assertAlmostEqual(q[0], 0.8, places=6)
        self.assertLess(q[1], 0.8)
        self.assertLessEqual(q[1] + q[2], 1.0)


class Normalisation(unittest.TestCase):
    def test_fold(self):
        self.assertEqual(fold("Société Générale S.A.R.L."), "societe generale sarl")
        self.assertEqual(fold("A & B"), "a and b")

    def test_abbrev(self):
        self.assertTrue(is_abbrev("blvd", "boulevard"))
        self.assertTrue(is_abbrev("llc", "limitedliabilitycompany"))
        self.assertFalse(is_abbrev("rd", "street"))
        pairs = [("12 Main Rd", "12 Main Road")] * 6 + [("Acme Sons", "Acme Solutions")]
        table = mine_abbreviations(pairs)
        self.assertEqual(table.get("rd"), "road")
        self.assertNotIn("sons", table)

    def test_legal_tail_and_identity(self):
        names = [["acme", "private", "limited"], ["zen", "limited"], ["orbit", "private", "limited"]] * 200
        legal = mine_legal_tokens(names)
        self.assertEqual(strip_tail(["acme", "foods", "private", "limited"], legal)[0], ["acme", "foods"])

    def test_legal_match(self):
        self.assertEqual(legal_match(("llc",), ("limited", "liability", "company")), 0.9)
        self.assertEqual(legal_match(("sas",), ("societe", "par", "actions", "simplifiee")), 0.9)
        self.assertEqual(legal_match((), ("ltd",)), -1.0)

    def test_address_slots(self):
        a = parse_address("Near Metro Pillar 42 33 Stonebridge Road, Indore, 653200")
        self.assertEqual((a.house, a.postcode, a.landmark), ("33", "653200", True))
        b = parse_address("23/33 Ravenwood Drive, Fresno")
        self.assertEqual(b.house, "23/33")
        self.assertIn("23", b.numbers)

    def test_soft_tfidf_abbreviation(self):
        idf = {"harbor": 3.0, "boulevard": 1.5, "bd": 2.5}
        s, _, _ = soft_tfidf(("harbor", "bd"), ("harbor", "boulevard"), idf, 3.0)
        self.assertGreater(s, 0.9)


if __name__ == "__main__":
    unittest.main()
