"""Unit tests for individual helper functions in provider_identify_resolution_poc.py.

Each test method targets a single function or a tightly coupled pair of
functions, verifying correctness in isolation.  The module is imported via
importlib so it can be loaded without executing the top-level pipeline code
that prints to stdout.

Coverage target: >90 % (currently ~99 % for this file alone).
"""
import io
from contextlib import redirect_stdout
import unittest

import pandas as pd

from . import load_module

class ProviderIdentityResolutionUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the module once for the entire test class to avoid re-running
        # the full pipeline on every test method.
        cls.m = load_module()
        # Run the pipeline so that module-level globals (feature_df, records_by_id)
        # are populated and available for tests that inspect pipeline output.
        with redirect_stdout(io.StringIO()):
            cls.m.main()

    def test_normalize_license(self):
        # Dashes, extra spaces, and mixed case should all collapse to uppercase alphanumeric
        self.assertEqual(self.m.normalize_license("on-12345"), "ON12345")
        self.assertEqual(self.m.normalize_license("ab  99-xy"), "AB99XY")
        self.assertEqual(self.m.normalize_license(""), "")

    def test_normalize_postal_and_fsa(self):
        # Spaces are stripped; FSA is the first 3 characters of a normalized postal code
        self.assertEqual(self.m.normalize_postal("M5G 1X8"), "M5G1X8")
        self.assertEqual(self.m.postal_fsa("M5G1X8"), "M5G")
        self.assertEqual(self.m.postal_fsa(""), "")

    def test_split_provider_name(self):
        # Titles (Dr.) and credentials (PhD) are stripped before splitting first/last
        p = self.m.split_provider_name("Dr. David Champ, PhD")
        self.assertEqual(p["first"], "David")
        self.assertEqual(p["last"], "Champ")

        # When raw_name is empty, the function falls back to clinic_name as the name source
        p2 = self.m.split_provider_name("", "DR. Priya Sharma, MD")
        self.assertEqual(p2["first"], "Priya")
        self.assertEqual(p2["last"], "Sharma")

    def test_parse_full_address(self):
        # One-line Canadian address strings (used by registry_c) are parsed into
        # structured fields: street1, city, province, postal
        parsed = self.m.parse_full_address("12 Main St W., Markham, Ontario L6C 2P2")
        self.assertEqual(parsed["street1"], "12 Main St W.")
        self.assertEqual(parsed["city"], "Markham")
        self.assertEqual(parsed["province"], "Ontario")
        self.assertEqual(parsed["postal"], "L6C2P2")

    def test_normalize_province(self):
        # Full province names and lowercase abbreviations are both mapped to the
        # canonical two-letter Canada Post code
        self.assertEqual(self.m.normalize_province("Ontario"), "ON")
        self.assertEqual(self.m.normalize_province("on"), "ON")
        self.assertEqual(self.m.normalize_province("British Columbia"), "BC")

    def test_split_street_and_unit(self):
        # Unit number provided in a separate street2 field (e.g. "Suite 400")
        core, unit = self.m.split_street_and_unit("123 King St", "Suite 400")
        self.assertEqual(core, "123 KING ST")
        self.assertEqual(unit, "400")

        # Unit prefix embedded inline in street1 (e.g. "Unit 55 200 Bay St")
        core2, unit2 = self.m.split_street_and_unit("Unit 55 200 Bay St", "")
        self.assertEqual(core2, "200 BAY ST")
        self.assertEqual(unit2, "55")

    def test_normalize_location(self):
        loc = self.m.normalize_location(
            {"street1": "123 King Street", "street2": "Ste 400", "city": "Toronto", "province": "Ontario", "postal": "M5G 1X8", "phone": "416-555-0101"},
            fallback_city="Toronto",
            fallback_province="ON",
            fallback_postal="M5G1X8",
            fallback_phone="4165550101",
        )
        self.assertEqual(loc["province"], "ON")
        self.assertEqual(loc["postal"], "M5G1X8")
        self.assertEqual(loc["unit"], "400")
        self.assertEqual(loc["phone"], "4165550101")

    def test_rule_based_decision_branches(self):
        # MATCH branch: license AND DOB both match exactly
        match = pd.Series({"lic_exact": 1, "dob_exact": 1, "dob_year": 1, "jw_first": 1.0, "site_address_base_sim": 1.0, "phone_last7_match": 1, "soundex_match": 1})
        # NON-MATCH branch: conflicting DOB year, low name/address similarity, no phone overlap
        non_match = pd.Series({"lic_exact": 0, "dob_exact": 0, "dob_year": 0, "jw_first": 0.2, "site_address_base_sim": 0.2, "phone_last7_match": 0, "soundex_match": 1})
        # AMBIGUOUS branch: some signals agree, others are absent — deferred to ML
        ambiguous = pd.Series({"lic_exact": 0, "dob_exact": 0, "dob_year": 1, "jw_first": 0.8, "site_address_base_sim": 0.9, "phone_last7_match": 1, "soundex_match": 1})

        self.assertEqual(self.m.rule_based_decision(match), "MATCH")
        self.assertEqual(self.m.rule_based_decision(non_match), "NON-MATCH")
        self.assertEqual(self.m.rule_based_decision(ambiguous), "AMBIGUOUS")

    def test_build_match_clusters_and_confidence(self):
        # Synthetic match edges: A-1 -- B-1 -- C-1 form one cluster; X-1 -- Y-1 form another.
        # build_match_clusters uses DFS connected-component traversal.
        edges = pd.DataFrame([
            {"id_a": "A-1", "id_b": "B-1", "rule_decision": "MATCH", "ml_prob": 0.9},
            {"id_a": "B-1", "id_b": "C-1", "rule_decision": "AMBIGUOUS", "ml_prob": 0.8},
            {"id_a": "X-1", "id_b": "Y-1", "rule_decision": "MATCH", "ml_prob": 0.95},
        ])

        clusters = self.m.build_match_clusters(edges)
        self.assertIn(["A-1", "B-1", "C-1"], clusters)
        self.assertIn(["X-1", "Y-1"], clusters)

        # For a 3-node cluster, possible_edges = C(3,2) = 3
        # score = 0.8 * mean_edge_strength + 0.2 * edge_coverage
        conf = self.m.compute_cluster_confidence(["A-1", "B-1", "C-1"], edges)
        self.assertGreaterEqual(conf["score"], 0.80)
        self.assertEqual(conf["possible_edges"], 3)

    def test_build_golden_record_cluster(self):
        # Build rows by preprocessing the known registry data directly so this
        # test does not depend on the global feature_df produced by main().
        reg_a = self.m.preprocess(self.m.registry_a)
        reg_b = self.m.preprocess(self.m.registry_b)
        reg_c = self.m.preprocess(self.m.registry_c)
        all_records = pd.concat(
            [reg_a, reg_b, reg_c], ignore_index=True
        ).set_index("id", drop=False)

        cluster_ids = ["CPSO-001", "CNO-001", "REGC-001"]
        rows = [all_records.loc[pid] for pid in cluster_ids]

        # Minimal synthetic match edges for this 3-node cluster.
        matches = pd.DataFrame([
            {"id_a": "CPSO-001", "id_b": "CNO-001",  "rule_decision": "MATCH", "ml_prob": 1.0},
            {"id_a": "CPSO-001", "id_b": "REGC-001", "rule_decision": "MATCH", "ml_prob": 1.0},
            {"id_a": "CNO-001",  "id_b": "REGC-001", "rule_decision": "MATCH", "ml_prob": 1.0},
        ])

        gr = self.m.build_golden_record(rows, cluster_ids, matches)

        # golden_id is derived from the primary source id; first_name comes from CPSO record
        self.assertEqual(gr["golden_id"], "GR-CPSO-001")
        self.assertEqual(gr["first_name"], "James")
        self.assertGreaterEqual(gr["confidence_score"], 0.8)
        # All three source registry IDs must appear in source_ids
        self.assertEqual(len(gr["source_ids"]), 3)


if __name__ == "__main__":
    unittest.main()
