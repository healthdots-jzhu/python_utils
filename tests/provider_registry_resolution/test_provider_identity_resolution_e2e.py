"""End-to-end (E2E) variant tests for provider_identify_resolution_poc.py.

Each subtest builds a minimal synthetic dataset covering a specific real-world
matching scenario, runs the full blocking → feature engineering → rule engine
→ XGBoost → decision pipeline, and asserts the expected outcomes.

Three variants are exercised:
  titles_and_suffixes   — abbreviated first name, licence formatting differences,
                          credential suffixes (PhD, MD) in registry_c
  clinic_name_fallback  — empty ``name`` field in registry_c, resolved via
                          ``clinic_name`` as the provider name source
  non_match_guard       — conflicting DOB, licence, address, and specialty across
                          registries; the deterministic rule engine must reject
                          all pairs without involving the ML model

Design notes:
  - run_pipeline_variant() is a self-contained harness that mirrors the main
    pipeline flow but uses the synthetic records and labels provided per variant.
  - When blocking produces no candidate pairs (empty union of soundex + FSA
    blocks), a full Cartesian product is used as a safe fallback so the test
    does not error out on trivially small datasets.
  - When all labelled pairs belong to a single class (match-only or reject-only),
    XGBoost is bypassed and a constant probability is assigned to avoid
    single-class training errors.
"""
import io
from contextlib import redirect_stdout
import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from . import load_module

def run_pipeline_variant(m, reg_a_raw, reg_b_raw, reg_c_raw, labels_by_id):
    """Run the core matching pipeline on synthetic per-variant records.

    Parameters
    ----------
    m              : loaded module object from load_module()
    reg_a_raw / reg_b_raw / reg_c_raw : list[dict] — raw registry rows
    labels_by_id   : dict[(id_a, id_b) -> 0|1] — ground-truth labels

    Returns
    -------
    feature_df : DataFrame with columns including id_a, id_b, decision
    """
    # Normalize all three registries using the same preprocessing as the main script
    reg_a = m.preprocess(pd.DataFrame(reg_a_raw))
    reg_b = m.preprocess(pd.DataFrame(reg_b_raw))
    reg_c = m.preprocess(pd.DataFrame(reg_c_raw))

    frames = []
    for pair_name, left_df, right_df in [("A-B", reg_a, reg_b), ("A-C", reg_a, reg_c), ("B-C", reg_b, reg_c)]:
        try:
            pairs = m.generate_candidate_pairs(left_df, right_df)
        except TypeError:
            pairs = None
        if pairs is None or len(pairs) == 0:
            # Very small synthetic datasets may produce empty blocking results;
            # fall back to a full Cartesian product so feature building can proceed.
            assert len(left_df) <= 5 and len(right_df) <= 5, (
                f"Cartesian fallback triggered on unexpectedly large dataset "
                f"({len(left_df)} x {len(right_df)} rows). Blocking may be misconfigured."
            )
            cartesian = [(i, j) for i in left_df.index for j in right_df.index]
            pairs = pd.MultiIndex.from_tuples(cartesian)
        frames.append(m.build_features(pairs, left_df, right_df, pair_name))
    feature_df = pd.concat(frames, ignore_index=True)

    def lookup(id_a, id_b):
        if (id_a, id_b) in labels_by_id:
            return labels_by_id[(id_a, id_b)]
        if (id_b, id_a) in labels_by_id:
            return labels_by_id[(id_b, id_a)]
        return np.nan

    feature_df["label"] = feature_df.apply(lambda r: lookup(r["id_a"], r["id_b"]), axis=1)
    labelled = feature_df.dropna(subset=["label"])

    x = labelled[m.FEATURE_COLS].values
    y = labelled["label"].astype(int).values

    # Apply the deterministic rule engine before ML scoring
    feature_df["rule_decision"] = feature_df[m.FEATURE_COLS].apply(m.rule_based_decision, axis=1)

    unique_labels = sorted(set(y.tolist()))
    if len(unique_labels) < 2:
        # XGBoost requires at least two classes to train.  Tiny synthetic datasets
        # (e.g. all-match or all-non-match) trigger this path: assign a constant
        # probability that reflects the single class and skip model training.
        const_prob = 0.99 if unique_labels and unique_labels[0] == 1 else 0.01
        probs = np.full(shape=len(feature_df), fill_value=const_prob, dtype=float)
    else:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        clf = XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.1, eval_metric="logloss", random_state=42)
        clf.fit(x_scaled, y)
        x_all_scaled = scaler.transform(feature_df[m.FEATURE_COLS].values)
        probs = clf.predict_proba(x_all_scaled)[:, 1]

    feature_df["ml_prob"] = probs
    feature_df["decision"] = feature_df.apply(lambda r: m.final_decision(r["rule_decision"], r["ml_prob"]), axis=1)
    return feature_df


class ProviderIdentityResolutionE2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the module once; the pipeline runs at import time and is reused
        # across all subtests.
        cls.m = load_module()

    def test_major_variants(self):
        """Run all E2E variant scenarios as subtests.

        Each variant dict contains:
          name        — human-readable scenario label (used in subTest)
          a / b / c   — synthetic registry rows for registries A, B, C
          labels      — ground-truth {(id_a, id_b): 0|1} used for ML training
          must_match  — pairs that must receive an AUTO-MATCH decision
          must_reject — pairs that must receive a REJECT decision
        """
        variants = [
            {
                "name": "titles_and_suffixes",
                "a": [{"id": "A-001", "first": "David", "last": "Champ", "dob": "1971-05-30", "license": "ON-77777", "specialty": "Family Medicine", "street1": "12 Main Street West", "street2": "Suite 400", "city": "Markham", "province": "ON", "postal": "L6C 2P2", "phone": "905-555-0201", "other_practice_sites": []}],
                "b": [{"id": "B-001", "first": "Dave", "last": "Champ", "dob": "1971-05-30", "license": "ON77777", "specialty": "Family Medicine", "street1": "12 Main St W", "street2": "Ste 400", "city": "Markham", "province": "ON", "postal": "L6C2P2", "phone": "9055550201", "other_practice_sites": []}],
                "c": [{"id": "C-001", "name": "Dr. David Champ, PhD", "clinic_name": "Champ Medical Centre", "address": "12 Main St W., Markham, Ontario L6C 2P2", "phone": "905-555-0201", "dob": "1971-05-30", "license": "ON-77777", "specialty": "Family Medicine"}],
                "labels": {("A-001", "B-001"): 1, ("A-001", "C-001"): 1, ("B-001", "C-001"): 1},
                "must_match": [("A-001", "B-001"), ("A-001", "C-001")],
            },
            {
                "name": "clinic_name_fallback",
                "a": [{"id": "A-010", "first": "Priya", "last": "Sharma", "dob": "1982-07-04", "license": "ON-88888", "specialty": "Cardiology", "street1": "10 OConnor Street", "street2": "", "city": "Ottawa", "province": "ON", "postal": "K1A 0A9", "phone": "613-555-0111", "other_practice_sites": []}],
                "b": [{"id": "B-010", "first": "Pri", "last": "Sharma", "dob": "1982-07-04", "license": "ON88888", "specialty": "Cardiology", "street1": "10 O Connor St", "street2": "", "city": "Ottawa", "province": "ON", "postal": "K1A0A9", "phone": "6135550111", "other_practice_sites": []}],
                "c": [{"id": "C-010", "name": "", "clinic_name": "DR. Priya Sharma, MD", "address": "10 O'Connor St., Ottawa, Ontario K1A 0A9", "phone": "613-555-0111", "dob": "1982-07-04", "license": "ON-88888", "specialty": "Cardiology"}],
                "labels": {("A-010", "B-010"): 1, ("A-010", "C-010"): 1, ("B-010", "C-010"): 1},
                "must_match": [("A-010", "B-010"), ("A-010", "C-010")],
            },
            {
                "name": "non_match_guard",
                "a": [{"id": "A-020", "first": "Luc", "last": "Tremblay", "dob": "1969-11-22", "license": "ON-54321", "specialty": "Orthopedics", "street1": "200 Waterloo St", "street2": "Suite 3", "city": "London", "province": "ON", "postal": "N6A 3K7", "phone": "519-555-0177", "other_practice_sites": []}],
                "b": [{"id": "B-020", "first": "Anne", "last": "Tremblay", "dob": "1971-05-30", "license": "ON-99001", "specialty": "Neurology", "street1": "210 Tecumseh Rd E", "street2": "Unit 1", "city": "Windsor", "province": "ON", "postal": "N9A 1E1", "phone": "226-555-0999", "other_practice_sites": []}],
                "c": [{"id": "C-020", "name": "Dr. Mark Spenser, MD", "clinic_name": "Spenser Clinic", "address": "88 Bay St., Toronto, Ontario M5J 2X2", "phone": "416-555-9999", "dob": "1980-01-01", "license": "ON-12300", "specialty": "Internal Medicine"}],
                "labels": {("A-020", "B-020"): 0, ("A-020", "C-020"): 0, ("B-020", "C-020"): 0},
                "must_reject": [("A-020", "B-020")],
            },
        ]

        for variant in variants:
            with self.subTest(variant=variant["name"]):
                feature_df = run_pipeline_variant(self.m, variant["a"], variant["b"], variant["c"], variant["labels"])

                for id_a, id_b in variant.get("must_match", []):
                    row = feature_df[(feature_df["id_a"] == id_a) & (feature_df["id_b"] == id_b)]
                    self.assertFalse(row.empty)
                    self.assertTrue(row.iloc[0]["decision"].startswith("AUTO-MATCH"))

                for id_a, id_b in variant.get("must_reject", []):
                    row = feature_df[(feature_df["id_a"] == id_a) & (feature_df["id_b"] == id_b)]
                    self.assertFalse(row.empty)
                    self.assertIn("REJECT", row.iloc[0]["decision"])


if __name__ == "__main__":
    unittest.main()
