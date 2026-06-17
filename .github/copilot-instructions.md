# Copilot Instructions

## Commands

**Install dependencies:**
```bash
pip install pandas scikit-learn jellyfish rapidfuzz recordlinkage xgboost coverage
```

**Run the main script:**
```bash
python provider_identify_resolution_poc.py
```

**Run all tests:**
```bash
python -m unittest discover -s tests -p "test_*.py"
```

**Run a single test file:**
```bash
python -m unittest tests.provider_registry_resolution.test_provider_identity_resolution_unit
python -m unittest tests.provider_registry_resolution.test_provider_identity_resolution_e2e
```

**Run a single test method:**
```bash
python -m unittest tests.provider_registry_resolution.test_provider_identity_resolution_unit.ProviderIdentityResolutionUnitTest.test_normalize_license
```

**Run coverage (enforced at 90%+):**
```bash
python -m coverage run -m unittest discover -s tests -p "test_*.py"
python -m coverage report -m --fail-under=90
```

> On Windows, set `PYTHONIOENCODING=utf-8` before running to avoid cp1252 errors with Unicode output.

## Architecture

The primary script is `provider_identify_resolution_poc.py` — a single-file pipeline for cross-registry Canadian healthcare provider identity resolution. It resolves provider records across up to three registry formats (A: CPSO-style structured, B: CNO-style structured with formatting variants, C: semi-structured with free-text name and one-line address).

At a high level, the pipeline has three phases: **Normalize** (stages 1–2), **Match** (stages 3–5), and **Merge** (stages 6–8). Each stage produces a well-defined output consumed only by the next stage.

**Pipeline stages (in order):**

| # | Stage | Key Function(s) | Input → Output |
|---|-------|-----------------|----------------|
| 1 | Preprocessing / normalization | `preprocess`, `resolve_person_name`, `build_primary_location` | Raw registry records → canonical `*_n` columns |
| 2 | Candidate blocking | `generate_candidate_pairs` | Normalized records → candidate pairs (Soundex + postal-FSA blocks across A↔B, A↔C, B↔C) |
| 3 | Feature engineering | `build_features` | Candidate pairs → ~27 similarity features (name, DOB, license, specialty, address, phone) |
| 4 | Rule engine | `rule_based_decision` | Feature rows → `MATCH` / `NON-MATCH` / `AMBIGUOUS` (deterministic: `lic_exact + dob_exact → MATCH`; conflicting signals → `NON-MATCH`) |
| 5 | ML fallback | `XGBClassifier` | `AMBIGUOUS` rows → match probability (≥0.75 AUTO-MATCH, 0.40–0.75 HUMAN REVIEW, <0.40 AUTO-REJECT) |
| 6 | Clustering | `build_match_clusters` | AUTO-MATCH edges → connected-component identity clusters |
| 7 | Golden record | `build_golden_record` | Cluster → single merged record (source trust: CPSO=1.00 > CNO=0.85 > REGC=0.80) |
| 8 | Confidence scoring | `compute_cluster_confidence` | Cluster edges → score (`0.8 × mean_edge_strength + 0.2 × edge_coverage`; −0.10 for `name_conflict`) |

**Missing / invalid data handling:** Records with missing or malformed required fields (e.g., no name, unparseable DOB) are logged as warnings during preprocessing and excluded from downstream stages. Downstream steps treat absent optional fields (e.g., phone, specialty) as null similarities (score = 0) rather than errors.

`detect_file_ext.py` is an independent standalone utility (not part of the pipeline).

## Prompt Validation

This workspace also includes a `prompt-tools` MCP server with a `validate_user_prompt` tool.

Use it when the user asks to validate, review, score, or improve a prompt before sending it to a model. If the validator reports `ready_for_model = false`, summarize the blocking issues first and use the suggested rewrite or follow-up questions to tighten the prompt.

## Key Conventions

**Normalized column naming:** All derived/normalized columns use a `_n` suffix (`first_n`, `last_n`, `license_n`, `postal_n`, etc.). Raw columns retain their original names. Feature columns are listed in the module-level `FEATURE_COLS` constant.

**Importing the main script in tests:** The script executes pipeline code at module level. Unit and E2E tests load it via `importlib.util.spec_from_file_location` with `redirect_stdout` to suppress output, then call `m.main()` explicitly when pipeline state is needed.

**Registry C name resolution:** Registry C uses free-text `name` and `clinic_name` fields. `split_provider_name` first tries `name`; if it is empty or contains any of the organization indicator keywords in `CLINIC_HINTS` — exactly: `'clinic'`, `'centre'`, `'center'`, `'health'`, `'medical'`, `'group'`, `'hospital'`, `'pharmacy'`, `'lab'`, `'laboratory'`, `'care'`, `'services'`, `'associates'` — it falls back to `clinic_name`. This is a frequent source of test variant coverage.

**Address matching uses base vs. full address:** `normalize_location` produces both `full_address` (with unit) and `base_address` (without unit). Matching on `base_address` is more robust because unit data is often missing or inconsistent across registries. Features expose both: `site_address_full_sim` and `site_address_base_sim`.

**Multi-site location scoring:** `best_location_similarity` compares every site in `location_sites_n` (primary + `other_practice_sites`) across both records and returns the best-scoring pair. This handles providers registered at their secondary location in one registry.

**E2E test harness:** `run_pipeline_variant()` in the E2E test file is a self-contained pipeline runner for synthetic datasets. When blocking yields no pairs, it falls back to a full Cartesian product. When all labelled pairs are single-class, XGBoost is bypassed and a constant probability is assigned.
