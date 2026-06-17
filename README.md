# Provider Identity Resolution Demo

This workspace contains a comprehensive demo for cross-registry identity resolution of healthcare providers.

Primary script:
- [provider_identify_resolution_poc.py](provider_identify_resolution_poc.py)

Additional utility:
- [rate_limiter.py](rate_limiter.py)

## Purpose

The script demonstrates how to reconcile provider identities across heterogeneous registry formats:
- registry A and B with structured first/last/address fields
- registry C with single name and single-line address fields

The result is a set of pairwise decisions and deduplicated golden records at cluster level.

## End-to-End Flow

The script runs through these stages:

1. Source ingestion
2. Parsing and normalization
3. Candidate blocking across registry pairs
4. Feature engineering
5. Rule-based fast path
6. ML fallback for ambiguous rows
7. Cross-pair clustering
8. Golden record construction
9. Cluster confidence scoring

## Data Models

### Registry A and B (structured)
Expected fields include:
- id
- first, last
- dob
- license
- specialty
- street1, street2, city, province, postal
- phone
- other_practice_sites (list)

### Registry C (semi-structured)
Expected fields include:
- id
- name (single provider display field)
- clinic_name (optional; may contain provider name in some sources)
- address (single full line)
- phone
- dob
- license
- specialty

## Parsing and Normalization Logic Details

### License normalization
Function: normalize_license
- Removes non-alphanumeric characters
- Uppercases result

Examples:
- ON-12345 -> ON12345
- on 99-xy -> ON99XY

### Postal normalization and FSA
Functions: normalize_postal, postal_fsa
- Postal code: uppercase, remove non-alphanumeric
- FSA: first 3 characters

Examples:
- M5G 1X8 -> M5G1X8
- FSA(M5G1X8) -> M5G

### Free-form provider name splitting
Function: split_provider_name
- Supports titles/suffixes and punctuation cleanup
- Falls back to clinic_name when name is empty/non-person-like

Examples:
- Dr. David Champ, PhD -> first=David, last=Champ
- DR. Mark Spenser, MD -> first=Mark, last=Spenser
- name="", clinic_name="DR. Priya Sharma, MD" -> first=Priya, last=Sharma

### Full-line address splitting
Function: parse_full_address
- Parses one-line addresses into street1, city, province, postal
- Uses Canadian postal regex extraction

Example:
- 12 Main St W., Markham, Ontario L6C 2P2
  - street1: 12 Main St W.
  - city: Markham
  - province: Ontario
  - postal: L6C2P2

### Province normalization
Function: normalize_province
- Maps long-form names to abbreviations

Examples:
- Ontario -> ON
- British Columbia -> BC

### Address line normalization
Functions: normalize_address_line, split_street_and_unit, normalize_location
- Standardizes road/street tokens
- Extracts/normalizes unit/suite independently from street core
- Produces both full address text and base address text (without unit)

Why base vs full:
- Missing/wrong suite is common in source data
- Matching on base address improves robustness while still tracking unit conflict

### Phone normalization
Function: normalize_phone
- Digits only
- Supports exact and last-7 matching as soft evidence

## Candidate Generation and Blocking

Function: generate_candidate_pairs
- Builds candidates by union of:
  - phonetic block on soundex
  - geographic block on postal FSA

Registry pair sets evaluated:
- A-B
- A-C
- B-C

This reduces full cartesian comparisons while keeping likely matches.

## FastAPI Rate Limiter Utility

The workspace also includes a small in-memory rate limiter that can be plugged
into a FastAPI route dependency.

Example:

```python
from fastapi import Depends, FastAPI

from rate_limiter import InMemoryRateLimiter

app = FastAPI()
limiter = InMemoryRateLimiter(limit=100, window_seconds=60)


@app.get("/providers", dependencies=[Depends(limiter.as_dependency())])
async def list_providers():
  return {"ok": True}
```

Behavior:
- fixed-window, in-memory counters keyed by client IP + method + path
- `X-RateLimit-*` and `Retry-After` headers on allowed and rejected requests
- raises HTTP 429 in FastAPI, or `RateLimitExceeded` when used without FastAPI

## Feature Engineering Details

Function: build_features

Feature groups:

1. Name signals
- jw_first, jw_last, jw_full
- fuzz_full
- clinic_name_sim
- soundex_match, metaphone_match

2. DOB signals
- dob_exact
- dob_year
- dob_month

3. License signals
- lic_exact
- lic_fuzzy

4. Specialty signal
- spec_sim

5. Address and location signals
- site_address_full_sim
- site_address_base_sim
- site_street_sim
- site_unit_match
- site_unit_conflict
- site_province_match
- postal_exact_match
- postal_fsa_match
- postal_sim
- site_city_sim

6. Phone and city signals
- phone_exact_match
- phone_last7_match
- phone_sim
- city_sim

## Rule Engine

Function: rule_based_decision

Deterministic decisions:
- MATCH when license exact and DOB exact
- NON-MATCH when key identity evidence diverges and contact/location is weak
- NON-MATCH when DOB year differs and phonetics diverge
- else AMBIGUOUS for ML

This protects precision and avoids obvious false positives from sparse training labels.

## ML Fallback

Model: XGBClassifier

For ambiguous rows only:
- model predicts match probability
- thresholds:
  - p >= 0.75 -> AUTO-MATCH
  - 0.40 <= p < 0.75 -> HUMAN REVIEW
  - p < 0.40 -> AUTO-REJECT

## Cross-Registry Clustering

After pairwise decisions:
- AUTO-MATCH edges form a graph
- connected components become identity clusters

Function: build_match_clusters

Why clustering matters:
- removes duplicate pairwise golden records
- yields one consolidated record per provider identity group

## Golden Record Logic

Function: build_golden_record

Rules:
- choose primary record by source trust priority
- merge other_practice_sites across all cluster members
- keep source_ids provenance
- flag name_conflict when cluster has multiple last names

Current source trust:
- CPSO: 1.00
- CNO: 0.85
- REGC: 0.80

## Cluster Confidence Scoring

Function: compute_cluster_confidence

Score inputs:
- mean edge strength in cluster
  - rule-based MATCH edge = 1.0
  - ambiguous edge = ml_prob
- edge coverage = supporting_edges / possible_edges

Formula:
- score = 0.8 * mean_edge_strength + 0.2 * edge_coverage

Level mapping:
- HIGH >= 0.90
- MEDIUM >= 0.75
- LOW < 0.75

Calibration:
- if name_conflict is true, subtract 0.10 from score (floor at 0.0)

Printed in output as:
- confidence level
- numeric score
- edges supporting/possible

## Address Validation Hook

Function: validate_business_address

The script includes a pluggable hook for business address validation.
Supported provider modes currently stubbed:
- canada_post
- smarty
- loqate
- melissa

Current behavior is placeholder status reporting until live API credentials and endpoints are wired.

## Running

Install deps:

```bash
pip install pandas scikit-learn jellyfish rapidfuzz recordlinkage xgboost coverage
```

Run script:

```bash
python provider_identify_resolution_poc.py
```

## Testing

Test modules:
- [tests/test_provider_identity_resolution.py](tests/test_provider_identity_resolution.py)
- [tests/test_provider_identity_resolution_unit.py](tests/test_provider_identity_resolution_unit.py)
- [tests/test_provider_identity_resolution_e2e.py](tests/test_provider_identity_resolution_e2e.py)

Run all tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Run coverage and enforce 90%+:

```bash
python -m coverage run -m unittest discover -s tests -p "test_*.py"
python -m coverage report -m --fail-under=90
```

## E2E Variant Coverage Included

The E2E suite covers major variants:
- title/suffix name parsing (Dr., MD, PhD)
- clinic-name fallback when provider name field is empty
- one-line address parsing across punctuation variants
- unit/suite formatting differences
- deterministic non-match guard under conflicting identity evidence
- cross-registry matching across A-B, A-C, B-C

# PDF Reader & Writer MCP Utility

This workspace also includes a layout-aware PDF reader utility and a local MCP
server so LLM clients can inspect PDFs without relying on plain-text extraction
alone.

The same MCP server also exposes a PDF writer utility for converting text,
Markdown, XML, YAML, HTML, CSV, Excel, and Word documents into a generated PDF.

Files:
- `pdf_reader.py` - standalone utility for plain-text and layout-aware PDF extraction
- `pdf_writer.py` - standalone utility for writing PDFs from text, markup, spreadsheet, and Word inputs
- `pdf_mcp_server.py` - stdio MCP server exposing the reader as tools
- `.vscode/mcp.json` - workspace MCP registration for GitHub Copilot in VS Code
- `.mcp.json` - project-scoped registration for Claude Code

Utility example:

```powershell
& .\.venv\Scripts\python.exe .\pdf_reader.py "C:\path\to\document.pdf" --pages 1 2 --max-chars 5000
& .\.venv\Scripts\python.exe .\pdf_writer.py ".\notes.md" ".\notes.pdf"
```

The utility returns:
- document metadata
- bookmark or outline entries when present
- plain text per page
- layout-aware text per page
- likely heading candidates
- likely table blocks based on preserved spacing

The writer utility returns:
- source path and detected source type
- output PDF path
- generated PDF page count

Writer dependencies:
- `reportlab` for PDF generation
- `beautifulsoup4` for robust HTML parsing
- `openpyxl` for Excel input
- `python-docx` for Word input

GitHub Copilot in VS Code:
- open Chat after trusting the server in `.vscode/mcp.json`
- the `pdf-tools` MCP server exposes `inspect_pdf`, `inspect_pdf_page`, and `write_pdf_from_file`

Claude Code:
- the project already includes `.mcp.json`
- or add it manually with:

```bash
claude mcp add --transport stdio pdf-tools -- ./.venv/Scripts/python.exe ./pdf_mcp_server.py
```

Other MCP clients, including Codex-compatible MCP setups:
- point the client at the same stdio command and args used above
- the server is standard MCP over stdio and does not depend on VS Code-specific APIs
