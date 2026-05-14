"""
Canadian Healthcare Provider Identity Resolution
=================================================
Fuzzy matching + ML classifier pipeline for deduplicating provider records
across provincial registries (CPSO, CPSBC, CNO, CPhA, etc.)

Dependencies:
    pip install pandas scikit-learn jellyfish rapidfuzz recordlinkage xgboost
"""

import os  # Environment variables for optional external address validation.
import re  # Regular expressions for normalizing licenses and postal codes.
import jellyfish  # Phonetic and string similarity helpers for name matching.
import recordlinkage  # Blocking and candidate-pair generation for record linkage.
import pandas as pd  # DataFrame operations for registry data and feature tables.
import numpy as np  # Numeric utilities and NaN handling for labels and arrays.
from rapidfuzz import fuzz  # Fast fuzzy string similarity scores for names and specialties.
from sklearn.preprocessing import StandardScaler  # Feature scaling before training the classifier.
from sklearn.model_selection import train_test_split  # Train/test splitting for production evaluation.
from sklearn.metrics import classification_report  # Precision/recall summary for match quality.
from xgboost import XGBClassifier  # Gradient-boosted classifier for match probability prediction.


# ─────────────────────────────────────────────
# 1. SAMPLE DATA  (two registries, same providers, messy real-world variation)
# ─────────────────────────────────────────────

registry_a = pd.DataFrame([
    {"id": "CPSO-001", "first": "James",     "last": "Okafor",    "dob": "1975-03-12", "license": "ON-12345", "specialty": "Family Medicine",  "street1": "123 King Street West", "street2": "Suite 400", "city": "Toronto",   "province": "ON", "postal": "M5G 1X8", "phone": "416-555-0101", "other_practice_sites": [{"street1": "25 Hurontario St", "street2": "Unit 6", "city": "Mississauga", "province": "ON", "postal": "L5B 2C9", "phone": "416-555-0101"}]},
    {"id": "CPSO-002", "first": "Priya",     "last": "Sharma",    "dob": "1982-07-04", "license": "ON-67890", "specialty": "Cardiology",        "street1": "10 O'Connor Street",  "street2": "Unit 805", "city": "Ottawa",    "province": "ON", "postal": "K1A 0A9", "phone": "613-555-0144", "other_practice_sites": [{"street1": "1100 Beaverbrook Road", "street2": "Suite 210", "city": "Kanata", "province": "ON", "postal": "K2K 3E7", "phone": "613-555-0144"}]},
    {"id": "CPSO-003", "first": "Luc",       "last": "Tremblay",  "dob": "1969-11-22", "license": "ON-54321", "specialty": "Orthopedics",       "street1": "200 Waterloo St",    "street2": "Suite 3",   "city": "London",    "province": "ON", "postal": "N6A 3K7", "phone": "519-555-0177", "other_practice_sites": [{"street1": "400 Dundas St", "street2": "", "city": "Woodstock", "province": "ON", "postal": "N4S 7V9", "phone": "519-555-0177"}]},
    {"id": "CPSO-004", "first": "Margaret",  "last": "Chen",      "dob": "1990-01-15", "license": "ON-11223", "specialty": "Pediatrics",        "street1": "88 Main Street East", "street2": "Unit 2A",   "city": "Hamilton",  "province": "ON", "postal": "L8N 3Z5", "phone": "905-555-0199", "other_practice_sites": [{"street1": "22 Plains Road East", "street2": "Suite 11", "city": "Burlington", "province": "ON", "postal": "L7S 1T7", "phone": "905-555-0199"}]},
])

registry_b = pd.DataFrame([
    # Slight name typo, license formatted differently, postal different format
    {"id": "CNO-001",  "first": "James",     "last": "Okafor",    "dob": "1975-03-12", "license": "ON12345",  "specialty": "Fam. Medicine",    "street1": "123 King St W",      "street2": "Ste 400",  "city": "Toronto",   "province": "ON", "postal": "M5G1X8", "phone": "4165550101", "other_practice_sites": [{"street1": "25 Hurontario Street", "street2": "#6", "city": "Etobicoke", "province": "ON", "postal": "M9C 5N3", "phone": "4165550101"}]},
    # Nickname vs full name
    {"id": "CNO-002",  "first": "Pri",       "last": "Sharma",    "dob": "1982-07-04", "license": "ON-67890", "specialty": "Cardiology",        "street1": "10 O Connor St",     "street2": "",         "city": "Ottawa",    "province": "ON", "postal": "K1A 0A9", "phone": "6135550144", "other_practice_sites": []},
    # Different person entirely
    {"id": "CNO-003",  "first": "Anne",      "last": "Tremblay",  "dob": "1971-05-30", "license": "ON-99001", "specialty": "Neurology",         "street1": "210 Tecumseh Rd E",  "street2": "Unit 1",  "city": "Windsor",   "province": "ON", "postal": "N9A 1E1", "phone": "2265550999", "other_practice_sites": [{"street1": "400 Dougal Ave", "street2": "", "city": "Essex", "province": "ON", "postal": "N8M 1A6", "phone": "2265550999"}]},
    # Married name change, postal drift
    {"id": "CNO-004",  "first": "Margaret",  "last": "Wong",      "dob": "1990-01-15", "license": "ON-11223", "specialty": "Pediatrics",        "street1": "88 Main St E",       "street2": "Suite 3",  "city": "Hamilton",  "province": "ON", "postal": "L8N 3Z5", "phone": "9055550199", "other_practice_sites": [{"street1": "22 Plains Road East", "street2": "Suite 11", "city": "Stoney Creek", "province": "ON", "postal": "L8J 1X1", "phone": "9055550199"}]},
])

registry_c = pd.DataFrame([
    {
        "id": "REGC-001",
        "name": "Dr. James Okafor, MD",
        "clinic_name": "Okafor Family Medicine",
        "address": "123 King St W., Toronto, Ontario M5G 1X8",
        "phone": "416-555-0101",
        "dob": "1975-03-12",
        "license": "ON-12345",
        "specialty": "Family Medicine",
    },
    {
        "id": "REGC-002",
        "name": "",
        "clinic_name": "DR. Priya Sharma, MD",
        "address": "10 O'Connor St., Ottawa, Ontario K1A 0A9",
        "phone": "613-555-0144",
        "dob": "1982-07-04",
        "license": "ON-67890",
        "specialty": "Cardiology",
    },
    {
        "id": "REGC-003",
        "name": "Dr. Margaret Chen",
        "clinic_name": "Hamilton Pediatrics",
        "address": "88 Main St E., Hamilton, Ontario L8N 3Z5",
        "phone": "905-555-0199",
        "dob": "1990-01-15",
        "license": "ON-11223",
        "specialty": "Pediatrics",
    },
    {
        "id": "REGC-004",
        "name": "Dr. David Champ, PhD",
        "clinic_name": "Champ Medical Centre",
        "address": "12 Main St W., Markham, Ontario L6C 2P2",
        "phone": "905-555-0210",
        "dob": "1971-05-30",
        "license": "ON-99001",
        "specialty": "General Practice",
    },
])


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def normalize_license(lic: str) -> str:
    """Strip non-alphanumeric chars and upper-case."""
    return re.sub(r"[^A-Z0-9]", "", lic.upper()) if lic else ""

def normalize_postal(postal: str) -> str:
    """Normalize to compact alphanumeric form (e.g., M5G 1X8 -> M5G1X8)."""
    return re.sub(r"[^A-Z0-9]", "", postal.upper()) if postal else ""

def postal_fsa(postal: str) -> str:
    """Return the FSA (first 3 chars) from a normalized postal code."""
    return postal[:3] if postal else ""

NAME_TITLES = {"DR", "DOCTOR", "MR", "MRS", "MS", "MISS", "PROF"}
NAME_SUFFIXES = {"MD", "PHD", "DO", "DDS", "DMD", "MBBS", "FRCSC", "FRCPC", "NP", "RN", "BSC", "MSC"}
CLINIC_HINTS = {"clinic", "centre", "center", "medical", "medicine", "practice", "health", "care", "hospital"}

def looks_like_person_name(text: str) -> bool:
    """Heuristic to tell whether a free-text field is likely a person name."""
    normalized = re.sub(r"[^A-Z0-9\s]", " ", text.upper()) if text else ""
    tokens = [token for token in normalized.split() if token]
    if len(tokens) < 2:
        return False
    if any(token.lower() in CLINIC_HINTS for token in tokens):
        return False
    return True

def split_provider_name(raw_name: str, clinic_name: str = "") -> dict[str, str]:
    """Split a provider display name into first and last name tokens."""
    candidate = raw_name.strip() if raw_name else ""
    if not candidate or not looks_like_person_name(candidate):
        if clinic_name and looks_like_person_name(clinic_name):
            candidate = clinic_name.strip()

    candidate = candidate.split(",")[0]
    candidate = re.sub(r"\b(?:DR|DOCTOR|MR|MRS|MS|MISS|PROF)\.?\b", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\b(?:MD|PHD|DO|DDS|DMD|MBBS|FRCSC|FRCPC|NP|RN|BSC|MSC)\b", " ", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"[^A-Za-z0-9\s\-']", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    tokens = candidate.split()

    if not tokens:
        return {"first": "", "last": "", "full": ""}
    if len(tokens) == 1:
        return {"first": tokens[0], "last": "", "full": tokens[0]}
    return {"first": tokens[0], "last": tokens[-1], "full": candidate}

CANADIAN_POSTAL_RE = re.compile(r"\b([A-Z]\d[A-Z])[\s-]?(\d[A-Z]\d)\b", re.IGNORECASE)

def parse_full_address(address_line: str) -> dict[str, str]:
    """Split a one-line Canadian address into street, city, province, and postal parts."""
    if not address_line:
        return {"street1": "", "street2": "", "city": "", "province": "", "postal": ""}

    text = re.sub(r"\s+", " ", address_line).strip()
    segments = [segment.strip() for segment in text.split(",") if segment.strip()]

    street = segments[0] if segments else text
    city = segments[1] if len(segments) > 1 else ""
    province_postal = ", ".join(segments[2:]) if len(segments) > 2 else ""

    postal_match = CANADIAN_POSTAL_RE.search(province_postal) or CANADIAN_POSTAL_RE.search(text)
    postal = postal_match.group(0).upper().replace(" ", "") if postal_match else ""
    province = province_postal[: postal_match.start()].strip() if postal_match else province_postal
    province = province.replace(",", " ").strip()

    return {
        "street1": street,
        "street2": "",
        "city": city,
        "province": province,
        "postal": postal,
    }

PROVINCE_ALIASES = {
    "ONTARIO": "ON",
    "QUEBEC": "QC",
    "NOVA SCOTIA": "NS",
    "NEW BRUNSWICK": "NB",
    "MANITOBA": "MB",
    "BRITISH COLUMBIA": "BC",
    "PRINCE EDWARD ISLAND": "PE",
    "SASKATCHEWAN": "SK",
    "ALBERTA": "AB",
    "NEWFOUNDLAND AND LABRADOR": "NL",
    "NEWFOUNDLAND": "NL",
    "YUKON": "YT",
    "NORTHWEST TERRITORIES": "NT",
    "NUNAVUT": "NU",
}

ADDRESS_TOKEN_MAP = {
    "STREET": "ST",
    "ST": "ST",
    "AVENUE": "AVE",
    "AVE": "AVE",
    "BOULEVARD": "BLVD",
    "BLVD": "BLVD",
    "ROAD": "RD",
    "RD": "RD",
    "DRIVE": "DR",
    "DR": "DR",
    "COURT": "CT",
    "CT": "CT",
    "LANE": "LN",
    "LN": "LN",
    "PLACE": "PL",
    "PL": "PL",
    "WEST": "W",
    "EAST": "E",
    "NORTH": "N",
    "SOUTH": "S",
    "SUITE": "STE",
    "STE": "STE",
    "APARTMENT": "APT",
    "APT": "APT",
    "UNIT": "UNIT",
    "FLOOR": "FL",
    "LEVEL": "FL",
}

def normalize_phone(phone: str) -> str:
    """Normalize to digits only so formatting differences do not matter."""
    return re.sub(r"\D", "", phone) if phone else ""

def normalize_province(province: str) -> str:
    """Normalize Canadian province/territory names and abbreviations."""
    if not province:
        return ""
    compact = re.sub(r"[^A-Z]", "", province.upper())
    words = re.sub(r"\s+", " ", province.upper()).strip()
    return PROVINCE_ALIASES.get(words, PROVINCE_ALIASES.get(compact, compact))

def normalize_address_line(text: str) -> str:
    """Normalize address text while preserving meaningful unit/street tokens."""
    if not text:
        return ""
    cleaned = text.upper().replace("#", " UNIT ")
    cleaned = re.sub(r"[.,]", " ", cleaned)
    cleaned = re.sub(r"[^A-Z0-9/\-\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    tokens = [ADDRESS_TOKEN_MAP.get(token, token) for token in cleaned.split()]
    return " ".join(tokens)

def split_street_and_unit(street1: str, street2: str) -> tuple[str, str]:
    """Extract a stable street core and unit/suite when one is present."""
    street_core = normalize_address_line(street1)
    unit_raw = normalize_address_line(street2)
    unit = re.sub(r"^(?:STE|SUITE|APT|APARTMENT|UNIT)\s*", "", unit_raw).strip()

    if not unit:
        match = re.match(r"^(?:STE|SUITE|APT|APARTMENT|UNIT)\s*([A-Z0-9\-]+)\s+(.*)$", street_core)
        if match:
            unit = match.group(1)
            street_core = match.group(2)

    return street_core, unit

def build_location_text(street_core: str, unit: str, city: str, province: str, postal: str) -> str:
    parts = [street_core]
    if unit:
        parts.append(f"UNIT {unit}")
    parts.extend([city, province, postal])
    return " | ".join(part for part in parts if part)

def normalize_location(site, fallback_city: str, fallback_province: str, fallback_postal: str, fallback_phone: str = "") -> dict[str, str]:
    """Normalize a business location record into comparison-ready fields."""
    street_core, unit = split_street_and_unit(site.get("street1", ""), site.get("street2", ""))
    city = normalize_name(site.get("city", fallback_city))
    province = normalize_province(site.get("province", fallback_province))
    postal = normalize_postal(site.get("postal", fallback_postal))
    phone = normalize_phone(site.get("phone", fallback_phone))
    full_address = build_location_text(street_core, unit, city, province, postal)
    base_address = build_location_text(street_core, "", city, province, postal)

    return {
        "street_core": street_core,
        "unit": unit,
        "city": city,
        "province": province,
        "postal": postal,
        "fsa": postal_fsa(postal),
        "phone": phone,
        "full_address": full_address,
        "base_address": base_address,
    }

def normalize_sites(sites, fallback_city: str, fallback_province: str, fallback_postal: str, fallback_phone: str = "") -> list[dict[str, str]]:
    """Normalize all secondary practice sites; fall back only when a field is missing."""
    raw_sites = sites if isinstance(sites, list) and sites else []
    normalized_sites = []
    for site in raw_sites:
        normalized_sites.append(
            normalize_location(site, fallback_city, fallback_province, fallback_postal, fallback_phone)
        )
    return normalized_sites

def build_primary_location(row: pd.Series) -> dict[str, str]:
    """Normalize either a structured primary location or a one-line address field."""
    if row.get("address"):
        parsed = parse_full_address(row["address"])
        parsed["phone"] = row.get("phone", "")
        return parsed

    return {
        "street1": row.get("street1", ""),
        "street2": row.get("street2", ""),
        "city": row.get("city", ""),
        "province": row.get("province", ""),
        "postal": row.get("postal", ""),
        "phone": row.get("phone", ""),
    }

def resolve_person_name(row: pd.Series) -> dict[str, str]:
    """Resolve provider name from explicit fields or from a name-like free-text field."""
    first = row.get("first", "") or ""
    last = row.get("last", "") or ""
    clinic_name = row.get("clinic_name", "") or ""
    raw_name = row.get("name", "") or ""

    parsed = split_provider_name(raw_name, clinic_name)
    if not first:
        first = parsed["first"]
    if not last:
        last = parsed["last"]

    return {"first": first, "last": last, "clinic_name": clinic_name}

def validate_business_address(location: dict[str, str], provider: str = "canada_post") -> dict[str, str | bool]:
    """Optional hook for Canada Post or another external address-validation service."""
    provider = provider.lower().strip()
    if provider == "canada_post":
        api_key = os.getenv("CANADA_POST_API_KEY", "")
        api_secret = os.getenv("CANADA_POST_API_SECRET", "")
        if not api_key or not api_secret:
            return {"provider": provider, "validated": False, "status": "not_configured"}
        return {"provider": provider, "validated": False, "status": "endpoint_not_wired"}
    if provider in {"smarty", "loqate", "melissa"}:
        return {"provider": provider, "validated": False, "status": "not_configured"}
    return {"provider": provider, "validated": False, "status": "unsupported"}

def best_location_similarity(locations_a: list[dict[str, str]], locations_b: list[dict[str, str]]) -> dict[str, float]:
    """Use the strongest matching location pair, preferring full address and base-address agreement."""
    if not locations_a or not locations_b:
        return {
            "site_address_full_sim": 0.0,
            "site_address_base_sim": 0.0,
            "site_street_sim": 0.0,
            "site_unit_match": 0,
            "site_unit_conflict": 0,
            "site_province_match": 0,
            "site_postal_exact_match": 0,
            "site_postal_fsa_match": 0,
            "site_postal_sim": 0.0,
            "site_city_sim": 0.0,
            "site_phone_exact_match": 0,
            "site_phone_last7_match": 0,
            "site_phone_sim": 0.0,
        }

    best = {
        "site_address_full_sim": 0.0,
        "site_address_base_sim": 0.0,
        "site_street_sim": 0.0,
        "site_unit_match": 0,
        "site_unit_conflict": 0,
        "site_province_match": 0,
        "site_postal_exact_match": 0,
        "site_postal_fsa_match": 0,
        "site_postal_sim": 0.0,
        "site_city_sim": 0.0,
        "site_phone_exact_match": 0,
        "site_phone_last7_match": 0,
        "site_phone_sim": 0.0,
    }
    best_score = -1.0

    for site_a in locations_a:
        for site_b in locations_b:
            street_full_sim = fuzz.token_sort_ratio(site_a["full_address"], site_b["full_address"]) / 100.0
            street_base_sim = fuzz.token_sort_ratio(site_a["base_address"], site_b["base_address"]) / 100.0
            street_sim = fuzz.token_sort_ratio(site_a["street_core"], site_b["street_core"]) / 100.0
            unit_match = int(bool(site_a["unit"] and site_b["unit"] and site_a["unit"] == site_b["unit"]))
            unit_conflict = int(bool(site_a["unit"] and site_b["unit"] and site_a["unit"] != site_b["unit"]))
            province_match = int(site_a["province"] == site_b["province"] and site_a["province"] != "")
            postal_exact_match = int(site_a["postal"] == site_b["postal"] and site_a["postal"] != "")
            postal_fsa_match = int(site_a["fsa"] == site_b["fsa"] and site_a["fsa"] != "")
            postal_sim = fuzz.ratio(site_a["postal"], site_b["postal"]) / 100.0
            city_sim = jellyfish.jaro_winkler_similarity(site_a["city"], site_b["city"])
            phone_exact_match = int(site_a["phone"] == site_b["phone"] and site_a["phone"] != "")
            phone_last7_match = int(
                len(site_a["phone"]) >= 7
                and len(site_b["phone"]) >= 7
                and site_a["phone"][-7:] == site_b["phone"][-7:]
            )
            phone_sim = fuzz.ratio(site_a["phone"], site_b["phone"]) / 100.0

            score = (
                street_base_sim * 3.0
                + street_full_sim * 1.5
                + street_sim
                + province_match * 1.0
                + postal_exact_match * 3.0
                + postal_fsa_match * 2.0
                + postal_sim
                + city_sim
                + phone_exact_match * 0.75
                + phone_last7_match * 0.25
                - unit_conflict * 0.25
            )

            if score > best_score:
                best_score = score
                best = {
                    "site_address_full_sim": street_full_sim,
                    "site_address_base_sim": street_base_sim,
                    "site_street_sim": street_sim,
                    "site_unit_match": unit_match,
                    "site_unit_conflict": unit_conflict,
                    "site_province_match": province_match,
                    "site_postal_exact_match": postal_exact_match,
                    "site_postal_fsa_match": postal_fsa_match,
                    "site_postal_sim": postal_sim,
                    "site_city_sim": city_sim,
                    "site_phone_exact_match": phone_exact_match,
                    "site_phone_last7_match": phone_last7_match,
                    "site_phone_sim": phone_sim,
                }

    return best

def normalize_name(name: str) -> str:
    return re.sub(r"[^\w\s]", "", re.sub(r"\s+", " ", name)).strip().lower() if name else ""

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    name_parts = df.apply(resolve_person_name, axis=1)
    location_parts = df.apply(build_primary_location, axis=1)
    df["first"] = name_parts.apply(lambda item: item["first"])
    df["last"] = name_parts.apply(lambda item: item["last"])
    df["clinic_name"] = name_parts.apply(lambda item: item["clinic_name"])
    df["street1"] = location_parts.apply(lambda item: item["street1"])
    df["street2"] = location_parts.apply(lambda item: item["street2"])
    df["city"] = location_parts.apply(lambda item: item["city"])
    df["province"] = location_parts.apply(lambda item: item["province"])
    df["postal"] = location_parts.apply(lambda item: item["postal"])
    df["phone"] = location_parts.apply(lambda item: item["phone"])
    df["first_n"]   = df["first"].apply(normalize_name)
    df["last_n"]    = df["last"].apply(normalize_name)
    df["full_n"]    = df["first_n"] + " " + df["last_n"]
    df["clinic_name_n"] = df["clinic_name"].fillna("").apply(normalize_name)
    df["license_n"] = df["license"].apply(normalize_license)
    df["postal_n"]  = df["postal"].apply(normalize_postal)
    df["postal_fsa"] = df["postal_n"].apply(postal_fsa)
    df["dob"]       = pd.to_datetime(df["dob"], errors="coerce")
    df["soundex"]   = df["last_n"].apply(lambda x: jellyfish.soundex(x) if x else "")
    df["metaphone"] = df["last_n"].apply(lambda x: jellyfish.metaphone(x) if x else "")
    df["primary_location_n"] = df.apply(
        lambda row: normalize_location(
            build_primary_location(row),
            row.get("city", ""),
            row.get("province", ""),
            row.get("postal", ""),
            row.get("phone", ""),
        ),
        axis=1,
    )
    df["primary_address_validation"] = df["primary_location_n"].apply(validate_business_address)
    df["primary_address_validation_status"] = df["primary_address_validation"].apply(lambda result: result["status"])
    df["other_practice_sites_n"] = df.apply(
        lambda row: normalize_sites(
            row.get("other_practice_sites"),
            row["city"],
            row.get("province", ""),
            row["postal"],
            row.get("phone", ""),
        ),
        axis=1,
    )
    df["location_sites_n"] = df.apply(
        lambda row: [row["primary_location_n"]] + row["other_practice_sites_n"],
        axis=1,
    )
    return df

reg_a = preprocess(registry_a)
reg_b = preprocess(registry_b)
reg_c = preprocess(registry_c)


# ─────────────────────────────────────────────
# 3. BLOCKING + CROSS-REGISTRY PAIR GENERATION
#    Compare across A↔B, A↔C, and B↔C using phonetic and postal blocks.
# ─────────────────────────────────────────────

def generate_candidate_pairs(a: pd.DataFrame, b: pd.DataFrame) -> pd.MultiIndex:
    """Union candidate pairs from phonetic and geographic blocking."""
    soundex_indexer = recordlinkage.Index()
    soundex_indexer.block(left_on="soundex", right_on="soundex")
    soundex_pairs = set(soundex_indexer.index(a, b).tolist())

    postal_indexer = recordlinkage.Index()
    postal_indexer.block(left_on="postal_fsa", right_on="postal_fsa")
    postal_pairs = set(postal_indexer.index(a, b).tolist())

    merged = sorted(soundex_pairs | postal_pairs)
    return pd.MultiIndex.from_tuples(merged)


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING  (one row per candidate pair)
# ─────────────────────────────────────────────

def build_features(pairs, a: pd.DataFrame, b: pd.DataFrame, pair_name: str) -> pd.DataFrame:
    """
    For each candidate pair (i, j) compute similarity features.
    Each feature encodes a different signal — the classifier learns weights.
    """
    rows = []
    for idx_a, idx_b in pairs:
        ra, rb = a.loc[idx_a], b.loc[idx_b]

        # ── Name similarities ──────────────────────────────────────────────
        jw_first   = jellyfish.jaro_winkler_similarity(ra["first_n"], rb["first_n"])
        jw_last    = jellyfish.jaro_winkler_similarity(ra["last_n"],  rb["last_n"])
        jw_full    = jellyfish.jaro_winkler_similarity(ra["full_n"],  rb["full_n"])
        clinic_name_sim = fuzz.token_sort_ratio(ra["clinic_name_n"], rb["clinic_name_n"]) / 100.0
        fuzz_full  = fuzz.token_sort_ratio(ra["full_n"], rb["full_n"]) / 100.0
        # Phonetic match (exact after encoding)
        soundex_match   = int(ra["soundex"]   == rb["soundex"])
        metaphone_match = int(ra["metaphone"] == rb["metaphone"])

        # ── Date of birth ─────────────────────────────────────────────────
        if pd.notna(ra["dob"]) and pd.notna(rb["dob"]):
            dob_exact  = int(ra["dob"] == rb["dob"])
            dob_year   = int(ra["dob"].year == rb["dob"].year)
            dob_month  = int(ra["dob"].month == rb["dob"].month)
        else:
            dob_exact = dob_year = dob_month = 0

        # ── License number ────────────────────────────────────────────────
        lic_exact  = int(ra["license_n"] == rb["license_n"])
        lic_fuzzy  = fuzz.ratio(ra["license_n"], rb["license_n"]) / 100.0

        # ── Specialty ─────────────────────────────────────────────────────
        spec_sim   = fuzz.token_set_ratio(
                         ra["specialty"].lower(), rb["specialty"].lower()
                     ) / 100.0

        # ── Geography / contact ─────────────────────────────────────────
        location_features = best_location_similarity(ra["location_sites_n"], rb["location_sites_n"])
        city_sim = jellyfish.jaro_winkler_similarity(ra["city"].lower(), rb["city"].lower())
        phone_a = normalize_phone(ra.get("phone", ""))
        phone_b = normalize_phone(rb.get("phone", ""))
        phone_exact_match = int(phone_a == phone_b and phone_a != "")
        phone_last7_match = int(
            len(phone_a) >= 7
            and len(phone_b) >= 7
            and phone_a[-7:] == phone_b[-7:]
        )
        phone_sim = fuzz.ratio(phone_a, phone_b) / 100.0

        rows.append({
            "pair_name": pair_name,
            "idx_a": idx_a,
            "idx_b": idx_b,
            "id_a": ra["id"],
            "id_b": rb["id"],
            # name
            "jw_first":        jw_first,
            "jw_last":         jw_last,
            "jw_full":         jw_full,
            "clinic_name_sim":  clinic_name_sim,
            "fuzz_full":       fuzz_full,
            "soundex_match":   soundex_match,
            "metaphone_match": metaphone_match,
            # dob
            "dob_exact":       dob_exact,
            "dob_year":        dob_year,
            "dob_month":       dob_month,
            # license
            "lic_exact":       lic_exact,
            "lic_fuzzy":       lic_fuzzy,
            # specialty
            "spec_sim":        spec_sim,
            # geography
            "site_address_full_sim": location_features["site_address_full_sim"],
            "site_address_base_sim": location_features["site_address_base_sim"],
            "site_street_sim":      location_features["site_street_sim"],
            "site_unit_match":      location_features["site_unit_match"],
            "site_unit_conflict":    location_features["site_unit_conflict"],
            "site_province_match":   location_features["site_province_match"],
            "postal_exact_match":    location_features["site_postal_exact_match"],
            "postal_fsa_match":      location_features["site_postal_fsa_match"],
            "postal_sim":            location_features["site_postal_sim"],
            "site_city_sim":         location_features["site_city_sim"],
            "phone_exact_match":      phone_exact_match,
            "phone_last7_match":      phone_last7_match,
            "phone_sim":              phone_sim,
            "city_sim":               city_sim,
        })

    return pd.DataFrame(rows)

PAIR_CONFIGS = [
    ("A-B", reg_a, reg_b),
    ("A-C", reg_a, reg_c),
    ("B-C", reg_b, reg_c),
]

feature_frames = []
for pair_name, left_df, right_df in PAIR_CONFIGS:
    pairs = generate_candidate_pairs(left_df, right_df)
    feature_frames.append(build_features(pairs, left_df, right_df, pair_name))
    print(f"{pair_name}: total pairs={len(left_df) * len(right_df)}  candidates={len(pairs)}")

feature_df = pd.concat(feature_frames, ignore_index=True)
print("\nFeature matrix (candidate pairs × features):")
print(feature_df.drop(columns=["idx_a","idx_b"]).round(2).to_string(), "\n")


# ─────────────────────────────────────────────
# 5. LABELLED TRAINING DATA
#    In production: stewards review a sample of pairs and label match/non-match.
#    Here we define ground truth manually for the demo.
#    Pair format: (provider id A, provider id B)
# ─────────────────────────────────────────────

ground_truth_by_id = {
    # A ↔ B
    ("CPSO-001", "CNO-001"): 1,
    ("CPSO-002", "CNO-002"): 1,
    ("CPSO-003", "CNO-003"): 0,
    ("CPSO-004", "CNO-004"): 1,
    # A ↔ C
    ("CPSO-001", "REGC-001"): 1,
    ("CPSO-002", "REGC-002"): 1,
    ("CPSO-004", "REGC-003"): 1,
    ("CPSO-003", "REGC-004"): 0,
    # B ↔ C
    ("CNO-001", "REGC-001"): 1,
    ("CNO-002", "REGC-002"): 1,
    ("CNO-004", "REGC-003"): 1,
    ("CNO-003", "REGC-004"): 0,
}

def lookup_label(id_a: str, id_b: str) -> float:
    if (id_a, id_b) in ground_truth_by_id:
        return ground_truth_by_id[(id_a, id_b)]
    if (id_b, id_a) in ground_truth_by_id:
        return ground_truth_by_id[(id_b, id_a)]
    return np.nan

# Attach labels to feature rows
feature_df["label"] = feature_df.apply(
    lambda r: lookup_label(r["id_a"], r["id_b"]), axis=1
)
labelled = feature_df.dropna(subset=["label"])
print(f"Labelled pairs: {len(labelled)}  (matches: {int(labelled['label'].sum())})\n")

FEATURE_COLS = [
    "jw_first","jw_last","jw_full","clinic_name_sim","fuzz_full",
    "soundex_match","metaphone_match",
    "dob_exact","dob_year","dob_month",
    "lic_exact","lic_fuzzy",
    "spec_sim",
    "site_address_full_sim","site_address_base_sim","site_street_sim",
    "site_unit_match","site_unit_conflict","site_province_match",
    "postal_exact_match","postal_fsa_match","postal_sim","site_city_sim",
    "phone_exact_match","phone_last7_match","phone_sim",
    "city_sim",
]

X = labelled[FEATURE_COLS].values
y = labelled["label"].astype(int).values


# ─────────────────────────────────────────────
# 6. RULE-BASED BASELINE (deterministic fast-path)
#    High-confidence matches bypass the classifier entirely.
#    Anything ambiguous is handed to ML.
# ─────────────────────────────────────────────

def rule_based_decision(row: pd.Series) -> str:
    """
    Returns 'MATCH', 'NON-MATCH', or 'AMBIGUOUS' (→ send to ML classifier).
    """
    # Definite match: same license + same DOB
    if row["lic_exact"] == 1 and row["dob_exact"] == 1:
        return "MATCH"
    # Strong non-match: key identity fields diverge and contact/location evidence is weak
    if (
        row["lic_exact"] == 0
        and row["dob_year"] == 0
        and row["jw_first"] < 0.6
        and row["site_address_base_sim"] < 0.75
        and row["phone_last7_match"] == 0
    ):
        return "NON-MATCH"
    # Definite non-match: completely different DOB year and last-name phonetics differ
    if row["dob_year"] == 0 and row["soundex_match"] == 0:
        return "NON-MATCH"
    return "AMBIGUOUS"

feature_df["rule_decision"] = feature_df[FEATURE_COLS].apply(rule_based_decision, axis=1)


# ─────────────────────────────────────────────
# 7. ML CLASSIFIER  (XGBoost — handles missing features, non-linear interactions)
#    Trained on steward-labelled pairs; predicts match probability for ambiguous cases.
# ─────────────────────────────────────────────

# With only 4 labelled pairs we skip train/test split (demo only).
# In production: hundreds/thousands of labelled pairs → proper split + cross-val.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

clf = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42,
)
clf.fit(X_scaled, y)


# ─────────────────────────────────────────────
# 8. PREDICT ON ALL CANDIDATE PAIRS  (rule-based → ML fallback)
# ─────────────────────────────────────────────

X_all        = feature_df[FEATURE_COLS].values
X_all_scaled = scaler.transform(X_all)
proba        = clf.predict_proba(X_all_scaled)[:, 1]    # P(match)

MATCH_THRESHOLD     = 0.75   # auto-merge
REVIEW_THRESHOLD    = 0.40   # human review queue

def final_decision(rule: str, prob: float) -> str:
    if rule == "MATCH":
        return "AUTO-MATCH  (rule)"
    if rule == "NON-MATCH":
        return "AUTO-REJECT (rule)"
    # Ambiguous → ML
    if prob >= MATCH_THRESHOLD:
        return f"AUTO-MATCH  (ML  p={prob:.2f})"
    if prob >= REVIEW_THRESHOLD:
        return f"HUMAN REVIEW     (p={prob:.2f})"
    return f"AUTO-REJECT (ML  p={prob:.2f})"

feature_df["ml_prob"]  = proba
feature_df["decision"] = feature_df.apply(
    lambda r: final_decision(r["rule_decision"], r["ml_prob"]), axis=1
)


# ─────────────────────────────────────────────
# 9. RESULTS
# ─────────────────────────────────────────────

print("=" * 72)
print("IDENTITY RESOLUTION RESULTS")
print("=" * 72)

records_by_id = pd.concat([reg_a, reg_b, reg_c], ignore_index=True).set_index("id", drop=False)

for _, row in feature_df.iterrows():
    a = records_by_id.loc[row["id_a"]]
    b = records_by_id.loc[row["id_b"]]
    print(f"\n  {a['id']}  '{a['first']} {a['last']}'")
    print(f"  {b['id']}  '{b['first']} {b['last']}'")
    print(f"  Pair set: {row['pair_name']}")
    print(f"  License match: {bool(row['lic_exact'])}  |  DOB exact: {bool(row['dob_exact'])}"
          f"  |  Name JW: {row['jw_full']:.2f}")
    print(f"  → {row['decision']}")


# ─────────────────────────────────────────────
# 10. FEATURE IMPORTANCE  (which signals matter most)
# ─────────────────────────────────────────────

print("\n\nFEATURE IMPORTANCE (XGBoost gain)")
print("-" * 40)
importances = pd.Series(clf.feature_importances_, index=FEATURE_COLS)
for feat, score in importances.sort_values(ascending=False).items():
    bar = "█" * int(score * 40)
    print(f"  {feat:<20} {bar}  {score:.3f}")


# ─────────────────────────────────────────────
# 11. GOLDEN RECORD BUILDER  (merge confirmed matches)
# ─────────────────────────────────────────────

SOURCE_TRUST = {
    # Provincial college registries are most authoritative
    "CPSO": 1.0,
    "CNO":  0.85,
    "REGC": 0.80,
}

def merge_other_practice_sites(primary: pd.Series, secondary: pd.Series) -> list[dict[str, str]]:
    """Union additional practice sites from both records."""
    merged_sites = []
    seen = set()
    for source_row in (primary, secondary):
        for site in source_row["other_practice_sites_n"]:
            key = (site["street_core"], site["unit"], site["city"], site["province"], site["postal"])
            if key in seen:
                continue
            seen.add(key)
            merged_sites.append({
                "street1": site["street_core"],
                "street2": f"Unit {site['unit']}" if site["unit"] else "",
                "city": site["city"].title(),
                "province": site["province"],
                "postal": site["postal"],
                "phone": site["phone"],
            })
    return merged_sites

def format_address(address: dict[str, str]) -> str:
    """Render a normalized address dict into a compact human-readable string."""
    parts = [address.get("street1", "")]
    if address.get("street2"):
        parts.append(address["street2"])
    parts.append(address.get("city", ""))
    if address.get("province"):
        parts.append(address["province"])
    if address.get("postal"):
        parts.append(address["postal"])
    return ", ".join(part for part in parts if part)

def build_golden_record(a: pd.Series, b: pd.Series) -> dict:
    """
    Merge two confirmed provider records.
    For each field, pick the value from the most-trusted source.
    Where they agree, flag as high-confidence.
    """
    src_a = a["id"].split("-")[0]
    src_b = b["id"].split("-")[0]
    trust_a = SOURCE_TRUST.get(src_a, 0.7)
    trust_b = SOURCE_TRUST.get(src_b, 0.7)
    primary, secondary = (a, b) if trust_a >= trust_b else (b, a)

    golden = {
        "golden_id":      f"GR-{primary['id']}",
        "first_name":     primary["first"],
        "last_name":      primary["last"],
        "dob":            str(primary["dob"].date()) if pd.notna(primary["dob"]) else None,
        "license":        primary["license_n"],
        "specialty":      primary["specialty"],
        "clinic_name":    primary.get("clinic_name", ""),
        "address": {
            "street1":   primary.get("street1", ""),
            "street2":   primary.get("street2", ""),
            "city":      primary["city"],
            "province":  primary.get("province", ""),
            "postal":    primary["postal"],
            "phone":     primary.get("phone", ""),
        },
        "other_practice_sites": merge_other_practice_sites(primary, secondary),
        "source_ids":     [a["id"], b["id"]],
        "name_conflict":  a["last_n"] != b["last_n"],  # flag married-name changes etc.
        "confidence":     "HIGH" if trust_a >= 0.9 else "MEDIUM",
    }
    return golden

print("\n\nGOLDEN RECORDS")
print("-" * 60)
matches = feature_df[feature_df["decision"].str.startswith("AUTO-MATCH")]
for _, row in matches.iterrows():
    gr = build_golden_record(records_by_id.loc[row["id_a"]], records_by_id.loc[row["id_b"]])
    print(f"\n  {gr['golden_id']}")
    print(f"  Name      : {gr['first_name']} {gr['last_name']}")
    print(f"  DOB       : {gr['dob']}  |  License: {gr['license']}")
    print(f"  Specialty : {gr['specialty']}")
    print(f"  Clinic    : {gr['clinic_name']}")
    print(f"  Address   : {format_address(gr['address'])}")
    print(f"  Phone     : {gr['address']['phone']}")
    print(f"  Other sites: {gr['other_practice_sites']}")
    print(f"  Sources   : {gr['source_ids']}")
    print(f"  Conflicts : name_conflict={gr['name_conflict']}  confidence={gr['confidence']}")