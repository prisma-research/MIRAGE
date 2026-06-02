"""
Deterministic scorer for state-conditioned probes.

Task types and their scoring:
  full:              ANSWERABLE match + SOURCE match + ANSWER match
  locate:            SOURCE match
  verify:            SUPPORTED match
  extract_in_state:  ANSWER match
"""

from __future__ import annotations
import re


def parse_full(text: str) -> dict:
    """Parse ANSWERABLE/SOURCE/ANSWER from full task response."""
    r = {"answerable": None, "source": None, "answer": None, "ok": False}
    m = re.search(r'ANSWERABLE\s*=\s*(YES|NO)', text, re.IGNORECASE)
    if m:
        r["answerable"] = m.group(1).upper()
    m = re.search(r'SOURCE\s*=\s*(\S+)', text, re.IGNORECASE)
    if m:
        r["source"] = m.group(1).strip()
    m = re.search(r'ANSWER\s*=\s*(.+)', text, re.IGNORECASE)
    if m:
        r["answer"] = m.group(1).strip()
    r["ok"] = r["answerable"] is not None and r["source"] is not None and r["answer"] is not None
    return r


def parse_locate(text: str) -> dict:
    r = {"source": None, "ok": False}
    m = re.search(r'SOURCE\s*=\s*(\S+)', text, re.IGNORECASE)
    if m:
        r["source"] = m.group(1).strip()
        r["ok"] = True
    return r


def parse_verify(text: str) -> dict:
    r = {"supported": None, "ok": False}
    m = re.search(r'SUPPORTED\s*=\s*(YES|NO)', text, re.IGNORECASE)
    if m:
        r["supported"] = m.group(1).upper()
        r["ok"] = True
    return r


def parse_extract(text: str) -> dict:
    r = {"answer": None, "ok": False}
    m = re.search(r'ANSWER\s*=\s*(.+)', text, re.IGNORECASE)
    if m:
        r["answer"] = m.group(1).strip()
        r["ok"] = True
    return r


# --- Source canonicalization (reused from branch_probe_runner) ---

_DERIVED_FILE_TO_ARTIFACT = {
    # (C) Derived data files
    "gdp_extracted_2000-2020": "cqa_9f9df328",
    "canada_digital_population_2021": "cqa_536c64e4",
    "nevada_poker_tables_2000-2020": "cqa_3dd0635a",
    "nba_fan_cost_index_2020-21": "cqa_92d6ef9a",
    "mi_fitness_profile_2026-03-27_14-57": "ss_324e28cd",
    "ios_notifications_google_maps_18-16": "ss_102dfb65",
    # (D) Memory surrogates (post-compaction memory files → original artifact)
    "2026-03-27-nj-real-gdp-2000-2020": "cqa_9f9df328",
    "2026-03-27-canada-digital-users-2021": "cqa_536c64e4",
    "2026-03-27-nevada-poker-tables-2000-2020": "cqa_3dd0635a",
    "2026-03-27-nba-fan-cost-index-2015": "cqa_92d6ef9a",
    "2026-03-27-android-fitness-profile-screenshot": "ss_324e28cd",
    "2026-03-27-ios-google-maps-notifications-screenshot": "ss_102dfb65",
}


def canonicalize_source(raw: str) -> str:
    if not raw or raw.upper() == "NONE":
        return "none"
    import os.path
    # Strip URL-style fragment anchors: #L1, #L5C3, #section-name
    s = re.sub(r'#.*$', '', raw)
    s = os.path.basename(s)
    s = re.sub(r'\.(png|jpg|jpeg|json|md)$', '', s)
    s_lower = s.lower()
    if s_lower in _DERIVED_FILE_TO_ARTIFACT:
        return _DERIVED_FILE_TO_ARTIFACT[s_lower]
    return s_lower


# --- Answer comparison ---

def _normalize_numeric(val: str) -> float | None:
    if not val or val.upper() == "NONE":
        return None
    cleaned = val.strip()
    cleaned = re.sub(r'^[\$€£~≈]+', '', cleaned).strip()
    cleaned = cleaned.rstrip('%').strip()
    cleaned = re.sub(r'^(approximately|about|around|roughly)\s+', '', cleaned, flags=re.IGNORECASE).strip()
    _UNITS = {'k': 1e3, 'thousand': 1e3, 'm': 1e6, 'million': 1e6, 'b': 1e9, 'billion': 1e9}
    multiplier = 1.0
    m = re.match(r'^([\d,.]+)\s*([a-zA-Z]+)', cleaned)
    if m:
        unit = m.group(2).lower()
        if unit in _UNITS:
            multiplier = _UNITS[unit]
            cleaned = m.group(1)
        else:
            cleaned = m.group(1)  # strip trailing unit word
    m2 = re.match(r'^([\d,.]+)\s+(million|billion|thousand)\b', cleaned, re.IGNORECASE)
    if m2:
        multiplier = _UNITS[m2.group(2).lower()]
        cleaned = m2.group(1)
    cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


def _raw_numeric(val: str) -> float | None:
    if not val or val.upper() == "NONE":
        return None
    cleaned = val.strip()
    cleaned = re.sub(r'^[\$€£~≈]+', '', cleaned).strip()
    cleaned = re.sub(r'\s*(million|billion|thousand|[MBKk])\b.*$', '', cleaned, flags=re.IGNORECASE).strip()
    m = re.match(r'^([\d,.]+)', cleaned)
    if m:
        cleaned = m.group(1)
    cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def answers_match(pred: str, gold: str, answer_type: str) -> bool:
    if not pred or not gold:
        return (pred or "NONE").upper() == (gold or "NONE").upper()
    if answer_type == "numeric":
        gn = _normalize_numeric(gold)
        pn = _normalize_numeric(pred)
        if gn is not None and pn is not None:
            if abs(gn - pn) <= max(abs(gn) * 0.01, 0.5):
                return True
        gr = _raw_numeric(gold)
        pr = _raw_numeric(pred)
        if gr is not None and pr is not None:
            if abs(gr - pr) <= max(abs(gr) * 0.01, 0.5):
                return True
        return False
    elif answer_type == "year":
        return pred.strip() == gold.strip()
    else:
        return pred.strip().lower() == gold.strip().lower()


# --- Main scoring ---

def score_probe(probe: dict, response_text: str) -> dict:
    """Score one probe. Returns dict with task-specific metric fields."""
    tt = probe["task_type"]
    result = {"task_type": tt, "parse_success": False}

    if tt == "full":
        p = parse_full(response_text)
        result["parse_success"] = p["ok"]
        if not p["ok"]:
            return result
        gold_ans = probe.get("gold_answerable", "").upper()
        result["full_answerable_correct"] = 1 if p["answerable"] == gold_ans else 0
        # source (only meaningful when answerable=YES)
        if gold_ans == "YES":
            cs = canonicalize_source(p["source"] or "NONE")
            gs = (probe.get("gold_source") or "NONE").lower()
            result["full_source_correct"] = 1 if cs == gs else 0
            at = probe.get("answer_type", "string")
            result["full_answer_correct"] = 1 if answers_match(
                p["answer"] or "NONE", probe.get("gold_answer") or "NONE", at
            ) else 0
        else:
            result["full_source_correct"] = None
            result["full_answer_correct"] = None

    elif tt == "locate":
        p = parse_locate(response_text)
        result["parse_success"] = p["ok"]
        if not p["ok"]:
            return result
        cs = canonicalize_source(p["source"] or "NONE")
        gs = (probe.get("gold_source") or "NONE").lower()
        result["locate_correct"] = 1 if cs == gs else 0

    elif tt == "verify":
        p = parse_verify(response_text)
        result["parse_success"] = p["ok"]
        if not p["ok"]:
            return result
        gold = (probe.get("gold_supported") or "").upper()
        result["verify_correct"] = 1 if p["supported"] == gold else 0

    elif tt == "extract_in_state":
        p = parse_extract(response_text)
        result["parse_success"] = p["ok"]
        if not p["ok"]:
            return result
        at = probe.get("answer_type", "string")
        result["extract_correct"] = 1 if answers_match(
            p["answer"] or "NONE", probe.get("gold_answer") or "NONE", at
        ) else 0

    return result
