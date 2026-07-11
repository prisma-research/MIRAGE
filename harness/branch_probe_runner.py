"""
Branch Probe Runner — restores a checkpoint, asks one question, saves results.

This runner does NOT do batch evaluation. It:
  1. Loads a checkpoint manifest
  2. Restores it into an isolated branch
  3. Sends exactly one natural-language follow-up question
  4. Extracts tool trace and retrieval signals
  5. Saves a ProbeResult with enough fields for later scoring

Usage:
    cd MIRAGE

    # Single probe
    python -m harness.branch_probe_runner \\
        --episode-id pilot_v1 \\
        --checkpoint s1_prequery_d0 \\
        --question-bank configs/question_banks/pilot_questions.json \\
        --question-id q_budget_revenue \\
        --experiment depth \\
        --continuation same_session

    # Batch all questions against one checkpoint
    python -m harness.branch_probe_runner \\
        --episode-id pilot_v1 \\
        --checkpoint postcomp_50k \\
        --question-bank configs/question_banks/pilot_questions.json \\
        --experiment reset \\
        --continuation fresh_session
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import logging
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from harness.checkpoint import (
    load_checkpoint,
    restore_checkpoint,
    continue_same_session,
    create_fresh_session,
    CHECKPOINTS_ROOT,
    ContinuationResult,
)
from harness.probe_result import ProbeResult, save_probe_result
from constants import RETRIEVAL_TOOL_NAMES, MEMORY_PATH_PATTERNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROBE_OUTPUT_DIR = Path(__file__).parent.parent / "logs" / "probes"


# ---------------------------------------------------------------------------
# Tool trace extraction
# ---------------------------------------------------------------------------

def _extract_tool_trace(jsonl: list[dict], split_index: int) -> list[dict]:
    """Extract tool calls/results from JSONL entries after split_index."""
    calls = []
    for entry in jsonl[split_index:]:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        role = msg.get("role", "")
        if role == "assistant":
            for c in msg.get("content", []):
                if isinstance(c, dict) and c.get("type") in ("toolCall", "tool_use"):
                    calls.append({
                        "kind": "call",
                        "name": c.get("name"),
                        "input": c.get("arguments") or c.get("input"),
                    })
        elif role == "toolResult":
            content = msg.get("content", [])
            text = ""
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        text += c.get("text", "")
            elif isinstance(content, str):
                text = content
            calls.append({
                "kind": "result",
                "name": msg.get("toolName"),
                "text": text[:500],
            })
    return calls


# ---------------------------------------------------------------------------
# R-path classification
# ---------------------------------------------------------------------------

def classify_r_path(
    tool_trace: list[dict],
    continuation: str,
    jsonl: list[dict] | None = None,
) -> str:
    """Classify retrieval path from tool trace.

    R_context — same-session, no retrieval tool used (artifact was in context)
    R_boot    — fresh-session, bootstrap loaded memory before query
    R_tool    — model explicitly called a retrieval tool
    R_none    — no retrieval observed
    """
    # Check for explicit retrieval tool calls
    for t in tool_trace:
        if t.get("kind") == "call":
            name = t.get("name", "")
            if name in RETRIEVAL_TOOL_NAMES:
                return "R_tool"
            # Generic read tool accessing memory paths
            if name in ("read", "Read", "exec"):
                inp = str(t.get("input", ""))
                for pattern in MEMORY_PATH_PATTERNS:
                    if pattern in inp:
                        return "R_tool"

    if continuation == "same_session":
        return "R_context"

    if continuation == "fresh_session":
        # Check bootstrap exposure — any tool results before the query
        # In fresh sessions, startup tool calls indicate R_boot
        if jsonl:
            for entry in jsonl:
                if entry.get("type") != "message":
                    continue
                msg = entry.get("message", {})
                if msg.get("role") == "toolResult":
                    return "R_boot"
        return "R_none"

    return "R_none"


# ---------------------------------------------------------------------------
# Auto-scoring
# ---------------------------------------------------------------------------

def _extract_prompt_tokens(prompt: str) -> set[str]:
    """Extract lowercase tokens from the prompt for echo filtering."""
    return {w.strip("?,.-'\"!()").lower()
            for w in prompt.split() if len(w.strip("?,.-'\"!()")) >= 2}


def auto_score(
    result: ProbeResult,
    question: dict,
) -> ProbeResult:
    """Apply automatic scoring to a probe result.

    Prompt-echo filtering:
    Gold keywords that also appear in the probe prompt are excluded from
    the correctness check. This prevents counting keyword echo (where
    the model merely restates the question) as evidence of grounded recall.

    For charts: numeric answer tokens (e.g., "556.73") are KEPT even if they
    appear in the prompt, because a numeric value in the response is strong
    evidence of recall regardless of echo.

    For screenshots: if ALL remaining gold keywords after echo filtering are
    empty, target_grounded_correct is set to None (judge-needed), because
    we cannot distinguish real recall from prompt restatement.

    Scoring logic by query_type:

    target_present:
      - target_grounded_correct: 1 if answer-specific gold keywords appear
      - wrong_source: 0
      - abstention_correct: 0 if model abstains on an answerable question

    wrong_source:
      - wrong_source: 1 only if wrong_source_signals appear
      - target_grounded_correct: 1 if gold keywords present and no ws signals

    unanswerable:
      - target_grounded_correct: 1 if model abstains
      - wrong_source: 0
    """
    response = result.response_text.lower()
    prompt_lower = result.prompt.lower() if result.prompt else question.get("prompt", "").lower()
    query_type = question.get("query_type", question.get("family", ""))
    expected = question.get("expected_answerability", "")
    gold_keywords = [k.lower() for k in question.get("gold_keywords", []) if k]
    visual_type = question.get("visual_type", "")

    # --- Abstention detection ---
    abstention_phrases = [
        "i don't have", "i don't recall", "i'm not sure",
        "no record", "not available", "i don't remember",
        "i couldn't find", "no information", "not in my",
        "i don't see", "don't have that",
    ]
    is_abstention = any(p in response for p in abstention_phrases)
    result.abstention = 1 if is_abstention else 0

    if expected == "unanswerable":
        result.abstention_correct = 1 if is_abstention else 0
    elif expected == "answerable":
        result.abstention_correct = 0 if is_abstention else 1
    else:
        result.abstention_correct = None

    # --- Prompt-echo-aware gold keyword matching ---
    prompt_tokens = _extract_prompt_tokens(prompt_lower)

    # Partition gold keywords into answer-specific vs prompt-echoed
    answer_specific = []
    for k in gold_keywords:
        # Numeric tokens are always answer-specific (strong recall signal)
        if k.replace(".", "").replace(",", "").replace("-", "").isdigit():
            answer_specific.append(k)
        # Non-numeric tokens that appear in the prompt are prompt-echoed
        elif k not in prompt_tokens:
            answer_specific.append(k)
        # else: this keyword is in the prompt → skip for scoring

    if answer_specific:
        hits = sum(1 for k in answer_specific if k in response)
        ratio = hits / len(answer_specific)
        has_gold = ratio >= 0.4
    else:
        # All gold keywords were prompt-echoed. For charts with numeric answers,
        # this shouldn't happen (numerics are kept). For screenshots, mark as
        # judge-needed since we can't distinguish echo from recall.
        has_gold = None  # sentinel: judge needed

    # --- Wrong-source signal detection (correction-aware) ---
    ws_signals = [s.lower() for s in question.get("wrong_source_signals", []) if s]
    ws_signal_hits = [s for s in ws_signals if s in response] if ws_signals else []
    raw_ws_present = len(ws_signal_hits) > 0

    # Strong correction patterns — these indicate the model is explicitly
    # rejecting or correcting the misleading premise, not confused by it.
    # Only patterns that are unambiguously corrective are included here;
    # vague phrases like "actually shows" are excluded because they can
    # appear in confused answers too.
    _CORRECTION_PATTERNS = [
        "does not show", "doesn't show", "did not show", "didn't show",
        "does not have", "doesn't have", "did not have", "didn't have",
        "does not contain", "doesn't contain",
        "is not about", "isn't about", "was not about", "wasn't about",
        "is not in", "isn't in", "was not in", "wasn't in",
        "not on that screen", "not on the screen", "not visible",
        "not from that chart", "not from the chart",
        "no such data", "no such information",
        "that was on the other", "that was from the other",
        "that was in the other", "that was the other",
        "different chart", "different screen", "different app",
        "wrong chart", "wrong screen",
    ]

    has_correction = any(p in response for p in _CORRECTION_PATTERNS) if raw_ws_present else False

    # Three-way classification for wrong-source questions:
    #
    # 1. CONFUSED: distractor signals present, no correction language
    #    → wrong_source=1, target_grounded_correct=0
    #
    # 2. CORRECTED: distractor signals present (model references distractor
    #    to explicitly deny/correct it), AND gold_keywords match (model
    #    gives correct target-grounded correction)
    #    → wrong_source=0, target_grounded_correct=1
    #
    # 3. AMBIGUOUS: distractor signals present with some correction language,
    #    but gold_keywords do not match (correction is unclear/incomplete)
    #    → wrong_source=None, target_grounded_correct=None

    # --- Apply by query_type ---
    if query_type == "target_present":
        if has_gold is None:
            result.target_grounded_correct = None
        else:
            result.target_grounded_correct = 1 if has_gold else 0
        result.wrong_source = 0

    elif query_type == "wrong_source":
        if is_abstention:
            result.wrong_source = 0
            result.target_grounded_correct = None
        elif raw_ws_present and not has_correction:
            # Case 1: CONFUSED — distractor content used without correction
            result.wrong_source = 1
            result.target_grounded_correct = 0
        elif raw_ws_present and has_correction and has_gold:
            # Case 2: CORRECTED — model rejected premise and gave correct info
            result.wrong_source = 0
            result.target_grounded_correct = 1
        elif raw_ws_present and has_correction:
            # Case 3: AMBIGUOUS — correction language but gold keywords missing
            result.wrong_source = None
            result.target_grounded_correct = None
        elif not raw_ws_present and has_gold is not None and has_gold:
            # No distractor signals, gold present → correct target answer
            result.wrong_source = 0
            result.target_grounded_correct = 1
        elif not raw_ws_present and has_gold is not None and not has_gold:
            # No distractor, no gold → unclear
            result.wrong_source = 0
            result.target_grounded_correct = None
        else:
            # Fallback: cannot determine
            result.wrong_source = None
            result.target_grounded_correct = None

    elif query_type == "unanswerable":
        result.target_grounded_correct = 1 if is_abstention else 0
        result.wrong_source = 0

    else:
        result.target_grounded_correct = None
        result.wrong_source = None

    return result


# ---------------------------------------------------------------------------
# Constrained-output parsing and scoring
# ---------------------------------------------------------------------------

import re as _re

def parse_structured_response(response_text: str) -> dict:
    """Parse SUPPORT/SOURCE/ANSWER from model's structured response.

    Returns dict with keys: support, source, answer, parse_success.
    parse_success requires ALL THREE fields to be extracted.
    """
    text = response_text.strip()
    result = {"support": None, "source": None, "answer": None, "parse_success": False}

    m = _re.search(r'SUPPORT\s*=\s*(TARGET|OTHER_ARTIFACT|NOT_PRESENT)', text, _re.IGNORECASE)
    if m:
        result["support"] = m.group(1).upper()

    m = _re.search(r'SOURCE\s*=\s*(\S+)', text, _re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        result["source"] = "NONE" if val.upper() == "NONE" else val

    m = _re.search(r'ANSWER\s*=\s*(.+)', text, _re.IGNORECASE)
    if m:
        val = m.group(1).strip()
        result["answer"] = "NONE" if val.upper() == "NONE" else val

    # Strict: all three fields must be present
    result["parse_success"] = all(result[k] is not None for k in ("support", "source", "answer"))
    return result


def _normalize_numeric(val: str) -> float | None:
    """Parse a string as a number for comparison.

    Handles:
      - plain: "556.73"
      - with prefix: "~556", "approximately 556", "$556"
      - with suffix unit: "35.63M", "35.63 million", "30.8K", "1.2B"
      - with separators: "1,234.56"
    """
    if not val or val.upper() == "NONE":
        return None
    cleaned = val.strip()
    # Strip currency, percentage, ~, ≈
    cleaned = _re.sub(r'^[\$€£~≈]+', '', cleaned).strip()
    cleaned = cleaned.rstrip('%').strip()
    # Strip prefix words
    cleaned = _re.sub(
        r'^(approximately|about|around|roughly|approx\.?)\s+',
        '', cleaned, flags=_re.IGNORECASE,
    ).strip()
    # Handle suffix units: M/million, K/thousand, B/billion
    _UNIT_MULTIPLIERS = {
        'k': 1e3, 'thousand': 1e3,
        'm': 1e6, 'million': 1e6, 'mn': 1e6,
        'b': 1e9, 'billion': 1e9, 'bn': 1e9,
    }
    multiplier = 1.0
    m = _re.match(r'^([\d,.]+)\s*([a-zA-Z]+)$', cleaned)
    if m:
        num_part, unit = m.group(1), m.group(2).lower()
        if unit in _UNIT_MULTIPLIERS:
            multiplier = _UNIT_MULTIPLIERS[unit]
            cleaned = num_part
    # Handle space-separated scale: "35.63 million" or "556.73 billion U.S. dollars"
    m2 = _re.match(r'^([\d,.]+)\s+(million|billion|thousand|mn|bn)\b', cleaned, _re.IGNORECASE)
    if m2:
        multiplier = _UNIT_MULTIPLIERS[m2.group(2).lower()]
        cleaned = m2.group(1)
    else:
        # Strip any trailing non-numeric words: "676.42 USD", "920 tables", "35.5 users"
        m3 = _re.match(r'^([\d,.]+)\s+\S', cleaned)
        if m3:
            cleaned = m3.group(1)
    # Remove commas
    cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned) * multiplier
    except ValueError:
        return None


def _normalize_numeric_raw(val: str) -> float | None:
    """Extract the raw numeric value, stripping unit suffixes WITHOUT applying multiplier.

    "35.63M" → 35.63, "35.63 million" → 35.63, "35.63" → 35.63
    Used as fallback when chart bank stores raw values (e.g., 35.63)
    but model may add units (e.g., "35.63 million").
    """
    if not val or val.upper() == "NONE":
        return None
    cleaned = val.strip()
    cleaned = _re.sub(r'^[\$€£~≈]+', '', cleaned).strip()
    cleaned = cleaned.rstrip('%').strip()
    cleaned = _re.sub(
        r'^(approximately|about|around|roughly|approx\.?)\s+',
        '', cleaned, flags=_re.IGNORECASE,
    ).strip()
    # Strip scale suffix and anything after it WITHOUT multiplying
    cleaned = _re.sub(r'\s*(million|billion|thousand|mn|bn|[MBKk])\b.*$', '', cleaned, flags=_re.IGNORECASE).strip()
    # Also strip any trailing non-numeric words: "676.42 USD" → "676.42"
    m = _re.match(r'^([\d,.]+)', cleaned)
    if m:
        cleaned = m.group(1)
    cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


# Explicit mapping: derived workspace files → canonical artifact IDs.
# GPT-5 extracts chart/screenshot data into JSON files during the session.
# When the model cites one of these files as SOURCE, we map it back to the
# original artifact ID. Only known deterministic derivations are mapped.
_DERIVED_FILE_TO_ARTIFACT = {
    "gdp_extracted_2000-2020": "cqa_9f9df328",
    "canada_digital_population_2021": "cqa_536c64e4",
    "nevada_poker_tables_2000-2020": "cqa_3dd0635a",
    "nba_fan_cost_index_2020-21": "cqa_92d6ef9a",
    "mi_fitness_profile_2026-03-27_14-57": "ss_324e28cd",
    "ios_notifications_google_maps_18-16": "ss_102dfb65",
}


def _canonicalize_source(raw: str) -> str:
    """Canonicalize a predicted SOURCE value to an artifact ID.

    Rules (applied in order, all deterministic):
    1. If raw is NONE, return "none"
    2. Take basename (strip path)
    3. Strip known file extensions (.png, .jpg, .json, .md)
    4. Check explicit derived-file → artifact-id mapping
    5. Return lowercased result
    """
    if not raw or raw.upper() == "NONE":
        return "none"

    import os.path
    s = os.path.basename(raw)                            # strip path
    s = _re.sub(r'\.(png|jpg|jpeg|json|md)$', '', s)    # strip extension
    s_lower = s.lower()

    # Check derived file mapping
    if s_lower in _DERIVED_FILE_TO_ARTIFACT:
        return _DERIVED_FILE_TO_ARTIFACT[s_lower]

    return s_lower


def _normalize_string(s: str) -> str:
    """Normalize a string answer for deterministic comparison.

    Rules:
    - Lowercase, strip whitespace and quotes, collapse internal spaces
    - Boolean/toggle equivalences:
        yes/true/correct     → "yes"
        no/false/incorrect   → "no"
        on/enabled/active    → "on"
        off/disabled/inactive → "off"
    - Does NOT apply to strings that don't match a known synonym
    """
    s = s.strip().strip("'\"").lower()
    s = _re.sub(r'\s+', ' ', s)
    _SYNONYM_MAP = {
        # Affirmative / negative
        "yes": "yes", "true": "yes", "correct": "yes",
        "no": "no", "false": "no", "incorrect": "no",
        # Toggle states
        "on": "on", "enabled": "on", "active": "on",
        "off": "off", "disabled": "off", "inactive": "off",
    }
    if s in _SYNONYM_MAP:
        s = _SYNONYM_MAP[s]
    return s


def score_constrained(result: ProbeResult, question: dict) -> ProbeResult:
    """Score a probe result using constrained SUPPORT/SOURCE/ANSWER protocol.

    Primary scoring path for the 100k experiment. Deterministic.

    Scope-aware raw fields:
    - support_correct: always scored (SUPPORT enum match)
    - answer_correct: only scored when gold_support == TARGET; else None
    - source_correct: only scored when gold_support == OTHER_ARTIFACT; else None
    """
    parsed = parse_structured_response(result.response_text)

    result.parsed_support = parsed["support"]
    result.parsed_source = parsed["source"]
    result.parsed_answer = parsed["answer"]
    result.parse_success = parsed["parse_success"]

    # Canonicalize source and store for debugging
    result.canonical_source = _canonicalize_source(parsed["source"] or "NONE")

    # Store gold fields for conditional analysis
    result.gold_support = question.get("gold_support")
    result.gold_source = question.get("gold_source")
    result.gold_answer = question.get("gold_answer")
    result.answer_type = question.get("answer_type", "")

    if not parsed["parse_success"]:
        result.support_correct = None
        result.answer_correct = None
        result.source_correct = None
        return result

    gold_support = (question.get("gold_support") or "").upper()
    gold_source = question.get("gold_source") or "NONE"
    gold_answer = question.get("gold_answer")
    answer_type = question.get("answer_type", "string")

    # --- support_correct (always scored) ---
    result.support_correct = 1 if parsed["support"] == gold_support else 0

    # --- source_correct (only scored for gold OTHER_ARTIFACT) ---
    if gold_support == "OTHER_ARTIFACT":
        gold_source_norm = gold_source.lower()
        result.source_correct = 1 if result.canonical_source == gold_source_norm else 0
    else:
        result.source_correct = None  # not in scope

    # --- answer_correct (only scored for gold TARGET) ---
    if gold_support != "TARGET":
        result.answer_correct = None  # not in scope
        return result

    pred_ans = parsed["answer"] or "NONE"
    if gold_answer is None or str(gold_answer).upper() == "NONE":
        result.answer_correct = 1 if pred_ans.upper() == "NONE" else 0
    elif answer_type == "numeric":
        gold_num = _normalize_numeric(str(gold_answer))
        pred_num = _normalize_numeric(pred_ans)
        if gold_num is not None and pred_num is not None:
            # Primary: compare at the same scale
            diff = abs(gold_num - pred_num)
            tol = max(abs(gold_num) * 0.01, 0.5)
            if diff <= tol:
                result.answer_correct = 1
            else:
                # Fallback: the bank stores raw chart values (e.g., 35.63)
                # while the model may add units (e.g., "35.63 million").
                # Compare the raw digit portion, stripping unit multipliers.
                gold_raw = _normalize_numeric_raw(str(gold_answer))
                pred_raw = _normalize_numeric_raw(pred_ans)
                if gold_raw is not None and pred_raw is not None:
                    diff2 = abs(gold_raw - pred_raw)
                    tol2 = max(abs(gold_raw) * 0.01, 0.5)
                    result.answer_correct = 1 if diff2 <= tol2 else 0
                else:
                    result.answer_correct = 0
        elif gold_num is None and pred_num is None:
            result.answer_correct = 1
        else:
            result.answer_correct = 0
    elif answer_type == "year":
        result.answer_correct = 1 if pred_ans.strip() == str(gold_answer).strip() else 0
    else:
        # String: normalized comparison
        result.answer_correct = 1 if _normalize_string(pred_ans) == _normalize_string(str(gold_answer)) else 0

    return result


# ---------------------------------------------------------------------------
# Single probe execution
# ---------------------------------------------------------------------------

async def run_probe(
    episode_id: str,
    checkpoint_id: str,
    question: dict,
    experiment: str,
    continuation: str,
    model: str = "",
    output_dir: Path | None = None,
    checkpoints_root: Path | None = None,
    proxy_port: int | None = None,
) -> ProbeResult:
    """Run one probe: restore checkpoint, ask question, save result.

    If proxy_port is provided, the branch's openclaw.json is patched to
    route API calls through the live proxy (required when checkpoints were
    built with a now-dead proxy).
    """
    ckpt_root = checkpoints_root or CHECKPOINTS_ROOT
    out_dir = output_dir or PROBE_OUTPUT_DIR / episode_id

    manifest = load_checkpoint(episode_id, checkpoint_id, ckpt_root)
    probe_id = f"probe_{checkpoint_id}_{question['question_id']}_{uuid.uuid4().hex[:6]}"
    branch_id = f"br_{probe_id}"

    t0 = time.monotonic()
    logger.info("Probe %s: checkpoint=%s, question=%s, continuation=%s",
                probe_id, checkpoint_id, question["question_id"], continuation)

    if proxy_port is not None:
        # Restore WITHOUT gateway, patch config with live proxy, then start gateway
        branch = await restore_checkpoint(manifest, branch_id=branch_id, start_gateway=False)
        import json as _json
        from client.usage_proxy import patch_openclaw_config as _patch
        cfg_path = branch.branch_state_dir / "openclaw.json"
        with cfg_path.open() as f:
            cfg = _json.load(f)
        _patch(cfg, proxy_port=proxy_port, compaction_model=None)
        with cfg_path.open("w") as f:
            _json.dump(cfg, f, indent=2)
        # Start gateway with patched config
        import subprocess as _sp
        import os as _os
        env = {**_os.environ, "OPENCLAW_STATE_DIR": str(branch.branch_state_dir)}
        (branch.branch_state_dir / "logs").mkdir(parents=True, exist_ok=True)
        gw = _sp.Popen(
            ["openclaw", "gateway", "run", "--port", str(branch.gateway_port), "--force"],
            stdout=(branch.branch_state_dir / "logs" / "gateway.log").open("a"),
            stderr=(branch.branch_state_dir / "logs" / "gateway.err.log").open("a"),
            env=env,
        )
        branch.gateway_proc = gw
        import time as _time
        deadline = _time.monotonic() + 60.0
        while _time.monotonic() < deadline:
            try:
                r = _sp.run(["openclaw", "gateway", "status"],
                            capture_output=True, timeout=15, env=env)
                if r.returncode == 0:
                    break
            except _sp.TimeoutExpired:
                pass
            await asyncio.sleep(0.5)
    else:
        branch = await restore_checkpoint(manifest, branch_id=branch_id)

    try:
        if continuation == "same_session":
            cr: ContinuationResult = await continue_same_session(
                branch, message=question["prompt"],
            )
        elif continuation == "fresh_session":
            cr = await create_fresh_session(
                branch, message=question["prompt"],
            )
        else:
            raise ValueError(f"Unknown continuation: {continuation}")

        duration = time.monotonic() - t0

        # Extract tool trace from the probe turn only
        split = cr.jsonl_count_before if cr.jsonl_count_before is not None else 0
        tool_trace = _extract_tool_trace(cr.jsonl, split)

        # Classify R-path
        r_path = classify_r_path(tool_trace, continuation, cr.jsonl)

        result = ProbeResult(
            probe_id=probe_id,
            episode_id=episode_id,
            checkpoint_id=checkpoint_id,
            question_id=question["question_id"],
            experiment=experiment,
            continuation=continuation,
            checkpoint_kind=manifest.checkpoint_kind,
            depth_label=manifest.history_depth_label,
            compaction_threshold=manifest.compaction_threshold,
            effective_input_tokens=manifest.effective_input_tokens,
            native_compaction_count=manifest.native_compaction_count,
            target_artifact_id=question.get("target_artifact_id"),
            query_type=question.get("query_type", question.get("family", "")),
            visual_type=question.get("visual_type", ""),
            confusion_group=question.get("confusion_group", ""),
            paraphrase_group_id=question.get("paraphrase_group_id", ""),
            prompt=question["prompt"],
            gold_answer=question.get("gold_answer"),
            expected_answerability=question.get("expected_answerability", ""),
            response_text=cr.response_text,
            session_id=cr.session_id,
            branch_id=branch_id,
            jsonl_entry_count=len(cr.jsonl),
            tool_trace=tool_trace,
            jsonl_count_before=cr.jsonl_count_before,
            jsonl_count_after=cr.jsonl_count_after,
            session_id_matches_checkpoint=cr.session_id_matches_checkpoint,
            R_path=r_path,
            R_binary=1 if r_path != "R_none" else 0,
            model=model,
            created_at=datetime.datetime.utcnow().isoformat() + "Z",
            duration_seconds=round(duration, 1),
        )

        # Score: use constrained-output scoring if question has response_format,
        # otherwise fall back to legacy free-form scoring.
        if question.get("response_format") == "SUPPORT/SOURCE/ANSWER":
            result = score_constrained(result, question)
            logger.info("Probe saved: %s (R=%s, support=%s, ans=%s)",
                        probe_id, r_path, result.support_correct, result.answer_correct)
        else:
            result = auto_score(result, question)
            logger.info("Probe saved: %s (R=%s, correct=%s, abstain=%s)",
                        probe_id, r_path, result.target_grounded_correct, result.abstention)

        # Save
        save_probe_result(result, out_dir)
        return result

    finally:
        branch.cleanup()


# ---------------------------------------------------------------------------
# Batch: all questions against one checkpoint
# ---------------------------------------------------------------------------

async def run_probe_batch(
    episode_id: str,
    checkpoint_id: str,
    question_bank_path: Path,
    experiment: str,
    continuation: str,
    question_id: str | None = None,
    model: str = "",
    output_dir: Path | None = None,
    checkpoints_root: Path | None = None,
    proxy_port: int | None = None,
) -> list[ProbeResult]:
    """Run probes for all (or one) question(s) against a checkpoint."""
    with question_bank_path.open() as f:
        bank = json.load(f)

    questions = bank["questions"]
    if question_id:
        questions = [q for q in questions if q["question_id"] == question_id]
        if not questions:
            raise ValueError(f"Question {question_id} not found in bank")

    results = []
    for q in questions:
        r = await run_probe(
            episode_id=episode_id,
            checkpoint_id=checkpoint_id,
            question=q,
            experiment=experiment,
            continuation=continuation,
            model=model,
            proxy_port=proxy_port,
            output_dir=output_dir,
            checkpoints_root=checkpoints_root,
        )
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run branch probes from saved checkpoints",
    )
    parser.add_argument("--episode-id", required=True, help="Episode ID")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint ID to restore")
    parser.add_argument("--question-bank", required=True, help="Path to question bank JSON")
    parser.add_argument("--experiment", required=True,
                        choices=["depth", "compaction", "reset"])
    parser.add_argument("--continuation", required=True,
                        choices=["same_session", "fresh_session"])
    parser.add_argument("--question-id", default=None,
                        help="Run only this question (default: all)")
    parser.add_argument("--model", default="", help="Model name for metadata")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--checkpoints-root", default=None)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else None
    ckpt_root = Path(args.checkpoints_root) if args.checkpoints_root else None

    results = asyncio.run(run_probe_batch(
        episode_id=args.episode_id,
        checkpoint_id=args.checkpoint,
        question_bank_path=Path(args.question_bank),
        experiment=args.experiment,
        continuation=args.continuation,
        question_id=args.question_id,
        model=args.model,
        output_dir=out_dir,
        checkpoints_root=ckpt_root,
    ))

    print(f"\n{'='*60}")
    print(f"PROBES COMPLETE: {len(results)} results")
    print(f"{'='*60}")
    for r in results:
        if r.support_correct is not None:
            print(f"  {r.question_id}: support={r.support_correct}, "
                  f"answer={r.answer_correct}, source={r.source_correct}")
        else:
            print(f"  {r.question_id}: R={r.R_path}, correct={r.target_grounded_correct}")


if __name__ == "__main__":
    main()
