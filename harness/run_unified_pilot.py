"""
Unified state-conditioned pilot runner.

Single prompt protocol: ANSWERABLE/SOURCE/ANSWER.
Runs the same 200 questions across 5 states, scores from one structured response.

Usage:
    cd GroundingBench
    python -m harness.run_unified_pilot --max-bq 6        # smoke test
    python -m harness.run_unified_pilot --max-bq 20       # small validation
    python -m harness.run_unified_pilot                    # full run (200 × 5 = 1000)
    python -m harness.run_unified_pilot --analysis-only
"""

from __future__ import annotations
import argparse, asyncio, json, logging, os, re, shutil, subprocess, sys, time, uuid
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from client.usage_proxy import UsageProxy, patch_openclaw_config
from constants import EXPERIMENT_MODEL_REGISTRY
from harness.checkpoint import (
    load_checkpoint, restore_checkpoint, CHECKPOINTS_ROOT,
    continue_same_session, create_fresh_session,
)
from harness.branch_probe_runner import classify_r_path, _extract_tool_trace
from mitigations.citation_force.bootstrap_hook import (
    inject_citation_force_into_bootstrap,
    CONDITION_MARKER_FILE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Cm query-time wrapper ────────────────────────────────────────────────
# For same-session continuations the system prompt is already fixed (set at
# session creation in the checkpoint). BOOTSTRAP.md edits do NOT propagate.
# The workspace-side marker + BOOTSTRAP.md injection only gates tool exposure
# (the TypeScript plugin reads the marker at request time).
#
# To get the citation-force constraint in front of the model during same-session
# probes we prepend a query-time wrapper to the user prompt.
CM_QUERY_PREFIX = (
    "[IMPORTANT — CitationForce protocol is active for this query]\n"
    "Before answering, you MUST call the `artifact_recall` tool with the "
    "artifact identifier or relevant keywords. Do NOT use `memory_search` "
    "or `read` — `artifact_recall` is the ONLY approved retrieval method.\n"
    "After calling `artifact_recall`, include [Source: <path>] in your response.\n"
    "If `artifact_recall` returns no results, say: "
    "\"I was unable to locate the artifact.\"\n\n"
)

# ── CitationForce mitigation ladder (query-time prefixes) ────────────────
# All probe states of interest (d0/d50k/d80k/S2) are same-session continuations,
# so the system prompt is fixed at session creation and BOOTSTRAP.md edits never
# reach the model. The constraint is therefore delivered at query time, mirroring
# CM_QUERY_PREFIX. Each rung adds exactly one thing over the previous:
#   C0  no mitigation (no prefix, no tool)
#   C1  prompt-only: ask for a [Source: <path>] citation, NO tool available
#   C2  C1 + the artifact_recall retrieval tool is available
#   C3  C2 + post-compaction-aware re-injection ("context was compacted, re-retrieve")
#   Cm  mandatory artifact_recall routing (CM_QUERY_PREFIX), memory_search prohibited
C1_QUERY_PREFIX = (
    "[CitationForce protocol — grounding]\n"
    "When you reference any artifact, file, screenshot, chart, or image from the "
    "earlier conversation or from memory, you MUST include its source path in your "
    "response as [Source: <path>]. Do NOT describe or paraphrase an artifact from "
    "memory without citing the specific source it came from.\n\n"
)
C2_QUERY_PREFIX = (
    "[CitationForce protocol — grounding via retrieval]\n"
    "An `artifact_recall` tool is available: call it with the artifact identifier or "
    "relevant keywords to retrieve the artifact and its source path. When you "
    "reference any artifact, retrieve it with `artifact_recall` and include its "
    "source path in your response as [Source: <path>]. Do NOT describe an artifact "
    "from memory without retrieving it first.\n\n"
)
C3_QUERY_PREFIX = (
    "[CitationForce protocol — grounding via retrieval]\n"
    "NOTE: the earlier conversation context has been compacted/summarized, so any "
    "artifact details you recall from it may be lossy. Before answering, RE-RETRIEVE "
    "the artifact with the `artifact_recall` tool (using its identifier or relevant "
    "keywords) rather than relying on the summary. Include the retrieved source path "
    "in your response as [Source: <path>], and do NOT cite from the compacted summary "
    "alone.\n\n"
)

# condition → query-time prefix (C0 has none → not in the map)
CF_QUERY_PREFIXES = {
    "C1": C1_QUERY_PREFIX,
    "C2": C2_QUERY_PREFIX,
    "C3": C3_QUERY_PREFIX,
    "Cm": CM_QUERY_PREFIX,
}
# conditions that need the workspace marker + BOOTSTRAP injection
# (marker gates the artifact_recall plugin: it exposes the tool for C2/C3/Cm)
CF_MITIGATION_CONDITIONS = ("C1", "C2", "C3", "Cm")

GB_ROOT = Path(__file__).parent.parent
BANK_PATH = GB_ROOT / "configs" / "study" / "unified_state_conditioned_bank.json"
OUTPUT_DIR = GB_ROOT / "logs" / "probes" / "unified_v1"

# Default postcomp episode. Use --postcomp-episode to switch between:
#   pilot_v3_100k_prewrite          (leaked prewrite — original, answer-bearing prompt)
#   pilot_v3_100k_prewrite_generic  (generic prewrite — no answer leakage)
DEFAULT_POSTCOMP_EPISODE = "pilot_v3_100k_prewrite_generic"

def _build_states(postcomp_episode: str, s1_episode: str = "pilot_v3_100k") -> list[tuple]:
    return [
        (s1_episode,        "s1_prequery_d0",   "same_session",  "d0"),
        (s1_episode,        "s1_prequery_d50k", "same_session",  "d50k"),
        (s1_episode,        "s1_prequery_d80k", "same_session",  "d80k"),
        (postcomp_episode,  "postcomp_100k",    "same_session",  "S2"),
        (postcomp_episode,  "postcomp_100k",    "fresh_session", "S3"),
    ]

# ── Source canonicalization (from branch_probe_runner) ────────────────────
_DERIVED = {
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

def _canon_source(raw: str) -> str:
    if not raw or raw.upper() == "NONE":
        return "none"
    # Strip URL-style fragment anchors: #L1, #L5C3, #section-name
    s = re.sub(r'#.*$', '', raw)
    s = os.path.basename(s)
    s = re.sub(r'\.(png|jpg|jpeg|json|md|py|txt|pdf|csv)$', '', s)
    sl = s.lower()
    return _DERIVED.get(sl, sl)

# ── Numeric normalization ─────────────────────────────────────────────────
def _norm_num(val: str) -> float | None:
    if not val or val.upper() == "NONE":
        return None
    c = val.strip()
    c = re.sub(r'^[\$€£~≈]+', '', c).strip().rstrip('%').strip()
    c = re.sub(r'^(approximately|about|around|roughly)\s+', '', c, flags=re.I).strip()
    _U = {'k':1e3,'thousand':1e3,'m':1e6,'million':1e6,'b':1e9,'billion':1e9}
    mul = 1.0
    m = re.match(r'^([\d,.]+)\s*([a-zA-Z]+)', c)
    if m:
        u = m.group(2).lower()
        if u in _U: mul = _U[u]
        c = m.group(1)
    m2 = re.match(r'^([\d,.]+)\s+(million|billion|thousand)\b', c, re.I)
    if m2:
        mul = _U[m2.group(2).lower()]; c = m2.group(1)
    else:
        m3 = re.match(r'^([\d,.]+)\s+\S', c)
        if m3: c = m3.group(1)
    c = c.replace(",","")
    try: return float(c)*mul
    except: return None

def _raw_num(val: str) -> float | None:
    if not val or val.upper() == "NONE": return None
    c = val.strip()
    c = re.sub(r'^[\$€£~≈]+', '', c).strip()
    c = re.sub(r'\s*(million|billion|thousand|[MBKk])\b.*$', '', c, flags=re.I).strip()
    m = re.match(r'^([\d,.]+)', c)
    if m: c = m.group(1)
    c = c.replace(",","")
    try: return float(c)
    except: return None

def _normalize_string_answer(s: str) -> str:
    """Deterministic string answer normalization.

    Rules applied in order:
    1. Strip quotes, collapse whitespace, lowercase
    2. Collapse unit-adjacent spaces: "181 cm" → "181cm", "5 kg" → "5kg"
    3. Expand CJK scale words: "200万" → "2000000", "3亿" → "300000000"
    4. Expand English scale words: "2 million" → "2000000"
    5. Strip commas from numbers: "2,000,000" → "2000000"
    6. Normalize step/步 suffix: "步" → "steps"
    7. Strip preamble like "Award for"
    8. Strip parenthetical glosses like "(2,000,000 steps)"
    9. Synonym map for boolean-like values
    """
    s = re.sub(r'\s+', ' ', s.strip().strip("'\"").lower())
    # Strip copyright/trademark symbols: © ® ™
    s = re.sub(r'[©®™]', '', s).strip()
    # Normalize dotted abbreviations: U.S. → US, U.K. → UK, etc.
    s = re.sub(r'\b([A-Za-z])\.([A-Za-z])\.', r'\1\2', s)
    # Strip redundant quantity glosses in parens: "(2,000,000 steps)" → ""
    # These contain commas (formatted numbers) + optional unit word.
    # Short parens like "(2021)" are NOT stripped — they carry information.
    s = re.sub(r'\s*\([\d,.\s]*,[\d,.\s]*(steps|步|thousand|million|billion|[kKmMbB])?\)', '', s)
    # Flatten remaining parens: "Statista (2021)" → "Statista 2021"
    s = s.replace('(', ' ').replace(')', ' ')
    # Re-collapse whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # Strip common preamble
    s = re.sub(r'^(award for|badge[:\s]+)\s*', '', s, flags=re.I).strip()
    # CJK scale: 万=10^4, 亿=10^8
    def _expand_cjk(m):
        num = float(m.group(1).replace(',', ''))
        unit = m.group(2)
        mul = {'万': 1e4, '亿': 1e8}[unit]
        v = num * mul
        return str(int(v)) if v == int(v) else str(v)
    s = re.sub(r'([\d,.]+)(万|亿)', lambda m: _expand_cjk(m) + ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()  # re-collapse any double spaces
    # English scale words within string answers
    _SCALE = {'thousand': 1e3, 'million': 1e6, 'billion': 1e9}
    def _expand_eng(m):
        num = float(m.group(1).replace(',', ''))
        mul = _SCALE[m.group(2).lower()]
        v = num * mul
        return str(int(v)) if v == int(v) else str(v)
    s = re.sub(r'([\d,.]+)\s+(thousand|million|billion)', _expand_eng, s, flags=re.I)
    # Strip commas from remaining numbers
    s = re.sub(r'(\d),(\d)', r'\1\2', s)
    # Normalize 步 → steps
    s = re.sub(r'步', 'steps', s)
    # Collapse unit-adjacent space: "181 cm" → "181cm"
    s = re.sub(r'(\d)\s+(cm|kg|lb|lbs|mm|m|km|ft|in|oz|g|mg|mph|bpm|%)\b', r'\1\2', s)
    # Synonym map
    _SYN = {"true":"yes","false":"no","correct":"yes","incorrect":"no",
            "enabled":"on","disabled":"off","active":"on","inactive":"off"}
    s = _SYN.get(s, s)
    return s

def _ans_match(pred, gold, at):
    if not pred or not gold:
        return (pred or "NONE").upper() == (gold or "NONE").upper()
    if at == "numeric":
        gn, pn = _norm_num(gold), _norm_num(pred)
        if gn is not None and pn is not None:
            if abs(gn-pn) <= max(abs(gn)*0.01, 0.5): return True
        gr, pr = _raw_num(gold), _raw_num(pred)
        if gr is not None and pr is not None:
            if abs(gr-pr) <= max(abs(gr)*0.01, 0.5): return True
        return False
    if at == "year": return pred.strip() == gold.strip()
    # string: deterministic normalization
    ps = _normalize_string_answer(pred)
    gs = _normalize_string_answer(gold)
    return ps == gs

# ── Parse + Score ─────────────────────────────────────────────────────────
def parse_and_score(response: str, bq: dict) -> dict:
    r = {"parse_success": False}
    m1 = re.search(r'ANSWERABLE\s*=\s*(YES|NO)', response, re.I)
    m2 = re.search(r'SOURCE\s*=\s*(\S+)', response, re.I)
    m3 = re.search(r'ANSWER\s*=\s*(.+)', response, re.I)
    # Fallback: JSON format (e.g. Gemma wraps in ```json {"ANSWERABLE": "YES", ...} ```)
    if not (m1 and m2 and m3):
        try:
            # Strip markdown code fences
            cleaned = re.sub(r'```(?:json)?\s*', '', response).strip()
            obj = json.loads(cleaned)
            if isinstance(obj, dict) and "ANSWERABLE" in obj:
                m1 = re.match(r'(YES|NO)', str(obj["ANSWERABLE"]), re.I)
                m2_val = str(obj.get("SOURCE", "NONE")).strip()
                m3_val = str(obj.get("ANSWER", "NONE")).strip()
                if m1:
                    # Create fake match-like values for downstream
                    class _M:
                        def __init__(self, v): self._v = v
                        def group(self, _=1): return self._v
                    m1 = _M(m1.group(1))
                    m2 = _M(m2_val)
                    m3 = _M(m3_val)
        except (json.JSONDecodeError, ValueError, KeyError):
            pass
    if not (m1 and m2 and m3):
        return r
    r["parse_success"] = True
    pred_ans_enum = m1.group(1).upper()
    pred_src = m2.group(1).strip()
    pred_ans = m3.group(1).strip()

    ga = bq["gold_answerable"]
    gs = (bq["gold_source"] or "NONE").lower()
    gv = bq["gold_answer"]
    at = bq.get("answer_type", "string")

    r["pred_answerable"] = pred_ans_enum
    r["pred_source"] = pred_src
    r["pred_answer"] = pred_ans
    r["canonical_source"] = _canon_source(pred_src)

    r["answerable_correct"] = 1 if pred_ans_enum == ga else 0
    # source + answer (scope: gold answerable=YES)
    if ga == "YES":
        r["source_correct"] = 1 if r["canonical_source"] == gs else 0
        r["answer_correct"] = 1 if _ans_match(pred_ans, gv, at) else 0
        # grounded_answer_correct: source AND answer both correct
        r["grounded_correct"] = 1 if r["source_correct"] == 1 and r["answer_correct"] == 1 else 0
    else:
        r["source_correct"] = None
        r["answer_correct"] = None
        r["grounded_correct"] = None
    # hallucination (gold=NO but model says YES or gives non-NONE answer)
    if ga == "NO":
        r["hallucinated"] = 1 if pred_ans_enum == "YES" or pred_ans.upper() != "NONE" else 0
    else:
        r["hallucinated"] = None
    # wrong_source: gold=YES but model gave a DIFFERENT source (including NONE).
    # Any source that doesn't match gold counts — whether the model said another
    # artifact or said NONE (missed the source entirely).
    if ga == "YES":
        r["wrong_source"] = 1 if r["canonical_source"] != gs else 0
    else:
        r["wrong_source"] = None
    return r

# ── Run one probe ─────────────────────────────────────────────────────────
def _patch_agent_model(cfg: dict, model_id: str) -> None:
    """Override the probing-model in a branch openclaw.json (in-place).

    Sets both agents.defaults.model.primary and the per-agent model entry so
    that both same-session continuation (which uses the current agent model)
    and fresh-session creation (which picks up the agent default) use the
    requested probing model instead of the checkpoint's original gpt-5.
    The source checkpoint is NOT touched — this only runs on the ephemeral
    branch copy created by restore_checkpoint.
    """
    agents = cfg.setdefault("agents", {})
    agents.setdefault("defaults", {}).setdefault("model", {})["primary"] = model_id
    for entry in agents.get("list", []):
        if "model" in entry:
            entry["model"] = model_id
    logger.info("Config patch: agent model → %s", model_id)


def _persist_raw_conversation(
    raw_root: Path,
    probe_id: str,
    episode_id: str,
    checkpoint_id: str,
    continuation: str,
    state_label: str,
    bq: dict,
    cr,
    duration_s: float,
    scores: dict,
) -> dict[str, str]:
    """Persist the full session transcript before branch cleanup removes it."""
    state_dir = raw_root / state_label
    state_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{state_label}__{bq['base_question_id']}"
    jsonl_path = state_dir / f"{stem}.jsonl"
    meta_path = state_dir / f"{stem}.meta.json"

    with jsonl_path.open("w") as f:
        for entry in cr.jsonl:
            f.write(json.dumps(entry) + "\n")

    meta = {
        "probe_id": probe_id,
        "episode_id": episode_id,
        "checkpoint_id": checkpoint_id,
        "continuation": continuation,
        "state": state_label,
        "base_question_id": bq["base_question_id"],
        "prompt": bq["prompt"],
        "session_id": cr.session_id,
        "duration_seconds": round(duration_s, 1),
        "jsonl_entry_count": len(cr.jsonl),
        "history_split_index": cr.jsonl_count_before if cr.jsonl_count_before is not None else 0,
        "jsonl_count_before": cr.jsonl_count_before,
        "jsonl_count_after": cr.jsonl_count_after,
        "jsonl_file_path": cr.jsonl_file_path,
        "session_id_matches_checkpoint": cr.session_id_matches_checkpoint,
        "response_text": cr.response_text,
        "response_payload": cr.response,
        "scores": scores,
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    return {
        "raw_session_jsonl": str(jsonl_path),
        "raw_session_meta": str(meta_path),
    }


async def run_one(ep, ckpt, cont, bq, state_label, proxy_port, model_id=None,
                  raw_root: Path | None = None, condition: str = "C0",
                  keep_branch_dir: Path | None = None, port_start: int = 19800):
    manifest = load_checkpoint(ep, ckpt)
    pid = f"u_{condition}_{state_label}_{bq['base_question_id']}_{uuid.uuid4().hex[:4]}"

    # Always restore with start_gateway=False so we can patch the branch config
    # (proxy baseUrl, probing model) before the gateway reads openclaw.json.
    # port_start is offset per concurrency slot so parallel probes don't race
    # on the same gateway port.
    branch = await restore_checkpoint(manifest, branch_id=f"br_{pid}",
                                      port_start=port_start, start_gateway=False)
    cfg_path = branch.branch_state_dir / "openclaw.json"
    with cfg_path.open() as f:
        cfg = json.load(f)

    if proxy_port is not None:
        patch_openclaw_config(cfg, proxy_port=proxy_port, compaction_model=None)

    if model_id is not None:
        _patch_agent_model(cfg, model_id)

    with cfg_path.open("w") as f:
        json.dump(cfg, f, indent=2)

    # ── CitationForce workspace setup (before gateway start) ──
    # Write condition marker + BOOTSTRAP.md so the TypeScript plugin exposes
    # artifact_recall for the tool-bearing rungs (C2/C3/Cm). C1 writes the marker
    # too (gates the tool OFF — prompt-only) so the plugin stays consistent.
    # For C0 nothing is written.
    if condition in CF_MITIGATION_CONDITIONS:
        ws_candidates = list(branch.branch_state_dir.glob(f"workspace-{branch.agent_id}"))
        if not ws_candidates:
            ws_candidates = list(branch.branch_state_dir.glob("workspace"))
        if ws_candidates:
            inject_citation_force_into_bootstrap(
                workspace_dir=ws_candidates[0], condition=condition,
            )
            logger.info("%s: injected CitationForce + marker into %s", condition, ws_candidates[0])

    env = {**os.environ, "OPENCLAW_STATE_DIR": str(branch.branch_state_dir)}
    (branch.branch_state_dir / "logs").mkdir(parents=True, exist_ok=True)
    gw = subprocess.Popen(
        ["openclaw", "gateway", "run", "--port", str(branch.gateway_port),
         "--force", "--allow-unconfigured"],
        stdout=(branch.branch_state_dir / "logs" / "gateway.log").open("a"),
        stderr=(branch.branch_state_dir / "logs" / "gateway.err.log").open("a"),
        env=env,
    )
    branch.gateway_proc = gw
    # Wait for gateway readiness by probing its HTTP endpoint.
    # `openclaw gateway status` requires systemd which isn't available on HPC nodes.
    import urllib.request, urllib.error
    dl = time.monotonic() + 60
    while time.monotonic() < dl:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{branch.gateway_port}/", timeout=2,
            )
            break
        except (urllib.error.URLError, OSError, ConnectionRefusedError):
            pass
        if gw.poll() is not None:
            raise RuntimeError(
                f"Gateway exited early (code={gw.returncode}). "
                f"Check {branch.branch_state_dir / 'logs' / 'gateway.err.log'}"
            )
        await asyncio.sleep(0.5)

    try:
        t0 = time.monotonic()
        # For same-session probes: prepend the condition's query-time prefix since
        # BOOTSTRAP.md edits don't retroactively affect the existing system prompt.
        # (C0 has no prefix; C1/C2/C3/Cm each prepend their CitationForce ladder text.)
        prompt = bq["prompt"]
        if cont == "same_session" and condition in CF_QUERY_PREFIXES:
            prompt = CF_QUERY_PREFIXES[condition] + prompt

        if cont == "same_session":
            cr = await continue_same_session(branch, message=prompt)
        else:
            cr = await create_fresh_session(branch, message=prompt)
        dur = time.monotonic() - t0

        split = cr.jsonl_count_before if cr.jsonl_count_before is not None else 0
        tt = _extract_tool_trace(cr.jsonl, split)
        rp = classify_r_path(tt, cont, cr.jsonl)

        scores = parse_and_score(cr.response_text, bq)
        raw_paths = {}
        if raw_root is not None:
            raw_paths = _persist_raw_conversation(
                raw_root=raw_root,
                probe_id=pid,
                episode_id=ep,
                checkpoint_id=ckpt,
                continuation=cont,
                state_label=state_label,
                bq=bq,
                cr=cr,
                duration_s=dur,
                scores=scores,
            )

        result = {
            "probe_id": pid,
            "base_question_id": bq["base_question_id"],
            "state": state_label,
            "condition": condition,
            "response_text": cr.response_text[:500],
            "R_path": rp,
            "duration": round(dur, 1),
            **raw_paths,
            **scores,
        }

        if keep_branch_dir:
            result["kept_branch"] = str(keep_branch_dir)

        return result
    finally:
        if keep_branch_dir:
            branch.stop_gateway()
            keep_branch_dir.mkdir(parents=True, exist_ok=True)
            if branch.branch_state_dir.exists():
                dest = keep_branch_dir / "state"
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.move(str(branch.branch_state_dir), str(dest))
                meta = {
                    "probe_id": pid,
                    "base_question_id": bq["base_question_id"],
                    "state": state_label,
                    "condition": condition,
                    "checkpoint": ckpt,
                    "episode": ep,
                    "continuation_mode": cont,
                    "branch_id": f"br_{pid}",
                    "kept_at": str(keep_branch_dir),
                }
                with (keep_branch_dir / "probe_meta.json").open("w") as f:
                    json.dump(meta, f, indent=2)
        else:
            branch.cleanup()

# ── Analysis ──────────────────────────────────────────────────────────────
def analyze(results):
    s = {"n": len(results)}
    # Dynamic state/condition discovery (no hardcoded S3)
    _state_order = ["d0", "d50k", "d80k", "S2", "S3"]
    seen_states = sorted(set(r["state"] for r in results),
                         key=lambda x: _state_order.index(x) if x in _state_order else 99)
    seen_conditions = sorted(set(r.get("condition", "C0") for r in results))
    metrics = ["answerable_correct","source_correct","answer_correct","grounded_correct","hallucinated","wrong_source"]

    def _rate(items, field):
        vals = [r[field] for r in items if r.get(field) is not None]
        return round(sum(vals)/len(vals),3) if vals else None

    # Per-condition × state table
    by_condition = {}
    for cond in seen_conditions:
        table = {}
        cond_results = [r for r in results if r.get("condition", "C0") == cond]
        for st in seen_states:
            rs = [r for r in cond_results if r["state"]==st and r.get("parse_success")]
            row = {m: _rate(rs, m) for m in metrics}
            pf = sum(1 for r in cond_results if r["state"]==st and not r.get("parse_success"))
            row["parse_failure"] = pf
            row["n"] = len(rs) + pf
            table[st] = row
        by_condition[cond] = table
    s["by_condition"] = by_condition

    # Flat by_state for backward compat (uses all results)
    table_flat = {}
    for st in seen_states:
        rs = [r for r in results if r["state"]==st and r.get("parse_success")]
        row = {m: _rate(rs, m) for m in metrics}
        pf = sum(1 for r in results if r["state"]==st and not r.get("parse_success"))
        row["parse_failure"] = pf
        row["n"] = len(rs) + pf
        table_flat[st] = row
    s["by_state"] = table_flat

    # Penalties (within each condition)
    pen = {}
    consecutive = [(seen_states[i], seen_states[i+1]) for i in range(len(seen_states)-1)]
    for cond in seen_conditions:
        tbl = by_condition[cond]
        for m in ["answerable_correct","source_correct","grounded_correct"]:
            for s1,s2 in consecutive:
                v1 = tbl.get(s1,{}).get(m)
                v2 = tbl.get(s2,{}).get(m)
                if v1 is not None and v2 is not None:
                    label = f"{cond}_{s1}_to_{s2}_{m}" if len(seen_conditions) > 1 else f"{s1}_to_{s2}_{m}"
                    pen[label] = round(v1-v2, 3)
    s["penalties"] = pen

    # R_path by condition × state
    rp = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in results:
        if r.get("R_path"):
            rp[r.get("condition","C0")][r["state"]][r["R_path"]] += 1
    s["rpath"] = {c: {st: dict(v) for st,v in sv.items()} for c,sv in rp.items()}

    s["parse_failure_rate"] = round(sum(1 for r in results if not r.get("parse_success"))/max(len(results),1), 3)
    s["states"] = seen_states
    s["conditions"] = seen_conditions
    return s

def print_analysis(s):
    print(f"\n{'='*80}")
    print("UNIFIED STATE-CONDITIONED ANALYSIS")
    print(f"{'='*80}")
    print(f"Total: {s['n']}, parse_failure_rate: {s['parse_failure_rate']}")

    states = s.get("states", ["d0","d50k","d80k","S2","S3"])
    conditions = s.get("conditions", ["C0"])

    def _f(v):
        return f"{v:.3f}" if v is not None else "-"

    for cond in conditions:
        tbl = s.get("by_condition", {}).get(cond, s.get("by_state", {}))
        print(f"\n--- {cond} by State ---")
        header = f"{'State':<7}{'n':>4} {'answerable':>11} {'source':>8} {'grounded':>9} {'answer':>8} {'halluc':>7} {'ws':>5} {'pf':>3}"
        print(header)
        for st in states:
            r = tbl.get(st, {})
            print(f"{st:<7}{r.get('n',0):>4} {_f(r.get('answerable_correct')):>11} "
                  f"{_f(r.get('source_correct')):>8} {_f(r.get('grounded_correct')):>9} "
                  f"{_f(r.get('answer_correct')):>8} "
                  f"{_f(r.get('hallucinated')):>7} {_f(r.get('wrong_source')):>5} {r.get('parse_failure',0):>3}")

    # C0 vs Cm delta table (if both present)
    if "C0" in conditions and len(conditions) > 1:
        c0_tbl = s["by_condition"]["C0"]
        for cond in conditions:
            if cond == "C0":
                continue
            cx_tbl = s["by_condition"][cond]
            print(f"\n--- Delta ({cond} − C0) ---")
            header = f"{'State':<7} {'Δground':>8} {'Δsource':>8} {'Δw_src':>7} {'ΔR_tool':>8}"
            print(header)
            for st in states:
                c0r = c0_tbl.get(st, {})
                cxr = cx_tbl.get(st, {})
                dg = round(cxr.get("grounded_correct",0) - c0r.get("grounded_correct",0), 3) if cxr.get("grounded_correct") is not None and c0r.get("grounded_correct") is not None else None
                ds = round(cxr.get("source_correct",0) - c0r.get("source_correct",0), 3) if cxr.get("source_correct") is not None and c0r.get("source_correct") is not None else None
                dw = round(cxr.get("wrong_source",0) - c0r.get("wrong_source",0), 3) if cxr.get("wrong_source") is not None and c0r.get("wrong_source") is not None else None
                # R_tool share delta
                c0_rp = s.get("rpath",{}).get("C0",{}).get(st,{})
                cx_rp = s.get("rpath",{}).get(cond,{}).get(st,{})
                c0_total = sum(c0_rp.values()) or 1
                cx_total = sum(cx_rp.values()) or 1
                c0_rt = c0_rp.get("R_tool",0)/c0_total
                cx_rt = cx_rp.get("R_tool",0)/cx_total
                drt = round(cx_rt - c0_rt, 3) if c0_total > 0 and cx_total > 0 else None
                print(f"{st:<7} {_f(dg):>8} {_f(ds):>8} {_f(dw):>7} {_f(drt):>8}")

    print(f"\n--- Penalties ---")
    for k,v in s.get("penalties",{}).items():
        print(f"  {k}: {v:+.3f}")

    print(f"\n--- R_path ---")
    for cond in conditions:
        rp_cond = s.get("rpath",{}).get(cond, {})
        for st in states:
            label = f"{cond}/{st}" if len(conditions) > 1 else st
            print(f"  {label}: {rp_cond.get(st,{})}")

# ── Main ──────────────────────────────────────────────────────────────────
async def main_async(args):
    import datetime as _dt

    # Run isolation: unique run_id → unique output dir
    run_id = args.run_id or _dt.datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / run_id

    if args.rescore:
        # Re-score existing raw results with updated canonicalization.
        # Does NOT re-run probes. Reads pred_source from JSONL, re-applies
        # _canon_source, recomputes source_correct/wrong_source/grounded_correct.
        bank = json.load(BANK_PATH.open())
        bq_map = {bq["base_question_id"]: bq for bq in bank["base_questions"]}
        all_results = []
        for rname in args.rescore:
            rd = OUTPUT_DIR / rname
            rf = rd / "results.jsonl"
            if not rf.exists():
                print(f"  SKIP {rname}: no results.jsonl")
                continue
            count = 0
            for line in rf.open():
                if not line.strip():
                    continue
                r = json.loads(line)
                # Re-canonicalize pred_source
                ps = r.get("pred_source", "")
                new_canon = _canon_source(ps)
                r["canonical_source"] = new_canon
                # Re-score source/grounded/wrong_source using gold from bank
                bq = bq_map.get(r.get("base_question_id"))
                if bq and r.get("parse_success"):
                    ga = bq["gold_answerable"]
                    gs = (bq["gold_source"] or "NONE").lower()
                    gv = bq["gold_answer"]
                    at = bq.get("answer_type", "string")
                    if ga == "YES":
                        r["source_correct"] = 1 if new_canon == gs else 0
                        r["answer_correct"] = 1 if _ans_match(r.get("pred_answer",""), gv, at) else 0
                        r["grounded_correct"] = 1 if r["source_correct"] == 1 and r["answer_correct"] == 1 else 0
                        r["wrong_source"] = 1 if new_canon != gs else 0
                all_results.append(r)
                count += 1
            print(f"  Loaded {rname}: {count} results")
        print(f"Total rescored: {len(all_results)}")
        summary = analyze(all_results)
        summary["rescored_runs"] = args.rescore
        print_analysis(summary)
        rescore_dir = OUTPUT_DIR / f"rescored_{'_'.join(args.rescore)}"
        rescore_dir.mkdir(parents=True, exist_ok=True)
        with (rescore_dir / "results.jsonl").open("w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")
        with (rescore_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"Rescored results: {rescore_dir / 'results.jsonl'}")
        print(f"Summary: {rescore_dir / 'summary.json'}")
        return

    if args.merge:
        # Merge multiple runs into one cumulative summary
        all_results = []
        for rname in args.merge:
            rd = OUTPUT_DIR / rname
            rf = rd / "results.jsonl"
            if rf.exists():
                for line in rf.open():
                    if line.strip(): all_results.append(json.loads(line))
                print(f"  Loaded {rname}: {sum(1 for l in (rd/'results.jsonl').open() if l.strip())} results")
        print(f"Total merged: {len(all_results)}")
        summary = analyze(all_results)
        summary["merged_runs"] = args.merge
        print_analysis(summary)
        merge_dir = OUTPUT_DIR / f"merged_{'_'.join(args.merge)}"
        merge_dir.mkdir(parents=True, exist_ok=True)
        with (merge_dir/"summary.json").open("w") as f: json.dump(summary, f, indent=2)
        print(f"Merged summary: {merge_dir/'summary.json'}")
        return

    if args.analysis_only:
        if not run_dir.exists():
            print(f"No run dir: {run_dir}")
            return
        results = []
        for line in (run_dir / "results.jsonl").open():
            if line.strip(): results.append(json.loads(line))
        summary = analyze(results)
        print_analysis(summary)
        with (run_dir/"summary.json").open("w") as f: json.dump(summary, f, indent=2)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_id:     {run_id}")
    print(f"output_dir: {run_dir}")

    bank_path = Path(args.bank) if getattr(args, "bank", None) else BANK_PATH
    if not bank_path.is_absolute():
        bank_path = GB_ROOT / bank_path if not bank_path.exists() else bank_path
    bank = json.load(bank_path.open())
    all_bqs = bank["base_questions"]
    print(f"bank: {bank_path} ({len(all_bqs)} questions)")

    # Stable stratified ordering: round-robin across strata (answerable × visual_type).
    # Generalized over whatever visual_types are present in the bank (combined modality
    # banks carry natural_photo / infographic / code_text / email_text / pdf_text in
    # addition to the original standalone_chart / app_ui), so no questions are dropped.
    vts = sorted(set(q.get("visual_type", "") for q in all_bqs))
    strata = []
    for ans in ["YES", "NO"]:
        for vt in vts:
            pool = [q for q in all_bqs if q["gold_answerable"] == ans and q.get("visual_type") == vt]
            if pool:
                strata.append(pool)
    ordered, idxs = [], [0]*len(strata)
    while True:
        added = False
        for si, pool in enumerate(strata):
            if idxs[si] < len(pool):
                ordered.append(pool[idxs[si]]); idxs[si] += 1; added = True
        if not added: break

    # Slice selection
    start = args.start_index or 0
    count = args.count or args.max_bq or len(ordered)
    bqs = ordered[start:start + count]

    # --question-ids filter (applied after slice)
    if args.question_ids:
        wanted = set(q.strip() for q in args.question_ids.split(","))
        bqs = [q for q in bqs if q["base_question_id"] in wanted]

    print(f"Question selection: ordered[{start}:{start+count}] → {len(bqs)} base questions")
    print(f"  IDs: {[q['base_question_id'] for q in bqs]}")

    # Conditions
    cf_conditions = [c.strip() for c in args.cf_conditions.split(",")]
    print(f"Conditions: {cf_conditions}")

    proxy = UsageProxy(patch_global=False) if any(
        s.get("needsUsageProxy") for s in EXPERIMENT_MODEL_REGISTRY.values()) else None
    pp = None
    if proxy: proxy.__enter__(); pp = proxy.port; print(f"Proxy on {pp}")

    rf = run_dir / "results.jsonl"
    raw_root = run_dir / "raw_conversations"
    all_r = []

    # ── Resume: load probes already completed for this run_id, skip them ────
    done_keys = set()
    if rf.exists():
        for line in rf.open():
            line = line.strip()
            if not line:
                continue
            try:
                rr = json.loads(line)
            except Exception:
                continue
            done_keys.add((rr.get("condition", "C0"), rr.get("state"), rr.get("base_question_id")))
            all_r.append(rr)
        if done_keys:
            print(f"RESUME: {len(done_keys)} probes already in results.jsonl — skipping them")

    import os as _os
    max_workers = max(1, getattr(args, "max_workers", None) or 1)
    # Per-process port base (PID-offset) so concurrent processes don't collide;
    # per-slot offset (below) so concurrent probes within this process don't race.
    base_port = 19500 + (_os.getpid() % 200) * 20
    slot_q: asyncio.Queue = asyncio.Queue()
    for _s in range(max_workers):
        slot_q.put_nowait(_s)
    write_lock = asyncio.Lock()

    try:
        # --family overrides BOTH the s1 and postcomp checkpoint families (e.g.
        # modality_ext_v1 builds all of d0/d50k/d80k/postcomp_100k in one family).
        fam = getattr(args, "family", None)
        s1_ep = fam or "pilot_v3_100k"
        postcomp_ep = fam or args.postcomp_episode
        states = _build_states(postcomp_ep, s1_ep)
        if args.skip_s3:
            states = [(ep, ck, co, sl) for ep, ck, co, sl in states if sl != "S3"]
        # --states filter
        if args.states:
            allowed = set(s.strip() for s in args.states.split(","))
            states = [(ep, ck, co, sl) for ep, ck, co, sl in states if sl in allowed]

        # Build worklist, skipping probes already done (resume)
        specs = []
        for cond in cf_conditions:
            for ep, ckpt, cont, sl in states:
                for bq in bqs:
                    if (cond, sl, bq["base_question_id"]) in done_keys:
                        continue
                    specs.append((cond, ep, ckpt, cont, sl, bq))
        total = len(specs)
        print(f"TO RUN: {total} probes (skipped {len(done_keys)} done) | max_workers={max_workers} base_port={base_port}")
        progress = {"n": 0, "ok": 0}

        async def _worker(cond, ep, ckpt, cont, sl, bq):
            slot = await slot_q.get()
            try:
                kb_dir = None
                if args.keep_branches:
                    kb_dir = run_dir / "branches" / cond / sl / bq["base_question_id"]
                r = await run_one(ep, ckpt, cont, bq, sl, pp, args.model,
                                  raw_root=raw_root, condition=cond,
                                  keep_branch_dir=kb_dir,
                                  port_start=base_port + slot * 6)
            except Exception as e:
                logger.error("FAILED %s/%s/%s: %s", cond, sl, bq["base_question_id"], e)
                slot_q.put_nowait(slot)
                return
            slot_q.put_nowait(slot)
            async with write_lock:
                all_r.append(r)
                with rf.open("a") as f:
                    f.write(json.dumps(r) + "\n")
                progress["n"] += 1
                if r.get("source_correct"):
                    progress["ok"] += 1
                logger.info("[%d/%d done | SC=%d] %s/%s %s parse=%s src_ok=%s R=%s",
                            progress["n"], total, progress["ok"], cond, sl,
                            bq["base_question_id"], r.get("parse_success"),
                            r.get("source_correct"), r.get("R_path"))

        if specs:
            await asyncio.gather(*[_worker(*s) for s in specs])
    finally:
        if proxy: proxy.__exit__(None,None,None)

    print(f"\nTotal: {len(all_r)}")
    summary = analyze(all_r)
    summary["run_id"] = run_id
    summary["n_base_questions"] = len(bqs)
    summary["cf_conditions"] = cf_conditions
    print_analysis(summary)
    with (run_dir/"summary.json").open("w") as f: json.dump(summary, f, indent=2)
    print(f"\nResults: {rf}")
    print(f"Summary: {run_dir/'summary.json'}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-bq", type=int, default=None, help="Select first N from stable order (shorthand for --start-index 0 --count N)")
    p.add_argument("--start-index", type=int, default=None, help="Start index in stable question order")
    p.add_argument("--max-workers", type=int, default=1,
                   help="Concurrent probes (each self-isolated gateway/branch). 1 = sequential.")
    p.add_argument("--count", type=int, default=None, help="Number of questions from start-index")
    p.add_argument("--run-id", type=str, default=None, help="Unique run identifier (auto-generated if omitted)")
    p.add_argument("--analysis-only", action="store_true")
    p.add_argument("--rescore", nargs="+", help="Re-score existing runs with updated canonicalization (no re-run)")
    p.add_argument("--merge", nargs="+", help="Merge multiple run dirs into cumulative summary")
    p.add_argument("--skip-s3", action="store_true",
                   help="Skip S3 (fresh_session) state — useful on HPC without systemd")
    p.add_argument("--model", type=str, default="local_qwen30b/qwen3-vl-30b-instruct",
                   help="Probing model override in provider/model-id format "
                        "(default: local_qwen30b/qwen3-vl-30b-instruct). "
                        "Applied to agents.defaults.model.primary and each agent's model "
                        "in the ephemeral branch config before gateway start. "
                        "The source checkpoint is never mutated.")
    p.add_argument("--postcomp-episode", type=str, default=DEFAULT_POSTCOMP_EPISODE,
                   help="Episode ID for S2/S3 post-compaction checkpoint "
                        "(default: %(default)s). "
                        "Use 'pilot_v3_100k_prewrite' for leaked-prewrite, "
                        "'pilot_v3_100k_prewrite_generic' for generic-prewrite.")
    p.add_argument("--bank", type=str, default=None,
                   help="Path to question bank JSON (default: unified_state_conditioned_bank.json). "
                        "Use configs/study/combined_modality_bank.json for the modality extension.")
    p.add_argument("--family", type=str, default=None,
                   help="Checkpoint family for ALL states (s1 d0/d50k/d80k + postcomp_100k). "
                        "E.g. modality_ext_v1. Overrides the default pilot_v3_100k / postcomp episode.")
    p.add_argument("--states", type=str, default=None,
                   help="Comma-separated state labels to run (default: all). "
                        "E.g. 'd0,d50k,d80k,S2'")
    p.add_argument("--question-ids", type=str, default=None,
                   help="Comma-separated base_question_ids to run "
                        "(default: all in slice)")
    p.add_argument("--keep-branches", action="store_true",
                   help="Preserve probe branch state dirs instead of deleting them. "
                        "Saved to <run_dir>/branches/<condition>/<state>/<base_question_id>/")
    p.add_argument("--cf-conditions", type=str, default="C0",
                   help="Comma-separated CitationForce conditions to run "
                        "(default: C0). E.g. 'C0,Cm' for baseline vs mitigation.")
    asyncio.run(main_async(p.parse_args()))

if __name__ == "__main__":
    main()
