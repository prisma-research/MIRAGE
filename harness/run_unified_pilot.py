"""
Unified state-conditioned pilot runner.

Single prompt protocol: ANSWERABLE/SOURCE/ANSWER.
Runs the same 200 questions across 5 states, scores from one structured response.

Usage:
    cd MIRAGE
    python -m harness.run_unified_pilot --max-bq 6        # smoke test
    python -m harness.run_unified_pilot --max-bq 20       # small validation
    python -m harness.run_unified_pilot                    # full run (200 × 5 = 1000)
    python -m harness.run_unified_pilot --analysis-only
"""

from __future__ import annotations
import argparse, asyncio, json, logging, os, re, subprocess, sys, time, uuid
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

GB_ROOT = Path(__file__).parent.parent
BANK_PATH = GB_ROOT / "configs" / "study" / "unified_state_conditioned_bank.json"
OUTPUT_DIR = GB_ROOT / "logs" / "probes" / "unified_v1"

ALL_STATE_LABELS = ["d0", "d50k", "d80k", "S2", "S3"]

# Default postcomp episode. Use --postcomp-episode to switch between:
#   pilot_v3_100k_prewrite          (leaked prewrite — original, answer-bearing prompt)
#   pilot_v3_100k_prewrite_generic  (generic prewrite — no answer leakage)
DEFAULT_POSTCOMP_EPISODE = "pilot_v3_100k_prewrite_generic"

def _build_states(postcomp_episode: str) -> list[tuple]:
    return [
        ("pilot_v3_100k",   "s1_prequery_d0",   "same_session",  "d0"),
        ("pilot_v3_100k",   "s1_prequery_d50k", "same_session",  "d50k"),
        ("pilot_v3_100k",   "s1_prequery_d80k", "same_session",  "d80k"),
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
    # Strip common markdown / formatting wrappers before basename mapping.
    s = raw.strip().strip("*`_ ").strip()
    if not s or s.upper() == "NONE":
        return "none"
    # Strip URL-style fragment anchors: #L1, #L5C3, #section-name
    s = re.sub(r'#.*$', '', s)
    s = os.path.basename(s)
    s = re.sub(r'\.(png|jpg|jpeg|json|md)$', '', s)
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

# ── Model override ────────────────────────────────────────────────────────
def _is_rate_limited(text: str) -> bool:
    t = (text or "").lower()
    return "rate limit" in t or "429" in t or "too many requests" in t

def _swap_api_key(cfg: dict, provider_id: str, new_key: str):
    """Replace the apiKey for a provider in the config dict (in-place)."""
    spec = cfg.get("models", {}).get("providers", {}).get(provider_id)
    if spec:
        spec["apiKey"] = new_key
        logger.info("API key swap: %s → backup key", provider_id)

def _patch_model_override(cfg: dict, model_id: str):
    """Override the primary model in an openclaw config dict (in-place).

    Patches:
      1. agents.defaults.model.primary
      2. Per-agent model field in agents.list[*]
    """
    cfg.setdefault("agents", {}).setdefault("defaults", {}).setdefault("model", {})["primary"] = model_id
    for agent in cfg.get("agents", {}).get("list", []):
        if "model" in agent:
            agent["model"] = model_id
    logger.info("Model override: primary → %s", model_id)

async def _restore_and_start(manifest, branch_id, proxy_port, model_override, api_key_override=None):
    """Restore checkpoint, apply patches, start gateway. Returns branch."""
    needs_manual_gw = proxy_port is not None or model_override or api_key_override
    if needs_manual_gw:
        branch = await restore_checkpoint(manifest, branch_id=branch_id,
                                          port_start=19800, start_gateway=False)
        cfg_path = branch.branch_state_dir / "openclaw.json"
        with cfg_path.open() as f: cfg = json.load(f)
        if proxy_port is not None:
            patch_openclaw_config(cfg, proxy_port=proxy_port, compaction_model=None)
        else:
            patch_openclaw_config(cfg, proxy_port=None, compaction_model=None)
        if model_override:
            _patch_model_override(cfg, model_override)
        if api_key_override:
            provider = model_override.split("/")[0] if model_override else "anthropic"
            _swap_api_key(cfg, provider, api_key_override)
        with cfg_path.open("w") as f: json.dump(cfg, f, indent=2)
        env = {**os.environ, "OPENCLAW_STATE_DIR": str(branch.branch_state_dir)}
        (branch.branch_state_dir / "logs").mkdir(parents=True, exist_ok=True)
        gw = subprocess.Popen(
            ["openclaw","gateway","run","--port",str(branch.gateway_port),"--force"],
            stdout=(branch.branch_state_dir/"logs"/"gateway.log").open("a"),
            stderr=(branch.branch_state_dir/"logs"/"gateway.err.log").open("a"), env=env)
        branch.gateway_proc = gw
        dl = time.monotonic() + 60
        while time.monotonic() < dl:
            try:
                r = subprocess.run(["openclaw","gateway","status"],capture_output=True,timeout=15,env=env)
                if r.returncode == 0: break
            except subprocess.TimeoutExpired: pass
            await asyncio.sleep(0.5)
    else:
        branch = await restore_checkpoint(manifest, branch_id=branch_id, port_start=19800)
    return branch

# ── Run one probe ─────────────────────────────────────────────────────────
async def run_one(ep, ckpt, cont, bq, state_label, proxy_port, model_override=None,
                  keep_branch_dir=None):
    """Run one probe. If keep_branch_dir is set, preserve the branch state there instead of deleting."""
    manifest = load_checkpoint(ep, ckpt)
    pid = f"u_{state_label}_{bq['base_question_id']}_{uuid.uuid4().hex[:4]}"
    backup_key = os.environ.get("ANTHROPIC_API_KEY_BACKUP")

    branch = await _restore_and_start(manifest, f"br_{pid}", proxy_port, model_override)
    try:
        t0 = time.monotonic()
        if cont == "same_session":
            cr = await continue_same_session(branch, message=bq["prompt"])
        else:
            cr = await create_fresh_session(branch, message=bq["prompt"])
        dur = time.monotonic() - t0

        # Rate-limit retry with backup key
        if _is_rate_limited(cr.response_text) and backup_key:
            logger.warning("Rate limit on primary key — retrying with backup key (%s/%s)",
                           state_label, bq["base_question_id"])
            branch.cleanup()
            pid2 = f"u_{state_label}_{bq['base_question_id']}_{uuid.uuid4().hex[:4]}_bk"
            branch = await _restore_and_start(manifest, f"br_{pid2}", proxy_port,
                                              model_override, api_key_override=backup_key)
            t0 = time.monotonic()
            if cont == "same_session":
                cr = await continue_same_session(branch, message=bq["prompt"])
            else:
                cr = await create_fresh_session(branch, message=bq["prompt"])
            dur = time.monotonic() - t0

        split = cr.jsonl_count_before if cr.jsonl_count_before is not None else 0
        tt = _extract_tool_trace(cr.jsonl, split)
        rp = classify_r_path(tt, cont, cr.jsonl)

        scores = parse_and_score(cr.response_text, bq)
        result = {
            "probe_id": pid,
            "base_question_id": bq["base_question_id"],
            "state": state_label,
            "response_text": cr.response_text[:500],
            "R_path": rp,
            "duration": round(dur, 1),
            **scores,
        }

        if keep_branch_dir:
            result["kept_branch"] = str(keep_branch_dir)

        return result
    finally:
        if keep_branch_dir:
            # Stop gateway but preserve state dir by moving it
            branch.stop_gateway()
            keep_branch_dir.mkdir(parents=True, exist_ok=True)
            if branch.branch_state_dir.exists():
                import shutil as _shutil
                dest = keep_branch_dir / "state"
                if dest.exists():
                    _shutil.rmtree(dest)
                _shutil.move(str(branch.branch_state_dir), str(dest))
                # Write probe metadata
                meta = {
                    "probe_id": pid,
                    "base_question_id": bq["base_question_id"],
                    "state": state_label,
                    "checkpoint": ckpt,
                    "episode": ep,
                    "continuation_mode": cont,
                    "branch_id": f"br_{pid}",
                    "kept_at": str(keep_branch_dir),
                }
                with (keep_branch_dir / "probe_meta.json").open("w") as f:
                    json.dump(meta, f, indent=2)
                logger.info("Branch kept: %s", keep_branch_dir)
        else:
            branch.cleanup()

# ── Analysis ──────────────────────────────────────────────────────────────
def analyze(results):
    s = {"n": len(results)}
    # Derive states from actual results, in canonical order
    present = {r["state"] for r in results}
    states = [st for st in ALL_STATE_LABELS if st in present]
    s["states"] = states
    metrics = ["answerable_correct","source_correct","answer_correct","grounded_correct","hallucinated","wrong_source"]

    def _rate(items, field):
        vals = [r[field] for r in items if r.get(field) is not None]
        return round(sum(vals)/len(vals),3) if vals else None

    # State table
    table = {}
    for st in states:
        rs = [r for r in results if r["state"]==st and r.get("parse_success")]
        row = {m: _rate(rs, m) for m in metrics}
        pf = sum(1 for r in results if r["state"]==st and not r.get("parse_success"))
        row["parse_failure"] = pf
        row["n"] = len(rs) + pf
        table[st] = row
    s["by_state"] = table

    # Penalties — only between adjacent states actually present
    pen = {}
    pairs = [(states[i], states[i+1]) for i in range(len(states)-1)]
    for m in ["answerable_correct","source_correct","grounded_correct"]:
        for s1,s2 in pairs:
            v1 = table.get(s1,{}).get(m)
            v2 = table.get(s2,{}).get(m)
            if v1 is not None and v2 is not None:
                pen[f"{s1}_to_{s2}_{m}"] = round(v1-v2, 3)
    s["penalties"] = pen

    # R_path by state
    rp = defaultdict(lambda: defaultdict(int))
    for r in results:
        if r.get("R_path"): rp[r["state"]][r["R_path"]] += 1
    s["rpath"] = {k: dict(v) for k,v in rp.items()}

    s["parse_failure_rate"] = round(sum(1 for r in results if not r.get("parse_success"))/max(len(results),1), 3)
    return s

def print_analysis(s):
    states = s.get("states", ALL_STATE_LABELS)
    print(f"\n{'='*80}")
    print("UNIFIED STATE-CONDITIONED ANALYSIS")
    print(f"{'='*80}")
    print(f"Total: {s['n']}, parse_failure_rate: {s['parse_failure_rate']}")

    print(f"\n--- By State ---")
    header = f"{'State':<7}{'n':>4} {'answerable':>11} {'source':>8} {'grounded':>9} {'answer':>8} {'halluc':>7} {'ws':>5} {'pf':>3}"
    print(header)
    for st in states:
        r = s["by_state"].get(st, {})
        def f(v): return f"{v:.3f}" if v is not None else "-"
        print(f"{st:<7}{r.get('n',0):>4} {f(r.get('answerable_correct')):>11} "
              f"{f(r.get('source_correct')):>8} {f(r.get('grounded_correct')):>9} "
              f"{f(r.get('answer_correct')):>8} "
              f"{f(r.get('hallucinated')):>7} {f(r.get('wrong_source')):>5} {r.get('parse_failure',0):>3}")

    print(f"\n--- Penalties ---")
    for k,v in s.get("penalties",{}).items():
        print(f"  {k}: {v:+.3f}")

    print(f"\n--- R_path ---")
    for st in states:
        print(f"  {st}: {s.get('rpath',{}).get(st,{})}")

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
    if args.model:
        print(f"model:      {args.model}")
    else:
        print(f"model:      (checkpoint default — no override)")

    bank = json.load(BANK_PATH.open())
    all_bqs = bank["base_questions"]

    # Stable stratified ordering: round-robin across 4 strata.
    # This order is deterministic and reproducible for any start/count slice.
    strata = []
    for ans in ["YES","NO"]:
        for vt in ["standalone_chart","app_ui"]:
            strata.append([q for q in all_bqs if q["gold_answerable"]==ans and q["visual_type"]==vt])
    ordered, idxs = [], [0]*len(strata)
    while True:
        added = False
        for si, pool in enumerate(strata):
            if idxs[si] < len(pool):
                ordered.append(pool[idxs[si]]); idxs[si] += 1; added = True
        if not added: break

    # Slice selection
    if args.question_ids:
        allowed_ids = set(qid.strip() for qid in args.question_ids.split(","))
        bqs = [q for q in ordered if q["base_question_id"] in allowed_ids]
        print(f"Question filter: {len(bqs)} specific IDs: {[q['base_question_id'] for q in bqs]}")
    else:
        start = args.start_index or 0
        count = args.count or args.max_bq or len(ordered)
        bqs = ordered[start:start + count]
        print(f"Question selection: ordered[{start}:{start+count}] → {len(bqs)} base questions")
        print(f"  IDs: {[q['base_question_id'] for q in bqs]}")

    proxy = UsageProxy(patch_global=False) if any(
        s.get("needsUsageProxy") for s in EXPERIMENT_MODEL_REGISTRY.values()) else None
    pp = None
    if proxy: proxy.__enter__(); pp = proxy.port; print(f"Proxy on {pp}")

    rf = run_dir / "results.jsonl"
    all_r = []
    try:
        states = _build_states(args.postcomp_episode)
        if args.states:
            allowed = [s.strip() for s in args.states.split(",")]
            states = [s for s in states if s[3] in allowed]
            print(f"State filter: {allowed} → running {[s[3] for s in states]}")
        for ep, ckpt, cont, sl in states:
            print(f"\n--- {sl} ({ckpt}, {cont}) ---")
            for i, bq in enumerate(bqs):
                logger.info("[%d/%d] %s", i+1, len(bqs), bq["base_question_id"])
                try:
                    kb_dir = None
                    if args.keep_branches:
                        kb_dir = run_dir / "branches" / sl / f"{bq['base_question_id']}"
                    r = await run_one(ep, ckpt, cont, bq, sl, pp,
                                      model_override=args.model, keep_branch_dir=kb_dir)
                    all_r.append(r)
                    with rf.open("a") as f: f.write(json.dumps(r)+"\n")
                    logger.info("  → parse=%s ans_ok=%s src_ok=%s R=%s",
                        r.get("parse_success"), r.get("answerable_correct"),
                        r.get("source_correct"), r.get("R_path"))
                except Exception as e:
                    logger.error("  FAILED: %s", e)
    finally:
        if proxy: proxy.__exit__(None,None,None)

    print(f"\nTotal: {len(all_r)}")
    summary = analyze(all_r)
    summary["run_id"] = run_id
    summary["model_override"] = args.model
    summary["states_filter"] = args.states
    summary["n_base_questions"] = len(bqs)
    print_analysis(summary)
    with (run_dir/"summary.json").open("w") as f: json.dump(summary, f, indent=2)
    print(f"\nResults: {rf}")
    print(f"Summary: {run_dir/'summary.json'}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-bq", type=int, default=None, help="Select first N from stable order (shorthand for --start-index 0 --count N)")
    p.add_argument("--start-index", type=int, default=None, help="Start index in stable question order")
    p.add_argument("--count", type=int, default=None, help="Number of questions from start-index")
    p.add_argument("--run-id", type=str, default=None, help="Unique run identifier (auto-generated if omitted)")
    p.add_argument("--analysis-only", action="store_true")
    p.add_argument("--rescore", nargs="+", help="Re-score existing runs with updated canonicalization (no re-run)")
    p.add_argument("--merge", nargs="+", help="Merge multiple run dirs into cumulative summary")
    p.add_argument("--postcomp-episode", type=str, default=DEFAULT_POSTCOMP_EPISODE,
                   help="Episode ID for S2/S3 post-compaction checkpoint "
                        "(default: %(default)s). "
                        "Use 'pilot_v3_100k_prewrite' for leaked-prewrite, "
                        "'pilot_v3_100k_prewrite_generic' for generic-prewrite.")
    p.add_argument("--model", type=str, default=None,
                   help="Override primary model in restored branch config "
                        "(e.g. 'anthropic/claude-haiku-4-5-20251001')")
    p.add_argument("--states", type=str, default=None,
                   help="Comma-separated state labels to run (default: all). "
                        "E.g. 'd0,d50k,d80k,S2'")
    p.add_argument("--question-ids", type=str, default=None,
                   help="Comma-separated base_question_ids to run (overrides --start-index/--count/--max-bq). "
                        "E.g. 'tp_c1_01,ua_s2_06'")
    p.add_argument("--keep-branches", action="store_true",
                   help="Preserve probe branch state dirs instead of deleting them. "
                        "Saved to <run_dir>/branches/<state>/<base_question_id>/")
    asyncio.run(main_async(p.parse_args()))

if __name__ == "__main__":
    main()
