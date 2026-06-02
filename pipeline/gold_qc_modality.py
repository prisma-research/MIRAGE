"""
Gold QC for the modality-ext answerable questions: ask Doubao Seed 2.0 each
question with its image (direct Ark API, no gateway) and compare its answer to
the authored gold using the SAME matcher as run_unified_pilot. Concurrent.

Flags disagreements for manual review (catches gold authored wrong / ambiguous).

Usage: python -m pipeline.gold_qc_modality [--workers 6]
"""
from __future__ import annotations
import argparse, base64, json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.request, urllib.error

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
from harness.run_unified_pilot import _ans_match  # reuse exact matcher

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "configs" / "study"
ARK = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
MODEL = "doubao-seed-2-0-pro-260215"
KEY = os.environ["VOLCENGINE"]

INSTR = ("Answer the question about the image with ONLY the short answer "
         "(a value, word, or short phrase) — no explanation.")

def _post(content, retries=2):
    body = {"model": MODEL, "max_tokens": 80, "messages": [{"role": "user", "content": content}]}
    last = None
    for _ in range(retries + 1):
        try:
            req = urllib.request.Request(ARK, data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
            r = json.load(urllib.request.urlopen(req, timeout=120))
            return r["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last = e
    raise last

def ask_image(img_path: str, question: str) -> str:
    b64 = base64.b64encode((ROOT / img_path).read_bytes()).decode()
    return _post([{"type": "text", "text": f"{INSTR}\n\nQuestion: {question}"},
                  {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}])

def ask_text(content: str, question: str) -> str:
    return _post(f"{INSTR}\n\nDocument:\n{content}\n\nQuestion: {question}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only first N answerable Qs (0=all)")
    ap.add_argument("--one-per-object", action="store_true", help="smoke: 1 question per object")
    args = ap.parse_args()

    objs = {o["artifact_id"]: o for o in json.loads((STUDY / "modality_ext_objects.json").read_text())}
    bank = json.loads((STUDY / "modality_ext_bank.json").read_text())
    ans_qs = [q for q in bank["base_questions"] if q["gold_answerable"] == "YES"]
    if args.one_per_object:
        seen = set(); pick = []
        for q in ans_qs:
            if q["gold_source"] not in seen:
                seen.add(q["gold_source"]); pick.append(q)
        ans_qs = pick
    if args.limit:
        ans_qs = ans_qs[:args.limit]

    def work(q):
        aid = q["gold_source"]
        obj = objs[aid]
        try:
            if obj.get("image_path"):
                pred = ask_image(obj["image_path"], q["natural_question"])
            else:
                pred = ask_text(obj["content"], q["natural_question"])
        except Exception as e:
            return {"q": q, "pred": f"<ERR {repr(e)[:60]}>", "match": False, "err": True}
        at = q.get("answer_type", "string"); g = q["gold_answer"]
        match = (_ans_match(pred, g, at)
                 or _ans_match(pred.strip().rstrip('.。!?,;:"\' '), g, at)
                 or (at == "string" and g.lower() in pred.lower()))  # gold substring of pred
        return {"q": q, "pred": pred, "match": match, "err": False}

    results = []
    print(f"QC: {len(ans_qs)} answerable Qs, {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, q) for q in ans_qs]
        for n, f in enumerate(as_completed(futs), 1):
            r = f.result(); results.append(r)
            mk = "OK " if r["match"] else "XX "
            print(f"  [{n}/{len(ans_qs)}] {mk} {r['q']['base_question_id']} "
                  f"gold={r['q']['gold_answer']!r} pred={r['pred'][:40]!r}", flush=True)

    results.sort(key=lambda r: r["q"]["base_question_id"])
    agree = sum(1 for r in results if r["match"])
    print(f"\n=== Gold QC: {agree}/{len(results)} Doubao answers match authored gold ===\n")
    print("--- DISAGREEMENTS (review these golds) ---")
    for r in results:
        if not r["match"]:
            q = r["q"]
            print(f"[{q['base_question_id']}] {q['natural_question']}")
            print(f"    gold={q['gold_answer']!r}  doubao={r['pred']!r}")
    out = STUDY / "modality_ext_gold_qc.json"
    out.write_text(json.dumps([{**{k: r[k] for k in ('pred','match','err')},
                                'bqid': r['q']['base_question_id'],
                                'question': r['q']['natural_question'],
                                'gold': r['q']['gold_answer']} for r in results],
                              indent=2, ensure_ascii=False))
    print(f"\nFull QC → {out}  (agreement {agree}/{len(results)} = {agree/len(results):.0%})")

if __name__ == "__main__":
    main()
