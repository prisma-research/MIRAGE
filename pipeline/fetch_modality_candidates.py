"""
Fetch candidate modality evidence images + seed Q&A via the HuggingFace
datasets-server REST API (no `datasets` library needed — base env lacks it).

Saves images to data/generated_images/modality_staging/<modality>/ and a
candidates.json with seed question/answer per image, for manual review +
selection of the final modality objects.

Usage:
    python -m pipeline.fetch_modality_candidates
"""
from __future__ import annotations
import json, io, sys
from pathlib import Path
import requests
from PIL import Image

BASE = "https://datasets-server.huggingface.co"
ROOT = Path(__file__).resolve().parent.parent
STAGE = ROOT / "data" / "generated_images" / "modality_staging"

# (modality, dataset, config, split, question_field, answer_extractor)
SOURCES = [
    ("photo",       "lmms-lab/VQAv2",  "default",         "validation",
     "question", lambda r: r.get("multiple_choice_answer", "")),
    ("document",    "lmms-lab/DocVQA", "DocVQA",           "validation",
     "question", lambda r: (r.get("answers") or [""])[0]),
    ("infographic", "lmms-lab/DocVQA", "InfographicVQA",   "validation",
     "question", lambda r: (r.get("answers") or [""])[0]),
]

def fetch_rows(dataset, config, split, offset, length=20):
    r = requests.get(f"{BASE}/rows", params={
        "dataset": dataset, "config": config, "split": split,
        "offset": offset, "length": length}, timeout=60)
    r.raise_for_status()
    return r.json().get("rows", [])

def download_image(src, out_path: Path) -> bool:
    try:
        resp = requests.get(src, timeout=60)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
        if img.mode not in ("RGB",):
            img = img.convert("RGB")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="PNG")
        return True
    except Exception as e:
        print(f"    download fail: {repr(e)[:120]}")
        return False

def main(n_per_modality=6, scan=60):
    catalog = []
    for modality, ds, cfg, split, qf, ans_fn in SOURCES:
        print(f"=== {modality} ← {ds}/{cfg} ===")
        out_dir = STAGE / modality
        got = 0
        offset = 0
        while got < n_per_modality and offset < scan:
            rows = fetch_rows(ds, cfg, split, offset, length=20)
            if not rows:
                break
            for item in rows:
                if got >= n_per_modality:
                    break
                row = item["row"]
                q = str(row.get(qf, "")).strip()
                a = str(ans_fn(row)).strip()
                src = (row.get("image") or {}).get("src")
                if not (q and a and src):
                    continue
                # skip trivial yes/no for richer evidence
                if a.lower() in ("yes", "no"):
                    continue
                stem = f"{modality}_{offset+got:03d}"
                img_path = out_dir / f"{stem}.png"
                if download_image(src, img_path):
                    rec = {
                        "stem": stem, "modality": modality, "dataset": ds, "config": cfg,
                        "image_path": str(img_path.relative_to(ROOT)),
                        "seed_question": q, "seed_answer": a,
                        "extra": {k: row.get(k) for k in
                                  ("question_types", "answer_type", "question_type") if k in row},
                    }
                    catalog.append(rec)
                    print(f"  [{stem}] Q={q[:60]!r} A={a[:30]!r}")
                    got += 1
            offset += 20
    STAGE.mkdir(parents=True, exist_ok=True)
    with (STAGE / "candidates.json").open("w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(catalog)} candidates → {STAGE/'candidates.json'}")

if __name__ == "__main__":
    main()
