from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

GB_ROOT = Path(__file__).resolve().parents[1]
if str(GB_ROOT) not in sys.path:
    sys.path.insert(0, str(GB_ROOT))

from harness.run_unified_pilot import _ans_match, _canon_source, analyze

LOCAL_BUNDLE = GB_ROOT / "results" / "mirage_local_model_bundle_20260331"
OLD_BUNDLE = GB_ROOT / "results" / "mirage_model_bundle_20260330"
BANK_PATH = GB_ROOT / "configs" / "study" / "unified_state_conditioned_bank.json"
OUT_DIR = LOCAL_BUNDLE / "curated_c0_cm_rescored_4state"
TARGET_STATES = ["d0", "d50k", "d80k", "S2"]


SOURCE_SPECS = {
    "gpt5": {
        "C0": {
            "results": [
                OLD_BUNDLE / "gpt5/probes/unified_v1/gpt5_generic_4state_full/results.jsonl",
            ],
            "source_hints": [
                OLD_BUNDLE / "gpt5/probes/unified_v1/gpt5_generic_4state_full",
            ],
        },
    },
    "haiku": {
        "C0": {
            "results": [
                OLD_BUNDLE / "haiku/probes/unified_v1/haiku_generic_4state_full/results.jsonl",
            ],
            "source_hints": [
                OLD_BUNDLE / "haiku/probes/unified_v1/haiku_generic_4state_full",
            ],
        },
    },
    "qwen3-vl-4b-instruct": {
        "C0": {
            "results": [
                LOCAL_BUNDLE / "qwen3-vl-4b-instruct/Cm/results.jsonl",
            ],
            "filter_condition": "C0",
            "source_hints": [
                LOCAL_BUNDLE / "qwen3-vl-4b-instruct/Cm",
            ],
        },
        "Cm": {
            "results": [
                LOCAL_BUNDLE / "qwen3-vl-4b-instruct/Cm/results.jsonl",
            ],
            "filter_condition": "Cm",
            "source_hints": [
                LOCAL_BUNDLE / "qwen3-vl-4b-instruct/Cm",
                LOCAL_BUNDLE / "qwen3-vl-4b-instruct/Cm/raw_conversations",
            ],
        },
    },
    "qwen3-vl-8b-instruct": {
        "C0": {
            "results": [
                LOCAL_BUNDLE / "qwen3-vl-8b-instruct/Cm/results.jsonl",
            ],
            "filter_condition": "C0",
            "source_hints": [
                LOCAL_BUNDLE / "qwen3-vl-8b-instruct/Cm",
            ],
        },
        "Cm": {
            "results": [
                LOCAL_BUNDLE / "qwen3-vl-8b-instruct/Cm/results.jsonl",
            ],
            "filter_condition": "Cm",
            "source_hints": [
                LOCAL_BUNDLE / "qwen3-vl-8b-instruct/Cm",
                LOCAL_BUNDLE / "qwen3-vl-8b-instruct/Cm/raw_qwen3vl8b_cm_full",
                LOCAL_BUNDLE / "qwen3-vl-8b-instruct/Cm/raw_qwen3vl8b_cm_d50k",
                LOCAL_BUNDLE / "qwen3-vl-8b-instruct/Cm/raw_qwen3vl8b_cm_d80k",
                LOCAL_BUNDLE / "qwen3-vl-8b-instruct/Cm/raw_qwen3vl8b_cm_S2",
            ],
        },
    },
    "qwen3-vl-30b-instruct": {
        "C0": {
            "results": [
                OLD_BUNDLE / "qwen3vl30b/probes/unified_v1/smoke_qwen3vl30b_generic_003/results.jsonl",
                OLD_BUNDLE / "qwen3vl30b/probes/unified_v1/qwen3vl30b_generic_007to050/results.jsonl",
                OLD_BUNDLE / "qwen3vl30b/probes/unified_v1/qwen3vl30b_generic_051to100/results.jsonl",
                OLD_BUNDLE / "qwen3vl30b/probes/unified_v1/qwen3vl30b_generic_101to200/results.jsonl",
            ],
            "source_hints": [
                OLD_BUNDLE / "qwen3vl30b/probes/unified_v1/smoke_qwen3vl30b_generic_003",
                OLD_BUNDLE / "qwen3vl30b/probes/unified_v1/qwen3vl30b_generic_007to050",
                OLD_BUNDLE / "qwen3vl30b/probes/unified_v1/qwen3vl30b_generic_051to100",
                OLD_BUNDLE / "qwen3vl30b/probes/unified_v1/qwen3vl30b_generic_101to200",
                OLD_BUNDLE / "qwen3vl30b/probes/unified_v1/merged_smoke_qwen3vl30b_generic_003_qwen3vl30b_generic_007to050_qwen3vl30b_generic_051to100_qwen3vl30b_generic_101to200",
            ],
        },
        "Cm": {
            "results": [
                LOCAL_BUNDLE / "qwen3-vl-30b-instruct/Cm/results.jsonl",
            ],
            "filter_condition": "Cm",
            "source_hints": [
                LOCAL_BUNDLE / "qwen3-vl-30b-instruct/Cm",
                LOCAL_BUNDLE / "qwen3-vl-30b-instruct/Cm/raw_qwen3vl30b_cm_d0",
                LOCAL_BUNDLE / "qwen3-vl-30b-instruct/Cm/raw_qwen3vl30b_cm_d50k",
                LOCAL_BUNDLE / "qwen3-vl-30b-instruct/Cm/raw_qwen3vl30b_cm_d80k",
                LOCAL_BUNDLE / "qwen3-vl-30b-instruct/Cm/raw_qwen3vl30b_cm_S2",
            ],
        },
    },
    "internvl3_5-20b-a4b": {
        "C0": {
            "results": [
                LOCAL_BUNDLE / "internvl3_5-20b-a4b/C0/results.jsonl",
            ],
            "source_hints": [
                LOCAL_BUNDLE / "internvl3_5-20b-a4b/C0",
            ],
        },
        "Cm": {
            "results": [
                LOCAL_BUNDLE / "internvl3_5-20b-a4b/Cm/results.jsonl",
            ],
            "source_hints": [
                LOCAL_BUNDLE / "internvl3_5-20b-a4b/Cm",
                LOCAL_BUNDLE / "internvl3_5-20b-a4b/raw_internvl20b_cm_d0",
                LOCAL_BUNDLE / "internvl3_5-20b-a4b/raw_internvl20b_cm_d50k",
                LOCAL_BUNDLE / "internvl3_5-20b-a4b/raw_internvl20b_cm_d80k",
                LOCAL_BUNDLE / "internvl3_5-20b-a4b/raw_internvl20b_cm_S2",
            ],
        },
    },
    "gemma-3-27b-it": {
        "C0": {
            "results": [
                OLD_BUNDLE / "gemma3_27b/probes/unified_v1/gemma3_27b_generic_full/results.jsonl",
            ],
            "source_hints": [
                OLD_BUNDLE / "gemma3_27b/probes/unified_v1/gemma3_27b_generic_full",
            ],
        },
        "Cm": {
            "results": [
                LOCAL_BUNDLE / "gemma-3-27b-it/Cm/results.jsonl",
            ],
            "filter_condition": "Cm",
            "source_hints": [
                LOCAL_BUNDLE / "gemma-3-27b-it/Cm",
                LOCAL_BUNDLE / "gemma-3-27b-it/Cm/raw_gemma27b_cm_d0",
                LOCAL_BUNDLE / "gemma-3-27b-it/Cm/raw_gemma27b_cm_d50k",
                LOCAL_BUNDLE / "gemma-3-27b-it/Cm/raw_gemma27b_cm_d80k",
                LOCAL_BUNDLE / "gemma-3-27b-it/Cm/raw_gemma27b_cm_S2",
            ],
        },
    },
}


def load_bank():
    bank = json.loads(BANK_PATH.read_text())
    return {bq["base_question_id"]: bq for bq in bank["base_questions"]}


def load_rows(paths: list[Path], condition: str, filter_condition: str | None = None):
    rows = []
    seen = set()
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("state") not in TARGET_STATES:
                continue
            row_condition = row.get("condition", condition)
            if filter_condition and row_condition != filter_condition:
                continue
            key = (row_condition, row.get("state"), row.get("base_question_id"))
            if key in seen:
                continue
            seen.add(key)
            row["condition"] = condition
            rows.append(row)
    return rows


def rescore_rows(rows: list[dict], bq_map: dict[str, dict]):
    rescored = []
    for row in rows:
        r = deepcopy(row)
        pred_src = r.get("pred_source", "")
        r["canonical_source"] = _canon_source(pred_src)
        bq = bq_map.get(r.get("base_question_id"))
        if bq and r.get("parse_success"):
            ga = bq["gold_answerable"]
            gs = (bq["gold_source"] or "NONE").lower()
            gv = bq["gold_answer"]
            at = bq.get("answer_type", "string")
            pred_ans_enum = r.get("pred_answerable", "")
            pred_ans = r.get("pred_answer", "")
            r["answerable_correct"] = 1 if pred_ans_enum == ga else 0
            if ga == "YES":
                r["source_correct"] = 1 if r["canonical_source"] == gs else 0
                r["answer_correct"] = 1 if _ans_match(pred_ans, gv, at) else 0
                r["grounded_correct"] = 1 if r["source_correct"] == 1 and r["answer_correct"] == 1 else 0
                r["hallucinated"] = None
                r["wrong_source"] = 1 if r["canonical_source"] != gs else 0
            else:
                r["source_correct"] = None
                r["answer_correct"] = None
                r["grounded_correct"] = None
                r["hallucinated"] = 1 if pred_ans_enum == "YES" or pred_ans.upper() != "NONE" else 0
                r["wrong_source"] = None
        rescored.append(r)
    return rescored


def enrich_summary(summary: dict, rows: list[dict], model: str, condition: str):
    summary = deepcopy(summary)
    summary["model"] = model
    summary["condition"] = condition
    summary["n_base_questions"] = len({r["base_question_id"] for r in rows})
    summary["states"] = [st for st in TARGET_STATES if st in {r["state"] for r in rows}]
    return summary


def write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def build_condition_comparison(model: str, cond_summaries: dict[str, dict]):
    out = {
        "model": model,
        "conditions": sorted(cond_summaries.keys()),
        "states": TARGET_STATES,
        "by_condition": cond_summaries,
    }
    if "C0" in cond_summaries and "Cm" in cond_summaries:
        deltas = {}
        c0 = cond_summaries["C0"]["by_state"]
        cm = cond_summaries["Cm"]["by_state"]
        for st in TARGET_STATES:
            if st not in c0 or st not in cm:
                continue
            row = {}
            for metric in [
                "answerable_correct",
                "source_correct",
                "answer_correct",
                "grounded_correct",
                "hallucinated",
                "wrong_source",
            ]:
                v0 = c0[st].get(metric)
                v1 = cm[st].get(metric)
                if v0 is not None and v1 is not None:
                    row[metric] = round(v1 - v0, 3)
            if row:
                deltas[st] = row
        out["cm_minus_c0"] = deltas
    return out


def main():
    bq_map = load_bank()
    if OUT_DIR.exists():
        import shutil
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {"target_states": TARGET_STATES, "models": {}}
    flat_rows = []

    for model, cond_map in SOURCE_SPECS.items():
        model_cond_summaries = {}
        manifest["models"][model] = {}
        for condition, spec in cond_map.items():
            rows = load_rows(spec["results"], condition, spec.get("filter_condition"))
            original_summary = enrich_summary(analyze(rows), rows, model, condition)
            rescored_rows = rescore_rows(rows, bq_map)
            rescored_summary = enrich_summary(analyze(rescored_rows), rescored_rows, model, condition)

            cond_dir = OUT_DIR / model / condition
            write_jsonl(cond_dir / "original" / "results.jsonl", rows)
            write_json(cond_dir / "original" / "summary.json", original_summary)
            write_jsonl(cond_dir / "rescored" / "results.jsonl", rescored_rows)
            write_json(cond_dir / "rescored" / "summary.json", rescored_summary)

            provenance = {
                "model": model,
                "condition": condition,
                "source_results": [str(p.relative_to(GB_ROOT)) for p in spec["results"]],
                "source_hints": [str(p.relative_to(GB_ROOT)) for p in spec.get("source_hints", []) if p.exists()],
                "filter_condition": spec.get("filter_condition"),
                "target_states": TARGET_STATES,
                "n_rows": len(rows),
                "n_base_questions": len({r["base_question_id"] for r in rows}),
            }
            write_json(cond_dir / "provenance.json", provenance)

            model_cond_summaries[condition] = rescored_summary
            manifest["models"][model][condition] = provenance

            for st, vals in rescored_summary["by_state"].items():
                flat_row = {
                    "model": model,
                    "condition": condition,
                    "state": st,
                    "n": vals.get("n"),
                    "parse_failure": vals.get("parse_failure"),
                    "answerable_correct": vals.get("answerable_correct"),
                    "source_correct": vals.get("source_correct"),
                    "answer_correct": vals.get("answer_correct"),
                    "grounded_correct": vals.get("grounded_correct"),
                    "hallucinated": vals.get("hallucinated"),
                    "wrong_source": vals.get("wrong_source"),
                }
                rpath = rescored_summary.get("rpath", {}).get(st, {})
                flat_row["R_context"] = rpath.get("R_context", 0)
                flat_row["R_tool"] = rpath.get("R_tool", 0)
                flat_rows.append(flat_row)

        comparison = build_condition_comparison(model, model_cond_summaries)
        write_json(OUT_DIR / model / "comparison_summary.json", comparison)

    write_json(OUT_DIR / "manifest.json", manifest)
    write_json(OUT_DIR / "all_metrics_by_model_condition_state.json", {"rows": flat_rows})

    with (OUT_DIR / "all_metrics_by_model_condition_state.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model", "condition", "state", "n", "parse_failure",
                "answerable_correct", "source_correct", "answer_correct",
                "grounded_correct", "hallucinated", "wrong_source",
                "R_context", "R_tool",
            ],
        )
        writer.writeheader()
        writer.writerows(flat_rows)

    readme = f"""# Curated C0/Cm Rescored 4-State Bundle

This folder standardizes the MIRAGE 4-state results requested for:

- `gpt5` (`C0`)
- `haiku` (`C0`)
- `qwen3-vl-4b-instruct` (`C0`, `Cm`)
- `qwen3-vl-8b-instruct` (`C0`, `Cm`)
- `qwen3-vl-30b-instruct` (`C0`, `Cm`)
- `internvl3_5-20b-a4b` (`C0`, `Cm`)
- `gemma-3-27b-it` (`C0`, `Cm`)

For each model/condition, this bundle stores:

- `original/results.jsonl`: original probe-level rows used as input
- `original/summary.json`: summary recomputed on the standardized 4-state subset
- `rescored/results.jsonl`: probe-level rows after current source canonicalization and metric recomputation
- `rescored/summary.json`: rescored summary
- `provenance.json`: source paths used to assemble the condition

Top-level aggregate files:

- `manifest.json`
- `all_metrics_by_model_condition_state.json`
- `all_metrics_by_model_condition_state.csv`

Target states: `{", ".join(TARGET_STATES)}`
"""
    (OUT_DIR / "README.md").write_text(readme)

    print(f"Wrote curated bundle: {OUT_DIR}")


if __name__ == "__main__":
    main()
