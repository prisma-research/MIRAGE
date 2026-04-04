"""
Derive state-conditioned probe bank from existing constrained-output bank.

Migration strategy:
  - target_present → base_question (supported) → full + locate + verify(pos) + verify(neg) + extract
  - unanswerable   → base_question (unsupported) → full + locate + verify(neg)
  - wrong_source   → reused as verify hard-negatives and locate gold

No questions are rewritten from scratch. All gold semantics inherited.
"""

from __future__ import annotations
import json
import re
from collections import defaultdict
from pathlib import Path

GB_ROOT = Path(__file__).parent.parent
OLD_BANK = GB_ROOT / "configs" / "study" / "question_bank_constrained_100k.json"
OUTPUT = GB_ROOT / "configs" / "study" / "state_conditioned_bank.json"

# Confusion group → hard-negative artifact pairings
_HARD_NEGATIVES = {
    "cqa_9f9df328": "cqa_536c64e4",   # NJ GDP ↔ Canada users
    "cqa_536c64e4": "cqa_9f9df328",
    "cqa_3dd0635a": "cqa_92d6ef9a",   # Nevada poker ↔ NBA FCI
    "cqa_92d6ef9a": "cqa_3dd0635a",
    "ss_324e28cd": "ss_102dfb65",     # fitness app ↔ Maps settings
    "ss_102dfb65": "ss_324e28cd",
}

_ARTIFACT_HANDLES = {
    "cqa_9f9df328": "the NJ GDP line chart",
    "cqa_536c64e4": "the Canada digital users bar chart",
    "cqa_3dd0635a": "the Nevada poker tables bar chart",
    "cqa_92d6ef9a": "the NBA Fan Cost Index chart",
    "ss_324e28cd": "the Android fitness app profile screenshot",
    "ss_102dfb65": "the iOS Google Maps notification settings screenshot",
}


def _strip_response_block(prompt: str) -> str:
    """Remove the structured response format block from the end of a prompt."""
    return re.split(r'\n\n\s*Return exactly:', prompt, maxsplit=1)[0].strip()


def derive():
    old = json.load(OLD_BANK.open())
    old_qs = old["questions"]

    base_questions = []
    derived_probes = []
    bq_id_counter = 0

    # --- Pass 1: target_present → supported base questions ---
    tp_qs = [q for q in old_qs if q["family"] == "target_present"]
    for q in tp_qs:
        bq_id_counter += 1
        bq_id = f"bq_{bq_id_counter:03d}"
        nat_q = _strip_response_block(q["prompt"])
        src = q["gold_source"]
        ans = q["gold_answer"]
        at = q.get("answer_type", "string")

        bq = {
            "base_question_id": bq_id,
            "natural_question": nat_q,
            "gold_source": src,
            "gold_answer": ans,
            "answer_type": at,
            "has_support": True,
            "confusion_group": q.get("confusion_group", ""),
            "visual_type": q.get("visual_type", ""),
            "artifact_handle": _ARTIFACT_HANDLES.get(src, src),
            # Inherited metadata
            "old_question_id": q["question_id"],
            "old_family": "target_present",
        }
        base_questions.append(bq)

        # Derive: full
        derived_probes.append({
            "probe_id": f"full_{bq_id}",
            "task_type": "full",
            "base_question_id": bq_id,
            "prompt": (
                f"{nat_q}\n\n"
                "Return exactly:\n"
                "ANSWERABLE=YES | NO\n"
                "SOURCE=<artifact_id or NONE>\n"
                "ANSWER=<short value or NONE>"
            ),
            "gold_answerable": "YES",
            "gold_source": src,
            "gold_answer": ans,
            "answer_type": at,
        })

        # Derive: locate
        derived_probes.append({
            "probe_id": f"locate_{bq_id}",
            "task_type": "locate",
            "base_question_id": bq_id,
            "prompt": (
                f"Which artifact I shared earlier contains the answer to: {nat_q}\n\n"
                "Return exactly:\n"
                "SOURCE=<artifact_id or NONE>"
            ),
            "gold_source": src,
        })

        # Derive: verify positive
        handle = _ARTIFACT_HANDLES.get(src, src)
        derived_probes.append({
            "probe_id": f"verify_{bq_id}_pos",
            "task_type": "verify",
            "base_question_id": bq_id,
            "candidate_artifact_id": src,
            "candidate_artifact_handle": handle,
            "prompt": (
                f"Does {handle} contain information to answer: {nat_q}\n\n"
                "Return exactly:\n"
                "SUPPORTED=YES | NO"
            ),
            "gold_supported": "YES",
        })

        # Derive: verify hard negative
        neg_id = _HARD_NEGATIVES.get(src)
        if neg_id:
            neg_handle = _ARTIFACT_HANDLES.get(neg_id, neg_id)
            derived_probes.append({
                "probe_id": f"verify_{bq_id}_neg",
                "task_type": "verify",
                "base_question_id": bq_id,
                "candidate_artifact_id": neg_id,
                "candidate_artifact_handle": neg_handle,
                "prompt": (
                    f"Does {neg_handle} contain information to answer: {nat_q}\n\n"
                    "Return exactly:\n"
                    "SUPPORTED=YES | NO"
                ),
                "gold_supported": "NO",
            })

        # Derive: extract_in_state
        derived_probes.append({
            "probe_id": f"extract_{bq_id}",
            "task_type": "extract_in_state",
            "base_question_id": bq_id,
            "source_artifact_id": src,
            "source_artifact_handle": handle,
            "prompt": (
                f"Using {handle} that I shared earlier, answer: {nat_q}\n\n"
                "Return exactly:\n"
                "ANSWER=<short value>"
            ),
            "gold_answer": ans,
            "answer_type": at,
        })

    # --- Pass 2: unanswerable → unsupported base questions ---
    ua_qs = [q for q in old_qs if q["family"] == "unanswerable"]
    for q in ua_qs:
        bq_id_counter += 1
        bq_id = f"bq_{bq_id_counter:03d}"
        nat_q = _strip_response_block(q["prompt"])

        bq = {
            "base_question_id": bq_id,
            "natural_question": nat_q,
            "gold_source": "NONE",
            "gold_answer": "NONE",
            "answer_type": "none",
            "has_support": False,
            "confusion_group": q.get("confusion_group", ""),
            "visual_type": q.get("visual_type", ""),
            "artifact_handle": None,
            "old_question_id": q["question_id"],
            "old_family": "unanswerable",
        }
        base_questions.append(bq)

        # full
        derived_probes.append({
            "probe_id": f"full_{bq_id}",
            "task_type": "full",
            "base_question_id": bq_id,
            "prompt": (
                f"{nat_q}\n\n"
                "Return exactly:\n"
                "ANSWERABLE=YES | NO\n"
                "SOURCE=<artifact_id or NONE>\n"
                "ANSWER=<short value or NONE>"
            ),
            "gold_answerable": "NO",
            "gold_source": "NONE",
            "gold_answer": "NONE",
            "answer_type": "none",
        })

        # locate
        derived_probes.append({
            "probe_id": f"locate_{bq_id}",
            "task_type": "locate",
            "base_question_id": bq_id,
            "prompt": (
                f"Which artifact I shared earlier contains the answer to: {nat_q}\n\n"
                "Return exactly:\n"
                "SOURCE=<artifact_id or NONE>"
            ),
            "gold_source": "NONE",
        })

        # verify negative (pick a plausible but wrong candidate)
        # Use the artifact from the same confusion group if available
        target_aid = q.get("target_artifact_id")
        neg_id = target_aid if target_aid and target_aid != "NONE" else "cqa_9f9df328"
        neg_handle = _ARTIFACT_HANDLES.get(neg_id, neg_id)
        derived_probes.append({
            "probe_id": f"verify_{bq_id}_neg",
            "task_type": "verify",
            "base_question_id": bq_id,
            "candidate_artifact_id": neg_id,
            "candidate_artifact_handle": neg_handle,
            "prompt": (
                f"Does {neg_handle} contain information to answer: {nat_q}\n\n"
                "Return exactly:\n"
                "SUPPORTED=YES | NO"
            ),
            "gold_supported": "NO",
        })

        # No extract_in_state for unsupported questions

    # --- Pass 3: wrong_source → NOT base questions, but used for additional verify probes ---
    # The wrong_source questions asked about content in artifact B while referencing artifact A.
    # In the new design, the natural question part can become a base question whose
    # gold_source is the REAL source (old gold_source), not the hidden target.
    ws_qs = [q for q in old_qs if q["family"] == "wrong_source"]
    for q in ws_qs:
        bq_id_counter += 1
        bq_id = f"bq_{bq_id_counter:03d}"
        nat_q = _strip_response_block(q["prompt"])
        # The old gold_source is the REAL source (the distractor that actually has the data)
        real_src = q.get("gold_source", "NONE")
        real_ans = q.get("gold_answer", "NONE")
        at = q.get("answer_type", "string")
        # The old target_artifact_id is the WRONG source the question references
        wrong_src = q.get("target_artifact_id", "")

        # This question IS answerable — the data exists in real_src
        if real_src and real_src != "NONE":
            has_support = True
        else:
            has_support = False

        bq = {
            "base_question_id": bq_id,
            "natural_question": nat_q,
            "gold_source": real_src if has_support else "NONE",
            "gold_answer": real_ans if has_support else "NONE",
            "answer_type": at if has_support else "none",
            "has_support": has_support,
            "confusion_group": q.get("confusion_group", ""),
            "visual_type": q.get("visual_type", ""),
            "artifact_handle": _ARTIFACT_HANDLES.get(real_src, real_src) if has_support else None,
            "old_question_id": q["question_id"],
            "old_family": "wrong_source",
        }
        base_questions.append(bq)

        # full — now scored against the REAL source, not the hidden target
        derived_probes.append({
            "probe_id": f"full_{bq_id}",
            "task_type": "full",
            "base_question_id": bq_id,
            "prompt": (
                f"{nat_q}\n\n"
                "Return exactly:\n"
                "ANSWERABLE=YES | NO\n"
                "SOURCE=<artifact_id or NONE>\n"
                "ANSWER=<short value or NONE>"
            ),
            "gold_answerable": "YES" if has_support else "NO",
            "gold_source": real_src if has_support else "NONE",
            "gold_answer": real_ans if has_support else "NONE",
            "answer_type": at if has_support else "none",
        })

        # locate — gold is the REAL source
        derived_probes.append({
            "probe_id": f"locate_{bq_id}",
            "task_type": "locate",
            "base_question_id": bq_id,
            "prompt": (
                f"Which artifact I shared earlier contains the answer to: {nat_q}\n\n"
                "Return exactly:\n"
                "SOURCE=<artifact_id or NONE>"
            ),
            "gold_source": real_src if has_support else "NONE",
        })

        if has_support:
            # verify positive — the REAL source supports this question
            real_handle = _ARTIFACT_HANDLES.get(real_src, real_src)
            derived_probes.append({
                "probe_id": f"verify_{bq_id}_pos",
                "task_type": "verify",
                "base_question_id": bq_id,
                "candidate_artifact_id": real_src,
                "candidate_artifact_handle": real_handle,
                "prompt": (
                    f"Does {real_handle} contain information to answer: {nat_q}\n\n"
                    "Return exactly:\n"
                    "SUPPORTED=YES | NO"
                ),
                "gold_supported": "YES",
            })

            # verify hard negative — the old hidden target does NOT support this question
            if wrong_src and wrong_src != real_src:
                wrong_handle = _ARTIFACT_HANDLES.get(wrong_src, wrong_src)
                derived_probes.append({
                    "probe_id": f"verify_{bq_id}_neg",
                    "task_type": "verify",
                    "base_question_id": bq_id,
                    "candidate_artifact_id": wrong_src,
                    "candidate_artifact_handle": wrong_handle,
                    "prompt": (
                        f"Does {wrong_handle} contain information to answer: {nat_q}\n\n"
                        "Return exactly:\n"
                        "SUPPORTED=YES | NO"
                    ),
                    "gold_supported": "NO",
                })

            # extract_in_state
            derived_probes.append({
                "probe_id": f"extract_{bq_id}",
                "task_type": "extract_in_state",
                "base_question_id": bq_id,
                "source_artifact_id": real_src,
                "source_artifact_handle": real_handle,
                "prompt": (
                    f"Using {real_handle} that I shared earlier, answer: {nat_q}\n\n"
                    "Return exactly:\n"
                    "ANSWER=<short value>"
                ),
                "gold_answer": real_ans,
                "answer_type": at,
            })

    # Build output
    result = {
        "bank_id": "state_conditioned_v1",
        "description": (
            "State-conditioned probe bank derived from constrained_100k. "
            f"{len(base_questions)} base questions → {len(derived_probes)} probes. "
            "No hidden-target scoring. Tasks: full, locate, verify, extract_in_state."
        ),
        "derived_from": str(OLD_BANK),
        "base_questions": base_questions,
        "derived_probes": derived_probes,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w") as f:
        json.dump(result, f, indent=2)

    # Report
    from collections import Counter
    by_task = Counter(p["task_type"] for p in derived_probes)
    by_old_fam = Counter(bq["old_family"] for bq in base_questions)
    sup = sum(1 for bq in base_questions if bq["has_support"])
    unsup = len(base_questions) - sup

    print(f"Base questions: {len(base_questions)} ({sup} supported, {unsup} unsupported)")
    print(f"  from old families: {dict(by_old_fam)}")
    print(f"Derived probes: {len(derived_probes)}")
    print(f"  by task: {dict(by_task)}")
    print(f"Saved to {OUTPUT}")


if __name__ == "__main__":
    derive()
