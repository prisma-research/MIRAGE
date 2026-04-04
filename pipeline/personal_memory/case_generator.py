"""Case generator for the T × M personal memory study.

Generates page-grounded benchmark cases from domain templates + random
sampling. Each case includes page_spec, all M-condition memory variants,
condition-specific gold actions, and aligned-action fields.

Usage:
    cd MIRAGE
    python -m pipeline.personal_memory.case_generator --seed 42 --cases-per-slot 3
    python -m pipeline.personal_memory.case_generator --seed 42 --output configs/study/personal_memory_tm_cases_v1.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.personal_memory.domain_templates import (
    ALL_DOMAINS, BACKGROUND_TRAITS, DOMAIN_IRRELEVANT,
    SHARING_DISTRACTORS, SHARING_DISTRACTORS_TAGGED,
)
from pipeline.personal_memory.memory_compiler import (
    compile_m0, compile_m1, compile_m2, compile_m3,
    sample_domain_safe_traits,
)


# ---------------------------------------------------------------------------
# M0 gold derivation (template-derived, not hardcoded)
# ---------------------------------------------------------------------------

def determine_m0_gold(domain: str, task_type: str, t3_scope: str | None,
                      decision_space: list[str], default_action: str | None = None) -> str:
    """Determine M0 gold action from template logic."""
    if task_type == "T3" and t3_scope == "out_of_scope" and default_action:
        return default_action
    return "ASK_USER"


# ---------------------------------------------------------------------------
# M3 distractor sampling (explicit split: background + domain-adjacent)
# ---------------------------------------------------------------------------

def _sample_m3_distractors(domain: str, general_traits: list[str],
                           rng: random.Random,
                           min_domain_adjacent: int = 2,
                           total: int = 4) -> tuple[list[str], list[dict]]:
    """Sample distractors for M3 with guaranteed domain-adjacent quota.

    Returns (flat_texts, tagged_items):
      - flat_texts: list of distractor strings for compile_m3
      - tagged_items: list of {"text": ..., "tag": ...} for structured_state
        (enables tag-based mitigation filtering)
    """
    general_set = set(general_traits)
    tagged_items = []

    # Domain-adjacent pool — tag convention: "{domain}_adjacent"
    adj_tag = f"{domain}_adjacent"
    if domain == "sharing":
        adj_tagged = [d for d in SHARING_DISTRACTORS_TAGGED if d["text"] not in general_set]
    else:
        adj_tagged = [{"text": t, "tag": adj_tag}
                      for t in DOMAIN_IRRELEVANT.get(domain, []) if t not in general_set]

    bg_tagged = [{"text": t, "tag": "general"}
                 for t in BACKGROUND_TRAITS if t not in general_set]

    # Sample domain-adjacent first
    n_adj = min(min_domain_adjacent, len(adj_tagged))
    adj_sample = rng.sample(adj_tagged, k=n_adj)
    tagged_items.extend(adj_sample)

    # Fill remainder from background
    used_texts = {d["text"] for d in adj_sample}
    remaining_bg = [d for d in bg_tagged if d["text"] not in used_texts]
    n_bg = min(total - n_adj, len(remaining_bg))
    bg_sample = rng.sample(remaining_bg, k=n_bg)
    tagged_items.extend(bg_sample)

    flat_texts = [d["text"] for d in tagged_items]
    return flat_texts, tagged_items


# ---------------------------------------------------------------------------
# Case generators by domain × task type
# ---------------------------------------------------------------------------

def generate_booking_t1(template: dict, rng: random.Random, idx: int) -> dict:
    """Generate a T1 (stable preference) booking case."""
    stable = rng.choice(template["t1_stable_pool"])
    general = sample_domain_safe_traits("booking", k=3, rng=rng)

    prefs = {"Seat preference": {"value": stable["value"], "reason": stable["reason"]}}
    gold_specific = stable["value"].upper()

    # T1 M2: stale = a different (plausible old) seat preference
    stale_value = _pick_stale_alternative(stable["value"], ["window", "aisle", "middle"], rng)
    prefs_stale = {"Seat preference": {"value": stale_value, "reason": "previous preference"}}

    return _assemble_case(
        case_id=f"booking_seat_t1_{idx:03d}",
        domain="booking",
        task_type="T1",
        general_traits=general,
        preferences_correct=prefs,
        preferences_stale=prefs_stale,
        gold_specific=gold_specific,
        stale_aligned=stale_value.upper(),
        noisy_aligned=_sample_non_gold(template["decision_space"], gold_specific, rng),
        decision_space=template["decision_space"],
        task_prompt=rng.choice(template["task_prompts"]),
        page_template_id=template["page_template_id"],
        page_spec={"seat_options": ["window", "aisle", "middle"],
                    "flight": f"SH→{'BJ' if rng.random() > 0.5 else 'GZ'}",
                    "date": "2026-04-05"},
        rng=rng,
    )


def generate_booking_t2(template: dict, rng: random.Random, idx: int) -> dict:
    """Generate a T2 (preference update) booking case."""
    update = rng.choice(template["t2_update_pool"])
    general = sample_domain_safe_traits("booking", k=3, rng=rng)

    prefs_correct = {"Seat preference": {
        "value": update["new"], "reason": update["new_reason"]}}
    prefs_stale = {"Seat preference": {
        "value": update["old"], "reason": update["old_reason"]}}

    return _assemble_case(
        case_id=f"booking_seat_t2_{idx:03d}",
        domain="booking",
        task_type="T2",
        general_traits=general,
        preferences_correct=prefs_correct,
        preferences_stale=prefs_stale,
        gold_specific=update["new"].upper(),
        stale_aligned=update["old"].upper(),
        noisy_aligned=_sample_non_gold(
            template["decision_space"], update["new"].upper(), rng),
        decision_space=template["decision_space"],
        task_prompt=rng.choice(template["task_prompts"]),
        page_template_id=template["page_template_id"],
        page_spec={"seat_options": ["window", "aisle", "middle"],
                    "flight": f"SH→{'BJ' if rng.random() > 0.5 else 'CD'}",
                    "date": "2026-04-08"},
        rng=rng,
    )


def generate_sharing_t3_pair(template: dict, rng: random.Random, idx: int) -> list[dict]:
    """Generate a T3 paired case (in-scope + out-of-scope) for sharing."""
    rule_tmpl = rng.choice(template["t3_rule_templates"])
    general = sample_domain_safe_traits("sharing", k=3, rng=rng)

    # T3 M2: stale version of the rule (e.g., rule didn't exist yet, or was more relaxed)
    stale_rules = [f"(outdated) {rule_tmpl['rule'].replace('Do not', 'It is usually fine to')}"]

    in_case = _assemble_case(
        case_id=f"sharing_t3_in_{idx:03d}",
        domain="sharing",
        task_type="T3",
        general_traits=general,
        preferences_correct={},
        preferences_stale={},
        gold_specific=rule_tmpl["constrained_action"],
        stale_aligned=rule_tmpl["default_action"],  # stale = no constraint → default
        noisy_aligned=None,
        decision_space=["SHARE", "COPY_LINK", "DECLINE", "ASK_USER"],
        task_prompt=rng.choice(template["task_prompts"]),
        page_template_id=template["page_template_id"],
        page_spec=rule_tmpl["in_scope_slots"],
        rules=[rule_tmpl["rule"]],
        rules_stale=stale_rules,
        t3_scope="in_scope",
        t3_default_action=rule_tmpl["default_action"],
        rng=rng,
    )

    out_case = _assemble_case(
        case_id=f"sharing_t3_out_{idx:03d}",
        domain="sharing",
        task_type="T3",
        general_traits=general,
        preferences_correct={},
        preferences_stale={},
        gold_specific=rule_tmpl["default_action"],
        stale_aligned=rule_tmpl["default_action"],  # stale still = default for out-of-scope
        noisy_aligned=rule_tmpl["constrained_action"],
        decision_space=["SHARE", "COPY_LINK", "DECLINE", "ASK_USER"],
        task_prompt=rng.choice(template["task_prompts"]),
        page_template_id=template["page_template_id"],
        page_spec=rule_tmpl["out_scope_slots"],
        rules=[rule_tmpl["rule"]],
        rules_stale=stale_rules,
        t3_scope="out_of_scope",
        t3_default_action=rule_tmpl["default_action"],
        rng=rng,
    )

    return [in_case, out_case]


# ---------------------------------------------------------------------------
# Core assembly
# ---------------------------------------------------------------------------

def _assemble_case(
    *, case_id, domain, task_type, general_traits, preferences_correct,
    preferences_stale, gold_specific, stale_aligned, noisy_aligned,
    decision_space, task_prompt, page_template_id, page_spec, rng,
    rules=None, rules_stale=None, t3_scope=None, t3_default_action=None,
) -> dict:
    """Assemble a complete case dict with all M-condition memory variants.

    Design rules:
    - page_spec always present (page-grounded case)
    - M2 has task-aware stale content for all task types (approach A)
    - M3 NEVER carries the target rule
    - M3 has guaranteed domain-adjacent distractor quota
    - Empty preferences rendered with placeholder text
    - structured_state preserved for mitigation pilots (not markdown-parsing)
    - All randomness via rng
    """
    distractor_texts, distractor_tagged = _sample_m3_distractors(domain, general_traits, rng)

    m0_gold = determine_m0_gold(domain, task_type, t3_scope, decision_space, t3_default_action)

    # structured_state: generation internals for mitigation pilots.
    # - current/stale_preferences: for recency overwrite (M2-mit)
    # - distractor_items: tagged items for relevance filtering (M3-mit)
    structured_state = {
        "current_preferences": {k: (v.get("value", v.get("restriction", str(v)))
                                    if isinstance(v, dict) else str(v))
                                for k, v in preferences_correct.items()},
        "stale_preferences": {k: (v.get("value", v.get("restriction", str(v)))
                                  if isinstance(v, dict) else str(v))
                              for k, v in preferences_stale.items()},
        "rules": rules or [],
        "rules_stale": rules_stale or rules or [],
        "distractor_items": distractor_tagged,
    }

    return {
        "case_id": case_id,
        "domain": domain,
        "task_type": task_type,
        "page_template_id": page_template_id,
        "page_spec": page_spec,
        "rendered_image_path": None,  # filled by screenshot renderer
        "task_prompt": task_prompt,
        "decision_space": decision_space,
        "gold_action": {
            "M0": m0_gold,
            "M1": gold_specific,
            "M2": gold_specific,  # correct answer regardless of stale memory
            "M3": gold_specific,  # correct answer regardless of noise
        },
        "stale_aligned_action": stale_aligned,
        "noisy_aligned_action": noisy_aligned,
        "t3_scope": t3_scope,
        "structured_state": structured_state,
        "memory_none": compile_m0(general_traits),
        "memory_correct": compile_m1(general_traits, preferences_correct, rules),
        "memory_stale": compile_m2(general_traits, preferences_stale, rules_stale or rules),
        "memory_irrelevant": compile_m3(general_traits, distractor_texts, rules=None),
    }


def _sample_non_gold(decision_space: list[str], gold: str,
                     rng: random.Random) -> str | None:
    candidates = [d for d in decision_space if d != gold and d != "ASK_USER"]
    return rng.choice(candidates) if candidates else None


def _pick_stale_alternative(current: str, options: list[str], rng: random.Random) -> str:
    """Pick a plausible old preference that differs from current."""
    alts = [o for o in options if o != current]
    return rng.choice(alts) if alts else current


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------

def generate_all(seed: int = 42, cases_per_slot: int = 3) -> list[dict]:
    """Generate a complete v1 case bank. Deterministic given seed."""
    rng = random.Random(seed)
    cases = []

    booking = ALL_DOMAINS["booking"]
    for i in range(cases_per_slot):
        cases.append(generate_booking_t1(booking, rng, i))
        cases.append(generate_booking_t2(booking, rng, i))

    sharing = ALL_DOMAINS["sharing"]
    for i in range(cases_per_slot):
        cases.extend(generate_sharing_t3_pair(sharing, rng, i))

    return cases


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate T×M personal memory case bank")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cases-per-slot", type=int, default=3)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args()

    cases = generate_all(seed=args.seed, cases_per_slot=args.cases_per_slot)
    bank = {
        "bank_id": "personal_memory_tm_v1",
        "description": "T×M personal memory study case bank (generated).",
        "conditions": ["M0", "M1", "M2", "M3"],
        "task_types": ["T1", "T2", "T3"],
        "cases": cases,
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(bank, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(cases)} cases to {args.output}")
    else:
        print(json.dumps(bank, indent=2, ensure_ascii=False))
