"""Lightweight mitigation policies for the T × M study.

These are small, transparent transforms applied to MEMORY.md content
before injection. NOT new memory architectures — just pilots to show
that targeted fixes can reduce stale/noisy memory errors.

Mitigation A: Recency-Aware Overwrite (targets M2 → T2)
    Overwrites stale preference values with current ones using
    structured_state, not markdown parsing.

Mitigation B: Tagged Distractor Pruning (targets M3 → T3 out-of-scope)
    Prunes distractor items tagged "general" (not domain-adjacent) from
    the M3 General section. Keeps domain-adjacent distractors (which may
    still tempt over-personalization) and all background traits.
    This is a lightweight distractor-level mitigation, not a full
    task-aware memory exposure system.
"""

from __future__ import annotations
import re

from pipeline.personal_memory.domain_templates import DOMAIN_RELEVANT_TAGS


def apply_recency_overwrite(memory_text: str, current_prefs: dict) -> str:
    """Mitigation A: overwrite stale preference values with current ones.

    Uses structured current_prefs dict (from structured_state), not
    markdown regex extraction.

    Args:
        memory_text: raw MEMORY.md content (potentially stale)
        current_prefs: {"Seat preference": "aisle", ...} from structured_state

    Returns:
        Updated MEMORY.md with current values replacing stale ones.
    """
    lines = memory_text.split("\n")
    updated = []
    matched_keys = set()

    for line in lines:
        replaced = False
        for key, value in current_prefs.items():
            pattern = rf'^(- {re.escape(key)}:\s*)(.+)$'
            m = re.match(pattern, line, re.IGNORECASE)
            if m:
                updated.append(f"- {key}: {value}.")
                matched_keys.add(key)
                replaced = True
                break
        if not replaced:
            updated.append(line)

    # Append any current prefs not found in existing memory
    for key, value in current_prefs.items():
        if key not in matched_keys:
            for i, line in enumerate(updated):
                if "Current Effective Preferences" in line:
                    updated.insert(i + 1, f"- {key}: {value}.")
                    break

    return "\n".join(updated)


def apply_relevance_filter(memory_text: str, domain: str,
                           distractor_items: list[dict]) -> str:
    """Mitigation B: tagged distractor pruning of M3 noisy memory.

    Removes distractor items tagged "general" (not domain-adjacent).
    Keeps items tagged "{domain}_adjacent" and all background traits
    (which are not distractors). This is a lightweight distractor-level
    mitigation — it does NOT filter background traits.

    Args:
        memory_text: raw M3 MEMORY.md content
        domain: current task domain
        distractor_items: list of {"text": ..., "tag": ...} from structured_state

    Returns:
        MEMORY.md with "general"-tagged distractors removed.
    """
    relevant_tags = DOMAIN_RELEVANT_TAGS.get(domain, set())

    # Build set of texts to remove (tagged "general", not domain-adjacent)
    texts_to_remove = set()
    for item in distractor_items:
        if item["tag"] not in relevant_tags:
            texts_to_remove.add(item["text"])

    # Filter lines
    lines = memory_text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- "):
            content = stripped[2:].strip()
            if content in texts_to_remove:
                continue  # prune this irrelevant distractor
        result.append(line)

    return "\n".join(result)
