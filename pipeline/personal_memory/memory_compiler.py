"""Deterministic memory compiler for the T × M personal memory study.

Compiles latent user state into MEMORY.md content for each M-condition.
All output is compact user-state (preferences, rules, facts), never chat summary.

Supports: M0 (minimal), M1 (correct), M2 (stale), M3 (noisy).
"""

from __future__ import annotations
import random
from pipeline.personal_memory.domain_templates import BACKGROUND_TRAITS, DOMAIN_IRRELEVANT


def _render_general_section(traits: list[str]) -> str:
    """Render the General Personal State section."""
    lines = ["## General Personal State"]
    for t in traits:
        lines.append(f"- {t}")
    return "\n".join(lines)


def _render_preferences_section(preferences: dict) -> str:
    """Render Current Effective Preferences section from a state dict.

    If preferences is empty, renders a placeholder so all M-conditions
    have consistent section shape.
    """
    lines = ["## Current Effective Preferences"]
    if not preferences:
        lines.append("- (no specific preferences recorded)")
    else:
        for key, info in preferences.items():
            if isinstance(info, dict):
                value = info.get("value", info.get("restriction", ""))
                reason = info.get("reason", "")
                line = f"- {key}: {value}"
                if reason:
                    line += f" ({reason})"
                line += "."
            else:
                line = f"- {key}: {info}."
            lines.append(line)
    return "\n".join(lines)


def _render_rules_section(rules: list[str]) -> str:
    """Render Rules / Boundaries section."""
    lines = ["## Rules / Boundaries"]
    if rules:
        for r in rules:
            lines.append(f"- {r}")
    else:
        lines.append("- (none currently recorded)")
    return "\n".join(lines)


def compile_m0(general_traits: list[str]) -> str:
    """M0: minimal memory — general background only, no task-relevant state."""
    return "\n\n".join([
        "# Memory",
        _render_general_section(general_traits),
        "## Current Effective Preferences\n- (no specific preferences recorded)",
        "## Rules / Boundaries\n- (none currently recorded)",
    ])


def compile_m1(general_traits: list[str], preferences: dict,
               rules: list[str] | None = None) -> str:
    """M1: correct compact memory — accurate current user state."""
    return "\n\n".join([
        "# Memory",
        _render_general_section(general_traits),
        _render_preferences_section(preferences),
        _render_rules_section(rules or []),
    ])


def compile_m2(general_traits: list[str], stale_preferences: dict,
               rules: list[str] | None = None) -> str:
    """M2: stale memory — key target slot uses outdated value.

    stale_preferences should contain the OLD values for fields that
    have been updated. Auxiliary clues remain correct.
    """
    return "\n\n".join([
        "# Memory",
        _render_general_section(general_traits),
        _render_preferences_section(stale_preferences),
        _render_rules_section(rules or []),
    ])


def compile_m3(general_traits: list[str], distractor_traits: list[str],
               neutral_auxiliary: dict | None = None,
               rules: list[str] | None = None) -> str:
    """M3: noisy memory — no direct target clue, plausible distractors.

    The General section contains the normal background traits PLUS extra
    distractor traits. Duplicates are removed (distractor items that
    already appear in general_traits are skipped).

    Args:
        general_traits: stable background traits (same as M0/M1/M2)
        distractor_traits: extra domain-adjacent but task-irrelevant facts
        neutral_auxiliary: optional non-diagnostic auxiliary preference
        rules: optional rules (should be None for T3 — no target rule in M3)
    """
    # Dedup: keep general first, add only distractors not already present
    seen = set(general_traits)
    unique_distractors = [t for t in distractor_traits if t not in seen]
    combined_general = general_traits + unique_distractors
    sections = ["# Memory", _render_general_section(combined_general)]

    if neutral_auxiliary:
        sections.append(_render_preferences_section(neutral_auxiliary))
    else:
        sections.append("## Current Effective Preferences\n- (no specific preferences recorded)")

    sections.append(_render_rules_section(rules or []))
    return "\n\n".join(sections)


def sample_domain_safe_traits(domain: str, k: int = 3,
                              rng: random.Random | None = None) -> list[str]:
    """Sample traits guaranteed irrelevant to the given domain."""
    r = rng or random.Random()
    pool = BACKGROUND_TRAITS + DOMAIN_IRRELEVANT.get(domain, [])
    return r.sample(pool, k=min(k, len(pool)))
