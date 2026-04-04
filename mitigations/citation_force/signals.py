"""
CitationForce — Reference Signal Detection

Regex patterns to detect when the agent's response contains a proper citation
vs. a hallucinated or uncited reference.
"""

from __future__ import annotations

import re


# Patterns indicating the agent cited a source path
CITATION_PATTERNS = [
    re.compile(r"\[Source:\s*([^\]]+)\]", re.IGNORECASE),
    re.compile(r"\(source:\s*([^\)]+)\)", re.IGNORECASE),
    re.compile(r"from\s+[`'\"]([^`'\"]+\.(md|jpg|png|pdf))[`'\"]", re.IGNORECASE),
    re.compile(r"artifact[_\-\s]id[:\s]+([a-zA-Z0-9_\-]+)", re.IGNORECASE),
]

# Patterns indicating a vague/hallucinated cross-session reference
VAGUE_REFERENCE_PATTERNS = [
    re.compile(r"\b(last session|previous session|earlier session)\b", re.IGNORECASE),
    re.compile(r"\b(the (chart|screenshot|image|figure|document) (you|I) (showed|mentioned|uploaded|saved|found))\b", re.IGNORECASE),
    re.compile(r"\b(that (chart|screenshot|image|figure))\b", re.IGNORECASE),
    re.compile(r"\bfrom (last|the previous|our last) (time|week|conversation|session)\b", re.IGNORECASE),
    re.compile(r"\b(as (I|we) discussed\b)", re.IGNORECASE),
    re.compile(r"\b(the one (you|I) (showed|mentioned|saved))\b", re.IGNORECASE),
]

# Patterns indicating refusal / inability to retrieve
REFUSAL_PATTERNS = [
    re.compile(r"(I don't have|I do not have)\s+(access|that|the|any|context|record|memory)", re.IGNORECASE),
    re.compile(r"(I'm unable|I am unable|I cannot|can't)(\s+to)?\s+(retrieve|access|find|locate)", re.IGNORECASE),
    re.compile(r"(not (available|found|accessible)|unable to locate)\b", re.IGNORECASE),
    re.compile(r"(don't have (memory|access) to previous sessions?)", re.IGNORECASE),
    re.compile(r"(each (conversation|session) (starts|begins) (fresh|new|anew))", re.IGNORECASE),
    re.compile(r"(no (record|artifact|file) (found|available))", re.IGNORECASE),
    re.compile(r"(fresh|new)\s+(evaluation\s+)?session\b", re.IGNORECASE),
    re.compile(r"without (prior|previous) context\b", re.IGNORECASE),
]


def detect_citations(text: str) -> list[str]:
    """Return list of cited source paths found in the response."""
    citations = []
    for pattern in CITATION_PATTERNS:
        for m in pattern.finditer(text):
            citations.append(m.group(1).strip())
    return citations


def detect_vague_references(text: str) -> list[str]:
    """Return list of vague/hallucinated reference strings found."""
    refs = []
    for pattern in VAGUE_REFERENCE_PATTERNS:
        for m in pattern.finditer(text):
            refs.append(m.group(0))
    return refs


def detect_refusal(text: str) -> bool:
    """
    Return True if the response is a refusal to retrieve the artifact.

    If the response initially hedges ("I don't have context") but then
    provides substantive content (data, numbers, findings), it is NOT
    a refusal — the agent recovered via retrieval.
    """
    has_refusal = any(p.search(text) for p in REFUSAL_PATTERNS)
    if not has_refusal:
        return False

    # Check if the response also contains substantive content after the refusal.
    # Recovery indicators: specific data, "Found it", "Here are", structured content.
    recovery_patterns = [
        re.compile(r"Found it|Here are|Here's what|Based on (the|my) (memory|records|analysis)", re.IGNORECASE),
        re.compile(r"\$[\d,.]+[MBK]?", re.IGNORECASE),  # dollar amounts
        re.compile(r"\d+(\.\d+)?%"),  # percentages
        re.compile(r"##\s+\w+"),  # markdown headers (structured content)
    ]
    for p in recovery_patterns:
        if p.search(text):
            return False

    return True


def analyze_response_signals(response_text: str) -> dict:
    """
    Full signal analysis of a response.

    Returns:
        {
            "has_citation": bool,
            "cited_sources": list[str],
            "vague_references": list[str],
            "is_refusal": bool,
            "grounding_quality": "cited" | "vague" | "refusal" | "none"
        }
    """
    citations = detect_citations(response_text)
    vague = detect_vague_references(response_text)
    refusal = detect_refusal(response_text)

    if refusal:
        quality = "refusal"
    elif citations:
        quality = "cited"
    elif vague:
        quality = "vague"
    else:
        quality = "none"

    return {
        "has_citation": bool(citations),
        "cited_sources": citations,
        "vague_references": vague,
        "is_refusal": refusal,
        "grounding_quality": quality,
    }
