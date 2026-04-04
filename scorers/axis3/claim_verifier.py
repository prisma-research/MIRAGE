"""
Axis 3 — Step B: Claim Verifier

Given (query, claim, artifact_repr), calls LLM API to label each claim
as supported / unsupported / indeterminate.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

from constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL as _LLM_MODEL
from scorers.axis3.claim_extractor import Claim


VerifyLabel = Literal["supported", "unsupported", "indeterminate"]


@dataclass
class VerifyResult:
    claim: Claim
    label: VerifyLabel
    critical: bool
    rationale: str


VERIFICATION_SYSTEM_PROMPT = """\
You are a precise factual verifier for an AI evaluation pipeline.

Your task: given a USER QUERY, an ARTIFACT REPRESENTATION (text, OCR output, or
description of an image), and a single CLAIM from an AI's response, determine
whether the claim is supported by the artifact.

Labels:
  "supported"      : The artifact contains evidence that directly supports the claim.
  "unsupported"    : The artifact contradicts the claim, or the claim cannot be
                     derived from the artifact.
  "indeterminate"  : The artifact is ambiguous or lacks sufficient information to
                     verify the claim.

Also indicate:
  "critical": true  if the claim is a core factual assertion about the artifact's
                    main content (e.g., a specific number, name, or finding).
  "critical": false if the claim is peripheral or contextual.

Output JSON only:
{
  "label": "supported" | "unsupported" | "indeterminate",
  "critical": true | false,
  "rationale": "<1-2 sentence explanation>"
}
"""


def verify_claim(
    query: str,
    claim: Claim,
    artifact_repr: str,
    api_key: str | None = None,
) -> VerifyResult:
    """
    Call LLM API to verify a single claim against the artifact representation.

    Args:
        query: The original user query.
        claim: The Claim to verify.
        artifact_repr: Text representation / OCR of the artifact.
        api_key: API key.

    Returns:
        VerifyResult with label, critical flag, and rationale.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key or LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )

    user_content = f"""USER QUERY: {query}

ARTIFACT REPRESENTATION:
{artifact_repr}

CLAIM TO VERIFY:
{claim.text}
"""

    response = client.chat.completions.create(
        model=_LLM_MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON
    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)
    except (json.JSONDecodeError, IndexError):
        result = {"label": "indeterminate", "critical": False, "rationale": "parse error"}

    label = result.get("label", "indeterminate")
    if label not in ("supported", "unsupported", "indeterminate"):
        label = "indeterminate"

    return VerifyResult(
        claim=claim,
        label=label,
        critical=bool(result.get("critical", False)),
        rationale=result.get("rationale", ""),
    )


def verify_claims(
    query: str,
    claims: list[Claim],
    artifact_repr: str,
    api_key: str | None = None,
) -> list[VerifyResult]:
    """Verify all claims in parallel (sequential for now, easy to parallelize)."""
    results = []
    for claim in claims:
        result = verify_claim(query, claim, artifact_repr, api_key=api_key)
        results.append(result)
    return results
