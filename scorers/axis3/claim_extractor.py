"""
Axis 3 — Step A: Claim Extractor

Given (query, response_text), calls the LLM API to extract atomic claims
that depend on the target artifact.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from constants import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL as _LLM_MODEL


@dataclass
class Claim:
    text: str
    critical: bool = False  # set by verifier


EXTRACTION_SYSTEM_PROMPT = """\
You are a precise claim extractor for an AI evaluation pipeline.

Your task: given a USER QUERY and an AI RESPONSE, extract the minimal set of
atomic factual claims in the response that REQUIRE the target artifact to verify.

Rules:
1. Only include claims that are artifact-dependent (would be different if the artifact
   were different or absent).
2. Split conjunctions — each claim must be a single atomic fact.
3. Ignore filler, hedging, meta-commentary ("I found...", "Based on the document...").
4. Each claim should be independently verifiable against the artifact.
5. If the response contains no artifact-dependent claims (refusal, generic answer), output [].

Output JSON only — an array of strings, one per claim.
Example output: ["The revenue in Q3 was $4.2M", "The chart shows a downward trend in October"]
"""


def extract_claims(
    query: str,
    response_text: str,
    artifact_description: str = "",
    api_key: str | None = None,
) -> list[Claim]:
    """
    Call LLM API to extract artifact-dependent atomic claims from response_text.

    Args:
        query: The user query sent in Session B.
        response_text: The agent's response text.
        artifact_description: Optional description of the artifact type/content for context.
        api_key: API key (falls back to VOLCENGINE_API_KEY env var).

    Returns:
        List of Claim objects.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key or LLM_API_KEY,
        base_url=LLM_BASE_URL,
    )

    user_content = f"""USER QUERY: {query}

AI RESPONSE:
{response_text}
"""
    if artifact_description:
        user_content = f"TARGET ARTIFACT: {artifact_description}\n\n" + user_content

    response = client.chat.completions.create(
        model=_LLM_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON array
    try:
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        claims_raw = json.loads(raw)
        if not isinstance(claims_raw, list):
            claims_raw = []
    except (json.JSONDecodeError, IndexError):
        claims_raw = []

    return [Claim(text=c) for c in claims_raw if isinstance(c, str) and c.strip()]
