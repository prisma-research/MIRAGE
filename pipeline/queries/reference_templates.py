"""
Reference query templates for Session B.

Each reference_style × plant_type combination generates a query that
a user might naturally send when referring back to an artifact from a previous session.
"""

from __future__ import annotations

import random


# Reference templates by style
# {artifact_noun} is replaced with a plant-type-specific noun
TEMPLATES: dict[str, list[str]] = {
    "definite": [
        "Can you remind me of the key findings from that {artifact_noun} you saved?",
        "What were the specific numbers in that {artifact_noun}?",
        "Please summarize the {artifact_noun} you analyzed earlier.",
        "What did the {artifact_noun} show about the main metrics?",
        "Can you pull up the data from that {artifact_noun}?",
    ],
    "demonstrative": [
        "The one you showed me — what were the key figures?",
        "The {artifact_noun} from our conversation — can you recall the details?",
        "That {artifact_noun} you worked on — what were the highlights?",
        "What were the main numbers in the {artifact_noun} we discussed?",
        "The figure from before — what did it say exactly?",
    ],
    "temporal": [
        "From last session, what were the figures in that {artifact_noun}?",
        "What did the {artifact_noun} from our previous session show?",
        "Can you recall the {artifact_noun} you analyzed last time?",
        "The {artifact_noun} we looked at last week — what were the key stats?",
        "Going back to that {artifact_noun} from before, what were the numbers?",
    ],
}

# Artifact nouns by plant type
ARTIFACT_NOUNS: dict[str, str] = {
    "screenshot":  "screenshot",
    "chart_image": "chart",
}

# ── Grounded query frames ───────────────────────────────────────────────────────
# Wrap a dataset-derived content_query inside a reference-style framing prefix.
GROUNDED_FRAMES: dict[str, list[str]] = {
    "definite": [
        "Based on the {noun} you analyzed and saved earlier — {content_query}",
        "Looking at the {noun} you stored in memory — {content_query}",
        "From the {noun} you documented before — {content_query}",
    ],
    "demonstrative": [
        "That {noun} from our conversation — {content_query}",
        "The {noun} we looked at together — {content_query}",
        "Referring to that {noun} you saved — {content_query}",
    ],
    "temporal": [
        "From the {noun} in last session — {content_query}",
        "The {noun} you analyzed last time — {content_query}",
        "Going back to the {noun} from before — {content_query}",
    ],
}


def make_reference_query(
    reference_style: str,
    plant_type: str,
    artifact_id: str | None = None,
    seed: int = 42,
) -> str:
    """
    Generate a reference query for Session B.

    Args:
        reference_style: "definite", "demonstrative", or "temporal"
        plant_type: "screenshot", "chart_image", "ui_state_image"
        artifact_id: Optional — include artifact_id for more specific queries
        seed: Random seed for template selection

    Returns:
        A natural language query string.
    """
    rng = random.Random(seed)
    templates = TEMPLATES.get(reference_style, TEMPLATES["definite"])
    template = rng.choice(templates)
    noun = ARTIFACT_NOUNS.get(plant_type, "artifact")
    query = template.format(artifact_noun=noun)

    # Optionally append the artifact_id for more precise reference
    # (simulates a user who knows the artifact was saved with a specific identifier)
    # Disabled by default — too easy for the agent
    # if artifact_id:
    #     query += f" (I think it was saved as '{artifact_id}')"

    return query


def make_grounded_query(
    reference_style: str,
    plant_type: str,
    content_query: str,
    seed: int = 42,
) -> str:
    """
    Wrap a content-grounded question with reference-style framing.

    Used when PlantSpec.content_query is set (real-dataset planters).
    Falls back to make_reference_query if content_query is empty.

    Args:
        reference_style: "definite", "demonstrative", or "temporal"
        plant_type: "screenshot", "chart_image", "ui_state_image"
        content_query: dataset-derived factual question
        seed: Random seed for frame selection

    Returns:
        A natural language query string.
    """
    if not content_query:
        return make_reference_query(reference_style, plant_type, seed=seed)

    rng = random.Random(seed)
    frames = GROUNDED_FRAMES.get(reference_style, GROUNDED_FRAMES["definite"])
    frame = rng.choice(frames)
    noun = ARTIFACT_NOUNS.get(plant_type, "artifact")
    return frame.format(noun=noun, content_query=content_query)


def get_all_queries(
    plant_type: str,
    artifact_id: str | None = None,
    seeds: list[int] | None = None,
) -> dict[str, list[str]]:
    """
    Generate all reference queries for a given plant_type across all styles.

    Returns dict mapping style → list of queries (one per seed).
    """
    if seeds is None:
        seeds = [42, 123, 456]

    result = {}
    for style in ("definite", "demonstrative", "temporal"):
        result[style] = [
            make_reference_query(style, plant_type, artifact_id, seed=s)
            for s in seeds
        ]
    return result


# Pilot corpus: 10 artifact instances × 3 reference styles = 30 trial configs
def build_pilot_corpus() -> list[dict]:
    """
    Build the initial 30-trial pilot corpus configuration.

    Returns list of trial config dicts with plant_type, reference_style, seed.
    """
    trials = []
    plant_types = ["screenshot", "chart_image"]
    styles = ["definite", "demonstrative", "temporal"]

    # 10 artifact instances: cycle through plant types with varying seeds
    artifact_seeds = [42, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    plant_type_seq = [plant_types[i % len(plant_types)] for i in range(10)]

    trial_n = 0
    for i, (ptype, aseed) in enumerate(zip(plant_type_seq, artifact_seeds)):
        for style in styles:
            trials.append({
                "trial_number": trial_n,
                "plant_type": ptype,
                "artifact_seed": aseed,
                "reference_style": style,
                "query_seed": aseed + ord(style[0]),
                "history_depth": 1,
                "interference_count": 0,
                "seed": aseed,
            })
            trial_n += 1

    return trials
