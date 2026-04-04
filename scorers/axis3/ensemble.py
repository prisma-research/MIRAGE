"""
Axis 3 — Ensemble Judge

Runs 3 independent judge calls (extract → verify → aggregate) and takes
majority vote on the final S_hat score.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Literal

from scorers.axis3.aggregator import AggregationResult, SHat, apply_decision_rule
from scorers.axis3.claim_extractor import extract_claims
from scorers.axis3.claim_verifier import verify_claims

logger = logging.getLogger(__name__)

EnsembleScore = float | Literal["indeterminate"]


def run_single_judge(
    query: str,
    response_text: str,
    artifact_repr: str,
    artifact_description: str = "",
    api_key: str | None = None,
) -> AggregationResult:
    """Run one complete extract → verify → aggregate pipeline."""
    claims = extract_claims(query, response_text, artifact_description, api_key=api_key)
    if not claims:
        from scorers.axis3.aggregator import AggregationResult
        return AggregationResult(
            gamma=0.0,
            s_hat="indeterminate",
            n_claims=0,
            n_supported=0,
            n_unsupported=0,
            n_indeterminate=0,
            n_critical=0,
            n_critical_supported=0,
            critical_contradiction=False,
            rationale="no artifact-dependent claims found",
        )

    verify_results = verify_claims(query, claims, artifact_repr, api_key=api_key)
    return apply_decision_rule(verify_results)


def score_axis3(
    query: str,
    response_text: str,
    artifact_repr: str,
    artifact_description: str = "",
    n_judges: int = 3,
    api_key: str | None = None,
) -> EnsembleScore:
    """
    Run n_judges independent pipelines and majority-vote on S_hat.

    Returns a value in {0.0, 0.5, 1.0, "indeterminate"}.
    """
    results = []
    for i in range(n_judges):
        try:
            agg = run_single_judge(
                query, response_text, artifact_repr,
                artifact_description=artifact_description,
                api_key=api_key,
            )
            results.append(agg.s_hat)
            logger.debug("Judge %d: s_hat=%s, γ=%.2f", i + 1, agg.s_hat, agg.gamma)
        except Exception as exc:
            logger.warning("Judge %d failed: %s", i + 1, exc)
            results.append("indeterminate")

    if not results:
        return "indeterminate"

    # Majority vote
    vote = Counter(results)
    winner, count = vote.most_common(1)[0]
    logger.info("Ensemble vote: %s → %s (count %d/%d)", dict(vote), winner, count, len(results))
    return winner


def score_axis3_detailed(
    query: str,
    response_text: str,
    artifact_repr: str,
    artifact_description: str = "",
    n_judges: int = 3,
    api_key: str | None = None,
) -> dict:
    """Return full detail including per-judge results."""
    judge_results = []
    for i in range(n_judges):
        try:
            agg = run_single_judge(
                query, response_text, artifact_repr,
                artifact_description=artifact_description,
                api_key=api_key,
            )
            judge_results.append({
                "judge": i + 1,
                "s_hat": agg.s_hat,
                "gamma": agg.gamma,
                "n_claims": agg.n_claims,
                "n_supported": agg.n_supported,
                "critical_contradiction": agg.critical_contradiction,
                "rationale": agg.rationale,
            })
        except Exception as exc:
            judge_results.append({"judge": i + 1, "error": str(exc), "s_hat": "indeterminate"})

    s_hats = [r.get("s_hat", "indeterminate") for r in judge_results]
    vote = Counter(s_hats)
    final_score = vote.most_common(1)[0][0] if vote else "indeterminate"

    return {
        "final_score": final_score,
        "votes": dict(vote),
        "judges": judge_results,
    }
