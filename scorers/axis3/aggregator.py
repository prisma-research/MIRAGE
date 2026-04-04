"""
Axis 3 — Aggregator

Computes Γ_λ score from a list of VerifyResults and applies decision rules.

Decision rule (with defaults calibrated to human set later):
  Γ_λ = (λ * frac_supported + (1-λ) * frac_critical_supported)
  where the critical weight λ biases toward critical claims.

  S_hat:
    1.0  if Γ_λ >= τ_sup
    0.5  if τ_unsup < Γ_λ < τ_sup
    0.0  if Γ_λ <= τ_unsup
    "indeterminate" if all claims are indeterminate

  Critical contradiction override:
    If any critical claim is "unsupported", S_hat is forced to 0.0
    unless the majority of critical claims are supported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scorers.axis3.claim_verifier import VerifyResult
from constants import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_LAMBDA,
    DEFAULT_TAU_SUP,
    DEFAULT_TAU_UNSUP,
)


SHat = float | Literal["indeterminate"]


@dataclass
class AggregationResult:
    gamma: float
    s_hat: SHat
    n_claims: int
    n_supported: int
    n_unsupported: int
    n_indeterminate: int
    n_critical: int
    n_critical_supported: int
    critical_contradiction: bool
    rationale: str


def compute_gamma(
    results: list[VerifyResult],
    lam: float = DEFAULT_LAMBDA,
) -> float:
    """
    Compute the Γ_λ blended support score.

    Γ_λ = λ * frac_critical_supported + (1-λ) * frac_all_supported
    (If no critical claims, falls back to frac_all_supported.)
    """
    if not results:
        return 0.0

    n_supported = sum(1 for r in results if r.label == "supported")
    n_total = len(results)
    n_indeterminate = sum(1 for r in results if r.label == "indeterminate")
    n_decisive = n_total - n_indeterminate

    frac_all = n_supported / n_total if n_total > 0 else 0.0

    critical = [r for r in results if r.critical]
    if critical:
        n_crit_sup = sum(1 for r in critical if r.label == "supported")
        frac_crit = n_crit_sup / len(critical)
    else:
        frac_crit = frac_all  # no critical claims → use overall

    return lam * frac_crit + (1 - lam) * frac_all


def apply_decision_rule(
    results: list[VerifyResult],
    lam: float = DEFAULT_LAMBDA,
    tau_sup: float = DEFAULT_TAU_SUP,
    tau_unsup: float = DEFAULT_TAU_UNSUP,
) -> AggregationResult:
    """
    Apply the full decision rule to produce S_hat.
    """
    if not results:
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
            rationale="no claims to evaluate",
        )

    n_sup = sum(1 for r in results if r.label == "supported")
    n_unsup = sum(1 for r in results if r.label == "unsupported")
    n_indet = sum(1 for r in results if r.label == "indeterminate")
    critical = [r for r in results if r.critical]
    n_crit = len(critical)
    n_crit_sup = sum(1 for r in critical if r.label == "supported")
    n_crit_unsup = sum(1 for r in critical if r.label == "unsupported")

    gamma = compute_gamma(results, lam=lam)

    # All indeterminate → indeterminate
    if n_indet == len(results):
        return AggregationResult(
            gamma=gamma,
            s_hat="indeterminate",
            n_claims=len(results),
            n_supported=n_sup,
            n_unsupported=n_unsup,
            n_indeterminate=n_indet,
            n_critical=n_crit,
            n_critical_supported=n_crit_sup,
            critical_contradiction=False,
            rationale="all claims indeterminate",
        )

    # Critical contradiction override
    critical_contradiction = False
    if n_crit > 0 and n_crit_unsup > n_crit_sup:
        critical_contradiction = True
        s_hat: SHat = 0.0
        rationale = f"critical contradiction: {n_crit_unsup}/{n_crit} critical claims unsupported"
    elif gamma >= tau_sup:
        s_hat = 1.0
        rationale = f"Γ={gamma:.2f} ≥ τ_sup={tau_sup}"
    elif gamma <= tau_unsup:
        s_hat = 0.0
        rationale = f"Γ={gamma:.2f} ≤ τ_unsup={tau_unsup}"
    else:
        s_hat = 0.5
        rationale = f"τ_unsup={tau_unsup} < Γ={gamma:.2f} < τ_sup={tau_sup}"

    return AggregationResult(
        gamma=gamma,
        s_hat=s_hat,
        n_claims=len(results),
        n_supported=n_sup,
        n_unsupported=n_unsup,
        n_indeterminate=n_indet,
        n_critical=n_crit,
        n_critical_supported=n_crit_sup,
        critical_contradiction=critical_contradiction,
        rationale=rationale,
    )
