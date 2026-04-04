"""Tests for Axis 3 aggregator."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scorers.axis3.claim_extractor import Claim
from scorers.axis3.claim_verifier import VerifyResult
from scorers.axis3.aggregator import compute_gamma, apply_decision_rule


def make_result(label, critical=False):
    return VerifyResult(
        claim=Claim(text="test claim"),
        label=label,
        critical=critical,
        rationale="test",
    )


class TestComputeGamma:
    def test_all_supported(self):
        results = [make_result("supported")] * 3
        assert compute_gamma(results) == 1.0

    def test_all_unsupported(self):
        results = [make_result("unsupported")] * 3
        assert compute_gamma(results) == 0.0

    def test_mixed(self):
        results = [
            make_result("supported"),
            make_result("supported"),
            make_result("unsupported"),
        ]
        gamma = compute_gamma(results)
        assert 0.0 < gamma < 1.0

    def test_empty(self):
        assert compute_gamma([]) == 0.0

    def test_critical_weight(self):
        # One critical supported, one non-critical unsupported
        results = [
            make_result("supported", critical=True),
            make_result("unsupported", critical=False),
        ]
        gamma_lam07 = compute_gamma(results, lam=0.7)
        gamma_lam03 = compute_gamma(results, lam=0.3)
        # Higher lambda weights critical claims more
        assert gamma_lam07 > gamma_lam03


class TestApplyDecisionRule:
    def test_s_hat_1_high_gamma(self):
        results = [make_result("supported")] * 5
        agg = apply_decision_rule(results)
        assert agg.s_hat == 1.0

    def test_s_hat_0_low_gamma(self):
        results = [make_result("unsupported")] * 5
        agg = apply_decision_rule(results)
        assert agg.s_hat == 0.0

    def test_s_hat_indeterminate_all_indeterminate(self):
        results = [make_result("indeterminate")] * 3
        agg = apply_decision_rule(results)
        assert agg.s_hat == "indeterminate"

    def test_critical_contradiction_override(self):
        # Majority critical unsupported → force 0
        results = [
            make_result("unsupported", critical=True),
            make_result("unsupported", critical=True),
            make_result("supported", critical=True),
            make_result("supported"),
            make_result("supported"),
        ]
        agg = apply_decision_rule(results)
        assert agg.s_hat == 0.0
        assert agg.critical_contradiction is True

    def test_s_hat_05_mid_gamma(self):
        # 2 supported, 2 unsupported → gamma ~0.5 → indeterminate territory
        results = [
            make_result("supported"),
            make_result("supported"),
            make_result("unsupported"),
            make_result("unsupported"),
        ]
        agg = apply_decision_rule(results, tau_sup=0.7, tau_unsup=0.3)
        assert agg.s_hat == 0.5

    def test_empty_results(self):
        agg = apply_decision_rule([])
        assert agg.s_hat == "indeterminate"
        assert agg.n_claims == 0
