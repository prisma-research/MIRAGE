"""
Tests for eval_summary.py — strict vs inclusive metrics and modality split.

Covers:
  8. Strict vs inclusive metrics differ: a batch with heuristic_R_context trials gives
     R_binary_strict_mean < R_binary_inclusive_mean and F_strict_strict < F_strict_inclusive
  9. Modality split: by_modality["screenshot"] and by_modality["chart_image"] both present
     and have correct trial counts
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from results.eval_summary import compute_summary, is_access_strict, is_access_inclusive


def _make_trial_dict(
    r_path: str,
    plant_type: str = "screenshot",
    scenario: str = "S2",
    i_strict: int = 1,
    s_hat: float = 1.0,
    d: int = 0,
    t_response: int = 1,
    artifact_id: str = "ss_test001",
) -> dict:
    """Build a minimal trial dict for summary testing."""
    return {
        "trial_id": f"trial_{artifact_id}_{scenario}",
        "scenario": scenario,
        "artifact_id": artifact_id,
        "R_path": r_path,
        "R_binary": 0 if r_path == "R_none" else 1,
        "I_strict": i_strict,
        "I_set": i_strict,
        "I_empty_return": False,
        "S_hat": s_hat,
        "D": d,
        "T_response": t_response,
        "T": t_response,
        "F_strict": 1 if (t_response == 1 and r_path != "R_none" and i_strict == 1 and s_hat == 1.0) else 0,
        "F_relaxed": 1 if (t_response == 1 and r_path != "R_none" and i_strict == 1 and s_hat >= 0.5) else 0,
        "planting_path": "explicit_write",
        "theta": {
            "plant_type": plant_type,
            "history_depth": 0,
            "reference_style": "definite",
            "backend": "test_backend",
            "citation_force_condition": "C0",
        },
        "created_at": "2026-03-21T10:00:00Z",
        "outcome": f"R={r_path}|I={i_strict}|S={s_hat}|F=1",
    }


class TestStrictVsInclusiveMetrics:
    def test_heuristic_r_context_lowers_strict_mean(self):
        """
        A batch containing heuristic_R_context trials should give
        R_binary_strict_mean < R_binary_inclusive_mean.
        """
        trials = [
            _make_trial_dict("R_context", plant_type="chart_image"),
            _make_trial_dict("heuristic_R_context", plant_type="screenshot"),
            _make_trial_dict("R_none", plant_type="screenshot"),
        ]
        summary = compute_summary(trials)

        strict = summary["R_binary_strict_mean"]
        inclusive = summary["R_binary_inclusive_mean"]

        assert strict is not None
        assert inclusive is not None
        assert strict < inclusive, (
            f"Expected strict ({strict}) < inclusive ({inclusive}) when heuristic_R_context present"
        )

    def test_f_strict_strict_le_f_strict_inclusive(self):
        """
        F_strict_strict <= F_strict_inclusive when heuristic_R_context trials are present.
        """
        trials = [
            # heuristic_R_context — counts in inclusive but not strict F
            _make_trial_dict("heuristic_R_context", plant_type="screenshot", i_strict=1, s_hat=1.0),
            # R_context — counts in both
            _make_trial_dict("R_context", plant_type="chart_image", i_strict=1, s_hat=1.0),
        ]
        summary = compute_summary(trials)

        assert summary["F_strict_strict"] <= summary["F_strict_inclusive"]
        # Specifically: heuristic_R_context adds to inclusive but not strict
        assert summary["F_strict_inclusive"] > summary["F_strict_strict"]

    def test_both_metrics_present_in_summary(self):
        """Both strict and inclusive metrics should appear in the summary dict."""
        trials = [_make_trial_dict("R_tool", plant_type="screenshot")]
        summary = compute_summary(trials)

        assert "R_binary_strict_mean" in summary
        assert "R_binary_inclusive_mean" in summary
        assert "F_strict_strict" in summary
        assert "F_strict_inclusive" in summary


class TestModalitySplit:
    def test_modality_split_both_present(self):
        """
        by_modality should contain both 'screenshot' and 'chart_image' keys
        when trials from both types exist.
        """
        trials = [
            _make_trial_dict("R_tool", plant_type="screenshot", artifact_id="ss_001"),
            _make_trial_dict("R_tool", plant_type="screenshot", artifact_id="ss_002"),
            _make_trial_dict("R_context", plant_type="chart_image", artifact_id="cqa_001"),
        ]
        summary = compute_summary(trials)

        assert "by_modality" in summary
        assert "screenshot" in summary["by_modality"]
        assert "chart_image" in summary["by_modality"]

    def test_modality_trial_counts(self):
        """by_modality entries should have correct trial counts."""
        trials = [
            _make_trial_dict("R_tool", plant_type="screenshot", artifact_id="ss_001"),
            _make_trial_dict("R_tool", plant_type="screenshot", artifact_id="ss_002"),
            _make_trial_dict("R_context", plant_type="chart_image", artifact_id="cqa_001"),
        ]
        summary = compute_summary(trials)

        assert summary["by_modality"]["screenshot"]["n"] == 2
        assert summary["by_modality"]["chart_image"]["n"] == 1

    def test_modality_stats_have_required_fields(self):
        """Each modality entry should contain R_binary_strict_mean, R_binary_inclusive_mean, etc."""
        trials = [
            _make_trial_dict("R_context", plant_type="screenshot"),
            _make_trial_dict("heuristic_R_context", plant_type="screenshot"),
        ]
        summary = compute_summary(trials)

        ss_stats = summary["by_modality"]["screenshot"]
        assert "R_binary_strict_mean" in ss_stats
        assert "R_binary_inclusive_mean" in ss_stats
        assert "F_strict_strict" in ss_stats
        assert "F_strict_inclusive" in ss_stats
        assert "failure_modes" in ss_stats


class TestIsAccessHelpers:
    def test_is_access_strict(self):
        assert is_access_strict("R_context") is True
        assert is_access_strict("R_boot") is True
        assert is_access_strict("R_tool") is True
        assert is_access_strict("heuristic_R_context") is False
        assert is_access_strict("R_none") is False
        assert is_access_strict(None) is False

    def test_is_access_inclusive(self):
        assert is_access_inclusive("R_context") is True
        assert is_access_inclusive("R_boot") is True
        assert is_access_inclusive("R_tool") is True
        assert is_access_inclusive("heuristic_R_context") is True
        assert is_access_inclusive("R_none") is False
        assert is_access_inclusive(None) is False
