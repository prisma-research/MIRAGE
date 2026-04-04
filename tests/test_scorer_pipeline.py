"""
Tests for scorer_pipeline.py — screenshot R_context semantics.

Covers the artifact_context_status-driven classification rules:
  1. Screenshot + in_context + no retrieval calls → R_context
  2. Screenshot + in_context + retrieval call in tool_trace → not R_context
  3. Screenshot + out_of_context + keyword match → not R_context (suppressed)
  4. Screenshot + unknown + 2+ keyword hits → heuristic_R_context
  5. Screenshot + unknown + 0 keyword hits → not heuristic_R_context
  6. chart_image + keyword hits in active_context_view → R_context (full classifier)
  7. R_binary for heuristic_R_context == 1
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.trial_log import GroundingTrial, TrialConfig
from harness.scorer_pipeline import score_trial


def _make_trial(
    plant_type: str = "screenshot",
    artifact_context_status: str = "unknown",
    tool_trace: list | None = None,
    active_context_view: list | None = None,
    startup_exposure_log: list | None = None,
    system_prompt: str = "",
    ocr_keywords: list | None = None,
    artifact_id: str = "ss_test001",
) -> GroundingTrial:
    return GroundingTrial(
        trial_id="trial_test_001",
        theta=TrialConfig(
            plant_type=plant_type,
            history_depth=0,
            reference_style="definite",
            seed=42,
        ),
        artifact_id=artifact_id,
        artifact_ocr_keywords=ocr_keywords or ["revenue", "Q3"],
        query="What are the key figures from the artifact?",
        planting_path="explicit_write",
        system_prompt_at_init=system_prompt,
        tool_trace=tool_trace or [],
        active_context_view=active_context_view or [],
        startup_exposure_log=startup_exposure_log or [],
        response_text="The Q3 revenue was $1.2M.",
        artifact_context_status=artifact_context_status,
    )


def _score_no_axis3(trial: GroundingTrial) -> GroundingTrial:
    """Score a trial without running Axis 3 (no API calls)."""
    with patch("harness.scorer_pipeline.score_axis3_detailed") as mock_ax3, \
         patch("harness.scorer_pipeline.analyze_response_signals") as mock_sig:
        mock_sig.return_value = {"is_refusal": False}
        return score_trial(trial, run_axis3=False)


class TestScreenshotInContext:
    def test_in_context_no_retrieval_gives_r_context(self):
        """Screenshot + in_context + no post-query retrieval → R_context."""
        trial = _make_trial(
            artifact_context_status="in_context",
            tool_trace=[],
        )
        with patch("harness.scorer_pipeline.analyze_response_signals") as mock_sig:
            mock_sig.return_value = {"is_refusal": False}
            scored = score_trial(trial, run_axis3=False)
        assert scored.R_path == "R_context"

    def test_in_context_with_memory_search_not_r_context(self):
        """Screenshot + in_context + memory_search in tool_trace → R_path != R_context."""
        trial = _make_trial(
            artifact_context_status="in_context",
            tool_trace=[
                {"kind": "call", "name": "memory_search", "input": {"query": "revenue"}},
                {"kind": "result", "name": "memory_search", "text": "ss_test001: Q3 revenue $1.2M"},
            ],
        )
        with patch("harness.scorer_pipeline.analyze_response_signals") as mock_sig:
            mock_sig.return_value = {"is_refusal": False}
            scored = score_trial(trial, run_axis3=False)
        assert scored.R_path != "R_context"


class TestScreenshotOutOfContext:
    def test_out_of_context_keyword_match_not_r_context(self):
        """Screenshot + out_of_context + keyword match in active_context_view → R_context suppressed."""
        # Even if active_context_view has keyword matches, out_of_context suppresses R_context
        trial = _make_trial(
            artifact_context_status="out_of_context",
            active_context_view=[
                {"text": "The Q3 revenue figure is shown here"},
            ],
            ocr_keywords=["revenue", "Q3"],
        )
        with patch("harness.scorer_pipeline.analyze_response_signals") as mock_sig:
            mock_sig.return_value = {"is_refusal": False}
            scored = score_trial(trial, run_axis3=False)
        assert scored.R_path != "R_context"
        assert scored.R_path != "heuristic_R_context"


class TestScreenshotUnknown:
    def test_unknown_with_keyword_hits_gives_heuristic_r_context(self):
        """Screenshot + unknown + 2+ keyword hits in active_context_view → heuristic_R_context."""
        trial = _make_trial(
            artifact_context_status="unknown",
            active_context_view=[
                {"text": "The revenue figures for Q3 show strong growth"},
            ],
            ocr_keywords=["revenue", "Q3"],
            tool_trace=[],  # no retrieval
        )
        with patch("harness.scorer_pipeline.analyze_response_signals") as mock_sig:
            mock_sig.return_value = {"is_refusal": False}
            scored = score_trial(trial, run_axis3=False)
        assert scored.R_path == "heuristic_R_context"

    def test_unknown_no_keyword_hits_not_heuristic_r_context(self):
        """Screenshot + unknown + 0 keyword hits → R_path not heuristic_R_context."""
        trial = _make_trial(
            artifact_context_status="unknown",
            active_context_view=[
                {"text": "This is some unrelated context about the weather"},
            ],
            ocr_keywords=["revenue", "Q3"],
            tool_trace=[],
        )
        with patch("harness.scorer_pipeline.analyze_response_signals") as mock_sig:
            mock_sig.return_value = {"is_refusal": False}
            scored = score_trial(trial, run_axis3=False)
        assert scored.R_path != "heuristic_R_context"


class TestChartImage:
    def test_chart_image_keyword_hits_gives_r_context(self):
        """chart_image + keyword hits in active_context_view → R_context (full classifier)."""
        trial = _make_trial(
            plant_type="chart_image",
            artifact_id="cqa_test001",
            artifact_context_status="unknown",
            active_context_view=[
                {"text": "Chart cqa_test001 shows Q2 revenue growth trend"},
            ],
            ocr_keywords=["revenue", "Q2"],
            tool_trace=[],
        )
        with patch("harness.scorer_pipeline.analyze_response_signals") as mock_sig:
            mock_sig.return_value = {"is_refusal": False}
            scored = score_trial(trial, run_axis3=False)
        assert scored.R_path == "R_context"


class TestRBinary:
    def test_heuristic_r_context_gives_r_binary_1(self):
        """R_binary for heuristic_R_context == 1."""
        trial = _make_trial(
            artifact_context_status="unknown",
            active_context_view=[
                {"text": "The revenue figures for Q3 show strong growth"},
            ],
            ocr_keywords=["revenue", "Q3"],
            tool_trace=[],
        )
        with patch("harness.scorer_pipeline.analyze_response_signals") as mock_sig:
            mock_sig.return_value = {"is_refusal": False}
            scored = score_trial(trial, run_axis3=False)
        assert scored.R_path == "heuristic_R_context"
        assert scored.R_binary == 1
