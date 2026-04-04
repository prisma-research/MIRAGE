"""Tests for GroundingTrial model serialization."""

import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.trial_log import GroundingTrial, TrialConfig, save_trial, load_trials


def make_sample_trial() -> GroundingTrial:
    return GroundingTrial(
        trial_id="trial_20260313T000000_abc12345",
        theta=TrialConfig(
            plant_type="screenshot",
            history_depth=1,
            reference_style="definite",
            seed=42,
        ),
        artifact_id="screenshot_abc12345",
        artifact_ocr_keywords=["revenue", "Q3"],
        query="What were the key figures in that screenshot?",
        planting_path="pre_compaction_flush",
        system_prompt_at_init="You are an assistant...",
        tool_trace=[
            {"kind": "call", "name": "memory_search", "input": {"query": "screenshot"}},
        ],
        response_text="The Q3 revenue was $1.2M.",
        R_path="R_tool",
        R_binary=1,
        I_strict=1,
        I_set=1,
        I_empty_return=False,
        S_hat=1.0,
        D=0,
        T=1,
        F_strict=1,
        F_relaxed=1,
        outcome="R=R_tool | I=1 | S=1.0 | F_strict=1 | F_relaxed=1",
        compaction_during_session_b=False,
        citation_force_reinjected=None,
        created_at="2026-03-13T00:00:00Z",
        session_a_id="session_a",
        session_b_id="session_b",
    )


class TestGroundingTrial:
    def test_create_trial(self):
        trial = make_sample_trial()
        assert trial.trial_id == "trial_20260313T000000_abc12345"
        assert trial.R_path == "R_tool"
        assert trial.F_strict == 1
        assert trial.F_relaxed == 1

    def test_serialize_deserialize(self):
        trial = make_sample_trial()
        json_str = trial.model_dump_json()
        data = json.loads(json_str)
        assert data["trial_id"] == trial.trial_id
        assert data["R_path"] == "R_tool"

        restored = GroundingTrial.model_validate_json(json_str)
        assert restored.trial_id == trial.trial_id
        assert restored.R_path == trial.R_path

    def test_save_and_load(self):
        trial = make_sample_trial()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_trial(trial, tmpdir)
            assert path.exists()

            loaded = load_trials(tmpdir)
            assert len(loaded) == 1
            assert loaded[0].trial_id == trial.trial_id
            assert loaded[0].F_strict == 1
            assert loaded[0].F_relaxed == 1

    def test_partial_trial_fields_optional(self):
        """Trial with only required fields should work."""
        trial = GroundingTrial(
            trial_id="t1",
            theta=TrialConfig(
                plant_type="chart_image",
                history_depth=0,
                reference_style="temporal",
                seed=0,
            ),
            artifact_id="chart_xyz",
            query="show me the chart",
            planting_path="none",
        )
        assert trial.R_path is None
        assert trial.S_hat is None
        assert trial.F_strict is None
        assert trial.F_relaxed is None
