"""Tests for Axis 2 — Artifact Identification scorer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from scorers.axis2_identification import (
    score_identification,
    is_empty_return,
    summarize_identification,
)

ARTIFACT_ID = "chart_abc12345"


class TestIsEmptyReturn:
    def test_empty_text(self):
        result = {"kind": "result", "name": "memory_get", "text": "", "details": {}}
        assert is_empty_return(result) is True

    def test_non_empty_text(self):
        result = {"kind": "result", "name": "memory_get", "text": "some content", "details": {}}
        assert is_empty_return(result) is False

    def test_empty_details_text(self):
        result = {"kind": "result", "name": "memory_get", "text": "", "details": {"text": "", "path": "/some/path"}}
        assert is_empty_return(result) is True

    def test_non_result_kind(self):
        result = {"kind": "call", "name": "memory_get", "text": ""}
        assert is_empty_return(result) is False


class TestScoreIdentification:
    def test_score_1_artifact_id_in_source_paths(self):
        tool_trace = [
            {"kind": "call", "name": "memory_search", "input": {"query": "chart"}, "entry_id": "e1"},
            {
                "kind": "result",
                "name": "memory_search",
                "text": "chart data about revenue",
                "details": {"source_paths": [f"/workspace/memory/2026-03-13.md#{ARTIFACT_ID}"]},
                "entry_id": "e2",
            },
        ]
        assert score_identification(tool_trace, ARTIFACT_ID) == 1

    def test_score_1_artifact_id_in_text(self):
        tool_trace = [
            {
                "kind": "result",
                "name": "memory_search",
                "text": f"Data for {ARTIFACT_ID}: revenue was 1.2M",
                "details": {},
                "entry_id": "e1",
            },
        ]
        assert score_identification(tool_trace, ARTIFACT_ID) == 1

    def test_score_0_non_empty_wrong_artifact(self):
        tool_trace = [
            {
                "kind": "result",
                "name": "memory_search",
                "text": "Found chart_xyz99999: some other chart",
                "details": {"source_paths": ["/workspace/memory/other.md"]},
                "entry_id": "e1",
            },
        ]
        assert score_identification(tool_trace, ARTIFACT_ID) == 0

    def test_score_empty_no_retrieval_calls(self):
        tool_trace = [
            {"kind": "call", "name": "web_search", "input": {}, "entry_id": "e1"},
        ]
        assert score_identification(tool_trace, ARTIFACT_ID) == 0

    def test_score_empty_return(self):
        tool_trace = [
            {
                "kind": "result",
                "name": "memory_get",
                "text": "",
                "details": {"path": "/workspace/memory/2026-03-13.md"},
                "entry_id": "e1",
            },
        ]
        assert score_identification(tool_trace, ARTIFACT_ID) == "empty"

    def test_summarize_identification(self):
        tool_trace = [
            {
                "kind": "result",
                "name": "memory_search",
                "text": f"Found {ARTIFACT_ID}",
                "details": {"source_paths": [f"/path/{ARTIFACT_ID}"]},
                "entry_id": "e1",
            },
        ]
        summary = summarize_identification(tool_trace, ARTIFACT_ID)
        assert summary["I_strict"] == 1
        assert summary["I_empty_return"] is False
