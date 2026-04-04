"""Tests for Axis 1 — R-Path classifier."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from scorers.axis1_rpath import classify_r_path, r_binary, extract_tool_calls


ARTIFACT_ID = "screenshot_abc12345"
OCR_KEYWORDS = ["revenue", "Q3", "2025"]


class TestRPathClassifier:
    def test_r_boot_artifact_id_in_early_tool_result(self):
        # Agent read MEMORY.md via exec cat in first turn — artifact_id appears in result text
        early_tool_results = [{"name": "exec", "text": f"# MEMORY\n- [{ARTIFACT_ID}]: chart showing revenue."}]
        result = classify_r_path("clean system prompt", [], ARTIFACT_ID, OCR_KEYWORDS, early_tool_results)
        assert result == "R_boot"

    def test_r_boot_ocr_keyword_in_early_tool_result(self):
        # memory_search in first turn returns content with OCR keyword
        early_tool_results = [{"name": "memory_search", "text": "Q3 revenue figures from last session."}]
        result = classify_r_path("clean system prompt", [], ARTIFACT_ID, ["Q3", "revenue"], early_tool_results)
        assert result == "R_boot"

    def test_r_boot_triggered_by_system_prompt_with_keywords(self):
        # BOOTSTRAP.md content injected into system prompt → OCR keyword match → R_boot
        system_prompt = "You are an assistant. Previous session context: revenue figures from Q3 2025."
        result = classify_r_path(system_prompt, [], ARTIFACT_ID, OCR_KEYWORDS, startup_exposure_log=None)
        assert result == "R_boot"

    def test_r_boot_not_triggered_by_generic_system_prompt(self):
        # System prompt with no artifact-related keywords must not trigger R_boot
        system_prompt = "You are a helpful assistant. Please respond concisely."
        result = classify_r_path(system_prompt, [], ARTIFACT_ID, OCR_KEYWORDS, startup_exposure_log=None)
        assert result == "R_none"

    def test_r_context_ocr_keywords_in_conversation_history(self):
        # S1 baseline: OCR content keywords in pre-query conversation, no retrieval calls → R_context
        conversation_history = [
            {"role": "assistant", "content": "I analyzed the revenue data: Q3 figures show strong growth."},
        ]
        result = classify_r_path("clean system prompt", [], ARTIFACT_ID, ["revenue", "Q3"],
                                 active_context_view=conversation_history)
        assert result == "R_context"

    def test_r_context_not_triggered_by_artifact_id_alone(self):
        # REGRESSION: artifact_id in conversation without OCR keyword evidence must NOT trigger R_context.
        # This was the source of screenshot false positives (planting metadata always contains artifact_id).
        conversation_history = [
            {"role": "assistant", "content": f"I created a chart for you: {ARTIFACT_ID}"},
        ]
        result = classify_r_path("clean system prompt", [], ARTIFACT_ID, OCR_KEYWORDS,
                                 active_context_view=conversation_history)
        assert result != "R_context"

    def test_r_context_ocr_keyword_in_conversation_history(self):
        # S1 baseline: OCR keyword in conversation history, no retrieval calls
        conversation_history = [
            {"role": "user", "content": "Here is the revenue chart"},
            {"role": "assistant", "content": "I can see the Q3 revenue data."},
        ]
        result = classify_r_path("clean system prompt", [], ARTIFACT_ID, ["Q3", "revenue"],
                                 active_context_view=conversation_history)
        assert result == "R_context"

    def test_r_context_not_triggered_when_retrieval_calls_present(self):
        # Artifact in history but retrieval calls exist → not R_context (R_tool takes over)
        conversation_history = [
            {"role": "assistant", "content": f"Chart saved: {ARTIFACT_ID}"},
        ]
        tool_trace = [
            {"kind": "call", "name": "memory_search", "input": {}, "entry_id": "e1"},
        ]
        result = classify_r_path("clean system prompt", tool_trace, ARTIFACT_ID, OCR_KEYWORDS,
                                 active_context_view=conversation_history)
        assert result == "R_tool"

    def test_r_context_not_triggered_when_artifact_not_in_history(self):
        # History present but does not contain artifact → fall through to R_none
        conversation_history = [
            {"role": "user", "content": "Hello, how are you?"},
        ]
        result = classify_r_path("clean system prompt", [], ARTIFACT_ID, OCR_KEYWORDS,
                                 active_context_view=conversation_history)
        assert result == "R_none"

    def test_r_context_takes_priority_over_r_boot(self):
        # R_context (keyword evidence in history) beats R_boot even when startup_exposure also matches
        conversation_history = [
            {"role": "assistant", "content": "The revenue figures for Q3 are visible in the chart."},
        ]
        early_tool_results = [{"name": "exec", "text": f"MEMORY.md: {ARTIFACT_ID}"}]
        result = classify_r_path("clean system prompt", [], ARTIFACT_ID, ["revenue", "Q3"],
                                 early_tool_results, conversation_history)
        assert result == "R_context"

    def test_r_tool_memory_search_call(self):
        tool_trace = [
            {"kind": "call", "name": "memory_search", "input": {"query": "chart"}, "entry_id": "e1"},
            {"kind": "result", "name": "memory_search", "text": "chart data", "entry_id": "e2"},
        ]
        result = classify_r_path("clean system prompt", tool_trace, ARTIFACT_ID, OCR_KEYWORDS)
        assert result == "R_tool"

    def test_r_tool_artifact_recall_call(self):
        tool_trace = [
            {"kind": "call", "name": "artifact_recall", "input": {"artifact_id": ARTIFACT_ID}, "entry_id": "e1"},
        ]
        result = classify_r_path("clean system prompt", tool_trace, ARTIFACT_ID, OCR_KEYWORDS)
        assert result == "R_tool"

    def test_r_none_no_retrieval(self):
        result = classify_r_path("clean system prompt with no artifact info", [], ARTIFACT_ID, OCR_KEYWORDS)
        assert result == "R_none"

    def test_r_none_irrelevant_tool_calls(self):
        tool_trace = [
            {"kind": "call", "name": "web_search", "input": {"query": "weather"}, "entry_id": "e1"},
            {"kind": "call", "name": "exec", "input": {"command": "ls"}, "entry_id": "e2"},
        ]
        result = classify_r_path("clean system prompt", tool_trace, ARTIFACT_ID, OCR_KEYWORDS)
        assert result == "R_none"

    def test_r_boot_takes_priority_over_r_tool(self):
        """R_boot should be returned even if there are retrieval tool calls after the query."""
        early_tool_results = [{"name": "exec", "text": f"MEMORY.md content: {ARTIFACT_ID}"}]
        tool_trace = [
            {"kind": "call", "name": "memory_search", "input": {}, "entry_id": "e1"},
        ]
        result = classify_r_path("clean system prompt", tool_trace, ARTIFACT_ID, OCR_KEYWORDS, early_tool_results)
        assert result == "R_boot"

    def test_r_binary_values(self):
        assert r_binary("R_context") == 1
        assert r_binary("R_boot") == 1
        assert r_binary("R_tool") == 1
        assert r_binary("R_none") == 0

    def test_empty_system_prompt(self):
        result = classify_r_path("", [], ARTIFACT_ID, OCR_KEYWORDS)
        assert result == "R_none"

    def test_keyword_case_insensitive(self):
        # OCR keyword match in early_tool_results should be case-insensitive
        early_tool_results = [{"name": "exec", "text": "previous context: REVENUE data from Q3 2025"}]
        result = classify_r_path("clean system prompt", [], ARTIFACT_ID, ["revenue"], early_tool_results)
        assert result == "R_boot"

    def test_r_context_list_content_format(self):
        # Conversation history turn with content as list of dicts (OpenClaw format) + OCR keyword evidence
        conversation_history = [
            {"role": "assistant", "content": [{"type": "text", "text": "The revenue data for Q3 is analyzed."}]},
        ]
        result = classify_r_path("clean system prompt", [], ARTIFACT_ID, ["revenue", "Q3"],
                                 active_context_view=conversation_history)
        assert result == "R_context"
