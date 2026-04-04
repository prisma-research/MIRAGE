"""Tests for CitationForce signal detection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mitigations.citation_force.signals import (
    detect_citations,
    detect_vague_references,
    detect_refusal,
    analyze_response_signals,
)


class TestDetectCitations:
    def test_bracket_source(self):
        text = "The revenue was $1.2M. [Source: /workspace/memory/2026-03-13.md#chart_abc]"
        citations = detect_citations(text)
        assert len(citations) == 1
        assert "chart_abc" in citations[0]

    def test_no_citation(self):
        text = "The revenue was $1.2M based on our analysis."
        assert detect_citations(text) == []

    def test_paren_source(self):
        text = "According to the data (source: chart_abc.md), revenue grew 12%."
        citations = detect_citations(text)
        assert len(citations) == 1


class TestDetectVagueReferences:
    def test_last_session(self):
        text = "Based on what we found last session, the trend was upward."
        refs = detect_vague_references(text)
        assert len(refs) > 0

    def test_that_chart(self):
        text = "That chart you showed me indicated a 23% increase."
        refs = detect_vague_references(text)
        assert len(refs) > 0

    def test_no_vague_ref(self):
        text = "According to the Q3 earnings report, revenue was $847M."
        refs = detect_vague_references(text)
        assert len(refs) == 0


class TestDetectRefusal:
    def test_refusal_cannot_access(self):
        text = "I'm unable to retrieve information from previous sessions."
        assert detect_refusal(text) is True

    def test_refusal_no_memory(self):
        text = "I don't have access to files or conversations from previous sessions."
        assert detect_refusal(text) is True

    def test_refusal_fresh_start(self):
        text = "Each conversation starts fresh for me, so I don't have that data."
        assert detect_refusal(text) is True

    def test_no_refusal(self):
        text = "The Q3 revenue was $1.2M as shown in the report."
        assert detect_refusal(text) is False


class TestAnalyzeResponseSignals:
    def test_cited_response(self):
        text = "Revenue was $1.2M. [Source: /workspace/memory/chart_abc.md]"
        result = analyze_response_signals(text)
        assert result["grounding_quality"] == "cited"
        assert result["has_citation"] is True
        assert result["is_refusal"] is False

    def test_vague_response(self):
        text = "From last session, that chart showed revenue at $1.2M."
        result = analyze_response_signals(text)
        assert result["grounding_quality"] == "vague"
        assert result["is_refusal"] is False

    def test_refusal_response(self):
        text = "I'm unable to access previous session data."
        result = analyze_response_signals(text)
        assert result["grounding_quality"] == "refusal"
        assert result["is_refusal"] is True
