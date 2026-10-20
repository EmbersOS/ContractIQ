"""Tests for deterministic clause-body span extraction."""
from __future__ import annotations

import pytest

from app.clauses.span_extractor import extract_candidate_passage, extract_span_for_clause
from ml.evaluate import VAL_BENCHMARK_SAMPLES


def test_candidate_passage_extends_short_heading_and_preserves_offsets():
    text = "1. TERMINATION. Either party may terminate with written notice."

    span, start, end = extract_candidate_passage(text, text.index("TERMINATION"), text.index("TERMINATION") + len("TERMINATION"))

    assert "Either party may terminate" in span
    assert span == text[start:end]


def test_candidate_passage_rejects_invalid_offsets():
    text = "1. TERMINATION. Body."

    assert extract_candidate_passage(text, -1, 2) == ("", 0, 0)
    assert extract_candidate_passage(text, 4, 4) == ("", 0, 0)
    assert extract_candidate_passage(text, 4, len(text) + 1) == ("", 0, 0)


@pytest.mark.parametrize(
    ("text", "clause_type"),
    [("", "Termination"), ("1. TERMINATION. Body", "Unknown Clause"), ("No headings here.", "Termination")],
)
def test_empty_unknown_and_missing_matches_return_sentinel(text: str, clause_type: str):
    assert extract_span_for_clause(text, clause_type) == ("", 0, 0)


def test_numbered_mixed_case_heading_returns_body_with_raw_offsets():
    text = "1. tErMiNaTiOn: Either party may terminate."
    span, start, end = extract_span_for_clause(text, "Termination")

    assert span == "Either party may terminate."
    assert text[start:end] == span


def test_unicode_and_whitespace_do_not_shift_offsets():
    text = "1. CONFIDENTIALITY.\n\tRéservé information remains confidential.  \n"
    span, start, end = extract_span_for_clause(text, "Confidentiality")

    assert span == "Réservé information remains confidential."
    assert text[start:end] == span


def test_multiple_occurrences_uses_first_structured_section():
    text = "1. TERMINATION. First body.\n2. TERMINATION. Second body."
    span, start, end = extract_span_for_clause(text, "Termination")

    assert span == "First body."
    assert text[start:end] == span


def test_adjacent_sections_are_not_swallowed():
    text = "1. PAYMENT TERMS. Invoices are due.\n2. DEFINITIONS. Terms are defined here."
    span, start, end = extract_span_for_clause(text, "Payment Terms")

    assert span == "Invoices are due."
    assert text[start:end] == span


def test_end_of_document_clause_extends_to_end():
    text = "7) DISPUTE RESOLUTION - Disputes shall be arbitrated."
    span, start, end = extract_span_for_clause(text, "Dispute Resolution")

    assert span == "Disputes shall be arbitrated."
    assert end == len(text)
    assert text[start:end] == span


def test_prose_mention_is_not_treated_as_a_heading():
    text = "The termination provisions appear in a later section."
    assert extract_span_for_clause(text, "Termination") == ("", 0, 0)


@pytest.mark.parametrize(
    ("text", "clause_type"),
    [("PAYMENT. Invoices are due.", "Payment Terms"), ("DISPUTE. Arbitration applies.", "Dispute Resolution")],
)
def test_abbreviated_aliases_require_numeric_section_prefix(text: str, clause_type: str):
    assert extract_span_for_clause(text, clause_type) == ("", 0, 0)


@pytest.mark.parametrize("sample", VAL_BENCHMARK_SAMPLES)
def test_evaluation_samples_extract_body_only_span(sample: dict[str, object]):
    text = str(sample["text"])
    span, start, end = extract_span_for_clause(text, str(sample["gold_type"]))

    assert span == sample["gold_span"]
    assert (start, end) == (sample["char_start"], sample["char_end"])
    assert text[start:end] == span
