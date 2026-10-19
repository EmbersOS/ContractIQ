"""Tests for clause candidate scanner."""
from __future__ import annotations

import pytest

from app.contracts.clause_candidates import CandidateClause, find_clause_candidates


SAMPLE_CONTRACT = """
MASTER SERVICE AGREEMENT

1. LIMITATION OF LIABILITY
In no event shall either party be liable for any indirect, incidental, or consequential damages.
The aggregate liability of either party shall not exceed the total fees paid in the prior 12 months.

2. INDEMNIFICATION
Each party shall indemnify, defend, and hold harmless the other party from and against any claims,
damages, losses, and expenses arising out of or resulting from the indemnifying party's acts.

3. GOVERNING LAW
This Agreement shall be governed by the laws of the State of New York without regard to conflict
of law principles. The jurisdiction of New York courts shall apply to any disputes.

4. CONFIDENTIALITY
All proprietary information and trade secrets disclosed under this Agreement are strictly
confidential and subject to non-disclosure obligations.

5. TERMINATION
Either party may terminate this Agreement upon 30 days written notice of termination.
Upon termination, all licenses granted hereunder shall cease immediately.

6. PAYMENT TERMS
Payment terms are net 30 days from the date of invoice. Late payment shall accrue interest
on unpaid amounts at 1.5% per month.

7. DISPUTE RESOLUTION
Any dispute arising under this Agreement shall first be submitted to mediation, and if
unresolved, to binding arbitration under the rules of the AAA.

8. INTELLECTUAL PROPERTY
All intellectual property created under this Agreement, including patents, copyrights, and
trademarks, shall remain the property of the creating party.

9. WARRANTY
THE SERVICES ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND. PROVIDER EXPRESSLY DISCLAIMS
ALL WARRANTIES, INCLUDING WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

10. NON-COMPETE
During the term and for 12 months thereafter, neither party shall compete directly with the
other party. This covenant not to compete is reasonable in scope and duration.
"""


def test_limitation_of_liability_detected():
    candidates = find_clause_candidates(SAMPLE_CONTRACT)
    types = [c.clause_type for c in candidates]
    assert "Limitation of Liability" in types


def test_governing_law_detected():
    text = "This agreement shall be governed by the laws of New York."
    candidates = find_clause_candidates(text)
    types = [c.clause_type for c in candidates]
    assert "Governing Law" in types


def test_span_text_present_in_source():
    """Each CandidateClause.span_text must be present in the source text at reported offsets."""
    candidates = find_clause_candidates(SAMPLE_CONTRACT)
    assert len(candidates) > 0
    for c in candidates:
        extracted = SAMPLE_CONTRACT[c.char_start:c.char_end]
        assert extracted == c.span_text, (
            f"span_text mismatch for {c.clause_type!r}: "
            f"expected {c.span_text!r}, got {extracted!r}"
        )


def test_char_offsets_valid():
    """char_start < char_end and both within text bounds."""
    candidates = find_clause_candidates(SAMPLE_CONTRACT)
    assert len(candidates) > 0
    for c in candidates:
        assert c.char_start >= 0, f"char_start negative: {c.char_start}"
        assert c.char_end > c.char_start, f"char_end not > char_start: {c}"
        assert c.char_end <= len(SAMPLE_CONTRACT), f"char_end out of range: {c}"


def test_indemnification_detected():
    candidates = find_clause_candidates(SAMPLE_CONTRACT)
    types = [c.clause_type for c in candidates]
    assert "Indemnification" in types


def test_confidentiality_detected():
    candidates = find_clause_candidates(SAMPLE_CONTRACT)
    types = [c.clause_type for c in candidates]
    assert "Confidentiality" in types


def test_termination_detected():
    candidates = find_clause_candidates(SAMPLE_CONTRACT)
    types = [c.clause_type for c in candidates]
    assert "Termination" in types


def test_payment_terms_detected():
    candidates = find_clause_candidates(SAMPLE_CONTRACT)
    types = [c.clause_type for c in candidates]
    assert "Payment Terms" in types


def test_dispute_resolution_detected():
    candidates = find_clause_candidates(SAMPLE_CONTRACT)
    types = [c.clause_type for c in candidates]
    assert "Dispute Resolution" in types


def test_intellectual_property_detected():
    candidates = find_clause_candidates(SAMPLE_CONTRACT)
    types = [c.clause_type for c in candidates]
    assert "Intellectual Property" in types


def test_warranty_detected():
    candidates = find_clause_candidates(SAMPLE_CONTRACT)
    types = [c.clause_type for c in candidates]
    assert "Warranty" in types


def test_no_candidates_for_empty_text():
    candidates = find_clause_candidates("")
    assert candidates == []


def test_candidates_sorted_by_char_start():
    candidates = find_clause_candidates(SAMPLE_CONTRACT)
    starts = [c.char_start for c in candidates]
    assert starts == sorted(starts)

