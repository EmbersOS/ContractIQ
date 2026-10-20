"""Candidate-passage clause inference shared by processing and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.clauses.span_extractor import extract_candidate_passage
from app.contracts.clause_candidates import find_clause_candidates
from ml.preprocess import clean_contract_text
from ml.train_classifier import TARGET_CLAUSES

SUPPORTED_CLAUSE_TYPES = frozenset(TARGET_CLAUSES)


@dataclass(frozen=True)
class CandidatePrediction:
    clause_type: str
    span_text: str
    char_start: int
    char_end: int
    confidence: float


def predict_candidate_clauses(classifier: Any, source_text: str) -> list[CandidatePrediction]:
    """Classify candidate passages and return predictions with raw-text offsets."""
    if not source_text.strip():
        return []

    passages: dict[tuple[int, int], str] = {}
    for candidate in find_clause_candidates(source_text):
        span_text, char_start, char_end = extract_candidate_passage(
            source_text, candidate.char_start, candidate.char_end
        )
        if span_text:
            passages.setdefault((char_start, char_end), span_text)

    if not passages:
        return []

    # Keyword matches inside a larger passage belong to the same candidate.
    offsets: list[tuple[int, int]] = []
    for key in sorted(passages, key=lambda item: (item[0], -item[1])):
        if any(start <= key[0] and key[1] <= end for start, end in offsets):
            continue
        offsets.append(key)

    texts = [clean_contract_text(passages[key]) for key in offsets]
    labels = classifier.predict(texts)
    probabilities = classifier.predict_proba(texts) if hasattr(classifier, "predict_proba") else None

    predictions: list[CandidatePrediction] = []
    for index, (char_start, char_end) in enumerate(offsets):
        clause_type = str(labels[index])
        if clause_type not in SUPPORTED_CLAUSE_TYPES:
            continue
        confidence = float(max(probabilities[index])) if probabilities is not None else 1.0
        predictions.append(
            CandidatePrediction(
                clause_type=clause_type,
                span_text=passages[(char_start, char_end)],
                char_start=char_start,
                char_end=char_end,
                confidence=confidence,
            )
        )
    return predictions
