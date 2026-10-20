"""Regression tests for candidate-passage clause inference."""
from __future__ import annotations

from app.clauses.inference import predict_candidate_clauses
from ml.preprocess import clean_contract_text


class _RecordingClassifier:
    def __init__(self, label: str = "Limitation of Liability", confidence: float = 0.9) -> None:
        self.label = label
        self.confidence = confidence
        self.inputs: list[str] = []

    def predict(self, texts: list[str]) -> list[str]:
        self.inputs.extend(texts)
        return [self.label for _ in texts]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[1.0 - self.confidence, self.confidence] for _ in texts]


def test_classifies_candidate_passage_with_raw_source_offsets():
    text = "1. LIMITATION OF LIABILITY. Supplier has unlimited liability for all claims."
    classifier = _RecordingClassifier()

    predictions = predict_candidate_clauses(classifier, text)

    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.clause_type == "Limitation of Liability"
    assert prediction.confidence == 0.9
    assert prediction.span_text == text[prediction.char_start : prediction.char_end]
    assert "unlimited liability" in prediction.span_text
    assert classifier.inputs == [clean_contract_text(prediction.span_text)]
    assert classifier.inputs[0] != text


def test_multiple_candidate_occurrences_keep_distinct_offsets():
    first = (
        "Supplier has unlimited liability for all claims arising from this agreement, "
        "including direct damages, indirect damages, and any other losses."
    )
    second = (
        "Customer has unlimited liability for all claims arising under this agreement, "
        "including direct damages, indirect damages, and any other losses."
    )
    text = f"1. LIMITATION OF LIABILITY. {first}\n2. LIMITATION OF LIABILITY. {second}"
    classifier = _RecordingClassifier()

    predictions = predict_candidate_clauses(classifier, text)

    assert len(predictions) == 2
    spans = {(prediction.char_start, prediction.char_end) for prediction in predictions}
    assert len(spans) == 2
    for prediction in predictions:
        assert text[prediction.char_start : prediction.char_end] == prediction.span_text


def test_empty_text_and_no_candidates_return_no_predictions():
    classifier = _RecordingClassifier()

    assert predict_candidate_clauses(classifier, "") == []
    assert predict_candidate_clauses(classifier, "This document contains no clause keywords.") == []
    assert classifier.inputs == []


def test_unsupported_classifier_label_is_dropped():
    classifier = _RecordingClassifier(label="Unseen Clause")

    assert predict_candidate_clauses(classifier, "1. TERMINATION. Either party may terminate.") == []


def test_classifier_receives_cleaned_passage_text():
    text = "1. LIMITATION OF LIABILITY.\n\nSupplier\t has  unlimited liability for all claims."
    classifier = _RecordingClassifier()

    predict_candidate_clauses(classifier, text)

    assert classifier.inputs
    assert all("\t" not in value and "  " not in value for value in classifier.inputs)
