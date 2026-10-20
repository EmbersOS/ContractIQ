"""Tests for the Stage 2 ML data and baseline utilities."""
from __future__ import annotations

import json

import pytest

from ml.dataset import get_split_contracts, load_cuad, make_splits, verify_no_overlap
from ml.metrics import exact_match_score, span_iou_score, token_f1_score
from ml.preprocess import GuardedSplitContext, LeakageError, chunk_text, clean_contract_text
from ml.train_classifier import train_classifier


def _cuad_fixture() -> dict[str, object]:
    return {
        "data": [
            {
                "title": "master-service-agreement",
                "paragraphs": [
                    {"context": "Termination clause.", "qas": []},
                    {"context": "Payment clause.", "qas": []},
                ],
            },
            {"title": "vendor-agreement", "paragraphs": [{"context": "Warranty clause.", "qas": []}]},
            {"title": "license-agreement", "paragraphs": [{"context": "Governing law clause.", "qas": []}]},
        ]
    }


def test_cuad_paragraphs_from_one_contract_share_a_split(tmp_path):
    path = tmp_path / "cuad.json"
    path.write_text(json.dumps(_cuad_fixture()), encoding="utf-8")

    contracts = load_cuad(path)
    master_contracts = [c for c in contracts if c["title"] == "master-service-agreement"]
    assert len({c["contract_id"] for c in master_contracts}) == 1

    splits = make_splits(contracts, seed=7)
    assert verify_no_overlap(splits)
    split_memberships = [
        split_name
        for split_name in splits
        if {c["contract_id"] for c in master_contracts} & set(splits[split_name])
    ]
    assert len(split_memberships) == 1
    assert len(get_split_contracts(contracts, splits, split_memberships[0])) == 2


def test_chunk_text_preserves_source_offsets():
    text = "alpha beta gamma delta"
    chunks = chunk_text(text, contract_id="contract-1", window_chars=10, stride_chars=5)

    assert chunks
    for chunk in chunks:
        assert text[chunk.char_start:chunk.char_end] == chunk.text
        assert chunk.contract_id == "contract-1"


def test_preprocessing_and_metrics():
    assert clean_contract_text("Page 1\nA\t\tB\n\n\nC") == "\nA B\n\nC"
    assert exact_match_score("Net-30", "net 30") == 1.0
    assert token_f1_score("a b", "a c") == 0.5
    assert span_iou_score((2, 8), (5, 10)) == 0.375


def test_training_guard_rejects_non_training_splits():
    with pytest.raises(LeakageError):
        with GuardedSplitContext("validation"):
            pass


def test_classifier_trains_on_training_split_only():
    pipeline = train_classifier(
        [
            ("Payment is due within thirty days.", "Payment Terms"),
            ("The parties must keep data confidential.", "Confidentiality"),
        ]
    )

    assert pipeline.predict(["Invoices are due in thirty days."])[0] in {
        "Payment Terms",
        "Confidentiality",
    }
