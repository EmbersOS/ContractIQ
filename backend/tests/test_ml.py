"""Tests for the Stage 2 ML data and baseline utilities."""
from __future__ import annotations

import json

import pytest

from ml.dataset import get_split_contracts, load_cuad, make_splits, verify_no_overlap
from ml.evaluate import extract_held_out_samples, load_cuad_evaluation_data
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


def _cuad_eval_fixture() -> dict[str, object]:
    clauses = [
        ("Limitation of Liability", "Aggregate liability shall not exceed the fees paid."),
        ("Indemnification", "Supplier shall indemnify and hold harmless Customer."),
        ("Termination", "Either party may terminate for material breach."),
        ("Governing Law", "This agreement is governed by the laws of New York."),
        ("Confidentiality", "Each party shall keep proprietary information confidential."),
        ("Warranty", "Services are provided without warranties of merchantability."),
    ]
    data = []
    for index, (clause_type, span) in enumerate(clauses):
        context = f"{index + 1}. {clause_type.upper()}. {span}\n{index + 2}. DEFINITIONS. Terms are defined here."
        data.append(
            {
                "title": f"contract-{index}",
                "paragraphs": [
                    {
                        "context": context,
                        "qas": [
                            {
                                "question": f"What is the {clause_type} clause?",
                                "answers": [{"text": span, "answer_start": context.index(span)}],
                            }
                        ],
                    }
                ],
            }
        )
    return {"data": data}


def test_cuad_evaluation_uses_only_held_out_contracts(tmp_path):
    path = tmp_path / "cuad-eval.json"
    path.write_text(json.dumps(_cuad_eval_fixture()), encoding="utf-8")
    contracts = load_cuad(path)
    splits = make_splits(contracts)

    train_examples, samples = load_cuad_evaluation_data(path, "val")

    assert train_examples
    assert samples
    sample_ids = {sample["contract_id"] for sample in samples}
    assert sample_ids <= set(splits["val"])
    assert not sample_ids & set(splits["train"])
    assert not sample_ids & set(splits["test"])
    for sample in samples:
        assert sample["text"][sample["char_start"] : sample["char_end"]] == sample["gold_span"]


def test_held_out_samples_repair_invalid_answer_offsets():
    context = "1. TERMINATION. Either party may terminate with notice."
    samples = extract_held_out_samples(
        [
            {
                "contract_id": "contract-1",
                "context": context,
                "qas": [
                    {
                        "question": "What is the termination clause?",
                        "answers": [{"text": "Either party may terminate with notice.", "answer_start": 9999}],
                    }
                ],
            }
        ]
    )

    assert samples[0]["char_start"] == context.index("Either party may terminate with notice.")


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
