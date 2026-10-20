"""Tests for persisted classifier artifact loading and application parity."""
from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from app.clauses.inference import predict_candidate_clauses
from app.config import settings
from app.contracts import service as contract_service
from ml.model_artifact import (
    BACKEND_DIR,
    ModelArtifactError,
    load_classifier_artifact,
    resolve_model_artifact_path,
)
from ml.train_classifier import SAMPLE_TRAIN_CORPUS, train_classifier


@pytest.fixture()
def trained_artifact(tmp_path: Path):
    artifact_path = tmp_path / "classifier.joblib"
    pipeline = train_classifier(SAMPLE_TRAIN_CORPUS, split="train", output_path=artifact_path)
    return artifact_path, pipeline


def test_loaded_artifact_matches_trained_pipeline(trained_artifact):
    artifact_path, pipeline = trained_artifact
    loaded = load_classifier_artifact(artifact_path)
    texts = [
        "Invoices are due within thirty days of receipt.",
        "The parties shall keep all proprietary information confidential.",
    ]

    assert loaded.predict(texts).tolist() == pipeline.predict(texts).tolist()
    assert loaded.predict_proba(texts).tolist() == pipeline.predict_proba(texts).tolist()


def test_application_path_loads_configured_artifact(monkeypatch, trained_artifact):
    artifact_path, pipeline = trained_artifact
    monkeypatch.setattr(settings, "CLAUSE_MODEL_PATH", str(artifact_path))
    contract_service._get_classifier.cache_clear()
    try:
        loaded = contract_service._get_classifier()
        assert loaded is contract_service._get_classifier()

        text = (
            "1. LIMITATION OF LIABILITY. Supplier has unlimited liability for all claims.\n"
            "2. GOVERNING LAW. This Agreement is governed by the laws of New York."
        )
        assert predict_candidate_clauses(loaded, text) == predict_candidate_clauses(pipeline, text)
    finally:
        contract_service._get_classifier.cache_clear()


def test_application_path_fails_clearly_when_artifact_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "CLAUSE_MODEL_PATH", str(tmp_path / "missing.joblib"))
    contract_service._get_classifier.cache_clear()
    try:
        with pytest.raises(ModelArtifactError) as exc_info:
            contract_service._get_classifier()
        assert "not found" in str(exc_info.value)
    finally:
        contract_service._get_classifier.cache_clear()


def test_relative_artifact_path_resolves_from_backend():
    resolved = resolve_model_artifact_path("ml/results/classifier.joblib")

    assert resolved == (BACKEND_DIR / "ml/results/classifier.joblib").resolve()


def test_missing_artifact_error_is_actionable(tmp_path):
    missing = tmp_path / "missing.joblib"

    with pytest.raises(ModelArtifactError) as exc_info:
        load_classifier_artifact(missing)

    message = str(exc_info.value)
    assert "not found" in message
    assert "ml.train_classifier" in message


def test_corrupt_artifact_error_is_clear(tmp_path):
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"this is not a pickle")

    with pytest.raises(ModelArtifactError) as exc_info:
        load_classifier_artifact(corrupt)

    message = str(exc_info.value)
    assert "could not be loaded" in message
    assert "Regenerate" in message


def test_incompatible_artifact_is_rejected(tmp_path):
    incompatible = tmp_path / "incompatible.joblib"
    joblib.dump({"not": "a classifier"}, incompatible)

    with pytest.raises(ModelArtifactError) as exc_info:
        load_classifier_artifact(incompatible)

    assert "Incompatible" in str(exc_info.value)
