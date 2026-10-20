"""Contract upload orchestration and background processing service."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from sqlalchemy.orm import Session

from app.clauses.models import ClausePrediction
from app.clauses.span_extractor import extract_span_for_clause
from app.contracts.clause_candidates import candidates_to_dict, find_clause_candidates
from app.contracts.extractor import extract_docx, extract_pdf
from app.contracts.models import Contract
from app.risks.service import persist_risk_findings
from app.storage.base import StorageBackend


@lru_cache(maxsize=1)
def _get_classifier() -> Any:
    """Build the deterministic Stage 2 baseline once per worker process."""
    from ml.train_classifier import SAMPLE_TRAIN_CORPUS, train_classifier

    return train_classifier(SAMPLE_TRAIN_CORPUS)


def _prediction_confidence(classifier: Any, span_text: str) -> float:
    if not hasattr(classifier, "predict_proba"):
        return 1.0
    return float(max(classifier.predict_proba([span_text])[0]))


def _persist_clause_analysis(
    db: Session, contract: Contract, text: str, candidate_types: set[str]
) -> list[ClausePrediction]:
    """Classify heading-bounded clause bodies and persist idempotent predictions."""
    if not text.strip() or not candidate_types:
        return []

    classifier = _get_classifier()
    predictions: list[ClausePrediction] = []
    for candidate_type in sorted(candidate_types):
        span_text, char_start, char_end = extract_span_for_clause(text, candidate_type)
        if not span_text:
            continue

        clause_type = str(classifier.predict([span_text])[0])
        prediction = (
            db.query(ClausePrediction)
            .filter(
                ClausePrediction.contract_id == contract.id,
                ClausePrediction.clause_type == clause_type,
                ClausePrediction.char_start == char_start,
                ClausePrediction.char_end == char_end,
                ClausePrediction.prediction_source == "tfidf_classifier",
            )
            .first()
        )
        if prediction is None:
            prediction = ClausePrediction(
                contract_id=contract.id,
                clause_type=clause_type,
                span_text=span_text,
                char_start=char_start,
                char_end=char_end,
                confidence=_prediction_confidence(classifier, span_text),
                prediction_source="tfidf_classifier",
                model_version="p1-tfidf-logreg",
            )
            db.add(prediction)
            db.flush()
        persist_risk_findings(db, prediction)
        predictions.append(prediction)
    return predictions


def process_contract(contract_id: str, storage: StorageBackend, db: Session | None = None) -> None:
    """Background task: extract text, analyze clauses, and update status.

    Runs synchronously in a BackgroundTask worker.
    """
    session_created = False
    if db is None:
        from app.db.session import SessionLocal
        db = SessionLocal()
        session_created = True

    try:
        contract = db.query(Contract).filter(Contract.id == contract_id).first()
        if contract is None:
            return

        try:
            # Read raw bytes from storage
            raw_bytes = storage.read_file(contract.storage_path)

            # Write to a temp file so extractors can open it by path
            suffix = ".pdf" if contract.mime_type == "application/pdf" else ".docx"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(raw_bytes)
                tmp_path = tmp.name

            try:
                if contract.mime_type == "application/pdf":
                    result = extract_pdf(tmp_path)
                elif contract.mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    result = extract_docx(tmp_path)
                else:
                    raise ValueError(f"Unsupported contract MIME type: {contract.mime_type}")
            finally:
                os.unlink(tmp_path)

            # Identify clause candidates with page offsets
            candidates = find_clause_candidates(result.text, result.page_offsets)
            _persist_clause_analysis(
                db,
                contract,
                result.text,
                {candidate.clause_type for candidate in candidates},
            )

            # Update contract in DB
            contract.extracted_text = result.text
            contract.clause_candidates = candidates_to_dict(candidates)
            contract.status = "READY"
            contract.updated_at = datetime.now(tz=timezone.utc)
            db.commit()

        except Exception as exc:  # noqa: BLE001
            contract.status = "FAILED"
            contract.error_message = str(exc)
            contract.updated_at = datetime.now(tz=timezone.utc)
            db.commit()
    finally:
        if session_created:
            db.close()

