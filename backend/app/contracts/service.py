"""Contract upload orchestration and background processing service."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from sqlalchemy.orm import Session

from app.clauses.inference import predict_candidate_clauses
from app.clauses.models import ClausePrediction
from app.config import settings
from app.contracts.clause_candidates import candidates_to_dict, find_clause_candidates
from app.contracts.extractor import extract_docx, extract_pdf
from app.contracts.models import Contract
from app.risks.service import persist_risk_findings
from app.storage.base import StorageBackend
from ml.model_artifact import load_classifier_artifact, resolve_model_artifact_path


@lru_cache(maxsize=1)
def _get_classifier() -> Any:
    """Load the configured classifier artifact once per worker process."""
    artifact_path = resolve_model_artifact_path(settings.CLAUSE_MODEL_PATH)
    return load_classifier_artifact(artifact_path)


def _persist_clause_analysis(
    db: Session, contract: Contract, text: str
) -> list[ClausePrediction]:
    """Classify candidate passages and persist idempotent predictions."""
    predictions: list[ClausePrediction] = []
    for candidate in predict_candidate_clauses(_get_classifier(), text):
        prediction = (
            db.query(ClausePrediction)
            .filter(
                ClausePrediction.contract_id == contract.id,
                ClausePrediction.clause_type == candidate.clause_type,
                ClausePrediction.char_start == candidate.char_start,
                ClausePrediction.char_end == candidate.char_end,
                ClausePrediction.prediction_source == "tfidf_classifier",
            )
            .first()
        )
        if prediction is None:
            prediction = ClausePrediction(
                contract_id=contract.id,
                clause_type=candidate.clause_type,
                span_text=candidate.span_text,
                char_start=candidate.char_start,
                char_end=candidate.char_end,
                confidence=candidate.confidence,
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
            _persist_clause_analysis(db, contract, result.text)

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

