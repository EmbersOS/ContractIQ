"""Contract upload orchestration and background processing service."""
from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.contracts.clause_candidates import candidates_to_dict, find_clause_candidates
from app.contracts.extractor import extract_docx, extract_pdf
from app.contracts.models import Contract
from app.storage.base import StorageBackend


def process_contract(contract_id: str, storage: StorageBackend, db: Session | None = None) -> None:
    """Background task: extract text, find clause candidates, update status.

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
                else:
                    result = extract_docx(tmp_path)
            finally:
                os.unlink(tmp_path)

            # Identify clause candidates with page offsets
            candidates = find_clause_candidates(result.text, result.page_offsets)

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

