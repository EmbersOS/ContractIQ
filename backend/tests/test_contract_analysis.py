"""Integration tests for contract clause analysis orchestration."""
from __future__ import annotations

from app.auth.models import User
from app.clauses.models import ClausePrediction
from app.contracts.extractor import ExtractionResult
from app.contracts.models import Contract
from app.contracts.service import process_contract
from app.risks.models import RiskFinding
from app.storage.local import LocalStorageBackend


class _LimitationClassifier:
    def predict(self, texts: list[str]) -> list[str]:
        return ["Limitation of Liability" for _ in texts]

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        return [[0.05, 0.95] for _ in texts]


def _contract(db, storage: LocalStorageBackend, path: str = "contract.pdf") -> Contract:
    user = User(email="analysis@example.com", password_hash="unused")
    db.add(user)
    db.flush()
    contract = Contract(
        owner_id=user.id,
        title="Analysis test",
        original_filename="contract.pdf",
        storage_path=path,
        mime_type="application/pdf",
        file_size_bytes=4,
    )
    db.add(contract)
    db.commit()
    storage.write_file(path, b"%PDF")
    return contract


def test_process_persists_predictions_and_risks(monkeypatch, db, tmp_upload_dir):
    storage = LocalStorageBackend(tmp_upload_dir)
    contract = _contract(db, storage)
    text = "1. LIMITATION OF LIABILITY. Supplier has unlimited liability for all claims."
    monkeypatch.setattr("app.contracts.service.extract_pdf", lambda _: ExtractionResult(text=text))
    monkeypatch.setattr("app.contracts.service._get_classifier", lambda: _LimitationClassifier())

    process_contract(contract.id, storage, db)

    prediction = db.query(ClausePrediction).one()
    finding = db.query(RiskFinding).one()
    assert prediction.span_text == text[prediction.char_start:prediction.char_end]
    assert prediction.confidence == 0.95
    assert finding.rule_id == "unlimited-liability"
    assert finding.clause_id == prediction.id
    assert db.get(Contract, contract.id).status == "READY"


def test_reprocessing_does_not_duplicate_analysis(monkeypatch, db, tmp_upload_dir):
    storage = LocalStorageBackend(tmp_upload_dir)
    contract = _contract(db, storage)
    text = "1. LIMITATION OF LIABILITY. Supplier has unlimited liability for all claims."
    monkeypatch.setattr("app.contracts.service.extract_pdf", lambda _: ExtractionResult(text=text))
    monkeypatch.setattr("app.contracts.service._get_classifier", lambda: _LimitationClassifier())

    process_contract(contract.id, storage, db)
    process_contract(contract.id, storage, db)

    assert db.query(ClausePrediction).count() == 1
    assert db.query(RiskFinding).count() == 1


def test_empty_extraction_is_ready_without_analysis(monkeypatch, db, tmp_upload_dir):
    storage = LocalStorageBackend(tmp_upload_dir)
    contract = _contract(db, storage)
    monkeypatch.setattr("app.contracts.service.extract_pdf", lambda _: ExtractionResult(text="  \n"))

    process_contract(contract.id, storage, db)

    assert db.get(Contract, contract.id).status == "READY"
    assert db.query(ClausePrediction).count() == 0


def test_processing_failure_marks_contract_failed(db, tmp_upload_dir):
    storage = LocalStorageBackend(tmp_upload_dir)
    contract = _contract(db, storage, path="missing.pdf")
    storage.delete_file(contract.storage_path)

    process_contract(contract.id, storage, db)

    stored = db.get(Contract, contract.id)
    assert stored.status == "FAILED"
    assert stored.error_message


def test_unsupported_stored_mime_marks_contract_failed(db, tmp_upload_dir):
    storage = LocalStorageBackend(tmp_upload_dir)
    contract = _contract(db, storage)
    contract.mime_type = "text/plain"
    db.commit()

    process_contract(contract.id, storage, db)

    stored = db.get(Contract, contract.id)
    assert stored.status == "FAILED"
    assert "Unsupported contract MIME type" in stored.error_message
