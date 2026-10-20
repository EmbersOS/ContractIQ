"""Tests for deterministic clause risk assessment and persistence."""
from __future__ import annotations

from app.auth.models import User
from app.clauses.models import ClausePrediction
from app.contracts.models import Contract
from app.risks.models import RiskFinding
from app.risks.rules import assess_clause_prediction, load_risk_rules
from app.risks.service import persist_risk_findings


def _prediction(db, *, clause_type: str, span_text: str) -> ClausePrediction:
    user = User(email="risk@example.com", password_hash="unused")
    db.add(user)
    db.flush()
    contract = Contract(
        owner_id=user.id,
        title="Risk test",
        original_filename="risk.pdf",
        storage_path="risk.pdf",
        mime_type="application/pdf",
        file_size_bytes=1,
    )
    db.add(contract)
    db.flush()
    prediction = ClausePrediction(
        contract_id=contract.id,
        clause_type=clause_type,
        span_text=span_text,
        char_start=12,
        char_end=12 + len(span_text),
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


def test_yaml_rules_assess_clause_without_reclassifying(db):
    prediction = _prediction(
        db,
        clause_type="Limitation of Liability",
        span_text="The supplier has unlimited liability for all claims.",
    )

    matches = assess_clause_prediction(prediction)

    assert [match.id for match in matches] == ["unlimited-liability"]
    assert matches[0].risk_level == "HIGH"
    assert load_risk_rules()


def test_persisted_finding_copies_source_span_and_is_idempotent(db):
    prediction = _prediction(
        db,
        clause_type="Termination",
        span_text="Either party may terminate immediately.",
    )

    findings = persist_risk_findings(db, prediction)
    repeated = persist_risk_findings(db, prediction)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.contract_id == prediction.contract_id
    assert finding.clause_id == prediction.id
    assert finding.span_text == prediction.span_text
    assert (finding.char_start, finding.char_end) == (prediction.char_start, prediction.char_end)
    assert repeated == []
    assert db.query(RiskFinding).filter(RiskFinding.clause_id == prediction.id).count() == 1


def test_nonmatching_clause_creates_no_finding(db):
    prediction = _prediction(
        db,
        clause_type="Payment Terms",
        span_text="Invoices are due net 30 days from receipt.",
    )

    assert persist_risk_findings(db, prediction) == []


def test_contract_risks_endpoint_returns_persisted_findings(client, db, auth_headers, sample_pdf_bytes):
    headers = auth_headers("viewer")
    upload = client.post(
        "/contracts",
        files={"file": ("risk.pdf", sample_pdf_bytes, "application/pdf")},
        headers=headers,
    )
    assert upload.status_code == 202
    contract_id = upload.json()["id"]
    prediction = ClausePrediction(
        contract_id=contract_id,
        clause_type="Payment Terms",
        span_text="Invoices are due net 90 days from receipt.",
        char_start=0,
        char_end=45,
    )
    db.add(prediction)
    db.commit()
    persist_risk_findings(db, prediction)

    response = client.get(f"/contracts/{contract_id}/risks", headers=headers)

    assert response.status_code == 200
    assert response.json()[0]["rule_id"] == "extended-payment-term"
    assert response.json()[0]["risk_level"] == "MEDIUM"
    assert response.json()[0]["span_text"] == prediction.span_text
