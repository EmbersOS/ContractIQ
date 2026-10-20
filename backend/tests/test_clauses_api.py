"""Tests for detected clause read endpoints and static frontend hosting."""
from __future__ import annotations

import re
from pathlib import Path

from app.auth.models import User
from app.clauses.models import ClausePrediction
from app.contracts.models import Contract


def _contract_with_clauses(db) -> tuple[Contract, list[ClausePrediction]]:
    owner = User(email="clauses-owner@example.com", password_hash="unused")
    db.add(owner)
    db.flush()
    contract = Contract(
        owner_id=owner.id,
        title="Clause API test",
        original_filename="clauses.pdf",
        storage_path="clauses.pdf",
        mime_type="application/pdf",
        file_size_bytes=1,
        status="READY",
    )
    db.add(contract)
    db.flush()
    clauses = [
        ClausePrediction(
            contract_id=contract.id,
            clause_type="Termination",
            span_text="Either party may terminate with notice.",
            char_start=40,
            char_end=80,
            confidence=0.91,
            prediction_source="tfidf_classifier",
            model_version="p1-tfidf-logreg",
        ),
        ClausePrediction(
            contract_id=contract.id,
            clause_type="Governing Law",
            span_text="This agreement is governed by New York law.",
            char_start=5,
            char_end=48,
            confidence=0.97,
            prediction_source="tfidf_classifier",
            model_version="p1-tfidf-logreg",
        ),
    ]
    db.add_all(clauses)
    db.commit()
    return contract, clauses


def test_list_clauses_returns_detected_spans_in_source_order(client, db, auth_headers):
    contract, _ = _contract_with_clauses(db)
    headers = auth_headers("reviewer")

    response = client.get(f"/contracts/{contract.id}/clauses", headers=headers)

    assert response.status_code == 200
    body = response.json()
    assert [clause["clause_type"] for clause in body] == ["Governing Law", "Termination"]
    assert body[0]["span_text"] == "This agreement is governed by New York law."
    assert body[0]["char_start"] == 5
    assert body[0]["confidence"] == 0.97


def test_list_clauses_requires_authentication(client, db):
    contract, _ = _contract_with_clauses(db)

    response = client.get(f"/contracts/{contract.id}/clauses")

    assert response.status_code == 401


def test_list_clauses_rejects_unrelated_viewer(client, db, auth_headers):
    contract, _ = _contract_with_clauses(db)
    headers = auth_headers("viewer")

    response = client.get(f"/contracts/{contract.id}/clauses", headers=headers)

    assert response.status_code == 403


def test_list_clauses_allows_reviewer(client, db, auth_headers):
    contract, _ = _contract_with_clauses(db)
    headers = auth_headers("reviewer")

    response = client.get(f"/contracts/{contract.id}/clauses", headers=headers)

    assert response.status_code == 200
    assert len(response.json()) == 2


def test_list_clauses_unknown_contract_returns_404(client, auth_headers):
    headers = auth_headers("reviewer")

    response = client.get("/contracts/does-not-exist/clauses", headers=headers)

    assert response.status_code == 404


def test_frontend_is_served_at_app_path(client):
    response = client.get("/app/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "ContractIQ" in response.text
    assert "not legal advice" in response.text.lower()


def test_frontend_root_redirects_to_app(client):
    response = client.get("/", follow_redirects=False)

    assert response.status_code in (307, 308)
    assert response.headers["location"] == "/app/"


def test_frontend_assets_are_served(client):
    for path in ("/app/styles.css", "/app/js/app.js", "/app/js/api.js", "/app/js/helpers.js", "/app/favicon.svg"):
        response = client.get(path)
        assert response.status_code == 200, path

def test_frontend_script_references_existing_element_ids():
    frontend = Path(__file__).resolve().parents[1] / "frontend"
    script = (frontend / "js" / "app.js").read_text(encoding="utf-8")
    markup = (frontend / "index.html").read_text(encoding="utf-8")
    referenced = set(re.findall(r'\$\("([a-z0-9-]+)"\)', script)) - {"source-mark"}
    available = set(re.findall(r'id="([a-z0-9-]+)"', markup))

    assert referenced <= available, sorted(referenced - available)
