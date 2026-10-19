"""Tests for review event endpoints."""
from __future__ import annotations

import uuid


def _register_login(client, email, password="password123", role="viewer"):
    client.post("/auth/register", json={"email": email, "password": password, "role": role})
    resp = client.post("/auth/login", json={"email": email, "password": password})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _get_user_id(client, headers):
    """Get the current user's ID via the /auth/register response stored in the token."""
    from jose import jwt
    from app.config import settings
    token = headers["Authorization"].split(" ")[1]
    payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    return payload["sub"]


# ── Create ────────────────────────────────────────────────────────────────────

def test_create_review_event(client, auth_headers):
    headers = auth_headers("reviewer")
    target_id = str(uuid.uuid4())
    resp = client.post(
        "/reviews",
        json={"target_id": target_id, "target_type": "contract", "action": "approve"},
        headers=headers,
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["target_id"] == target_id
    assert body["action"] == "approve"


def test_create_review_event_with_notes(client, auth_headers):
    headers = auth_headers("reviewer")
    target_id = str(uuid.uuid4())
    resp = client.post(
        "/reviews",
        json={
            "target_id": target_id,
            "target_type": "clause",
            "action": "flag",
            "notes": "This clause needs legal review",
        },
        headers=headers,
    )
    assert resp.status_code == 201
    assert resp.json()["notes"] == "This clause needs legal review"


def test_create_review_event_invalid_target_type(client, auth_headers):
    headers = auth_headers("reviewer")
    resp = client.post(
        "/reviews",
        json={"target_id": str(uuid.uuid4()), "target_type": "INVALID", "action": "approve"},
        headers=headers,
    )
    assert resp.status_code == 422


def test_reviewer_id_recorded_from_jwt(client, auth_headers):
    """reviewer_id in the event should match the authenticated user's ID."""
    headers = auth_headers("reviewer")
    user_id = _get_user_id(client, headers)

    target_id = str(uuid.uuid4())
    resp = client.post(
        "/reviews",
        json={"target_id": target_id, "target_type": "contract", "action": "approve"},
        headers=headers,
    )
    assert resp.status_code == 201
    assert resp.json()["reviewer_id"] == user_id


# ── List ──────────────────────────────────────────────────────────────────────

def test_list_reviews(client, auth_headers):
    headers = auth_headers("reviewer")
    target_id = str(uuid.uuid4())
    client.post(
        "/reviews",
        json={"target_id": target_id, "target_type": "contract", "action": "approve"},
        headers=headers,
    )
    resp = client.get("/reviews", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_list_reviews_filter_by_action(client, auth_headers):
    headers = auth_headers("reviewer")

    # Create two events with different actions
    client.post(
        "/reviews",
        json={"target_id": str(uuid.uuid4()), "target_type": "contract", "action": "approve"},
        headers=headers,
    )
    client.post(
        "/reviews",
        json={"target_id": str(uuid.uuid4()), "target_type": "contract", "action": "reject"},
        headers=headers,
    )

    resp = client.get("/reviews?action=approve", headers=headers)
    assert resp.status_code == 200
    events = resp.json()
    assert all(e["action"] == "approve" for e in events)


# ── Audit trail ───────────────────────────────────────────────────────────────

def test_contract_audit_returns_chronological_order(client, auth_headers):
    headers = auth_headers("reviewer")
    contract_id = str(uuid.uuid4())

    # Create three events
    for action in ["submit", "review", "approve"]:
        client.post(
            "/reviews",
            json={"target_id": contract_id, "target_type": "contract", "action": action},
            headers=headers,
        )

    resp = client.get(f"/contracts/{contract_id}/audit", headers=headers)
    assert resp.status_code == 200
    events = resp.json()
    assert len(events) == 3
    # Should be chronological (created_at ascending)
    times = [e["created_at"] for e in events]
    assert times == sorted(times)
    assert [e["action"] for e in events] == ["submit", "review", "approve"]


def test_audit_only_returns_events_for_contract(client, auth_headers):
    headers = auth_headers("reviewer")
    contract_id = str(uuid.uuid4())
    other_id = str(uuid.uuid4())

    client.post(
        "/reviews",
        json={"target_id": contract_id, "target_type": "contract", "action": "approve"},
        headers=headers,
    )
    client.post(
        "/reviews",
        json={"target_id": other_id, "target_type": "contract", "action": "reject"},
        headers=headers,
    )

    resp = client.get(f"/contracts/{contract_id}/audit", headers=headers)
    events = resp.json()
    assert all(e["target_id"] == contract_id for e in events)


def test_audit_viewer_forbidden_on_others_contract(client, auth_headers, sample_pdf_bytes):
    owner_headers = auth_headers("reviewer")
    viewer_headers = auth_headers("viewer")

    # Owner uploads a contract
    upload_resp = client.post(
        "/contracts",
        files={"file": ("contract.pdf", sample_pdf_bytes, "application/pdf")},
        headers=owner_headers,
    )
    contract_id = upload_resp.json()["id"]

    # Add a review event for this contract
    client.post(
        "/reviews",
        json={"target_id": contract_id, "target_type": "contract", "action": "approve"},
        headers=owner_headers,
    )

    # Different user (viewer) tries to read the audit trail
    resp = client.get(f"/contracts/{contract_id}/audit", headers=viewer_headers)
    assert resp.status_code == 403

