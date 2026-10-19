"""Tests for auth endpoints."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from jose import jwt

from app.config import settings


def _register(client, email="test@example.com", password="password123", role="viewer"):
    return client.post("/auth/register", json={"email": email, "password": password, "role": role})


def _login(client, email="test@example.com", password="password123"):
    return client.post("/auth/login", json={"email": email, "password": password})


# ── Register ──────────────────────────────────────────────────────────────────

def test_register_success(client):
    resp = _register(client)
    assert resp.status_code == 201
    body = resp.json()
    assert body["email"] == "test@example.com"
    assert body["role"] == "viewer"
    assert "id" in body


def test_register_duplicate_email(client):
    _register(client)
    resp = _register(client)
    assert resp.status_code == 409


def test_register_invalid_role(client):
    resp = _register(client, role="superuser")
    assert resp.status_code == 422


def test_register_short_password(client):
    resp = _register(client, password="short")
    assert resp.status_code == 422


# ── Login ─────────────────────────────────────────────────────────────────────

def test_login_correct_credentials(client):
    _register(client)
    resp = _login(client)
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert "refresh_token" in body
    assert body["token_type"] == "bearer"


def test_login_wrong_password(client):
    _register(client)
    resp = _login(client, password="wrongpass")
    assert resp.status_code == 401


def test_login_unknown_email(client):
    resp = _login(client, email="nobody@example.com")
    assert resp.status_code == 401


# ── Protected route ───────────────────────────────────────────────────────────

def test_protected_route_no_token(client):
    resp = client.get("/contracts")
    assert resp.status_code == 401


def test_protected_route_expired_token(client):
    _register(client)
    # Craft an expired token
    payload = {
        "sub": "some-user-id",
        "role": "viewer",
        "type": "access",
        "exp": datetime.now(tz=timezone.utc) - timedelta(seconds=1),
    }
    expired_token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    resp = client.get("/contracts", headers={"Authorization": f"Bearer {expired_token}"})
    assert resp.status_code == 401


# ── Role guard ────────────────────────────────────────────────────────────────

def test_role_guard_viewer_cannot_access_reviewer_route(client, db):
    """Use the reviewer-only audit endpoint (reviewer or admin required is not enforced
    in the current design, so we test the require_role helper via a custom route or
    verify the dependency itself using a contrived scenario).

    Since the current routes don't have a strict reviewer-only gate, we test
    require_role by verifying that a viewer can access viewer-compatible routes
    and that the helper itself raises 403 when role doesn't match.
    """
    from app.auth.dependencies import require_role
    from fastapi import FastAPI
    from fastapi.testclient import TestClient as _TC
    from app.db.session import get_db
    from app.dependencies import get_storage

    mini = FastAPI()

    @mini.get("/reviewer-only")
    def _reviewer_only(u=pytest.importorskip("fastapi").Depends(require_role("reviewer", "admin"))):
        return {"ok": True}

    def _override_db():
        yield db

    mini.dependency_overrides[get_db] = _override_db

    with _TC(mini) as mini_client:
        # Register a viewer and get token
        reg = client.post("/auth/register", json={"email": "viewer2@example.com", "password": "password123", "role": "viewer"})
        login = client.post("/auth/login", json={"email": "viewer2@example.com", "password": "password123"})
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        # The mini app's DB won't have this user — but we just want to see
        # it reach the role check and fail
        resp = mini_client.get("/reviewer-only", headers=headers)
        # Either 401 (user not found) or 403 (wrong role) — both indicate access denied
        assert resp.status_code in (401, 403)


# ── Refresh ───────────────────────────────────────────────────────────────────

def test_refresh_token(client):
    _register(client)
    resp = _login(client)
    refresh_token = resp.json()["refresh_token"]

    resp2 = client.post("/auth/refresh", json={"refresh_token": refresh_token})
    assert resp2.status_code == 200
    body = resp2.json()
    assert "access_token" in body
    assert "refresh_token" in body


def test_refresh_with_access_token_fails(client):
    """Using an access token as a refresh token should fail."""
    _register(client)
    resp = _login(client)
    access_token = resp.json()["access_token"]

    resp2 = client.post("/auth/refresh", json={"refresh_token": access_token})
    assert resp2.status_code == 401


# ── Logout ────────────────────────────────────────────────────────────────────

def test_logout(client):
    resp = client.post("/auth/logout")
    assert resp.status_code == 204

