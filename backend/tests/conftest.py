"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import io
import os
import tempfile
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.dependencies import get_storage
from app.main import create_app
from app.storage.local import LocalStorageBackend

# ---------------------------------------------------------------------------
# Shared classifier artifact
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _test_classifier_artifact(tmp_path_factory):
    """Provide a disposable classifier artifact so tests never depend on local models."""
    from app.config import settings
    from app.contracts import service as contract_service
    from ml.train_classifier import SAMPLE_TRAIN_CORPUS, train_classifier

    artifact_path = tmp_path_factory.mktemp("classifier") / "test_classifier.joblib"
    train_classifier(SAMPLE_TRAIN_CORPUS, split="train", output_path=artifact_path)

    original_path = settings.CLAUSE_MODEL_PATH
    settings.CLAUSE_MODEL_PATH = str(artifact_path)
    contract_service._get_classifier.cache_clear()
    try:
        yield artifact_path
    finally:
        settings.CLAUSE_MODEL_PATH = original_path
        contract_service._get_classifier.cache_clear()


# ---------------------------------------------------------------------------
# In-memory SQLite engine
# ---------------------------------------------------------------------------

TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="function")
def engine():
    """Create a fresh in-memory SQLite engine with all tables for each test."""
    _engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    # Enable WAL mode isn't needed for in-memory; just create tables
    Base.metadata.create_all(bind=_engine)
    yield _engine
    Base.metadata.drop_all(bind=_engine)
    _engine.dispose()


@pytest.fixture(scope="function")
def db(engine) -> Generator[Session, None, None]:
    """Yield a DB session bound to the test engine."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def tmp_upload_dir():
    """Temporary directory for file storage during tests."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture(scope="function")
def client(db, tmp_upload_dir) -> Generator[TestClient, None, None]:
    """TestClient with DB and storage overridden."""
    storage = LocalStorageBackend(base_dir=tmp_upload_dir)

    app = create_app()

    def _override_db():
        try:
            yield db
        finally:
            pass

    def _override_storage():
        return storage

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_storage] = _override_storage

    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _register_and_login(client: TestClient, email: str, password: str, role: str = "viewer") -> dict:
    """Register a user and return auth headers."""
    resp = client.post("/auth/register", json={"email": email, "password": password, "role": role})
    assert resp.status_code == 201, resp.text
    resp2 = client.post("/auth/login", json={"email": email, "password": password})
    assert resp2.status_code == 200, resp2.text
    token = resp2.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def auth_headers(client):
    """Return a callable that registers+logs in a user with a given role."""
    _counter = [0]

    def _make(role: str = "viewer") -> dict:
        _counter[0] += 1
        email = f"user{_counter[0]}@example.com"
        return _register_and_login(client, email, "password123", role)

    return _make


# ---------------------------------------------------------------------------
# Sample file fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_pdf_bytes() -> bytes:
    """Generate a minimal valid PDF using reportlab, or fall back to raw bytes."""
    try:
        from reportlab.pdfgen import canvas  # type: ignore

        buf = io.BytesIO()
        c = canvas.Canvas(buf)
        c.drawString(100, 750, "This Agreement is subject to Limitation of Liability.")
        c.drawString(100, 730, "The governing law of New York shall apply.")
        c.drawString(100, 710, "All parties agree to indemnify and hold harmless.")
        c.drawString(100, 690, "Confidentiality obligations survive termination.")
        c.drawString(100, 670, "Payment terms are net 30 days from invoice date.")
        c.save()
        return buf.getvalue()
    except ImportError:
        # Minimal valid PDF (no reportlab needed)
        return (
            b"%PDF-1.4\n"
            b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
            b"4 0 obj\n<< /Length 44 >>\nstream\n"
            b"BT /F1 12 Tf 100 700 Td (Limitation of Liability) Tj ET\n"
            b"endstream\nendobj\n"
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
            b"xref\n0 6\n0000000000 65535 f\n0000000009 00000 n\n"
            b"0000000058 00000 n\n0000000115 00000 n\n0000000266 00000 n\n"
            b"0000000360 00000 n\n"
            b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n441\n%%EOF\n"
        )


@pytest.fixture(scope="session")
def sample_docx_bytes() -> bytes:
    """Create a DOCX with contract-like text using python-docx."""
    from docx import Document  # type: ignore

    doc = Document()
    doc.add_paragraph("MASTER SERVICE AGREEMENT")
    doc.add_paragraph("This Agreement is subject to Limitation of Liability clauses.")
    doc.add_paragraph("The governing law of New York shall apply to all disputes.")
    doc.add_paragraph("Indemnification: Each party shall indemnify and hold harmless the other.")
    doc.add_paragraph("Confidentiality: All proprietary information is strictly confidential.")
    doc.add_paragraph("Termination: Either party may terminate this agreement with 30 days notice.")
    doc.add_paragraph("Payment Terms: Invoice due within 30 days of receipt.")
    doc.add_paragraph("Dispute Resolution: Disputes shall be resolved by arbitration.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()

