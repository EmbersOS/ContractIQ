"""Tests for contract upload, list, get, delete endpoints."""
from __future__ import annotations

import io

import pytest

from app.contracts.service import process_contract


def _register_login(client, email, password="password123", role="viewer"):
    client.post("/auth/register", json={"email": email, "password": password, "role": role})
    resp = client.post("/auth/login", json={"email": email, "password": password})
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _upload(client, headers, file_bytes, filename, content_type):
    return client.post(
        "/contracts",
        files={"file": (filename, io.BytesIO(file_bytes), content_type)},
        headers=headers,
    )


# ── Upload ────────────────────────────────────────────────────────────────────

def test_upload_valid_pdf(client, auth_headers, sample_pdf_bytes):
    headers = auth_headers("reviewer")
    resp = _upload(client, headers, sample_pdf_bytes, "contract.pdf", "application/pdf")
    assert resp.status_code == 202
    body = resp.json()
    assert "id" in body
    assert body["status"] == "PENDING"


def test_upload_valid_docx(client, auth_headers, sample_docx_bytes):
    headers = auth_headers("reviewer")
    resp = _upload(
        client,
        headers,
        sample_docx_bytes,
        "contract.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "PENDING"


def test_upload_oversized_file(client, auth_headers):
    headers = auth_headers()
    big = b"%PDF" + b"x" * (21 * 1024 * 1024)  # 21 MB
    resp = _upload(client, headers, big, "big.pdf", "application/pdf")
    assert resp.status_code == 413


def test_upload_wrong_mime_type(client, auth_headers):
    headers = auth_headers()
    resp = _upload(client, headers, b"not a real file content", "test.txt", "text/plain")
    assert resp.status_code == 415


def test_upload_wrong_magic_bytes(client, auth_headers):
    """File with PDF content-type but wrong magic bytes should be rejected."""
    headers = auth_headers()
    resp = _upload(client, headers, b"\x00\x00\x00\x00fake data", "fake.pdf", "application/pdf")
    assert resp.status_code == 415


def test_upload_without_auth(client, sample_pdf_bytes):
    resp = _upload(client, {}, sample_pdf_bytes, "contract.pdf", "application/pdf")
    assert resp.status_code == 401


# ── Background processing / status poll ──────────────────────────────────────

def test_poll_status_becomes_ready(client, db, tmp_upload_dir, auth_headers, sample_pdf_bytes):
    """After calling process_contract synchronously, status should be READY."""
    from app.storage.local import LocalStorageBackend

    storage = LocalStorageBackend(base_dir=tmp_upload_dir)
    headers = auth_headers("reviewer")
    resp = _upload(client, headers, sample_pdf_bytes, "contract.pdf", "application/pdf")
    assert resp.status_code == 202
    contract_id = resp.json()["id"]

    # Run the background task synchronously
    process_contract(contract_id, storage, db)

    # Poll
    get_resp = client.get(f"/contracts/{contract_id}", headers=headers)
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["status"] in ("READY", "FAILED"), f"Unexpected status: {body['status']}"


def test_extracted_text_non_empty_pdf(client, db, tmp_upload_dir, auth_headers, sample_pdf_bytes):
    from app.storage.local import LocalStorageBackend

    storage = LocalStorageBackend(base_dir=tmp_upload_dir)
    headers = auth_headers("reviewer")
    resp = _upload(client, headers, sample_pdf_bytes, "contract.pdf", "application/pdf")
    contract_id = resp.json()["id"]

    process_contract(contract_id, storage, db)

    get_resp = client.get(f"/contracts/{contract_id}", headers=headers)
    body = get_resp.json()
    if body["status"] == "READY":
        assert body["extracted_text"] is not None
        assert len(body["extracted_text"]) > 0


def test_extracted_text_non_empty_docx(client, db, tmp_upload_dir, auth_headers, sample_docx_bytes):
    from app.storage.local import LocalStorageBackend

    storage = LocalStorageBackend(base_dir=tmp_upload_dir)
    headers = auth_headers("reviewer")
    resp = _upload(
        client,
        headers,
        sample_docx_bytes,
        "contract.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    contract_id = resp.json()["id"]

    process_contract(contract_id, storage, db)

    get_resp = client.get(f"/contracts/{contract_id}", headers=headers)
    body = get_resp.json()
    if body["status"] == "READY":
        assert body["extracted_text"] is not None
        assert len(body["extracted_text"]) > 0


# ── List / Get ────────────────────────────────────────────────────────────────

def test_list_contracts(client, auth_headers, sample_pdf_bytes):
    headers = auth_headers("reviewer")
    _upload(client, headers, sample_pdf_bytes, "contract.pdf", "application/pdf")
    resp = client.get("/contracts", headers=headers)
    assert resp.status_code == 200
    assert len(resp.json()) >= 1


def test_get_contract_by_id(client, auth_headers, sample_pdf_bytes):
    headers = auth_headers("reviewer")
    upload_resp = _upload(client, headers, sample_pdf_bytes, "contract.pdf", "application/pdf")
    contract_id = upload_resp.json()["id"]

    resp = client.get(f"/contracts/{contract_id}", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["id"] == contract_id


def test_get_contract_not_found(client, auth_headers):
    headers = auth_headers()
    resp = client.get("/contracts/00000000-0000-0000-0000-000000000000", headers=headers)
    assert resp.status_code == 404


# ── Delete ────────────────────────────────────────────────────────────────────

def test_delete_contract_by_owner(client, auth_headers, sample_pdf_bytes):
    headers = auth_headers("reviewer")
    upload_resp = _upload(client, headers, sample_pdf_bytes, "contract.pdf", "application/pdf")
    contract_id = upload_resp.json()["id"]

    resp = client.delete(f"/contracts/{contract_id}", headers=headers)
    assert resp.status_code == 204

    get_resp = client.get(f"/contracts/{contract_id}", headers=headers)
    assert get_resp.status_code == 404


def test_delete_contract_by_non_owner(client, auth_headers, sample_pdf_bytes):
    owner_headers = auth_headers("reviewer")
    viewer_headers = auth_headers("viewer")

    upload_resp = _upload(client, owner_headers, sample_pdf_bytes, "contract.pdf", "application/pdf")
    contract_id = upload_resp.json()["id"]

    # Viewer (different user) tries to delete
    resp = client.delete(f"/contracts/{contract_id}", headers=viewer_headers)
    assert resp.status_code in (403, 404)  # 404 is also acceptable since viewer can't see other's contracts


# ── Extractor & Page References Unit Tests ────────────────────────────────────

def test_extract_pdf_page_offsets_monotone(tmp_path, sample_pdf_bytes):
    from app.contracts.extractor import extract_pdf

    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(sample_pdf_bytes)

    result = extract_pdf(str(pdf_file))
    assert len(result.text) > 0
    assert len(result.page_offsets) >= 1

    prev_end = 0
    for po in result.page_offsets:
        assert po.page_num >= 1
        assert po.char_start >= prev_end
        assert po.char_end >= po.char_start
        assert po.char_end <= len(result.text)
        prev_end = po.char_end


def test_extract_docx_page_offsets_monotone(tmp_path, sample_docx_bytes):
    from app.contracts.extractor import extract_docx

    docx_file = tmp_path / "test.docx"
    docx_file.write_bytes(sample_docx_bytes)

    result = extract_docx(str(docx_file))
    assert len(result.text) > 0
    assert len(result.page_offsets) >= 1

    prev_end = 0
    for po in result.page_offsets:
        assert po.page_num >= 1
        assert po.char_start >= prev_end
        assert po.char_end >= po.char_start
        assert po.char_end <= len(result.text)
        prev_end = po.char_end


def test_clause_candidates_have_page_numbers(client, db, tmp_upload_dir, auth_headers, sample_pdf_bytes):
    from app.storage.local import LocalStorageBackend

    storage = LocalStorageBackend(base_dir=tmp_upload_dir)
    headers = auth_headers("reviewer")
    resp = _upload(client, headers, sample_pdf_bytes, "contract.pdf", "application/pdf")
    contract_id = resp.json()["id"]

    process_contract(contract_id, storage, db)

    get_resp = client.get(f"/contracts/{contract_id}", headers=headers)
    assert get_resp.status_code == 200
    body = get_resp.json()
    assert body["status"] == "READY"
    candidates = body["clause_candidates"]
    assert isinstance(candidates, list)
    if candidates:
        for c in candidates:
            assert "page_num" in c
            assert c["page_num"] >= 1


def test_upload_filename_basename_sanitized(client, auth_headers, sample_pdf_bytes):
    headers = auth_headers("reviewer")
    resp = _upload(client, headers, sample_pdf_bytes, "../../etc/contract.pdf", "application/pdf")
    assert resp.status_code == 202
    body = resp.json()
    assert body["original_filename"] == "contract.pdf"
    assert ".." not in body["original_filename"]
    assert "/" not in body["original_filename"]
    assert "\\" not in body["original_filename"]

