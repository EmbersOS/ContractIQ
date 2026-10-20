"""Browser E2E coverage for the main contract review workflow.

The test drives the installed Chrome/Edge binary through the Chrome DevTools
Protocol, so it does not need Playwright or Selenium. The FastAPI app runs in a
background thread with an isolated SQLite database and upload directory.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest
import uvicorn
from docx import Document
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from websockets.sync.client import connect

from app.db.base import Base
from app.db.session import get_db
from app.dependencies import get_storage
from app.main import create_app
from app.storage.local import LocalStorageBackend

pytestmark = pytest.mark.browser

_CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _browser_path() -> str:
    configured = os.environ.get("CONTRACTIQ_BROWSER")
    candidates = [configured, shutil.which("chrome"), shutil.which("msedge"), *_CHROME_CANDIDATES]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    pytest.skip("Chrome or Edge is required for the browser E2E test")


def _e2e_classifier() -> Any:
    """Train a real classifier on the train split so Stage 4 gates yield analysis."""
    cuad_path = Path(__file__).resolve().parents[1] / "data" / "cuad" / "CUAD_v1.json"
    if not cuad_path.is_file():
        pytest.skip("CUAD v1 data is required to build the browser E2E classifier fixture")

    from ml.dataset import get_split_contracts, load_cuad, make_splits
    from ml.train_classifier import extract_training_examples, train_classifier

    contracts = load_cuad(cuad_path)
    splits = make_splits(contracts)
    train_contracts = get_split_contracts(contracts, splits, "train")
    train_examples = extract_training_examples(train_contracts)
    if not train_examples:
        pytest.skip("CUAD train split did not provide classifier examples")
    return train_classifier(train_examples, split="train")


def _wait_http(url: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=0.5)
            if response.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for {url}")


class _CdpClient:
    """Small synchronous Chrome DevTools Protocol client for this workflow."""

    def __init__(self, websocket: Any) -> None:
        self._websocket = websocket
        self._next_id = 0

    def call(self, method: str, **params: Any) -> dict[str, Any]:
        self._next_id += 1
        request_id = self._next_id
        self._websocket.send(json.dumps({"id": request_id, "method": method, "params": params}))
        while True:
            message = json.loads(self._websocket.recv(timeout=15))
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise AssertionError(f"CDP {method} failed: {message['error']}")
            return message.get("result", {})

    def evaluate(self, expression: str) -> Any:
        result = self.call(
            "Runtime.evaluate",
            expression=expression,
            returnByValue=True,
            userGesture=True,
        )
        if "exceptionDetails" in result:
            raise AssertionError(f"Browser evaluation failed: {result['exceptionDetails']}")
        return result.get("result", {}).get("value")

    def wait_for(self, expression: str, *, description: str, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                if self.evaluate(f"Boolean({expression})"):
                    return
            except AssertionError:
                pass
            time.sleep(0.1)
        body = self.evaluate("document.body ? document.body.innerText.slice(0, 2000) : ''")
        raise AssertionError(f"Timed out waiting for {description}; page text: {body!r}")

    def navigate(self, url: str) -> None:
        self.call("Page.navigate", url=url)
        self.wait_for(
            "document.readyState === 'complete'",
            description=f"page load for {url}",
            timeout=15,
        )

    def set_file_input(self, selector: str, path: Path) -> None:
        document = self.call("DOM.getDocument", depth=0)
        node = self.call(
            "DOM.querySelector",
            nodeId=document["root"]["nodeId"],
            selector=selector,
        )
        if not node.get("nodeId"):
            raise AssertionError(f"File input not found: {selector}")
        self.call("DOM.setFileInputFiles", nodeId=node["nodeId"], files=[str(path)])


@pytest.fixture()
def e2e_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, Any]]:
    """Run the real app with an isolated DB/storage and a fresh browser profile."""
    browser = _browser_path()
    classifier = _e2e_classifier()
    from app.contracts import service as contract_service

    # The bundled sample-corpus fallback scores below Stage 4's confidence gate.
    # Use a real CUAD train-only model so the E2E exercises non-empty analysis.
    monkeypatch.setattr(contract_service, "_get_classifier", lambda: classifier)

    database_path = tmp_path / "e2e.db"
    upload_dir = tmp_path / "uploads"
    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"check_same_thread": False},
    )
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    app = create_app()
    Base.metadata.create_all(bind=engine)
    storage = LocalStorageBackend(base_dir=str(upload_dir))

    def override_db() -> Iterator[Any]:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_storage] = lambda: storage

    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    )
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    profile_dir = tmp_path / "browser-profile"
    debug_port = _free_port()
    creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    browser_process: subprocess.Popen[bytes] | None = None
    websocket = None
    try:
        _wait_http(f"{base_url}/health")
        browser_process = subprocess.Popen(
            [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--no-first-run",
                "--no-default-browser-check",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={debug_port}",
                f"--user-data-dir={profile_dir}",
                "about:blank",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
        )

        _wait_http(f"http://127.0.0.1:{debug_port}/json/version")
        targets = httpx.get(f"http://127.0.0.1:{debug_port}/json/list", timeout=5).json()
        page = next(target for target in targets if target.get("type") == "page")
        websocket = connect(
            page["webSocketDebuggerUrl"],
            open_timeout=10,
            close_timeout=5,
            legacy=True,
        )
        cdp = _CdpClient(websocket)
        cdp.call("Page.enable")
        cdp.call("Runtime.enable")
        cdp.call("DOM.enable")
        cdp.navigate(f"{base_url}/app/")
        yield base_url, cdp  # type: ignore[misc]
    finally:
        if websocket is not None:
            websocket.close()
        if browser_process is not None:
            browser_process.terminate()
            try:
                browser_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                browser_process.kill()
                browser_process.wait(timeout=5)
        server.should_exit = True
        server_thread.join(timeout=10)
        engine.dispose()


def _write_contract_fixture(path: Path) -> None:
    document = Document()
    document.add_paragraph("MASTER SERVICE AGREEMENT")
    document.add_paragraph(
        "1. LIMITATION OF LIABILITY. Supplier has unlimited liability for all claims arising from this Agreement."
    )
    document.add_paragraph(
        "2. GOVERNING LAW. This Agreement shall be governed by and construed in accordance with the laws of the State of New York."
    )
    document.add_paragraph(
        "3. TERMINATION. This contract may be terminated immediately upon material breach following a ten day cure period."
    )
    document.add_paragraph(
        "4. INTELLECTUAL PROPERTY. All patents, copyrights, trademarks, and trade secrets developed hereunder belong exclusively to Client."
    )
    document.add_paragraph(
        "5. WARRANTY. Software is provided AS IS with no express or implied warranty of merchantability."
    )
    document.save(path)


def test_browser_main_workflow_and_invalid_upload(e2e_stack: tuple[str, Any], tmp_path: Path) -> None:
    base_url, cdp = e2e_stack
    email = "browser-e2e@example.com"
    password = "password123"

    api = httpx.Client(base_url=base_url, timeout=15)
    register = api.post("/auth/register", json={"email": email, "password": password, "role": "viewer"})
    assert register.status_code == 201, register.text
    login = api.post("/auth/login", json={"email": email, "password": password})
    assert login.status_code == 200, login.text
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    cdp.wait_for("!document.getElementById('auth-view').hidden", description="sign-in form")
    cdp.evaluate(
        f"""
        document.getElementById('auth-email').value = {json.dumps(email)};
        document.getElementById('auth-password').value = {json.dumps(password)};
        document.getElementById('auth-submit').click();
        """
    )
    cdp.wait_for("!document.getElementById('workspace').hidden", description="signed-in workspace")

    contract_path = tmp_path / "e2e_contract.docx"
    _write_contract_fixture(contract_path)
    cdp.set_file_input("#file-input", contract_path)
    cdp.wait_for(
        "!document.getElementById('analysis-grid').hidden",
        description="clause and risk analysis to finish",
        timeout=45,
    )

    contracts = api.get("/contracts", headers=headers).json()
    assert len(contracts) == 1
    contract_id = contracts[0]["id"]
    assert contracts[0]["status"] == "READY"
    clauses = api.get(f"/contracts/{contract_id}/clauses", headers=headers).json()
    risks = api.get(f"/contracts/{contract_id}/risks", headers=headers).json()
    assert clauses, "Fixture should produce at least one clause prediction"
    assert risks, "Fixture should produce at least one deterministic risk finding"

    cdp.wait_for(
        f"document.getElementById('clause-count').textContent.trim() === {json.dumps(str(len(clauses)))}",
        description="clause count from API",
    )
    cdp.wait_for(
        f"document.getElementById('risk-count').textContent.trim() === {json.dumps(str(len(risks)))}",
        description="risk count from API",
    )
    clause_cards = cdp.evaluate("document.querySelectorAll('[data-clause-id]').length")
    risk_cards = cdp.evaluate("document.querySelectorAll('#risk-list .risk-card').length")
    assert clause_cards == len(clauses)
    assert risk_cards == len(risks)

    first_clause = clauses[0]
    assert cdp.evaluate(
        f"document.getElementById('clause-list').innerText.includes({json.dumps(first_clause['clause_type'])})"
    )
    assert cdp.evaluate(
        f"document.getElementById('risk-list').innerText.includes({json.dumps(risks[0]['explanation'])})"
    )
    assert cdp.evaluate(
        f"document.querySelector('#source-mark')?.textContent === {json.dumps(first_clause['span_text'])}"
    )

    notes = "E2E approval notes"
    cdp.evaluate(
        f"""
        const notes = {json.dumps(notes)};
        document.getElementById('review-notes').value = notes;
        document.querySelector('[data-review-target-type="contract"][data-review-action="approve"]').click();
        """
    )
    cdp.wait_for(
        f"document.getElementById('review-timeline').innerText.includes('Approve') && "
        f"document.getElementById('review-timeline').innerText.includes({json.dumps(notes)})",
        description="review event in the timeline",
    )
    audit = api.get(f"/contracts/{contract_id}/audit", headers=headers).json()
    assert any(event["action"] == "approve" and event["notes"] == notes for event in audit)

    invalid_path = tmp_path / "invalid.pdf"
    invalid_path.write_bytes(b"this is not a pdf")
    cdp.set_file_input("#file-input", invalid_path)
    cdp.wait_for(
        "!document.getElementById('upload-error').hidden",
        description="invalid upload error",
    )
    upload_error = cdp.evaluate("document.getElementById('upload-error').textContent")
    assert "Unsupported file type" in upload_error
    assert len(api.get("/contracts", headers=headers).json()) == 1
