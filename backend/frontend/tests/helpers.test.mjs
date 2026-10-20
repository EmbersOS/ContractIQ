import assert from "node:assert/strict";
import test from "node:test";

import {
  actionLabel,
  clampPercent,
  decodeJwtPayload,
  escapeHtml,
  filterContractReviews,
  findingsForClause,
  formatBytes,
  highlightSegments,
  latestReviewByTarget,
  roleFromAccessToken,
  severityMeta,
  statusMeta,
  summarizeRisks,
  truncateText,
  validateUpload,
} from "../js/helpers.js";

test("escapeHtml neutralizes markup in backend data", () => {
  assert.equal(escapeHtml('<img src=x onerror="alert(1)">'), "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;");
});

test("formatBytes and clampPercent handle normal and invalid input", () => {
  assert.equal(formatBytes(512), "512 B");
  assert.equal(formatBytes(1536), "1.5 KB");
  assert.equal(formatBytes(-1), "—");
  assert.equal(clampPercent(0.934), 93);
  assert.equal(clampPercent(2), 100);
  assert.equal(clampPercent("bad"), 0);
});

test("status and severity labels map to UI tones", () => {
  assert.deepEqual(statusMeta("READY"), { label: "Ready", tone: "ready" });
  assert.deepEqual(statusMeta("PENDING"), { label: "Processing", tone: "pending" });
  assert.deepEqual(statusMeta("FAILED"), { label: "Failed", tone: "failed" });
  assert.deepEqual(severityMeta("HIGH"), { label: "High risk", tone: "high" });
  assert.deepEqual(severityMeta("MEDIUM"), { label: "Medium risk", tone: "medium" });
  assert.equal(actionLabel("request_changes"), "Request changes");
});

test("contract review filters exclude unrelated events", () => {
  const reviews = [
    { target_id: "contract-1", created_at: "2026-01-01T00:00:00Z" },
    { target_id: "clause-1", created_at: "2026-01-02T00:00:00Z" },
    { target_id: "other", created_at: "2026-01-03T00:00:00Z" },
  ];
  const filtered = filterContractReviews(reviews, ["contract-1", "clause-1"]);
  assert.deepEqual(
    filtered.map((event) => event.target_id),
    ["contract-1", "clause-1"],
  );
});

test("latestReviewByTarget picks the newest event per target", () => {
  const latest = latestReviewByTarget([
    { target_id: "clause-1", created_at: "2026-01-01T00:00:00Z", action: "flag" },
    { target_id: "clause-1", created_at: "2026-01-02T00:00:00Z", action: "accept" },
    { target_id: "finding-1", created_at: "2026-01-01T00:00:00Z", action: "flag" },
  ]);
  assert.equal(latest.get("clause-1").action, "accept");
  assert.equal(latest.get("finding-1").action, "flag");
});

test("risk summary and clause linking count persisted findings", () => {
  const findings = [
    { clause_id: "clause-1", risk_level: "HIGH" },
    { clause_id: "clause-1", risk_level: "MEDIUM" },
    { clause_id: "clause-2", risk_level: "LOW" },
  ];
  assert.deepEqual(summarizeRisks(findings), { total: 3, high: 1, medium: 1, low: 1 });
  assert.equal(findingsForClause(findings, "clause-1").length, 2);
});

test("JWT helpers read role without verifying the token client-side", () => {
  const payload = Buffer.from(JSON.stringify({ sub: "user-1", role: "reviewer" })).toString("base64url");
  const token = `header.${payload}.signature`;
  assert.deepEqual(decodeJwtPayload(token), { sub: "user-1", role: "reviewer" });
  assert.equal(roleFromAccessToken(token), "reviewer");
  assert.equal(decodeJwtPayload("not-a-token"), null);
});

test("source highlighting uses raw backend offsets", () => {
  const text = "1. TERMINATION. Either party may terminate.";
  const start = text.indexOf("Either");
  const end = text.indexOf(".", start) + 1;
  assert.deepEqual(highlightSegments(text, start, end), {
    before: "1. TERMINATION. ",
    match: "Either party may terminate.",
    after: "",
  });
  assert.deepEqual(highlightSegments(text, -1, 3), { before: text, match: "", after: "" });
});

test("truncateText collapses whitespace and adds an ellipsis", () => {
  assert.equal(truncateText("a\n\n  b   c", 20), "a b c");
  assert.equal(truncateText("abcdefghij", 5), "abcd…");
});

test("upload validation rejects unsupported and oversized files", () => {
  assert.equal(validateUpload(null), "Choose a PDF or DOCX file.");
  assert.equal(validateUpload({ name: "contract.txt", size: 10 }), "Only PDF and DOCX files are supported.");
  assert.equal(validateUpload({ name: "contract.pdf", size: 21 * 1024 * 1024 }), "File exceeds the 20 MB upload limit.");
  assert.equal(validateUpload({ name: "contract.docx", size: 1024 }), "");
});
