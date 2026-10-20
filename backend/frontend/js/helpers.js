const HTML_ESCAPES = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

export const SUPPORTED_UPLOAD_EXTENSIONS = [".pdf", ".docx"];
export const MAX_UPLOAD_BYTES = 20 * 1024 * 1024;

export function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => HTML_ESCAPES[character]);
}

export function formatBytes(bytes) {
  const value = Number(bytes);
  if (!Number.isFinite(value) || value < 0) return "—";
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB"];
  let size = value / 1024;
  let unit = units[0];
  for (let index = 1; index < units.length && size >= 1024; index += 1) {
    size /= 1024;
    unit = units[index];
  }
  return `${size.toFixed(size >= 10 ? 0 : 1)} ${unit}`;
}

export function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown date";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export function relativeTime(value, now = Date.now()) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "unknown";
  const seconds = Math.round((now - date.getTime()) / 1000);
  if (seconds < 45) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} hr ago`;
  const days = Math.round(hours / 24);
  if (days < 30) return `${days} day${days === 1 ? "" : "s"} ago`;
  return formatDate(value);
}

export function clampPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.round(Math.min(1, Math.max(0, number)) * 100);
}

export function statusMeta(status) {
  const normalized = String(status || "").toUpperCase();
  if (normalized === "READY") return { label: "Ready", tone: "ready" };
  if (normalized === "FAILED") return { label: "Failed", tone: "failed" };
  if (normalized === "PENDING" || normalized === "PROCESSING") {
    return { label: "Processing", tone: "pending" };
  }
  return { label: status || "Unknown", tone: "neutral" };
}

export function severityMeta(level) {
  const normalized = String(level || "").toUpperCase();
  if (normalized === "HIGH") return { label: "High risk", tone: "high" };
  if (normalized === "MEDIUM") return { label: "Medium risk", tone: "medium" };
  if (normalized === "LOW") return { label: "Low risk", tone: "low" };
  return { label: level || "Unrated", tone: "neutral" };
}

export function actionLabel(action) {
  const value = String(action || "").replace(/[_-]+/g, " ").trim();
  if (!value) return "Reviewed";
  return value.charAt(0).toUpperCase() + value.slice(1);
}

export function filterContractReviews(reviews, targetIds) {
  const ids = targetIds instanceof Set ? targetIds : new Set(targetIds || []);
  return (reviews || []).filter((event) => ids.has(event.target_id));
}

export function latestReviewByTarget(reviews) {
  const latest = new Map();
  for (const event of reviews || []) {
    const current = latest.get(event.target_id);
    if (!current || new Date(event.created_at) > new Date(current.created_at)) {
      latest.set(event.target_id, event);
    }
  }
  return latest;
}

export function summarizeRisks(findings) {
  const summary = { total: 0, high: 0, medium: 0, low: 0 };
  for (const finding of findings || []) {
    summary.total += 1;
    const level = String(finding.risk_level || "").toUpperCase();
    if (level === "HIGH") summary.high += 1;
    else if (level === "MEDIUM") summary.medium += 1;
    else if (level === "LOW") summary.low += 1;
  }
  return summary;
}

export function decodeJwtPayload(token) {
  if (!token || typeof token !== "string") return null;
  try {
    const part = token.split(".")[1];
    if (!part) return null;
    const normalized = part.replace(/-/g, "+").replace(/_/g, "/");
    const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "=");
    return JSON.parse(globalThis.atob(padded));
  } catch {
    return null;
  }
}

export function roleFromAccessToken(token) {
  return decodeJwtPayload(token)?.role || "user";
}

export function truncateText(value, maxLength = 180) {
  const text = String(value ?? "").replace(/\s+/g, " ").trim();
  if (text.length <= maxLength) return text;
  return `${text.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`;
}

export function findingsForClause(findings, clauseId) {
  return (findings || []).filter((finding) => finding.clause_id === clauseId);
}

export function highlightSegments(text, charStart, charEnd) {
  const source = String(text ?? "");
  const start = Number(charStart);
  const end = Number(charEnd);
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start || end > source.length) {
    return { before: source, match: "", after: "" };
  }
  return {
    before: source.slice(0, start),
    match: source.slice(start, end),
    after: source.slice(end),
  };
}

export function validateUpload(file) {
  if (!file) return "Choose a PDF or DOCX file.";
  const name = String(file.name || "").toLowerCase();
  if (!SUPPORTED_UPLOAD_EXTENSIONS.some((extension) => name.endsWith(extension))) {
    return "Only PDF and DOCX files are supported.";
  }
  if (Number(file.size) > MAX_UPLOAD_BYTES) {
    return "File exceeds the 20 MB upload limit.";
  }
  return "";
}
