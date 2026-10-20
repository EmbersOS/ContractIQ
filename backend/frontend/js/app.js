import { ApiError, createApiClient, createTokenStore } from "./api.js";
import {
  actionLabel,
  clampPercent,
  escapeHtml,
  filterContractReviews,
  findingsForClause,
  formatBytes,
  formatDate,
  highlightSegments,
  latestReviewByTarget,
  relativeTime,
  roleFromAccessToken,
  severityMeta,
  statusMeta,
  summarizeRisks,
  truncateText,
  validateUpload,
} from "./helpers.js";

const api = createApiClient({ tokenStore: createTokenStore(globalThis.localStorage) });
const $ = (id) => document.getElementById(id);

const state = {
  authMode: "login",
  contracts: [],
  selectedId: null,
  contract: null,
  clauses: [],
  risks: [],
  reviews: [],
  selectedClauseId: null,
  clauseFilter: "all",
  detailLoading: false,
  analysisError: "",
  pollTimer: null,
  pollStartedAt: 0,
  polling: false,
  uploading: false,
  reviewingKey: "",
  email: "",
};

function setText(id, value) {
  const node = $(id);
  if (node) node.textContent = value;
}

function showError(message) {
  const banner = $("global-banner");
  banner.textContent = message;
  banner.hidden = !message;
}

function showToast(message, tone = "info") {
  const toast = document.createElement("div");
  toast.className = `toast${tone === "error" ? " error" : ""}`;
  toast.textContent = message;
  $("toast-region").append(toast);
  window.setTimeout(() => toast.remove(), 4200);
}

function errorMessage(error, fallback) {
  if (error instanceof ApiError && error.detail) return error.detail;
  if (error instanceof Error && error.message) return error.message;
  return fallback;
}

function setAuthMode(mode) {
  state.authMode = mode;
  const isLogin = mode === "login";
  $("login-tab").classList.toggle("is-active", isLogin);
  $("register-tab").classList.toggle("is-active", !isLogin);
  $("login-tab").setAttribute("aria-selected", String(isLogin));
  $("register-tab").setAttribute("aria-selected", String(!isLogin));
  setText("auth-form-title", isLogin ? "Welcome back" : "Create your account");
  setText(
    "auth-form-copy",
    isLogin ? "Review uploaded contracts and supporting analysis." : "Accounts are created with viewer access by default.",
  );
  setText("auth-submit", isLogin ? "Sign in" : "Create account");
  $("auth-password").autocomplete = isLogin ? "current-password" : "new-password";
  $("auth-error").hidden = true;
}

function showAuth() {
  stopPolling();
  $("workspace").hidden = true;
  $("auth-view").hidden = false;
  $("auth-password").value = "";
}

function showWorkspace() {
  $("auth-view").hidden = true;
  $("workspace").hidden = false;
}

async function handleAuthSubmit(event) {
  event.preventDefault();
  const email = $("auth-email").value.trim();
  const password = $("auth-password").value;
  const errorNode = $("auth-error");
  errorNode.hidden = true;

  if (!email || !password) {
    errorNode.textContent = "Enter both email and password.";
    errorNode.hidden = false;
    return;
  }
  if (state.authMode === "register" && password.length < 8) {
    errorNode.textContent = "Password must be at least 8 characters.";
    errorNode.hidden = false;
    return;
  }

  const submit = $("auth-submit");
  submit.disabled = true;
  submit.textContent = state.authMode === "register" ? "Creating account…" : "Signing in…";
  try {
    if (state.authMode === "register") {
      await api.register(email, password);
    }
    await api.login(email, password);
    state.email = email;
    $("role-label").textContent = roleFromAccessToken(api.getAccessToken());
    showWorkspace();
    showError("");
    await loadContracts();
  } catch (error) {
    errorNode.textContent = errorMessage(error, "Unable to sign in.");
    errorNode.hidden = false;
  } finally {
    submit.disabled = false;
    setAuthMode(state.authMode);
  }
}

async function signOut() {
  stopPolling();
  await api.logout();
  state.contracts = [];
  state.contract = null;
  state.selectedId = null;
  state.clauses = [];
  state.risks = [];
  state.reviews = [];
  state.email = "";
  showAuth();
}

function updateContractInList(contract) {
  const index = state.contracts.findIndex((item) => item.id === contract.id);
  if (index >= 0) state.contracts[index] = { ...state.contracts[index], ...contract };
}

function renderSidebar() {
  setText("contract-count", String(state.contracts.length));
  const list = $("contract-list");
  if (!state.contracts.length) {
    list.innerHTML = '<p class="sidebar-empty">No contracts yet. Upload one to begin.</p>';
    return;
  }
  list.innerHTML = state.contracts
    .map((contract) => {
      const status = statusMeta(contract.status);
      const active = contract.id === state.selectedId ? " is-active" : "";
      return `
        <button class="contract-item${active}" type="button" data-contract-id="${escapeHtml(contract.id)}" aria-pressed="${contract.id === state.selectedId}">
          <span class="contract-item-top">
            <span class="contract-name">${escapeHtml(contract.title || contract.original_filename)}</span>
            <span class="status-pill ${status.tone}">${escapeHtml(status.label)}</span>
          </span>
          <span class="contract-meta">${escapeHtml(formatBytes(contract.file_size_bytes))} · ${escapeHtml(relativeTime(contract.created_at))}</span>
        </button>`;
    })
    .join("");
}

function renderHeader() {
  const contract = state.contract;
  if (!contract) {
    $("detail-header").innerHTML = "";
    return;
  }
  const status = statusMeta(contract.status);
  $("detail-header").innerHTML = `
    <div>
      <p class="eyebrow">Contract review</p>
      <h2 class="detail-title">${escapeHtml(contract.title || contract.original_filename)}</h2>
      <div class="detail-meta">
        <span>${escapeHtml(contract.original_filename)}</span>
        <span>${escapeHtml(formatBytes(contract.file_size_bytes))}</span>
        <span>Uploaded ${escapeHtml(formatDate(contract.created_at))}</span>
      </div>
    </div>
    <span class="status-pill ${status.tone}">${escapeHtml(status.label)}</span>`;
}

function renderStats() {
  const riskSummary = summarizeRisks(state.risks);
  const reviewed = latestReviewByTarget(state.reviews).size;
  const stats = [
    { label: "Detected clauses", value: state.clauses.length, tone: "" },
    { label: "High risk", value: riskSummary.high, tone: "high" },
    { label: "Medium risk", value: riskSummary.medium, tone: "medium" },
    { label: "Reviewed items", value: reviewed, tone: "" },
  ];
  $("stat-grid").innerHTML = stats
    .map(
      (stat) => `
      <div class="stat-card ${stat.tone}">
        <div class="stat-value">${stat.value}</div>
        <div class="stat-label">${stat.label}</div>
      </div>`,
    )
    .join("");
}

function renderFilters() {
  const types = [...new Set(state.clauses.map((clause) => clause.clause_type))].sort();
  if (!types.length) {
    $("clause-filters").innerHTML = "";
    return;
  }
  const chips = ["all", ...types]
    .map((value) => {
      const label = value === "all" ? "All" : value;
      const active = state.clauseFilter === value ? " is-active" : "";
      return `<button type="button" class="filter-chip${active}" data-clause-filter="${escapeHtml(value)}" aria-pressed="${state.clauseFilter === value}">${escapeHtml(label)}</button>`;
    })
    .join("");
  $("clause-filters").innerHTML = chips;
}

function clauseReviewMeta(clauseId, latestReviews) {
  const event = latestReviews.get(clauseId);
  return event ? `<span class="review-tag">${escapeHtml(actionLabel(event.action))}</span>` : "";
}

function renderClauses() {
  const latestReviews = latestReviewByTarget(state.reviews);
  const clauses = state.clauses.filter(
    (clause) => state.clauseFilter === "all" || clause.clause_type === state.clauseFilter,
  );
  setText("clause-count", String(clauses.length));
  if (!clauses.length) {
    $("clause-list").innerHTML = state.clauses.length
      ? '<div class="empty-inline">No clauses match this filter.</div>'
      : '<div class="empty-inline">No supported clauses were detected in this contract.</div>';
    return;
  }
  $("clause-list").innerHTML = clauses
    .map((clause) => {
      const findings = findingsForClause(state.risks, clause.id);
      const selected = clause.id === state.selectedClauseId ? " is-selected" : "";
      const confidence = clampPercent(clause.confidence);
      const reviewing = state.reviewingKey === `clause:${clause.id}`;
      return `
        <article class="clause-card${selected}" data-clause-id="${escapeHtml(clause.id)}" tabindex="0" role="button" aria-label="View ${escapeHtml(clause.clause_type)} source text">
          <div class="clause-top">
            <h3>${escapeHtml(clause.clause_type)}</h3>
            <span class="confidence">${confidence}%</span>
          </div>
          <div class="confidence-bar" aria-hidden="true"><span style="width:${confidence}%"></span></div>
          <p class="clause-quote">${escapeHtml(truncateText(clause.span_text, 170))}</p>
          <div class="clause-footer">
            ${findings.length ? `<span class="mini-tag">${findings.length} risk${findings.length === 1 ? "" : "s"}</span>` : ""}
            ${clauseReviewMeta(clause.id, latestReviews)}
            <button class="link-button" type="button" data-review-target-type="clause" data-review-target-id="${escapeHtml(clause.id)}" data-review-action="accept" ${reviewing ? "disabled" : ""}>Accept</button>
            <button class="link-button" type="button" data-review-target-type="clause" data-review-target-id="${escapeHtml(clause.id)}" data-review-action="flag" ${reviewing ? "disabled" : ""}>Flag</button>
          </div>
        </article>`;
    })
    .join("");
}

function renderSource(scrollToSelection = false) {
  const text = state.contract?.extracted_text || "";
  const selected = state.clauses.find((clause) => clause.id === state.selectedClauseId);
  setText("source-label", selected ? selected.clause_type : text ? "Full document" : "");
  if (!text) {
    $("source-view").innerHTML = '<div class="empty-inline">No extracted text is available for this contract.</div>';
    return;
  }
  const segments = selected ? highlightSegments(text, selected.char_start, selected.char_end) : { before: text, match: "", after: "" };
  $("source-view").innerHTML = `<div class="source-text">${escapeHtml(segments.before)}${
    segments.match ? `<mark id="source-mark">${escapeHtml(segments.match)}</mark>` : ""
  }${escapeHtml(segments.after)}</div>`;
  if (scrollToSelection && segments.match) {
    window.requestAnimationFrame(() => {
      $("source-mark")?.scrollIntoView({ block: "center", behavior: "smooth" });
    });
  } else {
    $("source-view").scrollTop = 0;
  }
}

function renderRisks() {
  const severityRank = { HIGH: 0, MEDIUM: 1, LOW: 2 };
  const risks = [...state.risks].sort(
    (left, right) => (severityRank[left.risk_level] ?? 3) - (severityRank[right.risk_level] ?? 3),
  );
  setText("risk-count", String(risks.length));
  if (!risks.length) {
    $("risk-list").innerHTML = '<div class="empty-inline">No deterministic risk rules matched the detected clauses.</div>';
    return;
  }
  const clauseById = new Map(state.clauses.map((clause) => [clause.id, clause]));
  $("risk-list").innerHTML = risks
    .map((risk) => {
      const severity = severityMeta(risk.risk_level);
      const clause = clauseById.get(risk.clause_id);
      const reviewing = state.reviewingKey === `finding:${risk.id}`;
      return `
        <article class="risk-card ${severity.tone}">
          <div class="risk-top">
            <h3>${escapeHtml(clause?.clause_type || risk.rule_id)}</h3>
            <span class="severity-badge ${severity.tone}">${escapeHtml(severity.label)}</span>
          </div>
          <p class="risk-explanation">${escapeHtml(risk.explanation)}</p>
          <p class="risk-quote">“${escapeHtml(truncateText(risk.span_text, 180))}”</p>
          <div class="risk-actions">
            ${clause ? `<button class="link-button" type="button" data-select-clause="${escapeHtml(clause.id)}">View clause</button>` : ""}
            <button class="link-button" type="button" data-review-target-type="finding" data-review-target-id="${escapeHtml(risk.id)}" data-review-action="accept" ${reviewing ? "disabled" : ""}>Accept</button>
            <button class="link-button" type="button" data-review-target-type="finding" data-review-target-id="${escapeHtml(risk.id)}" data-review-action="flag" ${reviewing ? "disabled" : ""}>Flag</button>
          </div>
        </article>`;
    })
    .join("");
}

function reviewTargetLabel(event) {
  if (event.target_type === "contract") return state.contract?.title || "Contract";
  if (event.target_type === "clause") {
    return state.clauses.find((clause) => clause.id === event.target_id)?.clause_type || "Clause";
  }
  const finding = state.risks.find((risk) => risk.id === event.target_id);
  return finding ? `Finding · ${finding.rule_id}` : "Finding";
}

function renderReviewPanel() {
  const latest = latestReviewByTarget(state.reviews);
  const contractEvent = state.contract ? latest.get(state.contract.id) : null;
  $("review-summary").innerHTML = contractEvent
    ? `Current contract decision: <strong>${escapeHtml(actionLabel(contractEvent.action))}</strong> · ${escapeHtml(relativeTime(contractEvent.created_at))}`
    : "No contract-level decision recorded yet.";

  const reviewingContract = state.reviewingKey === `contract:${state.contract?.id}`;
  $("review-actions").innerHTML = `
    <button class="button button-soft button-small" type="button" data-review-target-type="contract" data-review-target-id="${escapeHtml(state.contract?.id || "")}" data-review-action="approve" ${reviewingContract ? "disabled" : ""}>Approve</button>
    <button class="button button-small" type="button" data-review-target-type="contract" data-review-target-id="${escapeHtml(state.contract?.id || "")}" data-review-action="request_changes" ${reviewingContract ? "disabled" : ""}>Request changes</button>
    <button class="button button-danger button-small" type="button" data-review-target-type="contract" data-review-target-id="${escapeHtml(state.contract?.id || "")}" data-review-action="flag" ${reviewingContract ? "disabled" : ""}>Flag</button>`;

  const events = [...state.reviews].sort((left, right) => new Date(right.created_at) - new Date(left.created_at));
  if (!events.length) {
    $("review-timeline").innerHTML = '<div class="empty-inline">No review decisions recorded yet.</div>';
    return;
  }
  $("review-timeline").innerHTML = events
    .map(
      (event) => `
      <article class="timeline-item">
        <div class="timeline-title">${escapeHtml(actionLabel(event.action))} · ${escapeHtml(reviewTargetLabel(event))}</div>
        <div class="timeline-meta">${escapeHtml(formatDate(event.created_at))} · ${escapeHtml(event.target_type)}</div>
        ${event.notes ? `<div class="timeline-notes">${escapeHtml(event.notes)}</div>` : ""}
      </article>`,
    )
    .join("");
}

function renderAnalysis() {
  renderStats();
  renderFilters();
  renderClauses();
  renderSource(false);
  renderRisks();
  renderReviewPanel();
}

function renderDetail() {
  const hasContract = Boolean(state.contract);
  $("empty-workspace").hidden = hasContract;
  $("detail-view").hidden = !hasContract;
  if (!hasContract) {
    $("analysis-grid").hidden = true;
    $("analysis-loading").hidden = true;
    $("analysis-error").hidden = true;
    return;
  }

  renderHeader();
  renderStats();

  const analyzing = state.detailLoading || state.contract.status === "PENDING" || state.contract.status === "PROCESSING";
  $("analysis-loading").hidden = !analyzing;
  $("analysis-error").hidden = !state.analysisError;
  if (state.analysisError) $("analysis-error").textContent = state.analysisError;

  if (analyzing) {
    setText(
      "analysis-loading-copy",
      state.contract.status === "READY" ? "Loading detected clauses and risk findings…" : "Extracting text, detecting clauses, and applying risk rules…",
    );
  }

  const showGrid = state.contract.status === "READY" && !state.detailLoading && !state.analysisError;
  $("analysis-grid").hidden = !showGrid;
  if (showGrid) renderAnalysis();
}

async function loadContracts() {
  try {
    state.contracts = await api.listContracts();
    renderSidebar();
    const stillExists = state.contracts.some((contract) => contract.id === state.selectedId);
    if (!stillExists) {
      state.selectedId = state.contracts[0]?.id || null;
      state.contract = null;
    }
    if (state.selectedId) {
      await selectContract(state.selectedId);
    } else {
      renderDetail();
      renderSidebar();
    }
  } catch (error) {
    showError(errorMessage(error, "Unable to load contracts."));
  }
}

function stopPolling() {
  if (state.pollTimer) window.clearInterval(state.pollTimer);
  state.pollTimer = null;
  state.polling = false;
}

function startPolling(contractId) {
  stopPolling();
  state.pollStartedAt = Date.now();
  state.pollTimer = window.setInterval(async () => {
    if (state.polling || state.selectedId !== contractId) return;
    if (Date.now() - state.pollStartedAt > 120000) {
      stopPolling();
      state.detailLoading = false;
      state.analysisError = "Processing is taking longer than expected. Refresh the contract to check again.";
      renderDetail();
      return;
    }
    state.polling = true;
    try {
      const contract = await api.getContract(contractId);
      if (state.selectedId !== contractId) return;
      state.contract = contract;
      updateContractInList(contract);
      renderSidebar();
      if (contract.status === "READY") {
        stopPolling();
        await loadAnalysis(contractId);
      } else if (contract.status === "FAILED") {
        stopPolling();
        state.detailLoading = false;
        state.analysisError = contract.error_message || "Contract processing failed.";
        renderDetail();
      } else {
        renderDetail();
      }
    } catch (error) {
      stopPolling();
      state.detailLoading = false;
      state.analysisError = errorMessage(error, "Unable to refresh processing status.");
      renderDetail();
    } finally {
      state.polling = false;
    }
  }, 1600);
}

async function selectContract(contractId) {
  stopPolling();
  state.selectedId = contractId;
  state.contract = state.contracts.find((item) => item.id === contractId) || null;
  state.clauses = [];
  state.risks = [];
  state.reviews = [];
  state.selectedClauseId = null;
  state.clauseFilter = "all";
  state.detailLoading = true;
  state.analysisError = "";
  showError("");
  renderSidebar();
  renderDetail();

  try {
    const contract = await api.getContract(contractId);
    if (state.selectedId !== contractId) return;
    state.contract = contract;
    updateContractInList(contract);
    renderSidebar();
    if (contract.status === "READY") {
      await loadAnalysis(contractId);
    } else if (contract.status === "FAILED") {
      state.detailLoading = false;
      state.analysisError = contract.error_message || "Contract processing failed.";
      renderDetail();
    } else {
      state.detailLoading = false;
      renderDetail();
      startPolling(contractId);
    }
  } catch (error) {
    state.detailLoading = false;
    state.analysisError = errorMessage(error, "Unable to load this contract.");
    renderDetail();
  }
}

async function loadAnalysis(contractId) {
  state.detailLoading = true;
  state.analysisError = "";
  renderDetail();
  try {
    const [clauses, risks, reviews] = await Promise.all([
      api.listClauses(contractId),
      api.listRisks(contractId),
      api.listReviews(500),
    ]);
    if (state.selectedId !== contractId) return;
    state.clauses = clauses;
    state.risks = risks;
    const targetIds = new Set([contractId, ...clauses.map((clause) => clause.id), ...risks.map((risk) => risk.id)]);
    state.reviews = filterContractReviews(reviews, targetIds);
    state.selectedClauseId = clauses[0]?.id || null;
    state.detailLoading = false;
    renderDetail();
  } catch (error) {
    state.detailLoading = false;
    state.analysisError = errorMessage(error, "Unable to load clause and risk analysis.");
    renderDetail();
  }
}

function selectClause(clauseId) {
  if (!state.clauses.some((clause) => clause.id === clauseId)) return;
  state.selectedClauseId = clauseId;
  renderClauses();
  renderSource(true);
}

async function postReview(targetType, targetId, action) {
  if (!state.contract || !targetId) return;
  const key = `${targetType}:${targetId}`;
  if (state.reviewingKey) return;
  state.reviewingKey = key;
  const notes = targetType === "contract" ? $("review-notes").value.trim() : "";
  renderClauses();
  renderRisks();
  renderReviewPanel();
  try {
    const event = await api.createReview({
      target_id: targetId,
      target_type: targetType,
      action,
      notes: notes || null,
      payload: { contract_id: state.contract.id },
    });
    state.reviews = [...state.reviews, event];
    if (targetType === "contract") $("review-notes").value = "";
    showToast(`Review recorded: ${actionLabel(action)}.`);
  } catch (error) {
    showToast(errorMessage(error, "Unable to record review decision."), "error");
  } finally {
    state.reviewingKey = "";
    renderAnalysis();
  }
}

async function handleUpload(file) {
  const uploadError = $("upload-error");
  uploadError.hidden = true;
  const validationError = validateUpload(file);
  if (validationError) {
    uploadError.textContent = validationError;
    uploadError.hidden = false;
    return;
  }
  if (state.uploading) return;

  state.uploading = true;
  $("upload-status").hidden = false;
  $("upload-status").textContent = `Uploading ${file.name}…`;
  $("upload-drop").classList.add("is-uploading");
  try {
    const contract = await api.uploadContract(file);
    state.contracts = [contract, ...state.contracts.filter((item) => item.id !== contract.id)];
    state.uploading = false;
    $("upload-status").textContent = "Upload complete. Processing started.";
    showToast("Contract uploaded. Analysis is processing.");
    await selectContract(contract.id);
    window.setTimeout(() => {
      $("upload-status").hidden = true;
      $("upload-drop").classList.remove("is-uploading");
    }, 2500);
  } catch (error) {
    state.uploading = false;
    $("upload-status").hidden = true;
    $("upload-drop").classList.remove("is-uploading");
    uploadError.textContent = errorMessage(error, "Upload failed.");
    uploadError.hidden = false;
    showToast(uploadError.textContent, "error");
  } finally {
    $("file-input").value = "";
  }
}

function handleWorkspaceClick(event) {
  const reviewButton = event.target.closest("[data-review-action]");
  if (reviewButton) {
    event.stopPropagation();
    postReview(
      reviewButton.dataset.reviewTargetType,
      reviewButton.dataset.reviewTargetId,
      reviewButton.dataset.reviewAction,
    );
    return;
  }

  const filter = event.target.closest("[data-clause-filter]");
  if (filter) {
    state.clauseFilter = filter.dataset.clauseFilter;
    renderFilters();
    renderClauses();
    return;
  }

  const viewClause = event.target.closest("[data-select-clause]");
  if (viewClause) {
    event.stopPropagation();
    selectClause(viewClause.dataset.selectClause);
    return;
  }

  const contractButton = event.target.closest("[data-contract-id]");
  if (contractButton) {
    if (contractButton.dataset.contractId !== state.selectedId) {
      selectContract(contractButton.dataset.contractId);
    }
    return;
  }

  const clauseCard = event.target.closest("[data-clause-id]");
  if (clauseCard) selectClause(clauseCard.dataset.clauseId);
}

function bindEvents() {
  $("login-tab").addEventListener("click", () => setAuthMode("login"));
  $("register-tab").addEventListener("click", () => setAuthMode("register"));
  $("auth-form").addEventListener("submit", handleAuthSubmit);
  $("sign-out").addEventListener("click", signOut);
  $("workspace").addEventListener("click", handleWorkspaceClick);
  $("workspace").addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const clauseCard = event.target.closest("[data-clause-id]");
    if (clauseCard) {
      event.preventDefault();
      selectClause(clauseCard.dataset.clauseId);
    }
  });

  $("file-input").addEventListener("change", (event) => {
    const [file] = event.target.files;
    if (file) handleUpload(file);
  });

  const drop = $("upload-drop");
  ["dragenter", "dragover"].forEach((name) =>
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.add("is-dragging");
    }),
  );
  ["dragleave", "drop"].forEach((name) =>
    drop.addEventListener(name, (event) => {
      event.preventDefault();
      drop.classList.remove("is-dragging");
    }),
  );
  drop.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    if (file) handleUpload(file);
  });
}

async function init() {
  bindEvents();
  setAuthMode("login");
  if (api.getAccessToken() || api.getRefreshToken()) {
    try {
      $("role-label").textContent = roleFromAccessToken(api.getAccessToken());
      showWorkspace();
      await loadContracts();
      return;
    } catch {
      api.clearTokens();
    }
  }
  showAuth();
}

init();
