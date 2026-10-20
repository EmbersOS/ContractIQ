const ACCESS_KEY = "contractiq.access_token";
const REFRESH_KEY = "contractiq.refresh_token";

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail || `Request failed with status ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function safeStorage(storage) {
  return storage || {
    getItem: () => null,
    setItem: () => {},
    removeItem: () => {},
  };
}

export function createTokenStore(storage = globalThis.localStorage) {
  const store = safeStorage(storage);
  return {
    getAccessToken() {
      try {
        return store.getItem(ACCESS_KEY);
      } catch {
        return null;
      }
    },
    getRefreshToken() {
      try {
        return store.getItem(REFRESH_KEY);
      } catch {
        return null;
      }
    },
    setTokens(tokens) {
      try {
        if (tokens?.access_token) store.setItem(ACCESS_KEY, tokens.access_token);
        if (tokens?.refresh_token) store.setItem(REFRESH_KEY, tokens.refresh_token);
      } catch {
        /* Storage can be unavailable in restricted browser contexts. */
      }
    },
    clear() {
      try {
        store.removeItem(ACCESS_KEY);
        store.removeItem(REFRESH_KEY);
      } catch {
        /* Ignore unavailable storage. */
      }
    },
  };
}

export function formatApiDetail(payload, fallback = "Something went wrong.") {
  const detail = payload?.detail;
  if (typeof detail === "string" && detail.trim()) return detail;
  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => (typeof item === "string" ? item : item?.msg))
      .filter(Boolean);
    if (messages.length) return messages.join(" ");
  }
  return fallback;
}

export function createApiClient({ baseUrl = "", fetchImpl = globalThis.fetch, tokenStore } = {}) {
  const tokens = tokenStore || createTokenStore();

  async function parseError(response) {
    let payload = null;
    try {
      payload = await response.json();
    } catch {
      payload = null;
    }
    return new ApiError(response.status, formatApiDetail(payload, response.statusText));
  }

  async function refreshTokens() {
    const refreshToken = tokens.getRefreshToken();
    if (!refreshToken) throw new ApiError(401, "Your session has expired. Sign in again.");
    const response = await fetchImpl(`${baseUrl}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    });
    if (!response.ok) throw await parseError(response);
    const pair = await response.json();
    tokens.setTokens(pair);
    return pair;
  }

  async function request(path, options = {}) {
    const { method = "GET", body, auth = true, retry = true, headers = {} } = options;
    const requestHeaders = { Accept: "application/json", ...headers };
    const init = { method, headers: requestHeaders };

    if (body !== undefined) {
      if (typeof FormData !== "undefined" && body instanceof FormData) {
        init.body = body;
      } else {
        requestHeaders["Content-Type"] = "application/json";
        init.body = JSON.stringify(body);
      }
    }

    const accessToken = tokens.getAccessToken();
    if (auth && accessToken) requestHeaders.Authorization = `Bearer ${accessToken}`;

    const response = await fetchImpl(`${baseUrl}${path}`, init);

    if (response.status === 401 && auth && retry && tokens.getRefreshToken()) {
      try {
        await refreshTokens();
      } catch (error) {
        tokens.clear();
        throw error;
      }
      return request(path, { ...options, retry: false });
    }

    if (!response.ok) throw await parseError(response);
    if (response.status === 204) return null;
    return response.json();
  }

  return {
    request,
    async login(email, password) {
      const pair = await request("/auth/login", {
        method: "POST",
        auth: false,
        body: { email, password },
      });
      tokens.setTokens(pair);
      return pair;
    },
    async register(email, password) {
      return request("/auth/register", {
        method: "POST",
        auth: false,
        body: { email, password, role: "viewer" },
      });
    },
    async logout() {
      try {
        await request("/auth/logout", { method: "POST" });
      } catch {
        /* Stateless JWT logout is best effort. */
      } finally {
        tokens.clear();
      }
    },
    listContracts() {
      return request("/contracts");
    },
    getContract(contractId) {
      return request(`/contracts/${encodeURIComponent(contractId)}`);
    },
    uploadContract(file) {
      const form = new FormData();
      form.append("file", file);
      return request("/contracts", { method: "POST", body: form });
    },
    listClauses(contractId) {
      return request(`/contracts/${encodeURIComponent(contractId)}/clauses`);
    },
    listRisks(contractId) {
      return request(`/contracts/${encodeURIComponent(contractId)}/risks`);
    },
    listReviews(limit = 200) {
      return request(`/reviews?limit=${encodeURIComponent(limit)}`);
    },
    createReview(body) {
      return request("/reviews", { method: "POST", body });
    },
    getAccessToken: tokens.getAccessToken,
    getRefreshToken: tokens.getRefreshToken,
    clearTokens: tokens.clear,
  };
}
