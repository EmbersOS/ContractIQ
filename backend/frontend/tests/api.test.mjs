import assert from "node:assert/strict";
import test from "node:test";

import { ApiError, createApiClient, createTokenStore } from "../js/api.js";

function memoryStore(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
    removeItem: (key) => values.delete(key),
  };
}

function jsonResponse(payload, status = 200, statusText = "OK") {
  return new Response(JSON.stringify(payload), {
    status,
    statusText,
    headers: { "Content-Type": "application/json" },
  });
}

test("login stores tokens and authenticated requests attach the bearer token", async () => {
  const calls = [];
  const storage = memoryStore();
  const api = createApiClient({
    baseUrl: "http://api.test",
    tokenStore: createTokenStore(storage),
    fetchImpl: async (url, init) => {
      calls.push({ url, init });
      if (url.endsWith("/auth/login")) return jsonResponse({ access_token: "access-1", refresh_token: "refresh-1" });
      if (url.endsWith("/contracts")) return jsonResponse([]);
      throw new Error(`Unexpected request: ${url}`);
    },
  });

  await api.login("reviewer@example.com", "password123");
  const contracts = await api.listContracts();

  assert.deepEqual(contracts, []);
  assert.equal(calls[0].init.headers.Authorization, undefined);
  assert.equal(calls[1].init.headers.Authorization, "Bearer access-1");
  assert.equal(api.getAccessToken(), "access-1");
  assert.equal(api.getRefreshToken(), "refresh-1");
});

test("a 401 triggers one refresh and retries the original request", async () => {
  let contractRequests = 0;
  const calls = [];
  const api = createApiClient({
    baseUrl: "http://api.test",
    tokenStore: createTokenStore(memoryStore({
      "contractiq.access_token": "expired",
      "contractiq.refresh_token": "refresh-1",
    })),
    fetchImpl: async (url, init) => {
      calls.push({ url, init });
      if (url.endsWith("/auth/refresh")) {
        return jsonResponse({ access_token: "access-2", refresh_token: "refresh-2" });
      }
      if (url.endsWith("/contracts")) {
        contractRequests += 1;
        if (contractRequests === 1) return jsonResponse({ detail: "expired" }, 401, "Unauthorized");
        return jsonResponse([{ id: "contract-1" }]);
      }
      throw new Error(`Unexpected request: ${url}`);
    },
  });

  const contracts = await api.listContracts();

  assert.deepEqual(contracts, [{ id: "contract-1" }]);
  assert.equal(contractRequests, 2);
  assert.equal(calls[1].url, "http://api.test/auth/refresh");
  assert.equal(calls[2].init.headers.Authorization, "Bearer access-2");
});

test("a failed refresh clears stored tokens", async () => {
  const storage = memoryStore({
    "contractiq.access_token": "expired",
    "contractiq.refresh_token": "bad-refresh",
  });
  const api = createApiClient({
    baseUrl: "http://api.test",
    tokenStore: createTokenStore(storage),
    fetchImpl: async (url) => {
      if (url.endsWith("/auth/refresh")) return jsonResponse({ detail: "invalid refresh" }, 401, "Unauthorized");
      return jsonResponse({ detail: "expired" }, 401, "Unauthorized");
    },
  });

  await assert.rejects(() => api.listContracts(), (error) => error instanceof ApiError);
  assert.equal(storage.getItem("contractiq.access_token"), null);
  assert.equal(storage.getItem("contractiq.refresh_token"), null);
});

test("upload sends multipart form data without a manual content type", async () => {
  let captured;
  const api = createApiClient({
    baseUrl: "http://api.test",
    tokenStore: createTokenStore(memoryStore({ "contractiq.access_token": "access-1" })),
    fetchImpl: async (url, init) => {
      captured = { url, init };
      return jsonResponse({ id: "contract-1", status: "PENDING" }, 202, "Accepted");
    },
  });
  const file = new Blob(["%PDF-1.4"], { type: "application/pdf" });

  await api.uploadContract(file);

  assert.equal(captured.url, "http://api.test/contracts");
  assert.equal(captured.init.method, "POST");
  assert.ok(captured.init.body instanceof FormData);
  assert.equal(captured.init.headers["Content-Type"], undefined);
  assert.equal(captured.init.headers.Authorization, "Bearer access-1");
});

test("API validation errors are surfaced with backend detail", async () => {
  const api = createApiClient({
    baseUrl: "http://api.test",
    tokenStore: createTokenStore(memoryStore()),
    fetchImpl: async () => jsonResponse({ detail: [{ msg: "field required" }, { msg: "invalid email" }] }, 422, "Unprocessable"),
  });

  await assert.rejects(
    () => api.register("bad", "short"),
    (error) => error instanceof ApiError && error.status === 422 && error.detail === "field required invalid email",
  );
});
