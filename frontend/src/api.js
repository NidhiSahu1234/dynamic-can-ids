const BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(BASE + path, {
      headers: { "Content-Type": "application/json", ...options.headers },
      ...options,
    });
  } catch (error) {
    const reason = error instanceof Error ? error.message : "network error";
    throw new Error(
      `Cannot reach the CAN IDS API at ${BASE}. Start the FastAPI backend and try again. (${reason})`
    );
  }
  if (!response.ok) {
    const body = await response.text();
    let message = body || response.statusText;
    try {
      const parsed = JSON.parse(body);
      message = typeof parsed.detail === "string" ? parsed.detail : JSON.stringify(parsed.detail ?? parsed);
    } catch {
      // Keep the server's plain-text error message.
    }
    throw new Error(message);
  }
  return response.json();
}

export const api = {
  health: () => request("/health"),
  stats: () => request("/stats"),
  metrics: () => request("/metrics"),
  alerts: (limit = 50) => request("/alerts?limit=" + encodeURIComponent(limit)),
  rules: () => request("/rules"),
  predict: (frame) => request("/predict", { method: "POST", body: JSON.stringify(frame) }),
  predictBatch: (frames) =>
    request("/predict/batch", { method: "POST", body: JSON.stringify({ frames }) }),
  trafficSample: (source, skip, limit) =>
    request("/traffic/sample?source=" + encodeURIComponent(source) +
      "&skip=" + encodeURIComponent(skip) + "&limit=" + encodeURIComponent(limit)),
  addRule: (body) => request("/rules", { method: "POST", body: JSON.stringify(body) }),
  deleteRule: (id) => request("/rules/" + encodeURIComponent(id), { method: "DELETE" }),
};
