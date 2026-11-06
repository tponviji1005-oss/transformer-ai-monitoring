const API_BASE = '/api';

/**
 * Robust JSON fetch wrapper with error handling and fallback reporting.
 */
async function fetchJson(endpoint, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${endpoint}`, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `HTTP ${res.status}: ${res.statusText}`);
    }
    return await res.json();
  } catch (err) {
    console.error(`API Error [${endpoint}]:`, err.message);
    throw err;
  }
}

export const api = {
  // System Health
  getHealth: () => fetchJson('/health'),

  // Transformer Specifications
  updateSpecs: (transformerId, specs) =>
    fetchJson(`/transformers/${transformerId}/specs`, {
      method: 'POST',
      body: JSON.stringify(specs),
    }),

  // Telemetry Aggregation
  getHourlyHistory: (transformerId, limit = 24) =>
    fetchJson(`/aggregation/${transformerId}/history?limit=${limit}`),
  triggerAggregation: () =>
    fetchJson('/aggregation/run', { method: 'POST', body: JSON.stringify({}) }),

  // 39-Feature Diagnostics
  getFeatureReadiness: (transformerId) =>
    fetchJson(`/features/${transformerId}/readiness`),
  getLatestFeatures: (transformerId) =>
    fetchJson(`/features/${transformerId}/latest`),
  generateFeatures: (transformerId) =>
    fetchJson(`/features/${transformerId}/generate`, { method: 'POST', body: JSON.stringify({}) }),

  // ML Predictions & Advisory Alerts
  getLatestPrediction: (transformerId) =>
    fetchJson(`/predictions/${transformerId}/latest`),
  getPredictionHistory: (transformerId, limit = 24) =>
    fetchJson(`/predictions/${transformerId}/history?limit=${limit}`),
  runPrediction: (transformerId) =>
    fetchJson(`/predictions/${transformerId}/predict`, { method: 'POST', body: JSON.stringify({}) }),
};

