import type {
  ExplainabilityResponse,
  HealthResponse,
  PredictionRequest,
  PredictionResponse,
  TelemetryBatch,
  WSMessage,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_BASE_URL ?? "ws://localhost:8000";

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, detail: unknown) {
    super(`API request failed with status ${status}`);
    this.status = status;
    this.detail = detail;
  }
}

async function parseJsonOrThrow<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail: unknown;
    try {
      detail = await response.json();
    } catch {
      detail = await response.text();
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/health`, { cache: "no-store" });
  return parseJsonOrThrow<HealthResponse>(res);
}

export async function predict(
  batch: TelemetryBatch,
  rechargeEfficiencyThreshold: number
): Promise<PredictionResponse> {
  const payload: PredictionRequest = {
    batch,
    recharge_efficiency_threshold: rechargeEfficiencyThreshold,
  };
  const res = await fetch(`${API_BASE_URL}/api/v1/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return parseJsonOrThrow<PredictionResponse>(res);
}

export async function explainStructure(
  structureId: string,
  batch: TelemetryBatch
): Promise<ExplainabilityResponse> {
  const res = await fetch(
    `${API_BASE_URL}/api/v1/explain/${encodeURIComponent(structureId)}`,
    {
      method: "GET",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(batch),
    }
  );
  return parseJsonOrThrow<ExplainabilityResponse>(res);
}

export interface TelemetryStreamHandle {
  send: (batch: TelemetryBatch, rechargeEfficiencyThreshold: number) => void;
  close: () => void;
}

/**
 * Opens the high-throughput streaming WebSocket. `onMessage` receives every
 * frame (predictions, errors, and heartbeats); callers typically ignore
 * heartbeats and update state on `type === "prediction"`.
 */
export function openTelemetryStream(
  onMessage: (msg: WSMessage) => void,
  onStatusChange?: (status: "connecting" | "open" | "closed" | "error") => void
): TelemetryStreamHandle {
  const ws = new WebSocket(`${WS_BASE_URL}/ws/telemetry`);
  onStatusChange?.("connecting");

  ws.onopen = () => onStatusChange?.("open");
  ws.onclose = () => onStatusChange?.("closed");
  ws.onerror = () => onStatusChange?.("error");
  ws.onmessage = (event: MessageEvent) => {
    try {
      const data = typeof event.data === "string" ? event.data : event.data;
      const parsed: WSMessage = JSON.parse(data);
      onMessage(parsed);
    } catch {
      // Malformed frame from a misbehaving proxy/server -- drop silently,
      // the heartbeat/next valid frame will re-sync UI state.
    }
  };

  return {
    send: (batch, rechargeEfficiencyThreshold) => {
      if (ws.readyState !== WebSocket.OPEN) return;
      const payload: PredictionRequest = {
        batch,
        recharge_efficiency_threshold: rechargeEfficiencyThreshold,
      };
      ws.send(JSON.stringify(payload));
    },
    close: () => ws.close(),
  };
}
