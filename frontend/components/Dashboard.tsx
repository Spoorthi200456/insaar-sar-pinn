"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ControlPanel from "./ControlPanel";
import ExplainabilityPanel from "./ExplainabilityPanel";
import MapView from "./MapView";
import TimeSeriesChart from "./TimeSeriesChart";
import { ApiError, getHealth, openTelemetryStream, predict } from "@/lib/api";
import { DEFAULT_SCENARIO, generateScenarioBatch } from "@/lib/synthetic";
import { DEFAULT_METHOD } from "@/lib/types";
import type {
  AssessmentMethod,
  HealthResponse,
  ScenarioControls,
  StructurePrediction,
  TelemetryBatch,
  WSMessage,
} from "@/lib/types";

type StreamStatus = "connecting" | "open" | "closed" | "error" | "idle";

export default function Dashboard() {
  const [controls, setControls] = useState<ScenarioControls>(DEFAULT_SCENARIO);
  const [method, setMethod] = useState<AssessmentMethod>(DEFAULT_METHOD);
  const [batch, setBatch] = useState<TelemetryBatch>(() => generateScenarioBatch(DEFAULT_SCENARIO));
  const [predictions, setPredictions] = useState<Map<string, StructurePrediction>>(new Map());
  const [selectedStructureId, setSelectedStructureId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [streamingEnabled, setStreamingEnabled] = useState(false);
  const [streamStatus, setStreamStatus] = useState<StreamStatus>("idle");

  const streamRef = useRef<ReturnType<typeof openTelemetryStream> | null>(null);

  useEffect(() => {
    getHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
    const interval = setInterval(() => {
      getHealth()
        .then(setHealth)
        .catch(() => setHealth(null));
    }, 15000);
    return () => clearInterval(interval);
  }, []);

  const runInference = useCallback(
    async (targetBatch: TelemetryBatch, threshold: number, chosen: AssessmentMethod) => {
      setIsLoading(true);
      setErrorMessage(null);
      try {
        const response = await predict(targetBatch, threshold, chosen);
        const next = new Map<string, StructurePrediction>();
        for (const p of response.predictions) next.set(p.structure_id, p);
        setPredictions(next);
      } catch (err) {
        if (err instanceof ApiError) {
          const detail =
            typeof err.detail === "object" && err.detail !== null && "detail" in err.detail
              ? JSON.stringify((err.detail as { detail?: unknown }).detail)
              : err.message;
          setErrorMessage(`Prediction failed (${err.status}): ${detail}`);
        } else {
          setErrorMessage("Prediction failed: unable to reach the backend.");
        }
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const regenerateAndRun = useCallback(() => {
    const nextBatch = generateScenarioBatch(controls);
    setBatch(nextBatch);
    setSelectedStructureId(null);
    if (streamingEnabled && streamRef.current) {
      streamRef.current.send(nextBatch, controls.rechargeEfficiencyThreshold, method);
    } else {
      void runInference(nextBatch, controls.rechargeEfficiencyThreshold, method);
    }
  }, [controls, method, streamingEnabled, runInference]);

  useEffect(() => {
    if (!streamingEnabled) {
      streamRef.current?.close();
      streamRef.current = null;
      setStreamStatus("idle");
      return;
    }

    const handle = openTelemetryStream(
      (msg: WSMessage) => {
        if (msg.type === "prediction") {
          const next = new Map<string, StructurePrediction>();
          for (const p of msg.predictions) next.set(p.structure_id, p);
          setPredictions(next);
          setIsLoading(false);
        } else if (msg.type === "error") {
          setErrorMessage(`Stream error: ${msg.detail}`);
          setIsLoading(false);
        }
      },
      (status) => setStreamStatus(status)
    );
    streamRef.current = handle;
    handle.send(batch, controls.rechargeEfficiencyThreshold, method);

    return () => {
      handle.close();
      streamRef.current = null;
    };
    // Intentionally reconnect only when streaming is toggled, not on every
    // control change -- individual scenario edits re-send over the existing
    // socket via regenerateAndRun / the threshold-change effect below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [streamingEnabled]);

  useEffect(() => {
    if (streamingEnabled && streamRef.current) {
      streamRef.current.send(batch, controls.rechargeEfficiencyThreshold, method);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [controls.rechargeEfficiencyThreshold, method]);

  const selectedRecord = useMemo(
    () => batch.records.find((r) => r.design.structure_id === selectedStructureId) ?? null,
    [batch, selectedStructureId]
  );
  const selectedPrediction = selectedStructureId
    ? predictions.get(selectedStructureId) ?? null
    : null;

  const summary = useMemo(() => {
    const values = Array.from(predictions.values());
    // Percentages exist only for structures with well data (author decision 5).
    const calibrated = values.filter((p) => p.recharge_efficiency !== null);
    const underperforming = calibrated.filter((p) => p.is_underperforming === true).length;
    const meanEfficiency =
      calibrated.length > 0
        ? calibrated.reduce((sum, p) => sum + (p.recharge_efficiency ?? 0), 0) / calibrated.length
        : null;
    const insarOnly = values.length - calibrated.length;
    const shownMethod = values[0]?.method ?? null;
    const methodNote = values[0]?.method_note ?? null;
    return {
      shownMethod,
      methodNote,
      total: batch.records.length,
      scored: values.length,
      underperforming,
      meanEfficiency,
      insarOnly,
    };
  }, [batch, predictions]);

  return (
    <div className="flex h-screen flex-col bg-base-950">
      <header className="flex items-center justify-between border-b border-base-700 bg-base-900 px-5 py-3">
        <div>
          <h1 className="text-base font-semibold text-slate-100">
            InSAR-PINN Artificial Recharge Structure Monitor
          </h1>
          <p className="text-xs text-slate-500">
            Recharge ranking from Sentinel-1 InSAR + rainfall telemetry (default: least
            squares, no neural network)
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs text-slate-400">
          <SummaryStat label="Structures" value={summary.total.toString()} />
          <SummaryStat label="Scored" value={summary.scored.toString()} />
          <SummaryStat
            label="Underperforming (wells)"
            value={summary.underperforming.toString()}
            tone={summary.underperforming > 0 ? "danger" : "accent"}
          />
          <SummaryStat
            label="Mean efficiency (wells)"
            value={
              summary.meanEfficiency === null
                ? "n/a"
                : `${(summary.meanEfficiency * 100).toFixed(1)}%`
            }
          />
          <HealthBadge health={health} />
        </div>
      </header>

      <div className="border-b border-warn-500/30 bg-warn-500/10 px-5 py-1.5 text-xs text-warn-400">
        SYNTHETIC DATA, PROVISIONAL PHYSICS. These outputs are a demonstration of the
        pipeline, not an assessment of any real recharge structure (see docs/STATUS.md).
        {summary.shownMethod && (
          <span
            className={`block ${
              summary.shownMethod === "neural_experimental" ? "font-semibold text-danger-400" : ""
            }`}
          >
            Method:{" "}
            {summary.shownMethod === "ols_baseline"
              ? "least squares (default)"
              : "neural network (EXPERIMENTAL)"}
            {" \u2014 "}
            {summary.methodNote}
          </span>
        )}
        {summary.insarOnly > 0 && (
          <span className="block">
            {summary.insarOnly} structure(s) have no well data: shown as a relative recharge
            ranking, because InSAR alone cannot give an efficiency percentage.
          </span>
        )}
      </div>

      {errorMessage && (
        <div className="border-b border-danger-500/30 bg-danger-500/10 px-5 py-2 text-xs text-danger-400">
          {errorMessage}
        </div>
      )}

      <main className="grid flex-1 grid-cols-12 gap-3 overflow-hidden p-3">
        <section className="col-span-3 min-h-0">
          <ControlPanel
            controls={controls}
            onChange={setControls}
            method={method}
            onMethodChange={setMethod}
            onRunInference={regenerateAndRun}
            isLoading={isLoading}
            streamingEnabled={streamingEnabled}
            onToggleStreaming={setStreamingEnabled}
            streamStatus={streamStatus}
          />
        </section>

        <section className="col-span-6 grid min-h-0 grid-rows-2 gap-3">
          <div className="panel min-h-0 overflow-hidden">
            <MapView
              records={batch.records}
              predictions={predictions}
              selectedStructureId={selectedStructureId}
              onSelectStructure={setSelectedStructureId}
            />
          </div>
          <div className="panel min-h-0 overflow-hidden p-3">
            <TimeSeriesChart record={selectedRecord} prediction={selectedPrediction} />
          </div>
        </section>

        <section className="col-span-3 min-h-0">
          <div className="panel h-full">
            <div className="panel-header">
              <h2 className="text-sm font-semibold text-slate-200">Explainability</h2>
              <span className="stat-pill bg-warn-500/20 text-warn-400">
                neural network, experimental
              </span>
            </div>
            <ExplainabilityPanel structureId={selectedStructureId} batch={batch} />
          </div>
        </section>
      </main>
    </div>
  );
}

function SummaryStat({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string;
  tone?: "default" | "danger" | "accent";
}) {
  const toneClass =
    tone === "danger" ? "text-danger-400" : tone === "accent" ? "text-accent-400" : "text-slate-200";
  return (
    <div className="text-right">
      <div className="text-[10px] uppercase tracking-wide text-slate-500">{label}</div>
      <div className={`font-mono text-sm ${toneClass}`}>{value}</div>
    </div>
  );
}

function HealthBadge({ health }: { health: HealthResponse | null }) {
  if (!health) {
    return <span className="stat-pill bg-danger-500/20 text-danger-400">backend unreachable</span>;
  }
  return (
    <span className="stat-pill bg-accent-500/20 text-accent-400">
      {health.backend ?? "model"} {"\u00b7"} {health.active_provider ?? "cpu"}
    </span>
  );
}
