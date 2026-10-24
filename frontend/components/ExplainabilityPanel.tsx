"use client";

import { useEffect, useState } from "react";
import { Bar } from "react-chartjs-2";
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  LinearScale,
  Tooltip,
} from "chart.js";
import { ApiError, explainStructure } from "@/lib/api";
import type { ExplainabilityResponse, TelemetryBatch } from "@/lib/types";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip);

interface ExplainabilityPanelProps {
  structureId: string | null;
  batch: TelemetryBatch | null;
}

const ATTRIBUTION_LABELS: Record<string, string> = {
  design_features_grad_l1: "Design parameters",
  rainfall_grad_l1: "Rainfall forcing",
  displacement_grad_l1: "InSAR displacement",
};

export default function ExplainabilityPanel({ structureId, batch }: ExplainabilityPanelProps) {
  const [result, setResult] = useState<ExplainabilityResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!structureId || !batch) {
      setResult(null);
      setError(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);

    explainStructure(structureId, batch)
      .then((res) => {
        if (!cancelled) setResult(res);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof ApiError) {
          const detail =
            typeof err.detail === "object" && err.detail !== null && "detail" in err.detail
              ? String((err.detail as { detail?: unknown }).detail)
              : err.message;
          setError(detail);
        } else {
          setError("Explainability request failed");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [structureId, batch]);

  if (!structureId) {
    return (
      <div className="flex h-full items-center justify-center px-4 text-center text-sm text-slate-500">
        Select a structure to inspect feature attribution, PDE-consistency, and latent drift.
      </div>
    );
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-slate-500">
        {"Computing attributions\u2026"}
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-full items-center justify-center px-4 text-center text-sm text-danger-400">
        {error}
      </div>
    );
  }

  if (!result) return null;

  const labels = Object.keys(result.feature_attributions).map(
    (k) => ATTRIBUTION_LABELS[k] ?? k
  );
  const values = Object.values(result.feature_attributions);

  return (
    <div className="flex h-full flex-col gap-4 px-4 py-3">
      <div className="grid grid-cols-2 gap-3">
        <DiagnosticStat
          label="PDE residual norm"
          value={result.pde_residual_norm.toFixed(4)}
          hint="Lower is more physically consistent"
        />
        <DiagnosticStat
          label="Latent drift score"
          value={
            result.latent_drift_score === null ? "n/a" : result.latent_drift_score.toFixed(2)
          }
          hint={
            result.latent_drift_score === null
              ? "No training reference in this model"
              : "RMS z-score vs training structures (0 = typical, >3 = unusual)"
          }
        />
      </div>

      <div className="min-h-0 flex-1">
        <div className="mb-2 text-xs font-medium text-slate-400">
          Feature attribution (|gradient|, structure {structureId})
        </div>
        <div className="h-40">
          <Bar
            data={{
              labels,
              datasets: [
                {
                  data: values,
                  backgroundColor: "#2dd4bf",
                  borderRadius: 4,
                  barThickness: 24,
                },
              ],
            }}
            options={{
              indexAxis: "y",
              responsive: true,
              maintainAspectRatio: false,
              plugins: { legend: { display: false } },
              scales: {
                x: { ticks: { color: "#94a3b8", font: { size: 10 } }, grid: { color: "#1f2e3a" } },
                y: { ticks: { color: "#cbd5e1", font: { size: 11 } }, grid: { display: false } },
              },
            }}
          />
        </div>
      </div>
    </div>
  );
}

function DiagnosticStat({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="rounded-lg border border-base-700 bg-base-800 px-3 py-2">
      <div className="text-[11px] text-slate-400">{label}</div>
      <div className="font-mono text-lg text-slate-100">{value}</div>
      <div className="text-[10px] text-slate-500">{hint}</div>
    </div>
  );
}
