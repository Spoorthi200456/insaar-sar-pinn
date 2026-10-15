"use client";

import { useMemo } from "react";
import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Title,
  Tooltip,
} from "chart.js";
import { Line } from "react-chartjs-2";
import type { ARSTelemetryRecord, StructurePrediction } from "@/lib/types";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler);

interface TimeSeriesChartProps {
  record: ARSTelemetryRecord | null;
  prediction: StructurePrediction | null;
}

export default function TimeSeriesChart({ record, prediction }: TimeSeriesChartProps) {
  const chartData = useMemo(() => {
    if (!record) return null;

    const labels = record.sar_series.map((o) =>
      new Date(o.timestamp).toLocaleDateString(undefined, { month: "short", day: "numeric" })
    );
    const observedDisplacement = record.sar_series.map((o) => o.los_displacement_cm);
    const rainfall = record.rainfall_series.map((o) => o.rainfall_mm);

    const datasets: {
      label: string;
      data: number[];
      borderColor: string;
      backgroundColor: string;
      pointRadius: number;
      borderWidth: number;
      borderDash?: number[];
      yAxisID: string;
      tension: number;
    }[] = [
      {
        label: "Observed InSAR displacement (cm)",
        data: observedDisplacement,
        borderColor: "#5eead4",
        backgroundColor: "rgba(94, 234, 212, 0.08)",
        pointRadius: 0,
        borderWidth: 1.5,
        yAxisID: "y",
        tension: 0.25,
      },
      {
        label: "Rainfall (mm)",
        data: rainfall,
        borderColor: "#60a5fa",
        backgroundColor: "rgba(96, 165, 250, 0.08)",
        pointRadius: 0,
        borderWidth: 1,
        borderDash: [4, 3],
        yAxisID: "y1",
        tension: 0.2,
      },
    ];

    if (prediction) {
      datasets.push({
        label: "Predicted displacement (cm)",
        data: prediction.predicted_displacement_cm,
        borderColor: "#f59e0b",
        backgroundColor: "rgba(245, 158, 11, 0.08)",
        pointRadius: 0,
        borderWidth: 1.5,
        yAxisID: "y",
        tension: 0.25,
        borderDash: [2, 2],
      });
    }

    return { labels, datasets };
  }, [record, prediction]);

  if (!record || !chartData) {
    return (
      <div className="flex h-full items-center justify-center text-sm text-slate-500">
        Select a structure on the map to inspect its time series.
      </div>
    );
  }

  return (
    <Line
      data={chartData}
      options={{
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            position: "top",
            labels: { color: "#cbd5e1", boxWidth: 12, font: { size: 11 } },
          },
          tooltip: { backgroundColor: "#16212b", borderColor: "#2c4152", borderWidth: 1 },
        },
        scales: {
          x: {
            ticks: { color: "#94a3b8", maxTicksLimit: 10, font: { size: 10 } },
            grid: { color: "#1f2e3a" },
          },
          y: {
            type: "linear",
            position: "left",
            title: { display: true, text: "Displacement (cm)", color: "#94a3b8" },
            ticks: { color: "#94a3b8", font: { size: 10 } },
            grid: { color: "#1f2e3a" },
          },
          y1: {
            type: "linear",
            position: "right",
            title: { display: true, text: "Rainfall (mm)", color: "#94a3b8" },
            ticks: { color: "#94a3b8", font: { size: 10 } },
            grid: { display: false },
          },
        },
      }}
    />
  );
}
