"use client";

import type { ScenarioControls } from "@/lib/types";

interface ControlPanelProps {
  controls: ScenarioControls;
  onChange: (controls: ScenarioControls) => void;
  onRunInference: () => void;
  isLoading: boolean;
  streamingEnabled: boolean;
  onToggleStreaming: (enabled: boolean) => void;
  streamStatus: "connecting" | "open" | "closed" | "error" | "idle";
}

interface SliderConfig {
  key: keyof ScenarioControls;
  label: string;
  min: number;
  max: number;
  step: number;
  format?: (v: number) => string;
}

const SLIDERS: SliderConfig[] = [
  { key: "numStructures", label: "Structures in batch", min: 4, max: 80, step: 1 },
  { key: "sequenceLength", label: "InSAR acquisitions (weeks)", min: 8, max: 104, step: 1 },
  {
    key: "skeletalStorageLog10",
    label: "Skeletal storage S_k (log\u2081\u2080)",
    min: -6,
    max: -2,
    step: 0.1,
    format: (v) => `1e${v.toFixed(1)}`,
  },
  {
    key: "meanRainfallMm",
    label: "Mean weekly rainfall (mm)",
    min: 0,
    max: 150,
    step: 1,
    format: (v) => `${v} mm`,
  },
  {
    key: "rainfallVolatility",
    label: "Rainfall volatility (\u03c3, mm)",
    min: 0,
    max: 80,
    step: 1,
    format: (v) => `${v} mm`,
  },
  {
    key: "meanDesignCapacityM3PerDay",
    label: "Mean design recharge capacity",
    min: 20,
    max: 500,
    step: 5,
    format: (v) => `${v} m\u00b3/day`,
  },
  {
    key: "rechargeEfficiencyThreshold",
    label: "Underperformance threshold",
    min: 0.05,
    max: 0.9,
    step: 0.01,
    format: (v) => `${Math.round(v * 100)}%`,
  },
];

const STREAM_STATUS_STYLE: Record<ControlPanelProps["streamStatus"], string> = {
  idle: "bg-base-700 text-slate-300",
  connecting: "bg-warn-500/20 text-warn-400",
  open: "bg-accent-500/20 text-accent-400",
  closed: "bg-base-700 text-slate-300",
  error: "bg-danger-500/20 text-danger-400",
};

export default function ControlPanel({
  controls,
  onChange,
  onRunInference,
  isLoading,
  streamingEnabled,
  onToggleStreaming,
  streamStatus,
}: ControlPanelProps) {
  const update = (key: keyof ScenarioControls, value: number) => {
    onChange({ ...controls, [key]: value });
  };

  const randomizeSeed = () => {
    onChange({ ...controls, seed: Math.floor(Math.random() * 1_000_000) });
  };

  return (
    <div className="panel flex h-full flex-col">
      <div className="panel-header">
        <h2 className="text-sm font-semibold text-slate-200">Boundary Conditions</h2>
        <span className={`stat-pill ${STREAM_STATUS_STYLE[streamStatus]}`}>
          {streamingEnabled ? streamStatus : "batch mode"}
        </span>
      </div>

      <div className="flex-1 space-y-5 overflow-y-auto px-4 py-4">
        {SLIDERS.map((slider) => {
          const value = controls[slider.key] as number;
          return (
            <div key={slider.key}>
              <div className="mb-1.5 flex items-center justify-between text-xs text-slate-400">
                <span>{slider.label}</span>
                <span className="font-mono text-slate-200">
                  {(slider.format ?? ((v: number) => v.toString()))(value)}
                </span>
              </div>
              <input
                type="range"
                min={slider.min}
                max={slider.max}
                step={slider.step}
                value={value}
                onChange={(e) => update(slider.key, Number(e.target.value))}
              />
            </div>
          );
        })}

        <button
          type="button"
          onClick={randomizeSeed}
          className="w-full rounded-lg border border-base-600 bg-base-800 px-3 py-2 text-xs text-slate-300 transition hover:border-accent-500 hover:text-accent-400"
        >
          Randomize scenario seed ({controls.seed})
        </button>
      </div>

      <div className="space-y-3 border-t border-base-700 px-4 py-4">
        <label className="flex items-center justify-between text-xs text-slate-300">
          <span>Live streaming (WebSocket)</span>
          <input
            type="checkbox"
            checked={streamingEnabled}
            onChange={(e) => onToggleStreaming(e.target.checked)}
            className="h-4 w-4 accent-accent-500"
          />
        </label>

        <button
          type="button"
          onClick={onRunInference}
          disabled={isLoading}
          className="w-full rounded-lg bg-accent-600 px-3 py-2.5 text-sm font-medium text-base-950 transition hover:bg-accent-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isLoading ? "Running inference\u2026" : "Run inference"}
        </button>
      </div>
    </div>
  );
}
