"use client";

import { useMemo, useState } from "react";
import DeckGL from "@deck.gl/react";
import { ScatterplotLayer } from "@deck.gl/layers";
import Map from "react-map-gl/maplibre";
import type { ARSTelemetryRecord, StructurePrediction } from "@/lib/types";

interface MapViewProps {
  records: ARSTelemetryRecord[];
  predictions: ReadonlyMap<string, StructurePrediction>;
  selectedStructureId: string | null;
  onSelectStructure: (structureId: string) => void;
}

interface PointDatum {
  structure_id: string;
  longitude: number;
  latitude: number;
  efficiency: number | null;
  underperforming: boolean | null;
  rank: number | null;
  percentile: number | null;
  note: string;
  method: StructurePrediction["method"] | null;
  hasPrediction: boolean;
}

// No Mapbox token required: this style is served publicly by the MapLibre
// project as a lightweight vector basemap, which keeps the dashboard
// runnable out of the box. Swap for a Mapbox style + access token by
// setting NEXT_PUBLIC_MAP_STYLE_URL if you prefer Mapbox's basemap tiles.
const DEFAULT_STYLE_URL =
  process.env.NEXT_PUBLIC_MAP_STYLE_URL ?? "https://demotiles.maplibre.org/style.json";

function colorForPoint(point: PointDatum): [number, number, number, number] {
  if (!point.hasPrediction) return [100, 116, 139, 160]; // slate-500, no prediction yet
  if (point.efficiency === null) {
    // InSAR only (author decision 5): relative ranking, no percentage.
    return (point.percentile ?? 1) < 1 / 3 ? [168, 85, 247, 210] : [96, 165, 250, 210];
  }
  if (point.underperforming) return [239, 68, 68, 210]; // danger-500
  if (point.efficiency < 0.6) return [245, 158, 11, 210]; // warn-500
  return [45, 212, 191, 210]; // accent-500
}

export default function MapView({
  records,
  predictions,
  selectedStructureId,
  onSelectStructure,
}: MapViewProps) {
  const [hoverInfo, setHoverInfo] = useState<{ x: number; y: number; point: PointDatum } | null>(
    null
  );

  const points: PointDatum[] = useMemo(
    () =>
      records.map((r) => {
        const pred = predictions.get(r.design.structure_id);
        return {
          structure_id: r.design.structure_id,
          longitude: r.design.longitude,
          latitude: r.design.latitude,
          efficiency: pred?.recharge_efficiency ?? null,
          underperforming: pred?.is_underperforming ?? null,
          rank: pred?.relative_recharge_rank ?? null,
          percentile: pred?.relative_recharge_percentile ?? null,
          note: pred?.assessment_note ?? "",
          method: pred?.method ?? null,
          hasPrediction: pred !== undefined,
        };
      }),
    [records, predictions]
  );

  const initialViewState = useMemo(() => {
    if (points.length === 0) {
      return { longitude: 77.59, latitude: 12.97, zoom: 8, pitch: 0, bearing: 0 };
    }
    const lons = points.map((p) => p.longitude);
    const lats = points.map((p) => p.latitude);
    return {
      longitude: (Math.min(...lons) + Math.max(...lons)) / 2,
      latitude: (Math.min(...lats) + Math.max(...lats)) / 2,
      zoom: 9,
      pitch: 0,
      bearing: 0,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [records.length]);

  const layers = [
    new ScatterplotLayer<PointDatum>({
      id: "ars-structures",
      data: points,
      pickable: true,
      opacity: 0.9,
      stroked: true,
      filled: true,
      radiusMinPixels: 6,
      radiusMaxPixels: 18,
      lineWidthMinPixels: 1,
      getPosition: (d) => [d.longitude, d.latitude],
      getRadius: (d) => (d.structure_id === selectedStructureId ? 220 : 140),
      getFillColor: (d) => colorForPoint(d),
      getLineColor: (d) =>
        d.structure_id === selectedStructureId ? [255, 255, 255, 255] : [10, 15, 20, 180],
      getLineWidth: (d) => (d.structure_id === selectedStructureId ? 3 : 1),
      onHover: (info) => {
        if (info.object) {
          setHoverInfo({ x: info.x, y: info.y, point: info.object as PointDatum });
        } else {
          setHoverInfo(null);
        }
      },
      onClick: (info) => {
        if (info.object) onSelectStructure((info.object as PointDatum).structure_id);
      },
    }),
  ];

  return (
    <div className="relative h-full w-full">
      <DeckGL initialViewState={initialViewState} controller layers={layers} style={{ borderRadius: "0.75rem" }}>
        <Map mapStyle={DEFAULT_STYLE_URL} reuseMaps />
      </DeckGL>

      {hoverInfo && (
        <div
          className="pointer-events-none absolute z-10 rounded-lg border border-base-600 bg-base-900/95 px-3 py-2 text-xs shadow-panel"
          style={{ left: hoverInfo.x + 12, top: hoverInfo.y + 12 }}
        >
          <div className="font-semibold text-slate-100">{hoverInfo.point.structure_id}</div>
          {hoverInfo.point.method === "neural_experimental" && (
            <div className="text-warn-400">Neural network (experimental)</div>
          )}
          {hoverInfo.point.hasPrediction && hoverInfo.point.efficiency === null ? (
            <>
              <div className="text-slate-400">
                Relative recharge rank:{" "}
                <span className="font-mono text-slate-200">
                  {hoverInfo.point.rank} of {predictions.size}
                </span>
              </div>
              <div className="max-w-[16rem] text-slate-500">{hoverInfo.point.note}</div>
            </>
          ) : hoverInfo.point.hasPrediction ? (
            <>
              <div className="text-slate-400">
                Efficiency:{" "}
                <span className="font-mono text-slate-200">
                  {((hoverInfo.point.efficiency ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className={hoverInfo.point.underperforming ? "text-danger-400" : "text-accent-400"}>
                {hoverInfo.point.underperforming ? "Underperforming" : "Nominal"}
              </div>
              <div className="max-w-[16rem] text-slate-500">{hoverInfo.point.note}</div>
            </>
          ) : (
            <div className="text-slate-500">No prediction yet</div>
          )}
        </div>
      )}

      <div className="absolute bottom-3 left-3 z-10 flex gap-3 rounded-lg border border-base-600 bg-base-900/90 px-3 py-2 text-[11px] text-slate-300">
        <LegendDot color="bg-accent-500" label="Nominal" />
        <LegendDot color="bg-warn-500" label="Reduced efficiency" />
        <LegendDot color="bg-danger-500" label="Underperforming" />
        <LegendDot color="bg-blue-400" label="No wells: upper 2/3 (relative)" />
        <LegendDot color="bg-purple-500" label="No wells: lowest 1/3 (relative)" />
        <LegendDot color="bg-slate-500" label="No prediction" />
      </div>
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className={`h-2 w-2 rounded-full ${color}`} />
      {label}
    </span>
  );
}
