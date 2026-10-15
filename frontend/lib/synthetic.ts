import type {
  ARSTelemetryRecord,
  RainfallObservation,
  SARObservation,
  ScenarioControls,
  TelemetryBatch,
} from "./types";

/** Deterministic pseudo-random generator (mulberry32) so a given seed reproduces a scenario. */
function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rng: () => number, mean: number, std: number): number {
  const u1 = Math.max(rng(), 1e-9);
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + std * z;
}

/**
 * Generates a physically-plausible synthetic `TelemetryBatch` client-side,
 * mirroring `src/training/train.py::make_synthetic_records`. Lets the
 * dashboard's control panel drive "what-if" scenarios (rainfall intensity,
 * Biot coupling, structure count) without requiring a live telemetry feed --
 * the same batch shape is sent to the real `/api/v1/predict` endpoint.
 */
export function generateScenarioBatch(controls: ScenarioControls): TelemetryBatch {
  const rng = makeRng(controls.seed);
  const records: ARSTelemetryRecord[] = [];
  const baseTimeMs = Date.UTC(2024, 0, 1);
  const weekMs = 7 * 24 * 60 * 60 * 1000;

  for (let i = 0; i < controls.numStructures; i++) {
    const structureId = `ARS-${String(i).padStart(4, "0")}`;
    const alpha = Math.min(
      0.95,
      Math.max(0.1, gaussian(rng, controls.meanBiotAlpha, 0.08))
    );
    const designCapacity = Math.max(
      5,
      gaussian(rng, controls.meanDesignCapacityM3PerDay, controls.meanDesignCapacityM3PerDay * 0.3)
    );
    const trueK = Math.max(0.05, gaussian(rng, 1.5, 1.0));

    const sarSeries: SARObservation[] = [];
    const rainfallSeries: RainfallObservation[] = [];
    let head = 0;

    for (let t = 0; t < controls.sequenceLength; t++) {
      const ts = new Date(baseTimeMs + t * weekMs).toISOString();
      const rainfall = Math.max(
        0,
        gaussian(rng, controls.meanRainfallMm, controls.rainfallVolatility)
      );
      const recharge = Math.min(rainfall * 0.5, designCapacity);
      head = 0.9 * head + recharge / (1 + trueK);
      const displacement = -alpha * head * 0.01 + gaussian(rng, 0, 0.05);
      const coherence = Math.min(1, Math.max(0, gaussian(rng, 0.85, 0.15)));

      sarSeries.push({
        structure_id: structureId,
        timestamp: ts,
        los_displacement_cm: displacement,
        coherence,
        track_direction: "descending",
        quality_flag: "ok",
      });
      rainfallSeries.push({
        structure_id: structureId,
        timestamp: ts,
        rainfall_mm: rainfall,
        quality_flag: "ok",
      });
    }

    records.push({
      design: {
        structure_id: structureId,
        name: `Recharge Structure ${i}`,
        latitude: 12.9 + gaussian(rng, 0, 0.4),
        longitude: 77.5 + gaussian(rng, 0, 0.4),
        height_m: Math.max(1, gaussian(rng, 6, 3)),
        storage_volume_m3: Math.max(200, gaussian(rng, 5000, 3000)),
        catchment_area_km2: Math.max(0.2, gaussian(rng, 6, 4)),
        construction_date: null,
        design_recharge_capacity_m3_per_day: designCapacity,
        biot_coefficient_alpha: alpha,
      },
      sar_series: sarSeries,
      rainfall_series: rainfallSeries,
      well_series: [],
    });
  }

  return {
    batch_id: crypto.randomUUID(),
    records,
    edges: [],
  };
}

export const DEFAULT_SCENARIO: ScenarioControls = {
  numStructures: 24,
  sequenceLength: 52,
  meanBiotAlpha: 0.6,
  meanRainfallMm: 40,
  rainfallVolatility: 25,
  meanDesignCapacityM3PerDay: 150,
  rechargeEfficiencyThreshold: 0.3,
  seed: 42,
};
