// Mirrors src/data/schemas.py. Keep these in sync with the backend contract
// -- they are intentionally structural copies, not generated, so a schema
// change on the backend should be reflected here deliberately.

export type DataQualityFlag =
  | "ok"
  | "low_coherence"
  | "missing"
  | "interpolated"
  | "out_of_bounds_clamped"
  | "anomaly_no_rainfall";

export interface ARSDesignParameters {
  structure_id: string;
  name?: string | null;
  latitude: number;
  longitude: number;
  height_m: number;
  storage_volume_m3: number;
  catchment_area_km2: number;
  construction_date?: string | null;
  design_recharge_capacity_m3_per_day: number;
  /** DEPRECATED, unused by the model (review finding 2). */
  biot_coefficient_alpha?: number;
}

export interface SARObservation {
  structure_id: string;
  timestamp: string;
  /** LOS displacement, positive = toward the satellite (PROVISIONAL). */
  los_displacement_cm: number;
  coherence: number;
  /** Radar incidence angle [deg], 20-50, from Sentinel-1 metadata. Required. */
  incidence_angle_deg: number;
  track_direction: "ascending" | "descending";
  quality_flag: DataQualityFlag;
}

export interface RainfallObservation {
  structure_id: string;
  timestamp: string;
  rainfall_mm: number;
  quality_flag: DataQualityFlag;
}

export interface WellHeadObservation {
  structure_id: string;
  timestamp: string;
  hydraulic_head_m: number;
  is_boundary_node: boolean;
}

export interface ARSTelemetryRecord {
  design: ARSDesignParameters;
  sar_series: SARObservation[];
  rainfall_series: RainfallObservation[];
  well_series: WellHeadObservation[];
}

export interface GraphEdge {
  source_id: string;
  target_id: string;
  weight: number;
}

export interface TelemetryBatch {
  batch_id: string;
  records: ARSTelemetryRecord[];
  edges: GraphEdge[];
}

export interface PredictionRequest {
  batch: TelemetryBatch;
  recharge_efficiency_threshold: number;
}

export interface StructurePrediction {
  structure_id: string;
  /** DEPRECATED name: holds transmissivity T [m2/day], not K. Use transmissivity_m2_per_day. */
  hydraulic_conductivity_m_per_day: number;
  transmissivity_m2_per_day: number;
  predicted_recharge_series_m3_per_day: number[];
  predicted_displacement_cm: number[];
  recharge_efficiency: number;
  is_underperforming: boolean;
  confidence_interval_m3_per_day: [number, number];
}

export interface PredictionResponse {
  batch_id: string;
  model_version: string;
  predictions: StructurePrediction[];
  generated_at: string;
}

export interface ExplainabilityResponse {
  structure_id: string;
  feature_attributions: Record<string, number>;
  latent_drift_score: number;
  pde_residual_norm: number;
}

export interface HealthResponse {
  status: "ok" | "starting";
  model_version: string;
  backend: string | null;
  active_provider: string | null;
  environment: string;
}

export interface WSPredictionMessage {
  type: "prediction";
  batch_id: string;
  model_version: string;
  predictions: StructurePrediction[];
  generated_at: string;
}

export interface WSErrorMessage {
  type: "error";
  detail: string;
  errors?: string;
}

export interface WSHeartbeatMessage {
  type: "heartbeat";
  ts: number;
}

export type WSMessage = WSPredictionMessage | WSErrorMessage | WSHeartbeatMessage;

/** Client-side scenario controls used to synthesize a demo TelemetryBatch. */
export interface ScenarioControls {
  numStructures: number;
  sequenceLength: number;
  /** log10 of the skeletal storage coefficient S_k (dimensionless), -6..-2. */
  skeletalStorageLog10: number;
  meanRainfallMm: number;
  rainfallVolatility: number;
  meanDesignCapacityM3PerDay: number;
  rechargeEfficiencyThreshold: number;
  seed: number;
}
