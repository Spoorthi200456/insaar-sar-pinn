// Mirrors src/data/schemas.py. Keep these in sync with the backend contract
// -- they are intentionally structural copies, not generated, so a schema
// change on the backend should be reflected here deliberately.

export type DataQualityFlag =
  | "ok"
  | "low_coherence"
  | "missing"
  | "interpolated"
  | "out_of_bounds_clamped"
  | "step_out_of_bounds"
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

/**
 * Assessment method (review 5). "ols_baseline" (DEFAULT): least squares on a
 * linear-reservoir model, no neural network. "neural_experimental": the
 * physics-informed network, which ranked unseen synthetic structures no better
 * than chance in reviews 3-4.
 */
export type AssessmentMethod = "ols_baseline" | "neural_experimental";
export const DEFAULT_METHOD: AssessmentMethod = "ols_baseline";

export interface PredictionRequest {
  batch: TelemetryBatch;
  recharge_efficiency_threshold: number;
  method?: AssessmentMethod;
}

export interface StructurePrediction {
  structure_id: string;
  method: AssessmentMethod;
  method_note: string;
  /** DEPRECATED name: holds transmissivity T [m2/day], not K. Null for ols_baseline. */
  hydraulic_conductivity_m_per_day: number | null;
  /** Null for ols_baseline (it does not estimate T). */
  transmissivity_m2_per_day: number | null;
  /** Null for ols_baseline (recharge only up to an unknown factor). */
  predicted_recharge_series_m3_per_day: number[] | null;
  predicted_displacement_cm: number[];
  /** Only with well data (author decision 5); null for InSAR-only structures. */
  recharge_efficiency: number | null;
  is_underperforming: boolean | null;
  /** Ranking score; only its order is meaningful. */
  relative_recharge_score: number;
  /** 1 = highest recharge relative to design among the submitted structures. */
  relative_recharge_rank: number;
  /** 1 = highest, 0 = lowest. */
  relative_recharge_percentile: number;
  assessment_basis: "relative_ranking_insar_only" | "well_calibrated_provisional";
  assessment_note: string;
  /** Null for ols_baseline. */
  confidence_interval_m3_per_day: [number, number] | null;
  /** False = InSAR only: absolute recharge is not identifiable. */
  has_well_data: boolean;
  well_head_rmse_m: number | null;
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
  /** RMS z-score vs the training reference; null if the model has none. */
  latent_drift_score: number | null;
  pde_residual_norm: number;
}

export interface HealthResponse {
  status: "ok" | "starting";
  model_version: string;
  backend: string | null;
  active_provider: string | null;
  environment: string;
  default_method?: AssessmentMethod;
  experimental_methods?: AssessmentMethod[];
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
