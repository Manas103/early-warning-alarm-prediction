export interface Contributor {
  channel: string;
  score: number;
}

export interface PredictionEvent {
  minute: number;
  probability: number;
  contributors: Contributor[];
}

export interface RunPrediction {
  run_id: string;
  is_upset_ground_truth: boolean;
  t0: number | null;
  alarm_minute: number | null;
  threshold: number;
  debounce_k: number;
  confirmed_prediction_minute: number | null;
  events: PredictionEvent[];
}

export interface EvalSummary {
  threshold: number;
  debounce_k: number;
  n_upset: number;
  n_healthy: number;
  n_caught: number;
  median_lead_minutes: number | null;
  n_false_alarms: number;
}

export interface PredictionLog {
  eval_summary: EvalSummary;
  runs: RunPrediction[];
}
