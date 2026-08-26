import { useEffect, useState } from "react";
import "./App.css";
import type { PredictionLog, RunPrediction } from "./types";

// Operator console: reads a JSON prediction log written by the C++ edge
// scorer (data/eval_runs/<run_id>_pred.json in the full pipeline; this demo
// reads a small curated sample of four real eval runs from
// public/sample_predictions.json, one of each outcome: caught upset, missed
// upset, false alarm, clean healthy run) and shows, for every confirmed
// prediction, which channels contributed most.
//
// Contribution score = mean absolute z-score over the scored window (see
// cpp/include/contribution.hpp). It is a magnitude-of-deviation-from-baseline
// ranking, not a gradient-based attribution.

function outcomeLabel(run: RunPrediction): { text: string; className: string } {
  const confirmed = run.confirmed_prediction_minute;
  if (run.is_upset_ground_truth) {
    if (confirmed !== null && run.alarm_minute !== null && confirmed < run.alarm_minute) {
      const lead = run.alarm_minute - confirmed;
      return { text: `caught, ${lead} min lead`, className: "badge badge-caught" };
    }
    return { text: "missed", className: "badge badge-missed" };
  }
  return confirmed !== null
    ? { text: "false alarm", className: "badge badge-false-alarm" }
    : { text: "clean (no alarm)", className: "badge badge-clean" };
}

function RunCard({ run }: { run: RunPrediction }) {
  const outcome = outcomeLabel(run);
  const confirmedEvent = run.events.find((e) => e.minute === run.confirmed_prediction_minute);
  const lastEvent = run.events.length > 0 ? run.events[run.events.length - 1] : undefined;
  const shownEvent = confirmedEvent ?? lastEvent;

  return (
    <div className="run-card">
      <div className="run-card-header">
        <span className="run-id">{run.run_id}</span>
        <span className={outcome.className}>{outcome.text}</span>
      </div>
      <div className="run-meta">
        <span>ground truth: {run.is_upset_ground_truth ? "upset" : "healthy"}</span>
        {run.t0 !== null && <span>ramp start t0={run.t0} min</span>}
        {run.alarm_minute !== null && <span>hard alarm at {run.alarm_minute} min</span>}
        <span>
          confirmed at{" "}
          {run.confirmed_prediction_minute !== null ? `${run.confirmed_prediction_minute} min` : "never"}
        </span>
      </div>

      {shownEvent ? (
        <div className="contributors">
          <div className="contributors-title">
            top contributing tags at minute {shownEvent.minute} (p={shownEvent.probability.toFixed(3)})
          </div>
          <ul>
            {shownEvent.contributors.map((c) => (
              <li key={c.channel}>
                <span className="channel-name">{c.channel}</span>
                <span className="channel-bar-track">
                  <span
                    className="channel-bar-fill"
                    style={{ width: `${Math.min(100, (c.score / 3) * 100)}%` }}
                  />
                </span>
                <span className="channel-score">{c.score.toFixed(2)} sigma</span>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <div className="contributors-empty">no threshold-crossing minutes in this run</div>
      )}
    </div>
  );
}

export default function App() {
  const [log, setLog] = useState<PredictionLog | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/sample_predictions.json")
      .then((r) => {
        if (!r.ok) throw new Error(`fetch failed: ${r.status}`);
        return r.json();
      })
      .then((data: PredictionLog) => setLog(data))
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <div className="app">
      <header>
        <h1>Early-Warning Alarm Prediction Console</h1>
        <p className="subtitle">
          Reads the C++ edge scorer's JSON prediction log. Contribution scores are a
          magnitude-of-deviation-from-baseline ranking, not gradient-based attribution.
        </p>
      </header>

      {error && <div className="error">failed to load prediction log: {error}</div>}

      {log && (
        <>
          <section className="summary">
            <div className="summary-stat">
              <div className="summary-value">
                {log.eval_summary.n_caught}/{log.eval_summary.n_upset}
              </div>
              <div className="summary-label">upsets caught</div>
            </div>
            <div className="summary-stat">
              <div className="summary-value">
                {log.eval_summary.median_lead_minutes !== null
                  ? log.eval_summary.median_lead_minutes.toFixed(1)
                  : "n/a"}
              </div>
              <div className="summary-label">median lead (min)</div>
            </div>
            <div className="summary-stat">
              <div className="summary-value">
                {log.eval_summary.n_false_alarms}/{log.eval_summary.n_healthy}
              </div>
              <div className="summary-label">false alarms</div>
            </div>
            <div className="summary-stat">
              <div className="summary-value">{log.eval_summary.threshold}</div>
              <div className="summary-label">threshold (debounce {log.eval_summary.debounce_k})</div>
            </div>
          </section>

          <section className="runs">
            {log.runs.map((run) => (
              <RunCard key={run.run_id} run={run} />
            ))}
          </section>
        </>
      )}
    </div>
  );
}
