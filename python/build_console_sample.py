"""Builds react-console/public/sample_predictions.json: a small, curated
subset of real eval_runs predictions (one caught upset, one missed upset,
one false alarm, one clean healthy run), stripped of the large
per-minute probability/scored_minutes arrays, for the operator console demo
data. Not part of the scoring pipeline; run after compute_eval_metrics.py.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "eval_runs"
OUT_PATH = ROOT / "react-console" / "public" / "sample_predictions.json"


def slim(pred: dict, ground_truth: dict) -> dict:
    return {
        "run_id": pred["run_id"],
        "is_upset_ground_truth": ground_truth["is_upset"],
        "t0": ground_truth.get("t0"),
        "alarm_minute": ground_truth.get("alarm_minute"),
        "threshold": pred["threshold"],
        "debounce_k": pred["debounce_k"],
        "confirmed_prediction_minute": pred["confirmed_prediction_minute"],
        "events": pred["events"],
    }


def main():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    by_id = {r["run_id"]: r for r in manifest["runs"]}

    summary = json.loads((ROOT / "docs" / "eval_summary.json").read_text())

    caught = missed = false_alarm = clean_healthy = None
    for r in manifest["runs"]:
        pred = json.loads((ROOT / r["pred_json"]).read_text())
        confirmed = pred["confirmed_prediction_minute"]
        if r["is_upset"] and confirmed is not None and confirmed < r["alarm_minute"] and caught is None:
            caught = (pred, r)
        elif r["is_upset"] and (confirmed is None or confirmed >= r["alarm_minute"]) and missed is None:
            missed = (pred, r)
        elif not r["is_upset"] and confirmed is not None and false_alarm is None:
            false_alarm = (pred, r)
        elif not r["is_upset"] and confirmed is None and clean_healthy is None:
            clean_healthy = (pred, r)

    picks = [p for p in (caught, missed, false_alarm, clean_healthy) if p is not None]
    out = {
        "eval_summary": summary,
        "runs": [slim(pred, gt) for pred, gt in picks],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"wrote {len(picks)} sample runs to {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
