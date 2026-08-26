"""Phase 3 of the eval: read every data/eval_runs/*_pred.json the C++
edge_scorer wrote, join against the ground truth manifest, and compute:
  - upsets caught out of 40 (confirmed_prediction_minute < alarm_minute)
  - median lead time in minutes (alarm_minute - confirmed_prediction_minute)
    over the caught upsets
  - false alarms over the 40 healthy runs (confirmed_prediction_minute is
    not null on a healthy run)
Writes a human-readable report to docs/eval_output.txt.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "eval_runs"
DOCS_DIR = ROOT / "docs"


def main():
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    runs = manifest["runs"]

    upset_results = []
    healthy_results = []

    for r in runs:
        pred_path = ROOT / r["pred_json"]
        pred = json.loads(pred_path.read_text())
        confirmed = pred["confirmed_prediction_minute"]

        if r["is_upset"]:
            caught = confirmed is not None and confirmed < r["alarm_minute"]
            lead = (r["alarm_minute"] - confirmed) if caught else None
            upset_results.append({
                "run_id": r["run_id"], "seed": r["seed"], "t0": r["t0"],
                "alarm_minute": r["alarm_minute"], "confirmed_minute": confirmed,
                "caught": caught, "lead_minutes": lead,
            })
        else:
            false_alarm = confirmed is not None
            healthy_results.append({
                "run_id": r["run_id"], "seed": r["seed"],
                "confirmed_minute": confirmed, "false_alarm": false_alarm,
            })

    n_upset = len(upset_results)
    n_healthy = len(healthy_results)
    n_caught = sum(1 for u in upset_results if u["caught"])
    leads = [u["lead_minutes"] for u in upset_results if u["caught"]]
    median_lead = statistics.median(leads) if leads else None
    n_false_alarms = sum(1 for h in healthy_results if h["false_alarm"])

    lines = []
    lines.append("Early-warning alarm prediction: eval harness results")
    lines.append(f"threshold={manifest['threshold']}  debounce_k={manifest['debounce_k']}  "
                  f"window_minutes=60  run_length_min={manifest['run_length_min']}")
    lines.append(f"n_upset_runs={n_upset}  n_healthy_runs={n_healthy}")
    lines.append("")
    lines.append(f"upsets caught: {n_caught}/{n_upset}")
    lines.append(f"median lead time (caught upsets only): "
                  f"{median_lead:.2f} minutes" if median_lead is not None else "median lead time: n/a (0 caught)")
    if leads:
        lines.append(f"lead time min/max: {min(leads)}/{max(leads)} minutes, "
                      f"mean={statistics.mean(leads):.2f}")
    lines.append(f"false alarms: {n_false_alarms}/{n_healthy}")
    lines.append("")
    lines.append("--- per-upset-run detail ---")
    for u in upset_results:
        lines.append(f"  {u['run_id']:28s} seed={u['seed']:7d} t0={u['t0']:3d} "
                      f"alarm_minute={u['alarm_minute']:3d} confirmed_minute="
                      f"{u['confirmed_minute'] if u['confirmed_minute'] is not None else 'none':>4} "
                      f"caught={u['caught']!s:5} lead={u['lead_minutes']}")
    lines.append("")
    lines.append("--- per-healthy-run detail ---")
    for h in healthy_results:
        lines.append(f"  {h['run_id']:28s} seed={h['seed']:7d} confirmed_minute="
                      f"{h['confirmed_minute'] if h['confirmed_minute'] is not None else 'none':>4} "
                      f"false_alarm={h['false_alarm']}")

    report = "\n".join(lines)
    print(report)
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "eval_output.txt").write_text(report + "\n")

    summary = {
        "threshold": manifest["threshold"],
        "debounce_k": manifest["debounce_k"],
        "n_upset": n_upset,
        "n_healthy": n_healthy,
        "n_caught": n_caught,
        "median_lead_minutes": median_lead,
        "n_false_alarms": n_false_alarms,
    }
    (DOCS_DIR / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
