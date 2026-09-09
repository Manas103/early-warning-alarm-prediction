"""Eval harness for the chamber-to-chamber drift check (see
chamber_drift.py). Not one of the six resume claims measured for the
excursion classifier; a separate, honestly-reported finding.

Three disjoint seed blocks:
  - calibration (CALIBRATION_SEED_OFFSET): 40 clean multi-chamber runs used
    only to estimate the per-quantity chamber-matching sigma.
  - eval (DRIFT_EVAL_SEED_OFFSET): 40 seeded-drift + 40 clean runs actually
    scored against that calibrated threshold.

Run: python eval_chamber_drift.py
Writes docs/chamber_drift_output.txt and docs/chamber_drift_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from chamber_drift import calibrate_sibling_spread, caught_the_seeded_drift, detect_drift
from simulate import QUANTITIES, make_chamber_drift_dataset

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"

CALIBRATION_SEED_OFFSET = 970000   # disjoint from train (1..), eval (900000..) ranges
DRIFT_EVAL_SEED_OFFSET = 980000
N_CALIBRATION_CLEAN = 40
N_DRIFT_SEEDED = 40
N_DRIFT_CLEAN = 40
K_SIGMA = 4.5


def main():
    calibration_runs = make_chamber_drift_dataset(
        n_drifted=0, n_clean=N_CALIBRATION_CLEAN, seed_offset=CALIBRATION_SEED_OFFSET
    )
    sibling_spread = calibrate_sibling_spread(calibration_runs)

    eval_runs = make_chamber_drift_dataset(
        n_drifted=N_DRIFT_SEEDED, n_clean=N_DRIFT_CLEAN, seed_offset=DRIFT_EVAL_SEED_OFFSET
    )
    drifted_runs = [r for r in eval_runs if r.drifted]
    clean_runs = [r for r in eval_runs if not r.drifted]

    lines = []
    lines.append("Chamber-to-chamber drift check: eval harness results")
    lines.append(f"k_sigma={K_SIGMA}  calibration_runs={N_CALIBRATION_CLEAN} (clean only)")
    lines.append(f"sibling_spread (per quantity, engineering units): "
                 + ", ".join(f"{q}={s:.4g}" for q, s in zip(QUANTITIES, sibling_spread)))
    lines.append("")

    n_caught = 0
    lines.append("--- seeded-drift runs ---")
    for run in drifted_runs:
        flags = detect_drift(run.channels, sibling_spread, K_SIGMA)
        caught = caught_the_seeded_drift(run, flags)
        n_caught += int(caught)
        flag_str = ", ".join(f"chamber{f.chamber + 1}_{f.quantity}(z={f.zscore:+.2f})" for f in flags) or "none"
        lines.append(
            f"  seed={run.seed:7d} injected=chamber{run.drift_chamber + 1}_{run.drift_quantity} "
            f"(+{run.drift_magnitude_std:.2f} sigma)  flagged=[{flag_str}]  caught={caught}"
        )

    n_false = 0
    lines.append("")
    lines.append("--- clean multi-chamber runs ---")
    for run in clean_runs:
        flags = detect_drift(run.channels, sibling_spread, K_SIGMA)
        falsely_flagged = len(flags) > 0
        n_false += int(falsely_flagged)
        flag_str = ", ".join(f"chamber{f.chamber + 1}_{f.quantity}(z={f.zscore:+.2f})" for f in flags) or "none"
        lines.append(f"  seed={run.seed:7d}  flagged=[{flag_str}]  false_flag={falsely_flagged}")

    lines.append("")
    lines.append(f"seeded drifts caught: {n_caught}/{len(drifted_runs)}")
    lines.append(f"false flags over clean multi-chamber runs: {n_false}/{len(clean_runs)}")

    report = "\n".join(lines)
    print(report)
    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "chamber_drift_output.txt").write_text(report + "\n")

    summary = {
        "k_sigma": K_SIGMA,
        "n_calibration_clean": N_CALIBRATION_CLEAN,
        "sibling_spread": {q: float(s) for q, s in zip(QUANTITIES, sibling_spread)},
        "n_drift_seeded": len(drifted_runs),
        "n_drift_caught": n_caught,
        "n_clean": len(clean_runs),
        "n_false_flags": n_false,
    }
    (DOCS_DIR / "chamber_drift_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
