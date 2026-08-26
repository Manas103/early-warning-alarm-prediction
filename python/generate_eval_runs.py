"""Phase 1 of the eval: generate the 40 seeded-upset + 40 healthy eval runs
(seeds disjoint from the training seed range in train.py), write each run's
raw channel matrix as a flat float32 binary the C++ scorer can read, write a
manifest of ground truth (t0, alarm_minute, etc.), and generate a bash
script that invokes edge_scorer once per run (used from WSL, see
scripts/wsl_run_eval.sh).
"""
from __future__ import annotations

import json
from pathlib import Path

from simulate import RUN_LENGTH_MIN, make_dataset

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "eval_runs"
SCRIPTS_DIR = ROOT / "scripts"

EVAL_SEED_OFFSET = 900000   # disjoint from train.py's TRAIN_SEED_OFFSET=1..140 range
N_EVAL_UPSET = 40
N_EVAL_HEALTHY = 40

THRESHOLD = 0.90
DEBOUNCE_K = 3


def to_wsl_path(windows_path: Path) -> str:
    s = str(windows_path.resolve())
    s = s.replace("\\", "/")
    assert s[1] == ":", s
    drive = s[0].lower()
    return f"/mnt/{drive}{s[2:]}"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    runs = make_dataset(N_EVAL_UPSET, N_EVAL_HEALTHY, seed_offset=EVAL_SEED_OFFSET)
    manifest = []
    script_lines = ["#!/bin/bash", "set -e", 'cd "$HOME/build/early-warning-alarm-prediction/cpp/build"', ""]

    weights_wsl = to_wsl_path(ROOT / "weights" / "weights.bin")
    norm_wsl = to_wsl_path(ROOT / "weights" / "norm.txt")

    for i, run in enumerate(runs):
        kind = "upset" if run.is_upset else "healthy"
        run_id = f"{kind}_{i:03d}_seed{run.seed}"
        bin_path = DATA_DIR / f"{run_id}.bin"
        run.channels.tofile(bin_path)
        out_json = DATA_DIR / f"{run_id}_pred.json"

        manifest.append({
            "run_id": run_id,
            "is_upset": run.is_upset,
            "seed": run.seed,
            "t0": run.t0,
            "alarm_minute": run.alarm_minute,
            "primary_channel": run.primary_channel,
            "secondary_channels": list(run.secondary_channels),
            "T": RUN_LENGTH_MIN,
            "pred_json": str(out_json.relative_to(ROOT)).replace("\\", "/"),
        })

        run_bin_wsl = to_wsl_path(bin_path)
        out_json_wsl = to_wsl_path(out_json)
        script_lines.append(
            f'./edge_scorer "{weights_wsl}" "{norm_wsl}" "{run_bin_wsl}" {RUN_LENGTH_MIN} '
            f'{THRESHOLD} {DEBOUNCE_K} "{out_json_wsl}" {run_id}'
        )

    (DATA_DIR / "manifest.json").write_text(json.dumps({
        "threshold": THRESHOLD,
        "debounce_k": DEBOUNCE_K,
        "n_upset": N_EVAL_UPSET,
        "n_healthy": N_EVAL_HEALTHY,
        "run_length_min": RUN_LENGTH_MIN,
        "runs": manifest,
    }, indent=2))

    script_path = SCRIPTS_DIR / "_wsl_run_eval_generated.sh"
    with open(script_path, "w", newline="\n") as f:
        f.write("\n".join(script_lines) + "\n")
    print(f"wrote {len(runs)} run binaries + manifest to {DATA_DIR}")
    print(f"wrote scorer-invocation script to {script_path}")


if __name__ == "__main__":
    main()
