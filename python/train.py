"""Train the early-warning CNN on simulated historian runs and export:

  - weights/weights.bin   raw float32 weights read by the C++ scorer
  - weights/norm.json     per-channel mean/std used to normalize inputs
  - docs/oracle_reference.json  a handful of real windows + PyTorch's exact
    output on them, used by cpp/tests/oracle_diff to prove the hand-rolled
    C++ forward pass matches PyTorch bit-for-bit (to float tolerance).

Train/eval seed ranges are disjoint by construction (see TRAIN_SEED_OFFSET
vs EVAL_SEED_OFFSET in run_eval.py) so the eval set is never seen in
training.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from features import build_training_arrays, channel_stats
from model import EarlyWarningCNN
from simulate import make_dataset

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
DOCS_DIR = ROOT / "docs"

TRAIN_SEED_OFFSET = 1          # upset seeds 1..N-1, healthy seeds 100001..
N_TRAIN_UPSET = 140
N_TRAIN_HEALTHY = 140
VAL_FRACTION = 0.2
TORCH_SEED = 1234


def export_weights(model: EarlyWarningCNN, mean: np.ndarray, std: np.ndarray) -> None:
    WEIGHTS_DIR.mkdir(exist_ok=True)
    sd = model.state_dict()

    def f32(t):
        return t.detach().cpu().numpy().astype(np.float32)

    # layout documented in cpp/include/model.hpp: conv1.w, conv1.b, conv2.w,
    # conv2.b, fc.w, fc.b, each a flat row-major float32 array with no
    # header, arrays concatenated in that exact order.
    arrays = [
        f32(sd["conv1.weight"]).reshape(-1),   # (16,24,5)
        f32(sd["conv1.bias"]).reshape(-1),     # (16,)
        f32(sd["conv2.weight"]).reshape(-1),   # (8,16,5)
        f32(sd["conv2.bias"]).reshape(-1),     # (8,)
        f32(sd["fc.weight"]).reshape(-1),      # (1,8)
        f32(sd["fc.bias"]).reshape(-1),        # (1,)
    ]
    blob = np.concatenate(arrays).astype(np.float32)
    (WEIGHTS_DIR / "weights.bin").write_bytes(blob.tobytes())

    norm = {"mean": mean.tolist(), "std": std.tolist()}
    (WEIGHTS_DIR / "norm.json").write_text(json.dumps(norm, indent=2))
    # plain-text form the C++ side reads (no JSON dependency in cpp/):
    # line 1 = channel count, then one "mean std" pair per channel
    lines = [str(len(mean))] + [f"{m:.8g} {s:.8g}" for m, s in zip(mean, std)]
    (WEIGHTS_DIR / "norm.txt").write_text("\n".join(lines) + "\n")

    shapes = {
        "conv1_weight": list(sd["conv1.weight"].shape),
        "conv1_bias": list(sd["conv1.bias"].shape),
        "conv2_weight": list(sd["conv2.weight"].shape),
        "conv2_bias": list(sd["conv2.bias"].shape),
        "fc_weight": list(sd["fc.weight"].shape),
        "fc_bias": list(sd["fc.bias"].shape),
        "total_floats": int(blob.shape[0]),
    }
    (WEIGHTS_DIR / "shapes.json").write_text(json.dumps(shapes, indent=2))
    print(f"exported {blob.shape[0]} float32 weights to {WEIGHTS_DIR/'weights.bin'} "
          f"({blob.nbytes} bytes)")


def export_oracle_reference(model: EarlyWarningCNN, val_runs, mean: np.ndarray, std: np.ndarray,
                             n: int = 16) -> None:
    """Dump n RAW (unnormalized) windows and PyTorch's exact logit/probability
    on each (PyTorch is fed the normalized version, same as training), so
    the C++ oracle_diff test -- which normalizes internally, exactly like
    the edge scorer does on a real run -- can replay the same raw inputs and
    compare outputs on equal footing. Written both as JSON (human-readable,
    for the README/docs) and as flat binary/text (what
    cpp/tests/oracle_diff_main.cpp actually reads, no JSON parser in C++)."""
    from model import WINDOW_MINUTES
    from features import build_windows

    DOCS_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)

    raw_windows = []
    for run in val_runs:
        norm_channels = (run.channels - mean) / std
        T = run.channels.shape[0]
        for t in range(WINDOW_MINUTES - 1, T):
            if run.is_upset and run.alarm_minute is not None and t >= run.alarm_minute:
                continue
            raw_windows.append(run.channels[t - WINDOW_MINUTES + 1 : t + 1, :].copy())
        if len(raw_windows) >= n * 20:
            break
    idx = rng.choice(len(raw_windows), size=n, replace=False)
    raw_sample = np.stack([raw_windows[i] for i in idx]).astype(np.float32)  # (n, 60, 24)
    norm_sample = ((raw_sample - mean) / std).astype(np.float32)

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(norm_sample)).numpy()
        probs = 1.0 / (1.0 + np.exp(-logits))

    raw_sample.tofile(DOCS_DIR / "oracle_windows.bin")
    with open(DOCS_DIR / "oracle_targets.txt", "w") as f:
        f.write(f"{n}\n")
        for i in range(n):
            f.write(f"{float(logits[i]):.8g} {float(probs[i]):.8g}\n")

    # (no separate human-readable JSON dump: oracle_windows.bin +
    # oracle_targets.txt are what cpp/tests/oracle_diff_main.cpp reads, and
    # a full-precision JSON dump of the same 16 windows was ~10x larger for
    # no extra test coverage; oracle_targets.txt alone is human-readable
    # enough to sanity check the logits/probabilities by eye)
    print(f"wrote {n} oracle reference windows to {DOCS_DIR/'oracle_windows.bin'} / "
          f"oracle_targets.txt")


def main():
    torch.manual_seed(TORCH_SEED)
    np.random.seed(TORCH_SEED)

    print("generating training runs...")
    runs = make_dataset(N_TRAIN_UPSET, N_TRAIN_HEALTHY, seed_offset=TRAIN_SEED_OFFSET)
    rng = np.random.default_rng(TORCH_SEED)
    rng.shuffle(runs)
    n_val_runs = int(len(runs) * VAL_FRACTION)
    val_runs, train_runs = runs[:n_val_runs], runs[n_val_runs:]

    mean, std = channel_stats(train_runs)
    x_train, y_train = build_training_arrays(train_runs, mean, std)
    x_val, y_val = build_training_arrays(val_runs, mean, std)
    print(f"train windows: {x_train.shape[0]} (pos={y_train.sum():.0f}), "
          f"val windows: {x_val.shape[0]} (pos={y_val.sum():.0f})")

    model = EarlyWarningCNN()
    # class weight: positives are a minority (only windows after t0), weight
    # them up so the model doesn't just predict "healthy" everywhere.
    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)

    n_epochs = 25
    batch_size = 256
    n_samples = x_train_t.shape[0]
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(n_samples)
        epoch_loss = 0.0
        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = x_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]
        epoch_loss /= n_samples

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_probs = torch.sigmoid(val_logits).numpy()
            val_pred = (val_probs >= 0.5).astype(np.float32)
            val_acc = (val_pred == y_val).mean()

        print(f"epoch {epoch+1:2d}/{n_epochs}  train_loss={epoch_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    export_weights(model, mean, std)
    export_oracle_reference(model, val_runs, mean, std)


if __name__ == "__main__":
    main()
