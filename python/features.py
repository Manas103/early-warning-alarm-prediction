"""Windowing, labeling, and normalization shared by training and eval.

The exact same normalization (per-channel mean/std baked into weights.bin)
and window layout are used by the C++ scorer, so training-time and
inference-time preprocessing never diverge.
"""
from __future__ import annotations

import numpy as np

from model import WINDOW_MINUTES
from simulate import N_CHANNELS, SimRun

# a window ending at minute t is a "positive" (upset-coming) training example
# if the run is an upset run and t falls inside [t0, alarm_minute), i.e. the
# ramp is physically in progress but the hard alarm has not fired yet.
# windows at or after alarm_minute are dropped from training: once the hard
# alarm has fired there is nothing left to predict early.


def channel_stats(runs: list[SimRun]) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std computed over every minute of every run passed
    in (typically the training set). Returned as float32 (N_CHANNELS,)."""
    all_vals = np.concatenate([r.channels for r in runs], axis=0)  # (T*n_runs, C)
    mean = all_vals.mean(axis=0)
    std = all_vals.std(axis=0)
    std = np.where(std < 1e-6, 1e-6, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(channels: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (channels - mean) / std


def build_windows(run: SimRun, mean: np.ndarray, std: np.ndarray):
    """Yield (window, label, minute) for every minute t >= WINDOW_MINUTES-1
    in the run, label in {0,1}, using window = channels[t-W+1 : t+1]."""
    norm = normalize(run.channels, mean, std)
    T = norm.shape[0]
    for t in range(WINDOW_MINUTES - 1, T):
        window = norm[t - WINDOW_MINUTES + 1 : t + 1, :]
        if run.is_upset:
            if run.alarm_minute is not None and t >= run.alarm_minute:
                continue  # drop post-alarm windows entirely
            label = 1 if (run.t0 is not None and t >= run.t0) else 0
        else:
            label = 0
        yield window.astype(np.float32), label, t


def build_training_arrays(runs: list[SimRun], mean: np.ndarray, std: np.ndarray):
    xs, ys = [], []
    for r in runs:
        for window, label, _t in build_windows(r, mean, std):
            xs.append(window)
            ys.append(label)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.float32)
