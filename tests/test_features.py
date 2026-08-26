import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from features import build_training_arrays, build_windows, channel_stats  # noqa: E402
from model import WINDOW_MINUTES  # noqa: E402
from simulate import N_CHANNELS, generate_run  # noqa: E402


def test_channel_stats_std_never_zero():
    runs = [generate_run(seed=s, is_upset=(s % 2 == 0)) for s in range(10)]
    mean, std = channel_stats(runs)
    assert mean.shape == (N_CHANNELS,)
    assert std.shape == (N_CHANNELS,)
    assert (std > 0).all()


def test_build_windows_shapes_and_no_post_alarm_windows():
    mean = np.zeros(N_CHANNELS, dtype=np.float32)
    std = np.ones(N_CHANNELS, dtype=np.float32)
    run = generate_run(seed=42, is_upset=True)
    windows = list(build_windows(run, mean, std))
    assert len(windows) > 0
    for window, label, t in windows:
        assert window.shape == (WINDOW_MINUTES, N_CHANNELS)
        assert label in (0, 1)
        assert t < run.alarm_minute  # post-alarm minutes must be dropped


def test_build_windows_labels_only_positive_after_t0():
    mean = np.zeros(N_CHANNELS, dtype=np.float32)
    std = np.ones(N_CHANNELS, dtype=np.float32)
    run = generate_run(seed=7, is_upset=True)
    for _window, label, t in build_windows(run, mean, std):
        if t < run.t0:
            assert label == 0
        else:
            assert label == 1


def test_healthy_run_never_labeled_positive():
    mean = np.zeros(N_CHANNELS, dtype=np.float32)
    std = np.ones(N_CHANNELS, dtype=np.float32)
    run = generate_run(seed=99, is_upset=False)
    labels = [label for _w, label, _t in build_windows(run, mean, std)]
    assert set(labels) == {0}


def test_build_training_arrays_matches_manual_window_count():
    mean = np.zeros(N_CHANNELS, dtype=np.float32)
    std = np.ones(N_CHANNELS, dtype=np.float32)
    runs = [generate_run(seed=s, is_upset=(s % 3 == 0)) for s in range(6)]
    x, y = build_training_arrays(runs, mean, std)
    manual_count = sum(1 for r in runs for _ in build_windows(r, mean, std))
    assert x.shape[0] == manual_count
    assert y.shape[0] == manual_count
    assert x.shape[1:] == (WINDOW_MINUTES, N_CHANNELS)
