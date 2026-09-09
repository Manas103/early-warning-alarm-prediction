import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from simulate import (  # noqa: E402
    CHANNEL_NAMES,
    N_CHANNELS,
    RUN_LENGTH_MIN,
    generate_run,
    make_dataset,
)


def test_channel_count_and_names():
    assert N_CHANNELS == 24
    assert len(CHANNEL_NAMES) == 24
    assert len(set(CHANNEL_NAMES)) == 24  # all unique
    assert CHANNEL_NAMES[0] == "chamber1_pressure"
    assert CHANNEL_NAMES[6] == "chamber2_pressure"
    assert CHANNEL_NAMES[12] == "chamber3_pressure"
    assert CHANNEL_NAMES[18] == "chamber4_pressure"
    assert CHANNEL_NAMES[5] == "chamber1_endpoint"


def test_generate_run_is_deterministic():
    r1 = generate_run(seed=12345, is_upset=True)
    r2 = generate_run(seed=12345, is_upset=True)
    assert np.array_equal(r1.channels, r2.channels)
    assert r1.t0 == r2.t0
    assert r1.alarm_minute == r2.alarm_minute
    assert r1.primary_channel == r2.primary_channel


def test_different_seeds_give_different_runs():
    r1 = generate_run(seed=1, is_upset=True)
    r2 = generate_run(seed=2, is_upset=True)
    assert not np.array_equal(r1.channels, r2.channels)


def test_healthy_run_has_no_alarm():
    r = generate_run(seed=777, is_upset=False)
    assert r.alarm_minute is None
    assert r.t0 is None
    assert r.primary_channel is None
    assert r.channels.shape == (RUN_LENGTH_MIN, N_CHANNELS)


def test_upset_run_always_has_a_defined_alarm_minute_inside_the_run():
    # every upset run must be a genuine true positive by construction: the
    # ramp always crosses the hard threshold strictly before the run ends
    for seed in range(50):
        r = generate_run(seed=seed, is_upset=True)
        assert r.alarm_minute is not None
        assert r.t0 is not None
        assert r.t0 < r.alarm_minute < RUN_LENGTH_MIN
        assert r.primary_channel is not None
        assert 1 <= len(r.secondary_channels) <= 2


def test_make_dataset_train_eval_seed_ranges_are_disjoint():
    train = make_dataset(5, 5, seed_offset=1)
    eval_ = make_dataset(5, 5, seed_offset=900000)
    train_seeds = {r.seed for r in train}
    eval_seeds = {r.seed for r in eval_}
    assert train_seeds.isdisjoint(eval_seeds)
