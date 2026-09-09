import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from chamber_drift import (  # noqa: E402
    DriftFlag,
    calibrate_sibling_spread,
    caught_the_seeded_drift,
    chamber_quantity_means,
    detect_drift,
    run_is_flagged,
)
from simulate import (  # noqa: E402
    N_CHAMBERS,
    N_QUANTITIES,
    QUANTITIES,
    RUN_LENGTH_MIN,
    generate_chamber_drift_run,
    make_chamber_drift_dataset,
)


def test_chamber_quantity_means_shape_and_reduction():
    run = generate_chamber_drift_run(seed=1, drifted=False)
    means = chamber_quantity_means(run.channels)
    assert means.shape == (N_CHAMBERS, N_QUANTITIES)
    # a manual reduction on one (chamber, quantity) must match exactly
    ch = 0 * N_QUANTITIES + 0  # chamber 0, quantity 0 (pressure)
    assert np.isclose(means[0, 0], run.channels[:, ch].mean())


def test_drifted_run_has_a_named_chamber_and_quantity():
    r = generate_chamber_drift_run(seed=42, drifted=True)
    assert r.drift_chamber is not None and 0 <= r.drift_chamber < N_CHAMBERS
    assert r.drift_quantity in QUANTITIES
    assert r.drift_magnitude_std is not None and r.drift_magnitude_std > 0


def test_clean_run_has_no_drift_metadata():
    r = generate_chamber_drift_run(seed=42, drifted=False)
    assert r.drift_chamber is None
    assert r.drift_quantity is None
    assert r.drift_magnitude_std is None


def test_generate_chamber_drift_run_is_deterministic():
    a = generate_chamber_drift_run(seed=777, drifted=True)
    b = generate_chamber_drift_run(seed=777, drifted=True)
    assert np.array_equal(a.channels, b.channels)
    assert a.drift_chamber == b.drift_chamber
    assert a.drift_quantity == b.drift_quantity


def test_calibration_is_deterministic_and_positive():
    calibration = make_chamber_drift_dataset(n_drifted=0, n_clean=10, seed_offset=5_000_000)
    spread_a = calibrate_sibling_spread(calibration)
    spread_b = calibrate_sibling_spread(calibration)
    assert np.array_equal(spread_a, spread_b)
    assert spread_a.shape == (N_QUANTITIES,)
    assert (spread_a > 0).all()


def test_detect_drift_returns_valid_flags():
    calibration = make_chamber_drift_dataset(n_drifted=0, n_clean=20, seed_offset=5_100_000)
    spread = calibrate_sibling_spread(calibration)
    run = generate_chamber_drift_run(seed=6_000_001, drifted=True)
    flags = detect_drift(run.channels, spread, k_sigma=4.5)
    for f in flags:
        assert isinstance(f, DriftFlag)
        assert 0 <= f.chamber < N_CHAMBERS
        assert f.quantity in QUANTITIES
        assert abs(f.zscore) >= 4.5


def test_caught_the_seeded_drift_requires_the_exact_channel():
    run = generate_chamber_drift_run(seed=99, drifted=True)
    correct = [DriftFlag(chamber=run.drift_chamber, quantity=run.drift_quantity, zscore=10.0)]
    wrong_chamber = [DriftFlag(chamber=(run.drift_chamber + 1) % N_CHAMBERS, quantity=run.drift_quantity, zscore=10.0)]
    wrong_quantity = [DriftFlag(chamber=run.drift_chamber, quantity="pressure" if run.drift_quantity != "pressure" else "gas_flow", zscore=10.0)]
    assert caught_the_seeded_drift(run, correct) is True
    assert caught_the_seeded_drift(run, wrong_chamber) is False
    assert caught_the_seeded_drift(run, wrong_quantity) is False
    assert caught_the_seeded_drift(run, []) is False


def test_run_is_flagged_matches_detect_drift_nonempty():
    calibration = make_chamber_drift_dataset(n_drifted=0, n_clean=20, seed_offset=5_200_000)
    spread = calibrate_sibling_spread(calibration)
    for seed in range(5_300_000, 5_300_010):
        run = generate_chamber_drift_run(seed=seed, drifted=False)
        flags = detect_drift(run.channels, spread, k_sigma=4.5)
        assert run_is_flagged(run.channels, spread, k_sigma=4.5) == (len(flags) > 0)


def test_clean_calibration_holdout_has_a_low_false_flag_rate():
    # calibrate on one clean block, evaluate on a disjoint clean block: the
    # false-flag rate should be low (this is the same property the honest
    # eval in eval_chamber_drift.py measures at full scale, 0/40 there).
    calibration = make_chamber_drift_dataset(n_drifted=0, n_clean=40, seed_offset=5_400_000)
    spread = calibrate_sibling_spread(calibration)
    holdout = make_chamber_drift_dataset(n_drifted=0, n_clean=20, seed_offset=5_500_000)
    false_flags = sum(1 for r in holdout if run_is_flagged(r.channels, spread, k_sigma=4.5))
    assert false_flags <= 2  # allow at most a small handful out of 20


def test_seeded_drift_run_channel_actually_shifted():
    # sanity check on the injection itself, independent of the detector: the
    # drifted (chamber, quantity)'s run-length mean must differ from the
    # cross-chamber mean by roughly the intended fraction of that quantity's
    # baseline std, not by construction accident
    from simulate import QUANTITY_BASELINE

    r = generate_chamber_drift_run(seed=321, drifted=True)
    means = chamber_quantity_means(r.channels)
    q_idx = QUANTITIES.index(r.drift_quantity)
    grand = means[:, q_idx].mean()
    drifted_dev = means[r.drift_chamber, q_idx] - grand
    _, std = QUANTITY_BASELINE[r.drift_quantity]
    assert drifted_dev > 0  # bias was added, not subtracted
    assert drifted_dev < r.drift_magnitude_std * std * 1.5  # same order of magnitude


def test_channel_layout_is_chamber_major():
    # channel index = chamber_idx * N_QUANTITIES + quantity_idx, asserted
    # against the public channel-index helper the check itself relies on
    from simulate import _channel_index

    assert _channel_index(0, "pressure") == 0
    assert _channel_index(1, "pressure") == N_QUANTITIES
    assert _channel_index(0, "endpoint") == N_QUANTITIES - 1


def test_run_length_and_channel_count_match_simulate():
    r = generate_chamber_drift_run(seed=1, drifted=False)
    assert r.channels.shape == (RUN_LENGTH_MIN, N_CHAMBERS * N_QUANTITIES)
