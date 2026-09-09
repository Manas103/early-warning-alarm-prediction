"""Chamber-to-chamber drift check.

A named real semiconductor-tool maintenance failure mode: one chamber on a
multi-chamber mainframe quietly drifts out of match with its siblings (a
seal degrades, an MFC calibration slips, a match network mistunes) while the
tool's own run-to-run control absorbs the difference wafer to wafer instead
of flagging it, so nothing ever crosses a per-chamber alarm threshold. This
is a different failure mode from the excursion classifier's ramp-to-alarm
scenario: chamber drift has no ramp and crosses no single-chamber threshold,
so a per-chamber classifier or hard alarm is blind to it by construction. It
needs its own check that looks *across* chambers instead of within one.

Method: for each physical quantity, compute every chamber's time-averaged
reading over the run, then express each chamber's mean as a z-score against
the cross-chamber mean, using a "chamber-matching sigma" per quantity
estimated once from a calibration set of clean multi-chamber runs (an SPC
control-limit style estimate from historical healthy data, not a per-run
statistic computed from only 4 chambers, which would be too small a sample
to be a stable threshold on its own).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from simulate import N_CHAMBERS, N_QUANTITIES, QUANTITIES, RUN_LENGTH_MIN, ChamberDriftRun


@dataclasses.dataclass
class DriftFlag:
    chamber: int          # 0-based
    quantity: str
    zscore: float


def chamber_quantity_means(channels: np.ndarray) -> np.ndarray:
    """(RUN_LENGTH_MIN, N_CHANNELS) -> (N_CHAMBERS, N_QUANTITIES) time-average
    per chamber/quantity. Assumes the chamber-major channel layout used by
    simulate.py (channel = chamber_idx * N_QUANTITIES + quantity_idx)."""
    reshaped = channels.reshape(RUN_LENGTH_MIN, N_CHAMBERS, N_QUANTITIES)
    return reshaped.mean(axis=0).astype(np.float64)


def calibrate_sibling_spread(calibration_runs: list[ChamberDriftRun]) -> np.ndarray:
    """Pooled, per-quantity std of (chamber mean - cross-chamber mean) across
    every chamber of every calibration run. Calibration runs must all be
    clean (drifted=False); mixing in drifted runs would inflate the
    threshold and make the check less sensitive. Returns (N_QUANTITIES,)."""
    deviations = []
    for run in calibration_runs:
        means = chamber_quantity_means(run.channels)      # (C, Q)
        grand = means.mean(axis=0)                          # (Q,)
        deviations.append(means - grand)                    # (C, Q)
    stacked = np.stack(deviations)                            # (R, C, Q)
    spread = stacked.reshape(-1, N_QUANTITIES).std(axis=0)   # (Q,)
    return np.where(spread < 1e-9, 1e-9, spread)


def detect_drift(channels: np.ndarray, sibling_spread: np.ndarray, k_sigma: float = 4.5) -> list[DriftFlag]:
    """Flags every (chamber, quantity) whose time-averaged reading is more
    than k_sigma chamber-matching-sigmas away from the cross-chamber mean
    for that quantity, on this run alone."""
    means = chamber_quantity_means(channels)     # (C, Q)
    grand = means.mean(axis=0)                     # (Q,)
    z = (means - grand) / sibling_spread[np.newaxis, :]

    flags = []
    for c in range(N_CHAMBERS):
        for q in range(N_QUANTITIES):
            if abs(z[c, q]) >= k_sigma:
                flags.append(DriftFlag(chamber=c, quantity=QUANTITIES[q], zscore=float(z[c, q])))
    return flags


def run_is_flagged(channels: np.ndarray, sibling_spread: np.ndarray, k_sigma: float = 4.5) -> bool:
    return len(detect_drift(channels, sibling_spread, k_sigma)) > 0


def caught_the_seeded_drift(run: ChamberDriftRun, flags: list[DriftFlag]) -> bool:
    """A seeded drift counts as caught only if the check names the exact
    (chamber, quantity) that was actually drifted, not merely "something is
    flagged"."""
    assert run.drifted
    return any(f.chamber == run.drift_chamber and f.quantity == run.drift_quantity for f in flags)
