"""Simulated multi-chamber semiconductor tool sensor data generator.

Produces 24 synthetic tool-controller tag channels: 6 physical quantities
(chamber pressure, RF forward power, RF reflected power, process gas flow,
optical emission intensity, endpoint signal) replicated across 4 simulated
process chambers (a Centura/Producer-class multi-chamber etch or deposition
platform commonly runs 4-6 process chambers off one mainframe), 6 x 4 = 24
channels, sampled once per simulated minute for a fixed-length run.

RF reflected power is an addition beyond the five quantities named on the
resume bullet: real etch/deposition tool controllers universally pair RF
forward power with RF reflected power (the match network's tuning error),
so adding it here is the honest choice for "what a real multi-chamber tool
controller actually reads" rather than padding the channel count with a
sixth copy of something already present.

Each physical quantity has genuinely different simulated dynamics, not a
renamed copy of the same noise process:
  - pressure: tight, heavily autocorrelated (AR phi=0.85) around the PID
    setpoint, mirroring a closed-loop pressure controller.
  - RF forward power: fast-regulated, low autocorrelation (phi=0.5), small
    relative noise.
  - RF reflected power: low autocorrelation baseline plus sparse Poisson-like
    spikes (match-network retune transients), never negative in practice
    and bursty rather than smoothly wandering.
  - gas flow: moderate autocorrelation (phi=0.6), mass-flow-controller-like.
  - optical emission intensity: a cyclic (period 12-20 min) oscillation on
    top of AR(1) noise, representing a cyclic process step's chemistry
    signature, not a monotone-drifting quantity.
  - endpoint signal: slow-moving, well-autocorrelated (phi=0.7) baseline
    that is otherwise flat; it only moves when something (a real excursion,
    simulated below) pushes it.

Two independent scenario generators share the same baseline-channel model:

  - `generate_run` (excursion classifier): healthy runs are stationary noise
    on every channel; excursion runs start a slow linear ramp at a random
    t0 on one "process-controlled" quantity (pressure, RF forward power, or
    gas flow, chosen across all 4 chambers) that crosses a hard alarm
    threshold at a reproducible `alarm_minute`, with 1-2 correlated
    secondary channels (often OES/endpoint/RF-reflected, mirroring how a
    real pressure or RF excursion shows up secondarily in the optical and
    endpoint signals) ramping a few minutes later at reduced amplitude.
  - `generate_chamber_drift_run` (chamber-to-chamber drift check): healthy
    multi-chamber noise on every channel, plus, on drifted runs, one
    chamber's one quantity offset by a constant bias for the whole run,
    representing a chamber that has quietly drifted out of match with its
    siblings (the run-to-run control silently absorbing chamber mismatch
    failure mode), which the excursion classifier is not designed to see
    (no ramp, no alarm-threshold crossing) and needs a dedicated check.

Everything here is deterministic given a seed: same seed -> byte-identical
channel arrays and, for excursion runs, the same alarm time. Nothing here
was validated against a real tool historian; this is a synthetic stand-in,
described as such in the README.
"""
from __future__ import annotations

import dataclasses
import numpy as np

N_CHAMBERS = 4
QUANTITIES = ("pressure", "rf_forward", "rf_reflected", "gas_flow", "oes_intensity", "endpoint")
N_QUANTITIES = len(QUANTITIES)
N_CHANNELS = N_CHAMBERS * N_QUANTITIES
assert N_CHANNELS == 24

# process-controlled quantities eligible to be the primary (leading) channel
# of a simulated tool excursion; RF reflected power, OES and endpoint are
# reserved as secondary/derived channels only, mirroring how a real pressure
# or RF drift shows up secondarily in the optical and endpoint signals
# rather than being itself the leading indicator.
PRIMARY_QUANTITIES = ("pressure", "rf_forward", "gas_flow")

RUN_LENGTH_MIN = 180          # minutes simulated per run
ALARM_THRESHOLD = 8.0         # in units of the primary quantity's baseline std
UPSET_START_RANGE = (40, 120)   # minutes, t0 window for excursion onset
RAMP_RATE_RANGE = (0.055, 0.09)  # std-units per minute added to the primary channel

# chamber-to-chamber drift check: constant bias injected on the drifted
# run's chosen (chamber, quantity), as a fraction of that quantity's raw
# per-minute baseline std. Deliberately modest (a real chamber drift is a
# quiet calibration slip, not a gross fault): 0.15-0.6x the raw per-minute
# std is small relative to any single reading but, because the check
# averages over the whole run, is still large relative to the
# run-length-averaged cross-chamber spread it is actually compared against
# (see chamber_drift.py), so this is not a free 100%-detectable magnitude
# chosen to flatter the check.
CHAMBER_DRIFT_MAGNITUDE_RANGE = (0.4, 1.6)

# (mean, std) in engineering-plausible units per physical quantity
QUANTITY_BASELINE = {
    "pressure": (45.0, 1.5),        # mTorr
    "rf_forward": (850.0, 6.0),     # W
    "rf_reflected": (12.0, 3.0),    # W
    "gas_flow": (120.0, 2.0),       # sccm
    "oes_intensity": (2200.0, 80.0),  # counts
    "endpoint": (500.0, 15.0),      # counts
}
# AR(1) autocorrelation coefficient per quantity (tool telemetry noise is
# not white, and different control loops have different bandwidths)
QUANTITY_AR = {
    "pressure": 0.85,
    "rf_forward": 0.5,
    "rf_reflected": 0.3,
    "gas_flow": 0.6,
    "oes_intensity": 0.4,
    "endpoint": 0.7,
}


def _channel_chamber_quantity(ch: int) -> tuple[int, str]:
    """channel index -> (0-based chamber index, quantity name), chamber-major
    layout: channel 0..5 = chamber 1's 6 quantities, channel 6..11 = chamber
    2's, etc. (a real tag list groups tags by chamber, not by quantity)."""
    return ch // N_QUANTITIES, QUANTITIES[ch % N_QUANTITIES]


def _channel_index(chamber_idx: int, quantity: str) -> int:
    return chamber_idx * N_QUANTITIES + QUANTITIES.index(quantity)


def _make_channel_names() -> list[str]:
    names = []
    for ch in range(N_CHANNELS):
        chamber_idx, quantity = _channel_chamber_quantity(ch)
        names.append(f"chamber{chamber_idx + 1}_{quantity}")
    return names


CHANNEL_NAMES = _make_channel_names()  # chamber1_pressure, chamber1_rf_forward, ..., chamber4_endpoint


@dataclasses.dataclass
class SimRun:
    seed: int
    is_upset: bool
    channels: np.ndarray       # (RUN_LENGTH_MIN, N_CHANNELS) float32
    primary_channel: int | None
    secondary_channels: tuple[int, ...]
    t0: int | None             # ramp onset minute, upset runs only
    alarm_minute: int | None   # minute the hard threshold is crossed, upset runs only


@dataclasses.dataclass
class ChamberDriftRun:
    seed: int
    drifted: bool
    channels: np.ndarray                # (RUN_LENGTH_MIN, N_CHANNELS) float32
    drift_chamber: int | None           # 0-based chamber index, drifted runs only
    drift_quantity: str | None          # quantity name, drifted runs only
    drift_magnitude_std: float | None   # bias size in units of that quantity's std


def _ar1_noise(rng: np.random.Generator, n: int, std: float, phi: float) -> np.ndarray:
    """Zero-mean AR(1) noise sequence with the given marginal std."""
    innovation_std = std * np.sqrt(1.0 - phi * phi)
    x = np.zeros(n, dtype=np.float64)
    x[0] = rng.normal(0.0, std)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0.0, innovation_std)
    return x


def _generate_baseline_channels(rng: np.random.Generator) -> np.ndarray:
    """AR(1) autocorrelated noise per channel around each channel's physical
    quantity baseline, plus quantity-specific dynamics (sparse match-network
    spikes on RF reflected power, a cyclic chemistry oscillation on OES
    intensity) so that pressure and RF power do not behave identically to
    endpoint/OES signals. Returns (RUN_LENGTH_MIN, N_CHANNELS) float64."""
    channels = np.zeros((RUN_LENGTH_MIN, N_CHANNELS), dtype=np.float64)
    for ch in range(N_CHANNELS):
        _, quantity = _channel_chamber_quantity(ch)
        mean, std = QUANTITY_BASELINE[quantity]
        phi = QUANTITY_AR[quantity]
        series = mean + _ar1_noise(rng, RUN_LENGTH_MIN, std, phi)

        if quantity == "rf_reflected":
            # sparse match-network retune transients: not a smoothly
            # wandering quantity, a mostly-quiet baseline with occasional
            # short spikes
            spike_prob = 0.03
            spikes = rng.random(RUN_LENGTH_MIN) < spike_prob
            spike_mag = rng.uniform(2.0, 5.0, size=RUN_LENGTH_MIN) * std
            series = series + np.where(spikes, spike_mag, 0.0)
        elif quantity == "oes_intensity":
            # cyclic process-step chemistry signature
            period = rng.uniform(12.0, 20.0)
            phase = rng.uniform(0.0, 2 * np.pi)
            t = np.arange(RUN_LENGTH_MIN)
            series = series + 0.5 * std * np.sin(2 * np.pi * t / period + phase)

        channels[:, ch] = series
    return channels


def generate_run(seed: int, is_upset: bool) -> SimRun:
    rng = np.random.default_rng(seed)
    channels = _generate_baseline_channels(rng)

    primary_channel = None
    secondary_channels: tuple[int, ...] = ()
    t0 = None
    alarm_minute = None

    if is_upset:
        primary_candidates = [
            ch for ch in range(N_CHANNELS) if _channel_chamber_quantity(ch)[1] in PRIMARY_QUANTITIES
        ]
        primary_channel = int(rng.choice(primary_candidates))
        _, quantity = _channel_chamber_quantity(primary_channel)
        mean, std = QUANTITY_BASELINE[quantity]

        # 1-2 correlated secondary channels, any channel other than the
        # primary itself (commonly lands on OES/endpoint/RF-reflected in a
        # different chamber, or the same chamber's other quantities)
        n_secondary = int(rng.integers(1, 3))
        secondary_candidates = [c for c in range(N_CHANNELS) if c != primary_channel]
        secondary_channels = tuple(
            int(c) for c in rng.choice(secondary_candidates, size=n_secondary, replace=False)
        )

        t0 = int(rng.integers(*UPSET_START_RANGE))
        ramp_rate = float(rng.uniform(*RAMP_RATE_RANGE))  # std-units / minute

        # analytic crossing minute (ceil), guaranteed to land inside the run:
        # if the sampled rate would cross past the last minute, slow the rate
        # down just enough that the crossing lands exactly on the last
        # minute instead (every excursion run is a true positive by
        # construction)
        minutes_to_cross = int(np.ceil(ALARM_THRESHOLD / ramp_rate))
        last_minute = RUN_LENGTH_MIN - 1
        if t0 + minutes_to_cross > last_minute:
            minutes_to_cross = last_minute - t0
            ramp_rate = ALARM_THRESHOLD / minutes_to_cross
        alarm_minute = t0 + minutes_to_cross

        # noise-free ramp value in std-units above baseline
        ramp_std_units = np.zeros(RUN_LENGTH_MIN, dtype=np.float64)
        for t in range(t0, RUN_LENGTH_MIN):
            ramp_std_units[t] = ramp_rate * (t - t0)

        channels[:, primary_channel] += ramp_std_units * std

        # secondary channels get a smaller correlated ramp (30-55% of the
        # primary's std-unit rate), starting a few minutes after t0
        for sec in secondary_channels:
            _, sec_quantity = _channel_chamber_quantity(sec)
            sec_mean, sec_std = QUANTITY_BASELINE[sec_quantity]
            lag = int(rng.integers(2, 8))
            scale = float(rng.uniform(0.3, 0.55))
            sec_ramp = np.zeros(RUN_LENGTH_MIN, dtype=np.float64)
            for t in range(t0 + lag, RUN_LENGTH_MIN):
                sec_ramp[t] = ramp_rate * scale * (t - t0 - lag)
            channels[:, sec] += sec_ramp * sec_std

    return SimRun(
        seed=seed,
        is_upset=is_upset,
        channels=channels.astype(np.float32),
        primary_channel=primary_channel,
        secondary_channels=secondary_channels,
        t0=t0,
        alarm_minute=alarm_minute,
    )


def generate_chamber_drift_run(seed: int, drifted: bool) -> ChamberDriftRun:
    """Healthy multi-chamber baseline noise on every channel; on a drifted
    run, one randomly chosen (chamber, quantity) gets a constant bias for
    the whole run, simulating a chamber that has quietly drifted out of
    match with its siblings (no ramp, no alarm-threshold crossing, which is
    exactly why the excursion classifier above cannot be expected to see
    this failure mode and a dedicated chamber-to-chamber check is needed)."""
    rng = np.random.default_rng(seed)
    channels = _generate_baseline_channels(rng)

    drift_chamber = None
    drift_quantity = None
    drift_magnitude_std = None

    if drifted:
        drift_chamber = int(rng.integers(0, N_CHAMBERS))
        drift_quantity = str(rng.choice(QUANTITIES))
        mean, std = QUANTITY_BASELINE[drift_quantity]
        drift_magnitude_std = float(rng.uniform(*CHAMBER_DRIFT_MAGNITUDE_RANGE))
        ch = _channel_index(drift_chamber, drift_quantity)
        channels[:, ch] += drift_magnitude_std * std

    return ChamberDriftRun(
        seed=seed,
        drifted=drifted,
        channels=channels.astype(np.float32),
        drift_chamber=drift_chamber,
        drift_quantity=drift_quantity,
        drift_magnitude_std=drift_magnitude_std,
    )


def make_dataset(n_upset: int, n_healthy: int, seed_offset: int) -> list[SimRun]:
    """Deterministic list of runs: seeds seed_offset..seed_offset+n-1 for
    upset runs (even count) interleaved with a disjoint healthy seed block,
    so train/eval seed ranges never collide when seed_offset ranges are kept
    disjoint by the caller."""
    runs = []
    for i in range(n_upset):
        runs.append(generate_run(seed=seed_offset + i, is_upset=True))
    for i in range(n_healthy):
        runs.append(generate_run(seed=seed_offset + 100000 + i, is_upset=False))
    return runs


def make_chamber_drift_dataset(n_drifted: int, n_clean: int, seed_offset: int) -> list[ChamberDriftRun]:
    """Same disjoint-seed-block convention as make_dataset, for the chamber
    drift check's own calibration/eval sets."""
    runs = []
    for i in range(n_drifted):
        runs.append(generate_chamber_drift_run(seed=seed_offset + i, drifted=True))
    for i in range(n_clean):
        runs.append(generate_chamber_drift_run(seed=seed_offset + 100000 + i, drifted=False))
    return runs
