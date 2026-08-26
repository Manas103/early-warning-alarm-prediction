"""Simulated industrial historian data generator.

Produces 24 synthetic historian tag channels (6 pressures, 6 temperatures,
6 flows, 6 vibration) sampled once per simulated minute for a fixed-length
run. Two kinds of runs:

  - healthy: every channel is stationary noise around its baseline for the
    whole run, no alarm ever fires.
  - upset: at a random start time t0, a primary channel (and one or two
    correlated secondary channels) begins a slow linear ramp toward a hard
    alarm threshold. The hard alarm fires at the first minute the primary
    channel's *true* (noise-free) ramp value crosses the threshold. Because
    the alarm is defined on the noise-free ramp, the ramp crossing minute is
    exactly reproducible from the seed, independent of the particular noise
    draw, which keeps the eval invariants stable.

Everything here is deterministic given a seed: same seed -> byte-identical
channel arrays and alarm time. There is nothing physically validated about
these dynamics; this is a synthetic stand-in for a real historian, described
as such in the README.
"""
from __future__ import annotations

import dataclasses
import numpy as np

N_CHANNELS = 24
CHANNEL_GROUPS = (["pressure"] * 6) + (["temperature"] * 6) + (["flow"] * 6) + (["vibration"] * 6)
def _make_channel_names() -> list[str]:
    names = []
    counts: dict[str, int] = {}
    for grp in CHANNEL_GROUPS:
        counts[grp] = counts.get(grp, 0) + 1
        names.append(f"{grp}_{counts[grp]:02d}")
    return names


CHANNEL_NAMES = _make_channel_names()  # pressure_01..06, temperature_01..06, flow_01..06, vibration_01..06

RUN_LENGTH_MIN = 180          # minutes simulated per run
ALARM_THRESHOLD = 8.0         # in units of baseline std, on the primary channel
UPSET_START_RANGE = (40, 120)  # minutes, t0 window for ramp onset
RAMP_RATE_RANGE = (0.055, 0.09)  # std-units per minute added to the primary channel

# baseline mean/std per channel group (arbitrary engineering-plausible units)
GROUP_BASELINE = {
    "pressure": (120.0, 2.5),
    "temperature": (75.0, 1.2),
    "flow": (400.0, 8.0),
    "vibration": (0.6, 0.05),
}
# AR(1) autocorrelation coefficient per group (historian noise is not white)
GROUP_AR = {
    "pressure": 0.55,
    "temperature": 0.75,
    "flow": 0.45,
    "vibration": 0.6,
}


@dataclasses.dataclass
class SimRun:
    seed: int
    is_upset: bool
    channels: np.ndarray       # (RUN_LENGTH_MIN, N_CHANNELS) float32
    primary_channel: int | None
    secondary_channels: tuple[int, ...]
    t0: int | None             # ramp onset minute, upset runs only
    alarm_minute: int | None   # minute the hard threshold is crossed, upset runs only


def _ar1_noise(rng: np.random.Generator, n: int, std: float, phi: float) -> np.ndarray:
    """Zero-mean AR(1) noise sequence with the given marginal std."""
    innovation_std = std * np.sqrt(1.0 - phi * phi)
    x = np.zeros(n, dtype=np.float64)
    x[0] = rng.normal(0.0, std)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0.0, innovation_std)
    return x


def generate_run(seed: int, is_upset: bool) -> SimRun:
    rng = np.random.default_rng(seed)
    channels = np.zeros((RUN_LENGTH_MIN, N_CHANNELS), dtype=np.float64)

    for ch in range(N_CHANNELS):
        grp = CHANNEL_GROUPS[ch]
        mean, std = GROUP_BASELINE[grp]
        phi = GROUP_AR[grp]
        noise = _ar1_noise(rng, RUN_LENGTH_MIN, std, phi)
        channels[:, ch] = mean + noise

    primary_channel = None
    secondary_channels: tuple[int, ...] = ()
    t0 = None
    alarm_minute = None

    if is_upset:
        # pick primary channel uniformly among pressure/temperature/flow
        # channels (vibration excluded as primary; used only as a secondary
        # correlate, mirroring how a real bearing/seal upset shows up as a
        # secondary vibration signature rather than the leading indicator)
        primary_channel = int(rng.integers(0, 18))
        grp = CHANNEL_GROUPS[primary_channel]
        mean, std = GROUP_BASELINE[grp]

        # 1-2 correlated secondary channels from a different group
        n_secondary = int(rng.integers(1, 3))
        candidates = [c for c in range(N_CHANNELS) if CHANNEL_GROUPS[c] != grp]
        secondary_channels = tuple(int(c) for c in rng.choice(candidates, size=n_secondary, replace=False))

        t0 = int(rng.integers(*UPSET_START_RANGE))
        ramp_rate = float(rng.uniform(*RAMP_RATE_RANGE))  # std-units / minute

        # analytic crossing minute (ceil), guaranteed to land inside the run:
        # if the sampled rate would cross past the last minute, slow the rate
        # down just enough that the crossing lands exactly on the last
        # minute instead (every upset run is a true positive by construction)
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
            sec_grp = CHANNEL_GROUPS[sec]
            sec_mean, sec_std = GROUP_BASELINE[sec_grp]
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
