"""
flat_region.py -- the ONE rule for where a running mean over replicates has gone flat.

Every replicate count chosen automatically goes through here: the MCTS node measurement
(mcts/MCTS.py) and run_kmc.py's adaptive mode.

THE RULE (2026-09-28)
    SEM     = the full mean's standard error: the replicates' spread / sqrt(N)
    band    = full mean +- z x SEM                                   (z = 1)
    n_flat  = the first n at which BOTH hold:
                1. the running mean stays inside the band for every later n, and
                2. at n it has reached the middle of the band: within z x SEM / 2 of the
                   full mean -- so a curve that has only just crossed into the band, still
                   on its way in, does not count.
    enough  = n_flat <= FLAT_FRAC x N: the curve settled with room to spare. Otherwise the
              mean was still moving near the end and more replicates are needed.

The value reported is always the FULL mean of all N replicates; n_flat says whether N was
enough, and where the curve settled.

WHY. It replaces a fixed +-1 % tolerance (MCTS n_rep_wiggle_pct). A single film's growth
rate scatters by ~0.7 % at 1125 K, so the whole running-mean curve sat inside +-1 % and the
old rule returned n_flat = 1: every node of the 1125 K Phase 2 MCTS was scored
on ONE film. The band here is set by each run's own scatter and replicate count, so no
number is typed in. Checked on that run's 240-film nodes: n_flat = 213, 171, 148, each where
the curve visibly levels off.
"""
import math

Z = 1.0             # band half-width in standard errors of the full mean
FLAT_FRAC = 0.8     # a curve that settles only after this share of the replicates is not flat


def running_mean(values):
    """Running mean after 1, 2, ... N replicates, in the order given."""
    out, s = [], 0.0
    for i, v in enumerate(values, 1):
        s += float(v)
        out.append(s / i)
    return out


def flat_point(values, z=Z):
    """(n_flat, full_mean, band_halfwidth) for replicate values IN REPLICATE ORDER.

    n_flat is 0 for no values; with one value, or no scatter at all, it is 1."""
    x = [float(v) for v in values]
    N = len(x)
    if N == 0:
        return 0, float("nan"), float("nan")
    f = sum(x) / N
    if N == 1:
        return 1, f, float("inf")
    sd = math.sqrt(sum((v - f) ** 2 for v in x) / (N - 1))
    band = z * sd / math.sqrt(N)
    run = running_mean(x)
    entry = 1                                  # 1. never leaves the band after this n
    for i, m in enumerate(run):
        if abs(m - f) > band:
            entry = i + 2
    entry = min(entry, N)
    n = next((i for i in range(entry, N + 1) if abs(run[i - 1] - f) <= band / 2.0), N)
    return n, f, band                          # 2. ... and has reached the middle at n


def settled(values, frac=FLAT_FRAC, z=Z):
    """True if the running mean went flat with room to spare: n_flat <= frac x N."""
    n = flat_point(values, z)[0]
    return len(values) > 1 and n <= frac * len(values)
