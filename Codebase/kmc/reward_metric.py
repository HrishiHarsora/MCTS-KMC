"""
reward_metric.py -- the ONE growth-rate definition every path reads.

    GR = (h_hi - h_lo) / (t(h_hi) - t(h_lo))                    [um/min]

h is the MEAN film height: the mean over lateral columns of the top real atom, measured
from the starting surface. t(h) is the simulated time at which it first reaches h,
linearly interpolated between the two checkpoints that bracket it. The window is
h_lo = 50 A to h_hi = 200 A (since 2026-09-27).

The engine stops a trajectory at the first checkpoint where the mean height reaches h_hi
(stop_mean_height = h_hi), so a trajectory costs only the events it needs to grow through
the window. A trajectory that never reaches h_hi -- the event cap or the stall rule ended
it first -- has no rate. It is reported as "did not grow through the window"
(reached = False), never as a number.

WHY A WINDOW
  * to 2026-08-22  GR = height at a fixed event count / time. A dead film decays as 1/t
                   onto any target.
  * to 2026-09-06  GR = 30 A / t(30 A). Read a transient in films that stalled at ~44 A.
  * to 2026-09-27  GR at the mean-height plateau. Films of the working engine do not
                   stall, so the plateau was the lattice ceiling: biased 15-20 %, noisy.
  * now            a window above the first 50 A (the start-up transient) and below the
                   lattice top. Lattice_10nm_h232.xyz has 231.8 A of room above the
                   starting surface; the front leads the mean height by ~7-12 A.

The sink rows are (cumulative_events, elapsed_sim_time_s, max_height_A, mean_height_A).
run_kmc.py, kmc_worker.py (srun fan-out) and MCTS._kmc_task (pool) all read the rate
here, so it cannot drift between them.
"""

WINDOW_A = (50.0, 200.0)


def check_window(lo, hi):
    """(lo, hi) as floats, refused unless 0 < lo < hi."""
    lo, hi = float(lo), float(hi)
    if not (0.0 < lo < hi):
        raise ValueError("reward_metric: the growth-rate window must satisfy 0 < h_lo < h_hi, "
                         "got %r-%r A" % (lo, hi))
    return lo, hi


def crossing_time(sink, h):
    """Simulated time [s] at which the mean height first reaches h, linearly interpolated
    between the last checkpoint below h and the first at or above it. The starting
    surface (t = 0, h = 0) comes before the first checkpoint. None if it never does."""
    t0, h0 = 0.0, 0.0
    for row in sink:
        t1, h1 = float(row[1]), float(row[3])
        if h1 >= h:
            return t0 + (t1 - t0) * (h - h0) / (h1 - h0)
        t0, h0 = t1, h1
    return None


def window_from_sink(sink, lo, hi):
    """The window read of one trajectory: (gr, reached, t_lo, t_hi, h_final).

    gr       growth rate through the window [um/min]; None unless reached
    reached  the mean height reached h_hi, so the whole window was grown through
    t_lo     time at h_lo [s], or None
    t_hi     time at h_hi [s], or None
    h_final  mean height at the last checkpoint [A]"""
    lo, hi = check_window(lo, hi)
    t_lo, t_hi = crossing_time(sink, lo), crossing_time(sink, hi)
    reached = t_lo is not None and t_hi is not None and t_hi > t_lo
    gr = (hi - lo) * 1e-4 / ((t_hi - t_lo) / 60.0) if reached else None
    h_final = float(sink[-1][3]) if sink else 0.0
    return gr, reached, t_lo, t_hi, h_final


def reward_ckpt_every(max_iter, requested):
    """Checkpoint cadence. It must DIVIDE max_iter: the engine writes a periodic
    checkpoint on the correct mean-height baseline (_z_init_mean) only when
    iteration % checkpoint_every == 0, while the guaranteed final row it appends
    otherwise uses the max-front baseline. Forcing the cadence to divide the cap
    guarantees every row shares one baseline."""
    if max_iter is None:
        return int(requested) if requested else 0
    req = int(requested) if requested else 0
    if req > 0 and max_iter % req == 0:
        return req
    return int(max_iter)
