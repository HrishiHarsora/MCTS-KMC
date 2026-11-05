"""
run_kmc.py -- the GENERAL KMC driver.

Runs a KMC sweep at ANY temperature set, ANY parameter set, ANY number of
replicates, ANY event cap. Everything comes from config.toml; every field can be
overridden on the command line. Nothing about a particular study is baked in.

    python run_kmc.py                                   # exactly what config.toml says
    python run_kmc.py --temps 900,1000,1125 --nrep 50   # override the sweep
    python run_kmc.py --params best_params.json --cap 15000000
    python run_kmc.py --dry-run                         # print the plan + cost, run nothing

THE GROWTH RATE (reward_metric.py)
    The mean film height is recorded every reward_checkpoint_events events. The rate is
    read through a thickness window, [kmc] gr_window_A = [50.0, 200.0]:

        GR = (h_hi - h_lo) / (t(h_hi) - t(h_lo))                    [um/min]

    with t(h) the time the mean height first reaches h. A trajectory that never reaches
    h_hi has no rate ("did not grow through the window"), never a number.

TERMINATION -- a trajectory ends at the FIRST of
    window   the mean height reaches h_hi (checked at every checkpoint)
    stall    the mean height has not advanced for max(stall_min_events,
             stall_idle_factor x events at its last advance); an advance is
             h > best + max(stall_rel_tol x |best|, stall_abs_tol_A). This only saves
             the cost of films that stopped growing; 0 turns it off
    cap      max_iterations events
    The growth FRONT never ends a run (target_growth is set out of reach).

IDENTICAL NUMBERS TO MCTS
    The task list is built T-major -- for T in temps: for rep in range(nrep) --
    with seed = seed_base + k, and each trajectory is read through the SAME
    reward_metric.window_from_sink as MCTS._kmc_task and kmc_worker.py. So a sweep at
    a parameter set reproduces, trajectory for trajectory, what MCTS would have
    measured at that node.

SHARDING (for SLURM arrays)
    --shard i --nshards N runs the tasks with (k % N == i). STRIDED, not
    contiguous, so a partially-complete run has replicates spread evenly across
    all temperatures instead of finishing the first temperature and none of the
    rest. Read partial results at any time; the CSV appends under an exclusive
    lock.

    Defaults come from SLURM: shard <- SLURM_ARRAY_TASK_ID,
    nshards <- SLURM_ARRAY_TASK_COUNT, workers <- SLURM_CPUS_PER_TASK.

OUTPUT
    <out>.csv        one row per checkpoint per trajectory:
                         temperature_K, seed, rep, events, sim_time_s, max_height_A,
                         mean_height_A, GR_max_umin, GR_mean_umin
                     (the last two are height / time at that checkpoint)
    <out>_traj.csv   one row per trajectory: the window read, how it ended, wall time
    <out>_waves.csv  one row per adaptive wave and temperature (only when target_sem_pct > 0)

ADAPTIVE n (target_sem_pct > 0, no --nrep)
    Replicates are added a wave at a time until, at EVERY temperature, the SEM of the
    window growth rate is at or below target_sem_pct AND its running mean has gone flat
    (kmc/flat_region.py, the rule MCTS uses: inside the full mean +- 1 SEM from n_flat on,
    at the middle of that band at n_flat, and n_flat before the last 20 % of the
    replicates), or n_rep_max is reached. A resumed run counts the replicates it already
    recorded.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

# Single-thread numpy BEFORE it is imported: this process runs a pool of
# trajectories, and per-process BLAS threads would oversubscribe the node.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMC_INCR_KHOP", "3")

import multiprocessing as mp
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import tomllib                      # py3.11+
except ModuleNotFoundError:             # pragma: no cover
    import tomli as tomllib

# The bucket engine: same interface, same chemistry (imports KMC.py), validated
# against KMC.py (tests/: Gate 1 menu == full re-enumeration, Gate 2 ensembles).
import KMC_bucket as KMC
from reward_metric import (WINDOW_A, check_window, reward_ckpt_every,
                           window_from_sink)
from flat_region import flat_point, settled, FLAT_FRAC

HEADER = ["temperature_K", "seed", "rep", "events", "sim_time_s",
          "max_height_A", "mean_height_A", "GR_max_umin", "GR_mean_umin"]

# The growth front never terminates a run; the window top, the stall rule and the cap do.
UNREACHABLE_GROWTH_A = 1e6

# Keys of the previous (plateau) definition. A config that still carries them was written
# for another read of the growth rate; refuse it rather than run it under this one.
LEGACY_KEYS = ("stop_mean_height", "stop_plateau_idle_factor", "stop_plateau_min_events",
               "stop_plateau_rel_tol", "stop_plateau_abs_tol_A")

# Measured per-event cost on the cluster, for the dry-run estimate only: 0.05 ms/event on
# the development machine with numba (10 nm lattice, including set-up), x2.2 for an HPC core.
MS_PER_EVENT_HPC = {True: 0.11, False: 0.30}


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def load_config(path):
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def resolve(args, cfg):
    """Merge config.toml and CLI into one settings dict. CLI always wins."""
    run = cfg.get("run", {})
    kmc = cfg.get("kmc", {})

    old = [k for k in LEGACY_KEYS if k in kmc]
    if old:
        sys.exit("[run_kmc] config sets %s -- keys of the plateau read, retired 2026-09-27.\n"
                 "          The rate is read through [kmc] gr_window_A = [lo, hi] and the run\n"
                 "          stops at hi; the stall rule is stall_idle_factor, stall_min_events,\n"
                 "          stall_rel_tol, stall_abs_tol_A." % ", ".join(old))

    if args.temps:
        temps = [float(x) for x in args.temps.replace(",", " ").split()]
    else:
        temps = [float(x) for x in run.get("temperatures", [])]
    if not temps:
        sys.exit("[run_kmc] no temperatures: set [run].temperatures or --temps")

    if args.params:
        with open(args.params) as f:
            params = json.load(f)
        params = {k: v for k, v in params.items() if not str(k).startswith("_")}
        params_src = args.params
    else:
        params = {str(k): float(v) for k, v in cfg.get("params", {}).items()}
        params_src = f"{args.config} [params]" if params else "CATALOG DEFAULTS"
    if not params:
        params_src = "CATALOG DEFAULTS (empty [params])"

    cap = int(args.cap if args.cap else kmc.get("max_iterations", 0))
    if cap <= 0:
        sys.exit("[run_kmc] no event cap: set [kmc].max_iterations or --cap")

    ckpt = int(args.ckpt if args.ckpt else kmc.get("reward_checkpoint_events", 20000))

    win = (args.window.replace(",", " ").split() if args.window
           else kmc.get("gr_window_A", list(WINDOW_A)))
    if len(win) != 2:
        sys.exit("[run_kmc] gr_window_A must be two heights [lo, hi] in angstrom")
    try:
        lo, hi = check_window(*win)
    except ValueError as e:
        sys.exit(f"[run_kmc] {e}")

    return {
        "temps":     temps,
        "params":    params,
        "params_src": params_src,
        "nrep":      int(args.nrep if args.nrep else kmc.get("n_replicates", 1)),
        "cap":       cap,
        "ckpt":      ckpt,
        "lattice":   args.lattice or kmc.get("lattice_file", "Lattice_10nm_h232.xyz"),
        "seed_base": int(args.seed_base if args.seed_base is not None
                         else kmc.get("seed_base", 1000)),
        "out":       args.out or run.get("output_csv", "results.csv"),
        "workers":   int(args.workers or os.environ.get("SLURM_CPUS_PER_TASK", 0)
                         or kmc.get("n_workers", 1)),
        "lo":        lo,
        "hi":        hi,
        # the stall rule -- see the module docstring. 0.0 turns it off.
        "stall_fac": float(kmc.get("stall_idle_factor", 0.0)),
        "stall_min": int(kmc.get("stall_min_events", 0)),
        "stall_rel": float(kmc.get("stall_rel_tol", 0.0)),
        "stall_abs": float(kmc.get("stall_abs_tol_A", 0.0)),
        # ADAPTIVE n. Replicates are added a wave at a time until the SEM of the
        # window growth rate meets target_sem_pct AND its running mean is flat
        # (flat_region.py), or n_rep_max is reached. --nrep on the command line forces
        # a FIXED n and turns the adaptation off.
        "wave":      int(kmc.get("n_rep_wave", 48)),
        "nmax":      int(kmc.get("n_rep_max", 240)),
        "sem_tgt":   float(kmc.get("target_sem_pct", 0.0)),
        "adaptive":  (args.nrep is None) and float(kmc.get("target_sem_pct", 0.0)) > 0,
        "shard":     max(0, int(args.shard)),
        "nshards":   max(1, int(args.nshards)),
    }


# ---------------------------------------------------------------------------
# one trajectory
# ---------------------------------------------------------------------------
def _task(a):
    """Run ONE trajectory. Returns (T, seed, rep, sink, wall). Top-level so it pickles."""
    (T, params, lattice, seed, cap, rce, rep, hi,
     stall_fac, stall_min, stall_rel, stall_abs) = a
    sink = []
    np.random.seed(seed)
    t0 = time.perf_counter()
    KMC.run_kmc_trajectory(target_T=T, params=params, node_label=f"T{T:.0f}r{rep}",
                           traj_freq=None, lattice_file=lattice,
                           target_growth=UNREACHABLE_GROWTH_A, max_iter=cap,
                           checkpoint_every=rce, checkpoint_sink=sink,
                           stop_mean_height=hi,
                           stop_plateau_idle_factor=stall_fac,
                           stop_plateau_min_events=stall_min,
                           stop_plateau_rel_tol=stall_rel,
                           stop_plateau_abs_tol=stall_abs)
    return (T, seed, rep, sink, time.perf_counter() - t0)


def rows_from_sink(T, seed, rep, sink):
    out = []
    for ev, ts, max_h, mean_h in sink:
        tmin = ts / 60.0
        gr_max = (max_h * 1e-4) / tmin if tmin > 0 else 0.0
        gr_mean = (mean_h * 1e-4) / tmin if tmin > 0 else 0.0
        out.append([f"{T:.1f}", seed, rep, int(ev), f"{ts:.9e}",
                    f"{max_h:.4f}", f"{mean_h:.4f}",
                    f"{gr_max:.6f}", f"{gr_mean:.6f}"])
    return out


def append_atomic(path, rows, header=None):
    """Append rows under one exclusive lock; write the header exactly once. Many
    array elements write the same CSV concurrently, so this must be atomic."""
    if not rows:
        return
    head = ",".join(header or HEADER) + "\n"
    block = "".join(",".join(str(x) for x in r) + "\n" for r in rows)
    try:
        import fcntl
        with open(path, "a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0)
            if not f.readline():
                f.write(head)
            f.seek(0, os.SEEK_END)
            f.write(block); f.flush(); os.fsync(f.fileno())
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except ImportError:                                  # Windows smoke tests
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a") as f:
            if new:
                f.write(head)
            f.write(block); f.flush()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.toml")
    ap.add_argument("--temps", help="comma/space list, e.g. 900,1000,1125")
    ap.add_argument("--nrep", type=int, help="replicates PER TEMPERATURE")
    ap.add_argument("--cap", type=int, help="event cap")
    ap.add_argument("--ckpt", type=int, help="checkpoint cadence; must divide --cap")
    ap.add_argument("--window", help="growth-rate window lo,hi in angstrom, e.g. 50,200")
    ap.add_argument("--params", help="JSON of parameter overrides; omit for [params]")
    ap.add_argument("--lattice")
    ap.add_argument("--out", help="output CSV")
    ap.add_argument("--seed-base", type=int, dest="seed_base")
    ap.add_argument("--workers", type=int, help="pool size within this shard")
    ap.add_argument("--shard", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int,
                    default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    s = resolve(args, load_config(args.config))
    rce = reward_ckpt_every(s["cap"], s["ckpt"])

    # Replicate k of temperature T always uses seed_base + (T-index * nmax + k).
    # The seed for a given (T, k) does NOT depend on how many replicates this arm
    # ends up running, which is what keeps CRN pairing valid between arms that
    # adapt to different n.
    def build(lo, hi):
        out, = [[]]
        for ti, T in enumerate(s["temps"]):
            for rep in range(lo, hi):
                out.append((T, s["params"], s["lattice"],
                            s["seed_base"] + ti * s["nmax"] + rep,
                            s["cap"], rce, rep, s["hi"],
                            s["stall_fac"], s["stall_min"],
                            s["stall_rel"], s["stall_abs"]))
        return [t for i, t in enumerate(out)
                if i % max(1, s["nshards"]) == s["shard"]]

    first = s["wave"] if s["adaptive"] else s["nrep"]
    mine = build(0, first)
    tasks = mine

    # --- RESUME ---------------------------------------------------------------
    # Seeds are a pure function of (temperature index, replicate index), so a
    # replicate already recorded in <out>_traj.csv is bit-identical to the one
    # about to be run. Skip it. This makes an interrupted or re-scoped arm cheap
    # to extend -- raising n_replicates re-runs only the NEW replicates -- and it
    # is why the same file can be appended to safely.
    _done_keys = set()
    _prior = []          # (T, rep, window GR or None) already recorded: the adaptive test counts them
    _traj_probe = s["out"].replace(".csv", "_traj.csv")
    if os.path.exists(_traj_probe):
        try:
            with open(_traj_probe) as fh:
                for r in csv.DictReader(fh):
                    _done_keys.add((int(r["seed"]), int(r["rep"])))
                    _prior.append((float(r["temperature_K"]), int(r["rep"]),
                                   float(r["GR_window_umin"]) if r["reached"] == "1" else None))
        except Exception as e:
            print(f"[run_kmc] resume: could not read {_traj_probe} ({e}); running everything")
            _done_keys, _prior = set(), []

    def drop_done(batch):
        keep = [t for t in batch if (t[3], t[6]) not in _done_keys]
        if len(keep) != len(batch):
            print(f"[run_kmc] resume: {len(batch) - len(keep)} of {len(batch)} "
                  f"replicates already recorded -> skipped")
        return keep

    n_ckpt = s["cap"] // rce if rce else 1
    print(f"[run_kmc] engine      : KMC_bucket, numba kernels "
          f"{'ON' if KMC.NUMBA_ON else 'OFF (Python path, ~4.6x slower)'}")
    print(f"[run_kmc] params      : {s['params_src']}  ({len(s['params'])} overrides)")
    print(f"[run_kmc] temperatures: {len(s['temps'])} -> "
          f"{', '.join(f'{t:.0f}' for t in s['temps'])}")
    print(f"[run_kmc] growth rate : mean height through {s['lo']:g}-{s['hi']:g} A; a run "
          f"stops at {s['hi']:g} A")
    print(f"[run_kmc] stall rule  : " + (
          f"idle >= max({s['stall_min']:,}, {s['stall_fac']:g} x events at the last advance)"
          if s["stall_fac"] > 0 else "off"))
    if s["adaptive"]:
        print(f"[run_kmc] n_replicates: ADAPTIVE -- waves of {s['wave']} until "
              f"SEM(GR) <= {s['sem_tgt']:g} % and the running mean is flat at every "
              f"temperature (flat_region.py), hard cap {s['nmax']}")
    print(f"[run_kmc] {'first wave  ' if s['adaptive'] else 'n_replicates'}: {first}"
          f"   cap: {s['cap']:,} events")
    print(f"[run_kmc] checkpoints : every {rce:,} -> at most {n_ckpt} per trajectory"
          + ("" if rce == s["ckpt"] else
             f"   [!] requested {s['ckpt']:,} does not divide the cap; "
             f"forced to {rce:,} so a correct-baseline sample lands ON the cap]"))
    print(f"[run_kmc] lattice     : {s['lattice']}")
    print(f"[run_kmc] total tasks : {len(tasks)}   this shard "
          f"{s['shard']}/{s['nshards']}: {len(mine)}   pool: {s['workers']}")
    print(f"[run_kmc] output      : {s['out']}")
    if s["ckpt"] and rce != s["ckpt"]:
        print(f"[run_kmc] WARNING: checkpoint cadence adjusted (see above).")

    if args.dry_run:
        ms = MS_PER_EVENT_HPC[bool(KMC.NUMBA_ON)]
        core_h = len(tasks) * s["cap"] * ms * 1e-3 / 3600.0
        print(f"[run_kmc] DRY RUN -- nothing executed. Cap-bound worst case: "
              f"{core_h:,.0f} core-h at ~{ms:g} ms/event on an HPC core.")
        print(f"[run_kmc]   A trajectory that reaches {s['hi']:g} A, or stalls, stops "
              f"there, so the real cost is lower. Measure ms/event at these parameters "
              f"before sizing a long run.")
        for i, t in enumerate(mine[:5]):
            print(f"           task {i}: T={t[0]:.0f} seed={t[3]} rep={t[6]}")
        if len(mine) > 5:
            print(f"           ... and {len(mine) - 5} more in this shard")
        return

    if not mine:
        print("[run_kmc] nothing to do in this shard.")
        return

    t0 = time.perf_counter()

    # STREAM the results: append each trajectory the moment it lands, so a half-done
    # arm shows its progress, and films failing to reach the window show up while the
    # job runs. Rows are order-independent: reducers group by (seed, rep).
    reads, nrows = [], 0
    win_gr = list(_prior)            # (T, rep, window GR or None), this run and any before it
    traj_path = s["out"].replace(".csv", "_traj.csv")
    wave_path = s["out"].replace(".csv", "_waves.csv")
    TRAJ_HDR = ["temperature_K", "seed", "rep", "events_total", "sim_time_s_total",
                "max_height_A_final", "mean_height_A_final", "window_lo_A", "window_hi_A",
                "reached", "t_lo_s", "t_hi_s", "GR_window_umin", "stop", "wall_s"]
    WAVE_HDR = ["wave", "n_total", "n_reached", "reached_frac",
                "GR_mean", "GR_sd", "CV_pct", "SEM_pct",
                "target_SEM_pct", "implied_n", "decision", "T_K", "n_flat"]

    def _emit(res):
        nonlocal nrows
        T, seed, rep, sink, wall = res
        r = rows_from_sink(T, seed, rep, sink)
        append_atomic(s["out"], r)
        nrows += len(r)
        gr, reached, t_lo, t_hi, h_fin = window_from_sink(sink, s["lo"], s["hi"])
        ev_tot = int(sink[-1][0]) if sink else 0
        how = "window" if reached else ("stall" if ev_tot < s["cap"] else "cap")
        reads.append((T, gr, reached, h_fin))
        win_gr.append((T, rep, gr if reached else None))
        append_atomic(traj_path,
                      [[f"{T:.1f}", seed, rep, ev_tot,
                        f"{sink[-1][1]:.9e}" if sink else "",
                        f"{sink[-1][2]:.4f}" if sink else "",
                        f"{h_fin:.4f}", f"{s['lo']:g}", f"{s['hi']:g}",
                        int(reached),
                        "" if t_lo is None else f"{t_lo:.9e}",
                        "" if t_hi is None else f"{t_hi:.9e}",
                        "" if gr is None else f"{gr:.6f}",
                        how, f"{wall:.1f}"]],
                      header=TRAJ_HDR)

    def run_batch(batch):
        n = max(1, min(s["workers"], len(batch)))
        if n == 1:
            for t in batch:
                _emit(_task(t))
        else:
            with mp.Pool(processes=n) as pool:
                for res in pool.imap_unordered(_task, batch):
                    _emit(res)

    wave_no = 0

    def per_T(T):
        """This temperature's replicates so far, in replicate order: (n run, reached GRs)."""
        rows = sorted((x for x in win_gr if x[0] == T), key=lambda x: x[1])
        return len(rows), [x[2] for x in rows if x[2] is not None]

    def log_wave(T, decision, m=None, sd=None, cv=None, sem=None, need=None, nf=None):
        n_T, g = per_T(T)
        append_atomic(wave_path, [[wave_no, n_T, len(g),
                                   f"{len(g)/n_T:.4f}" if n_T else "",
                                   "" if m is None else f"{m:.6f}",
                                   "" if sd is None else f"{sd:.6f}",
                                   "" if cv is None else f"{cv:.4f}",
                                   "" if sem is None else f"{sem:.4f}",
                                   s["sem_tgt"],
                                   "" if need is None else need, decision,
                                   f"{T:.0f}", "" if nf is None else nf]],
                      header=WAVE_HDR)

    run_batch(drop_done(mine))
    done = first
    if s["adaptive"] and win_gr:             # a resumed run: carry on from what is recorded
        done = max(done, 1 + max(x[1] for x in win_gr))

    # --- ADAPTIVE TOP-UP -----------------------------------------------------
    # n is a property of the operating point, so it cannot be known before the
    # operating point is measured. Add a wave, re-measure, stop when, at EVERY
    # temperature, the SEM of the window growth rate meets the target AND the running
    # mean has gone flat (flat_region.py). Only trajectories that grew through the
    # window have a rate to average.
    while s["adaptive"] and done < s["nmax"]:
        wave_no += 1
        ok_all = True
        for T in s["temps"]:
            n_T, g = per_T(T)
            if len(g) < 3:
                print(f"[run_kmc] adaptive: T={T:.0f}: only {len(g)} of {n_T} grew through the "
                      f"window -- adding a wave to get a measurable CV")
                log_wave(T, "too few reached the window top to size")
                ok_all = False
                continue
            m = float(np.mean(g)); sd = float(np.std(g, ddof=1))
            cv = 100.0 * sd / m if m else float("inf")
            sem = cv / math.sqrt(len(g))
            need = math.ceil((cv / s["sem_tgt"]) ** 2)
            nf = flat_point(g)[0]
            flat = settled(g)
            print(f"[run_kmc] adaptive: T={T:.0f} n={n_T} reached={len(g)} "
                  f"({100.0*len(g)/n_T:.0f} %)  GR={m:.4f}  CV={cv:.2f} %  "
                  f"SEM={sem:.2f} % (target {s['sem_tgt']:g} %)  running mean flat from "
                  f"n={nf} of {len(g)}")
            if sem <= s["sem_tgt"] and flat:
                log_wave(T, "SEM target met and running mean flat", m, sd, cv, sem, need, nf)
            else:
                ok_all = False
                why = ("SEM above target (CV %.2f %% implies n ~ %d)" % (cv, need)
                       if sem > s["sem_tgt"] else
                       "running mean flat only from n=%d of %d (after %.0f %%)"
                       % (nf, len(g), 100 * FLAT_FRAC))
                print(f"[run_kmc] adaptive: T={T:.0f}: {why}")
                log_wave(T, why + " -- add a wave", m, sd, cv, sem, need, nf)
        if ok_all:
            print(f"[run_kmc] adaptive: SEM target met and every running mean flat at "
                  f"n={done}. stopping.")
            break
        nxt = min(done + s["wave"], s["nmax"])
        print(f"[run_kmc] adaptive: running replicates {done}..{nxt - 1}")
        run_batch(drop_done(build(done, nxt)))
        done = nxt
    else:
        if s["adaptive"]:
            print(f"[run_kmc] adaptive: hit n_rep_max = {s['nmax']}; "
                  f"reporting what was reached.")
            wave_no += 1
            for T in s["temps"]:
                n_T, g = per_T(T)
                log_wave(T, "hit n_rep_max", nf=flat_point(g)[0] if g else None)

    print(f"[run_kmc] per-trajectory summary -> {traj_path}")
    if s["adaptive"]:
        print(f"[run_kmc] adaptive sizing history -> {wave_path}")
    print(f"[run_kmc] {len(reads)} trajectories in "
          f"{(time.perf_counter() - t0) / 60:.1f} min -> {nrows} rows")
    for T in sorted({x[0] for x in reads}):
        v = [x for x in reads if x[0] == T]
        g = [x[1] for x in v if x[2]]
        hf = float(np.mean([x[3] for x in v]))
        if g:
            sd = float(np.std(g, ddof=1)) if len(g) > 1 else 0.0
            print(f"          T={T:>6.0f}  n={len(v):>3}  reached {len(g):>3}  "
                  f"GR={np.mean(g):.6f} +/- {sd:.6f} um/min  "
                  f"(final mean height {hf:.1f} A)")
        else:
            print(f"          T={T:>6.0f}  n={len(v):>3}  reached   0  "
                  f"did not grow through {s['lo']:g}-{s['hi']:g} A "
                  f"(final mean height {hf:.2f} A)")


if __name__ == "__main__":
    main()
