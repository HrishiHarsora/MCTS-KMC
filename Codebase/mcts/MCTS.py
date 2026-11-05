"""
MCTS.py — Monte Carlo Tree Search optimizer for KMC parameter fitting.

Reads ALL settings from config.toml. Do not hard-code anything here.

Usage:
    python MCTS.py                          # uses config.toml
    python MCTS.py --config my_config.toml  # custom config file
    python MCTS.py --C 5000                 # override UCB constant

The optimizer minimises the relative sum-of-squared-errors (Rel SSE) of the
simulated growth rate curve vs. the experimental [target_curve] across all
temperatures specified there. The best parameters are saved to a JSON file
(best_params.json by default) and can be loaded directly by KMC.py via
  --params best_params.json

DEPENDENCIES: numpy, scipy (via KMC.py), tomllib/tomli (for config.toml)
"""
from __future__ import annotations

import math
from math import log as _m_log, exp as _m_exp
import random
import os
import sys
import datetime
import json
import argparse
import shutil
import subprocess
import time
import multiprocessing as mp

# Force single-threaded numpy BEFORE importing it. Each node runs a pool of
# n_workers KMC trajectories; if numpy also spawned per-process BLAS threads the
# 48-core pool would oversubscribe and crawl. Must precede `import numpy`.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

# The bucket engine. Same public interface, same chemistry (it imports KMC.py). Found next
# to this file (a stage folder, everything flat) or in ../kmc (the repository layout);
# _KMC_DIR also locates kmc_worker.py for the srun fan-out.
_HERE_MCTS = os.path.dirname(os.path.abspath(__file__))
_KMC_DIR = next((os.path.abspath(d) for d in (_HERE_MCTS, os.path.join(_HERE_MCTS, os.pardir, "kmc"))
                 if os.path.exists(os.path.join(d, "KMC_bucket.py"))), _HERE_MCTS)
if _KMC_DIR not in sys.path:
    sys.path.insert(0, _KMC_DIR)
import KMC_bucket as KMC
from reward_metric import WINDOW_A, check_window, reward_ckpt_every, window_from_sink
# How many replicates is enough, and where the running mean went flat: ONE rule, shared with
# run_kmc.py's adaptive mode (kmc/flat_region.py, 2026-09-28).
from flat_region import flat_point, FLAT_FRAC

# Force UTF-8 stdout/stderr so non-ASCII chars (±, →) in print() don't crash on
# Windows' cp1252 console. No-op on systems already using UTF-8 (e.g. HPC Linux).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# Lightweight simulation logging (READ-ONLY; never perturbs a trajectory).
# Two CSVs, both tagged with the run's b<pct>_C<C> tag and appended across every
# node evaluation of the whole tree:
#   sim_log_<tag>.csv   one row per trajectory: events + per-reaction breakdown
#   sim_checkpoints_<tag>.csv  (growth, event) samples, gated by log_checkpoints
# The numbers come from KMC module globals set at the end of run_kmc_trajectory
# (read via getattr, so a future engine that omits them logs blanks, not crashes).
# =============================================================================
_NUM_RID_LOG = int(getattr(KMC, "_NUM_RID", 34))
# after the engine-ID columns, one column per numbered reaction (n_R1 ..): the sum of its
# engine IDs, so a reader sees counts by reaction without knowing the engine's IDs
_ENG = getattr(KMC, "K", KMC)                  # KMC_bucket exposes the engine module as K
_RXN_COLS = (_ENG._reaction_count_columns()
             if hasattr(_ENG, "_reaction_count_columns") else [])


def _sim_log_header() -> str:
    rids = ",".join([f"n_rid_{r:02d}" for r in range(1, _NUM_RID_LOG)] + _RXN_COLS)
    return ("experiment_label,replicate_index,T_K,seed,total_events,total_fired,"
            "final_thickness_A,GR_umin,wall_s," + rids + "\n")


_SIM_CKPT_HEADER = ("experiment_label,replicate_index,T_K,seed,"
                    "cumulative_events,thickness_A\n")


def _summ_row(exp_lbl, rep_idx, T, seed, gr, wall_s) -> str:
    """Build one sim_log row from the KMC read-only log globals (post-call)."""
    total_events = int(getattr(KMC, "_LAST_ITERATIONS", 0))
    thickness_A  = getattr(KMC, "_LAST_THICKNESS_A", None)
    counts       = getattr(KMC, "_LAST_EVENT_COUNTS", None)
    total_fired  = getattr(KMC, "_LAST_TOTAL_FIRED", None)
    if counts is not None:
        rid_cells = [str(int(counts[r])) for r in range(1, _NUM_RID_LOG)]
        rid_cells += [str(n) for n in _ENG._reaction_counts(counts)] if _RXN_COLS else []
    else:
        rid_cells = ["" for _ in range(1, _NUM_RID_LOG)] + [""] * len(_RXN_COLS)
    thick = f"{float(thickness_A):.6f}" if thickness_A is not None else ""
    fired = str(int(total_fired)) if total_fired is not None else ""
    return ",".join([exp_lbl, str(rep_idx), str(int(T)), str(int(seed)),
                     str(total_events), fired, thick, repr(float(gr)),
                     f"{wall_s:.3f}"] + rid_cells)


def _ckpt_lines(exp_lbl, rep_idx, T, seed) -> list:
    cps = getattr(KMC, "_LAST_CHECKPOINTS", None) or []
    return [f"{exp_lbl},{rep_idx},{int(T)},{int(seed)},{int(ce)},{float(th):.6f}"
            for (ce, th) in cps]


def _append_lines(path, header, lines):
    """Append `lines` to `path`, writing `header` first iff the file is new. The
    controller is single-process and evaluates nodes serially, so these sequential
    appends need no lock."""
    if not lines:
        return
    new = not os.path.exists(path)
    with open(path, "a") as fh:
        if new:
            fh.write(header)
        for ln in lines:
            fh.write(ln if ln.endswith("\n") else ln + "\n")


# =============================================================================
# Parallel KMC evaluation — one process per (temperature, replicate) trajectory.
# A node's 6 temps x n_replicates trajectories run across a process pool, so the
# node-evaluation cost is ~ (n_tasks / n_workers) instead of the full sum.
# =============================================================================
def _kmc_task(args):
    """Worker: run ONE KMC trajectory with an independent RNG seed. Top-level so
    it is picklable for multiprocessing. Returns (T, gr, summary_row|None,
    ckpt_lines|None) — the logging fields are read-only observation of what the
    trajectory already computed, so they cannot change the GR."""
    (T, params, lattice_file, seed, target_growth, max_iter,
     log_on, ckpt_on, ckpt_ev, exp_lbl, rep_idx, rce, win, stall) = args
    lo, hi = win
    # Legacy base-engine sampler left off; the bucket engine uses the checkpoint_sink.
    if hasattr(KMC, "_CHECKPOINT_EVERY"):
        KMC._CHECKPOINT_EVERY = ckpt_ev if (log_on and ckpt_on) else 0
    # THE GROWTH RATE = the mean height's rate through the window [lo, hi]
    # (reward_metric.py); the run stops at the first checkpoint at or above hi. The
    # engine's max-height return value is not used.
    sink = []                # (events, elapsed_s, max_height_A, mean_height_A)
    np.random.seed(seed)                      # independent stream per trajectory
    t0 = time.perf_counter()
    KMC.run_kmc_trajectory(
        target_T=T, params=params, node_label="p",
        traj_freq=None, lattice_file=lattice_file,
        target_growth=target_growth, max_iter=max_iter,
        checkpoint_every=rce, checkpoint_sink=sink,
        stop_mean_height=hi,
        stop_plateau_idle_factor=stall[0], stop_plateau_min_events=stall[1],
        stop_plateau_rel_tol=stall[2], stop_plateau_abs_tol=stall[3])
    wall_s = time.perf_counter() - t0
    # A film that did not grow through the window scores 0.0: it did not grow at the
    # rate being fitted. h is the mean height the film ended at.
    g, reached, _tlo, _thi, h_fin = window_from_sink(sink, lo, hi)
    gr = g if reached else 0.0
    summ = _summ_row(exp_lbl, rep_idx, T, seed, gr, wall_s) if log_on else None
    cps  = _ckpt_lines(exp_lbl, rep_idx, T, seed) if (log_on and ckpt_on) else None
    return (T, gr, h_fin, 1.0 if reached else 0.0, summ, cps)


def _eval_grs_parallel(target_temps, params, lattice_file, n_rep, n_workers,
                       base_seed, target_growth, max_iter, log_ctx=None,
                       reward_ckpt=100000):
    """
    Run every (temperature x replicate) trajectory in parallel and return
    {T: (mean_gr, std_gr)}. n_workers processes; node time ~ ceil(N/n_workers).
    Every trajectory grows to `target_growth` A or stops at `max_iter` events —
    the SAME termination as production, so dead nodes hit the cap instead of
    looping forever.
    """
    lc = log_ctx or {}
    log_on  = bool(lc.get("enabled", False))
    ckpt_on = bool(lc.get("log_checkpoints", False))
    ckpt_ev = int(lc.get("checkpoint_events", 50000))
    exp_lbl = str(lc.get("experiment_label", "node"))
    rce = reward_ckpt_every(max_iter, reward_ckpt)
    tasks, k = [], 0
    for T in target_temps:
        for r in range(n_rep):                # r = within-temperature replicate index
            tasks.append((T, params, lattice_file, base_seed + k,
                          target_growth, max_iter,
                          log_on, ckpt_on, ckpt_ev, exp_lbl, r, rce,
                          _WIN, _STALL))
            k += 1
    nproc = max(1, min(int(n_workers), len(tasks)))
    if nproc == 1:
        results = [_kmc_task(t) for t in tasks]
    else:
        with mp.Pool(processes=nproc) as pool:
            results = pool.map(_kmc_task, tasks)
    grouped = {T: {'grs': [], 'hs': [], 'reached': []} for T in target_temps}
    sum_rows, ckpt_rows = [], []
    for T, gr, hfin, reach, summ, cps in results:
        grouped[T]['grs'].append(gr)
        grouped[T]['hs'].append(hfin)
        grouped[T]['reached'].append(reach)
        if summ is not None: sum_rows.append(summ)
        if cps:              ckpt_rows.extend(cps)
    if log_on:
        _append_lines(lc["sim_log_path"], _sim_log_header(), sum_rows)
        if ckpt_on:
            _append_lines(lc["sim_ckpt_path"], _SIM_CKPT_HEADER, ckpt_rows)
    # RAW per-trajectory values, never a mean: the CALLER decides how many
    # trajectories this node needs, so it has to be able to pool waves.
    return grouped


# =============================================================================
# MULTI-NODE evaluation — srun fan-out.
# The controller (this process, running once on the batch node) writes a manifest
# of every (T, seed) trajectory for the node being evaluated, then launches them
# across the WHOLE allocation via `srun`, one single-core task per trajectory. At
# n_rep=130 across 3 nodes (144 single-core slots) all 130 run in ONE wave instead
# of the 48+48+34 three-wave pool. Waves auto-batch if tasks exceed the slot count
# (e.g. the future multi-temperature ladder), so this stays correct for any N.
#
# Same task ordering (T-major) and same seed (base_seed+k) as the pool path, so
# the gathered GRs — and thus reward and tree — are IDENTICAL to single-node.
# =============================================================================
def _run_wave(worker, manifest, evaldir, offset, count, per_node, retries):
    """Launch one wave of `count` trajectories (global indices offset..offset+count-1)
    and block until every gr_<gidx>.txt exists. Retry the wave on missing files so a
    transient node hiccup can't kill a 10-day job."""
    local = os.environ.get("MCTS_LOCAL_FANOUT", "") == "1"
    attempt = 0
    while True:
        if local:
            # Test harness: no SLURM. Spawn each task as its own process with
            # MCTS_PROCID set — exercises the EXACT manifest/worker/gather path.
            for pid in range(count):
                env = dict(os.environ); env["MCTS_PROCID"] = str(pid)
                subprocess.call([sys.executable, worker, manifest, evaldir,
                                 str(offset)], env=env)
            rc = 0
        else:
            cmd = ["srun", f"--ntasks={count}", f"--ntasks-per-node={per_node}",
                   "--cpus-per-task=1", "--cpu-bind=cores",
                   sys.executable, worker, manifest, evaldir, str(offset)]
            rc = subprocess.call(cmd)
        missing = [k for k in range(offset, offset + count)
                   if not os.path.exists(os.path.join(evaldir, f"gr_{k}.txt"))]
        if not missing:
            return
        attempt += 1
        if attempt > retries:
            raise RuntimeError(
                f"[MCTS/srun] wave (offset={offset}, count={count}) FAILED: "
                f"{len(missing)} missing GR files after {retries} retries "
                f"(srun rc={rc}); first missing gidx={missing[:5]}")
        print(f"[MCTS/srun] wave retry {attempt}/{retries}: {len(missing)} of "
              f"{count} GR files missing (srun rc={rc})", flush=True)


def _eval_grs_srun(target_temps, params, lattice_file, n_rep, base_seed,
                   target_growth, max_iter, log_ctx=None, reward_ckpt=100000):
    """srun-fan-out equivalent of _eval_grs_parallel. Returns {T:(mean,std)} —
    same shape, same numbers as the pool path for the same seeds."""
    lc = log_ctx or {}
    log_on  = bool(lc.get("enabled", False))
    ckpt_on = bool(lc.get("log_checkpoints", False))
    # Task list: identical ordering (T-major) and seed (base_seed+k) to the pool.
    tasks, k = [], 0
    for T in target_temps:
        for _ in range(n_rep):
            tasks.append((T, base_seed + k))
            k += 1
    N = len(tasks)

    # Allocation geometry: n_nodes * per_node single-core slots => one wave when
    # N <= slots (130 <= 144). Bigger N auto-batches into ceil(N/slots) waves.
    n_nodes  = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
    per_node = int(os.environ.get("MCTS_TASKS_PER_NODE", "48"))
    slots    = max(1, n_nodes * per_node)
    retries  = int(os.environ.get("MCTS_WAVE_RETRIES", "2"))

    here    = os.path.dirname(os.path.abspath(__file__))
    worker  = os.path.join(_KMC_DIR, "kmc_worker.py")
    scratch = os.environ.get("MCTS_SCRATCH", os.path.join(here, "mcts_scratch"))
    # Unique per-eval dir (pid+base_seed) on the shared FS so every node sees it.
    evaldir = os.path.join(scratch, f"eval_{os.getpid()}_{base_seed}")
    os.makedirs(evaldir, exist_ok=True)

    manifest = os.path.join(evaldir, "manifest.json")
    with open(manifest, "w") as fh:
        json.dump({
            "lattice_file":  lattice_file,
            "params":        params,
            "target_growth": target_growth,
            "max_iter":      max_iter,
            # The growth rate is read through the window [lo, hi] and the run stops
            # at hi; workers sample the mean height at this cadence (divides max_iter).
            "reward_ckpt_every": reward_ckpt_every(max_iter, reward_ckpt),
            # _WIN / _STALL: the module-level values the pool path reads too, so the
            # two paths cannot drift.
            "gr_window_A":       [float(_WIN[0]), float(_WIN[1])],
            "stall_idle_factor": float(_STALL[0]),
            "stall_min_events":  int(_STALL[1]),
            "stall_rel_tol":     float(_STALL[2]),
            "stall_abs_tol_A":   float(_STALL[3]),
            # T kept as-is (int) so gather keys match the pool's int keys exactly.
            "tasks":         [[T, int(s)] for (T, s) in tasks],
            # logging: workers write sum_<k>.txt / ckpt_<k>.csv into evaldir; the
            # controller gathers them below. n_rep lets a worker recover its
            # within-temperature replicate index (T-major tasks => k % n_rep).
            "n_rep":         int(n_rep),
            "log": {
                "enabled":           log_on,
                "log_checkpoints":   ckpt_on,
                "checkpoint_events": int(lc.get("checkpoint_events", 50000)),
                "experiment_label":  str(lc.get("experiment_label", "node")),
            },
        }, fh)

    waves = (N + slots - 1) // slots
    for w in range(waves):
        offset = w * slots
        count  = min(slots, N - offset)
        _run_wave(worker, manifest, evaldir, offset, count, per_node, retries)

    # Gather — group by the SAME (int) T the pool used, from the task list.
    grouped = {T: {'grs': [], 'hs': [], 'reached': []} for T in target_temps}
    for k, (T, _seed) in enumerate(tasks):
        with open(os.path.join(evaldir, f"gr_{k}.txt")) as fh:
            f = fh.read().split()        # 'gr h reached'
        grouped[T]['grs'].append(float(f[0]))
        grouped[T]['hs'].append(float(f[1]) if len(f) > 1 else float('nan'))
        grouped[T]['reached'].append(float(f[2]) if len(f) > 2 else 0.0)
    # Gather the per-trajectory log fragments the workers wrote, then append to the
    # tree-wide master CSVs BEFORE the scratch dir is deleted.
    if log_on:
        sum_rows, ckpt_rows = [], []
        for k in range(N):
            sp = os.path.join(evaldir, f"sum_{k}.txt")
            if os.path.exists(sp):
                with open(sp) as fh:
                    line = fh.read().strip()
                    if line:
                        sum_rows.append(line)
            if ckpt_on:
                cp = os.path.join(evaldir, f"ckpt_{k}.csv")
                if os.path.exists(cp):
                    with open(cp) as fh:
                        txt = fh.read().strip()
                        if txt:
                            ckpt_rows.extend(txt.splitlines())
        _append_lines(lc["sim_log_path"], _sim_log_header(), sum_rows)
        if ckpt_on:
            _append_lines(lc["sim_ckpt_path"], _SIM_CKPT_HEADER, ckpt_rows)

    shutil.rmtree(evaldir, ignore_errors=True)   # keep nothing between evals
    return grouped


# The growth-rate window (lo, hi) [A] and the stall rule (idle_factor, min_events,
# rel_tol, abs_tol_A); set by MCTSOptimizer.__init__ from [kmc].
_WIN = WINDOW_A
_STALL = (0.0, 0, 0.0, 0.0)


def _running_mean(vals):
    """Running mean after 1, 2, ... N trajectories, in replicate order."""
    a = np.asarray(vals, dtype=float)
    return np.cumsum(a) / np.arange(1, a.size + 1)


def _n_flat(vals):
    """Where the running mean of these replicates (in replicate order) went flat:
    kmc/flat_region.py -- the first n from which it stays inside the full mean +- 1 SEM
    AND has reached the middle of that band. A DIAGNOSTIC and the stop test for adding
    replicates; the node's rate is always the mean of ALL its replicates.

    Until 2026-09-28 this was a fixed +-1 % tolerance and the rate was read at n_flat. With
    the films' own scatter below 1 % that returned n_flat = 1 and scored every node on one
    film (1125 K Phase 2)."""
    return flat_point(vals)[0]


def _reward(L, tau):
    """The search reward. ONE definition, called by _evaluate and by _auto_C,
    so the constant that sets exploration can never drift from the quantity it
    is meant to be scaled against.

    L is in E-FOLDS, |ln(GR/target)|, matching tau, error_threshold, the DRC
    D = d ln GR / d ln k, and max_step_efolds. Bounded in (0, 1], monotone.
    """
    import math as _m
    if not (L == L) or L == float("inf"):
        return 0.0
    return _m.exp(-abs(L) / tau)


def _fit_effects(S_mat, y, v, b0):
    """RUN 5: learn one output's lever effects from the tree's own moves.

    Each move m is a parent -> child pair: S_mat[m] = the step actually taken on
    each lever (e-folds), y[m] = the measured change in ln GR or ln h, v[m] = its
    measurement variance (parent + child SEM^2). Model

        y = S b + e,   b ~ N(b0, a I),   e_m ~ N(0, v_m + r)

    b0 = the DRC's effects (the prior), a = how far the true effects may sit from
    the DRC, r = what a straight-line model cannot explain (curvature, the film
    thinning). a and r are NOT chosen: both are set to the values that make the
    observed moves most probable (maximum marginal likelihood). With no moves,
    or moves the DRC already explains, a -> 0 and the answer is the DRC.

    Returns (b_post, sd_post, a, r)."""
    S_mat = np.asarray(S_mat, float)
    y = np.asarray(y, float)
    v = np.asarray(v, float)
    b0 = np.asarray(b0, float)
    M = y.size
    K = b0.size
    if M == 0:
        return b0.copy(), np.zeros(K), 0.0, 0.0
    res = y - S_mat @ b0
    G = S_mat @ S_mat.T

    def nlml(la, lr):
        C = (10.0 ** la) * G + np.diag(v + 10.0 ** lr)
        try:
            Lc = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            return float("inf")
        z = np.linalg.solve(Lc, res)
        return 0.5 * float(z @ z) + float(np.sum(np.log(np.diag(Lc))))

    # Search in log space. The range is numerical, not a modelling choice: 1e-8 is
    # indistinguishable from "exactly the DRC", 1e2 allows effects ten e-folds off.
    best = (float("inf"), 0.0, 0.0)
    grid = [None, (0.5, 11), (0.1, 11)]        # coarse half-decade grid, then zoom twice
    for la in np.linspace(-8.0, 2.0, 21):
        for lr in np.linspace(-8.0, 2.0, 21):
            f = nlml(la, lr)
            if f < best[0]:
                best = (f, la, lr)
    for half, n in grid[1:]:
        _, la0, lr0 = best
        for la in np.linspace(la0 - half, la0 + half, n):
            for lr in np.linspace(lr0 - half, lr0 + half, n):
                f = nlml(la, lr)
                if f < best[0]:
                    best = (f, la, lr)
    _, la, lr = best
    a, r = 10.0 ** la, 10.0 ** lr
    C = a * G + np.diag(v + r)
    Ci = np.linalg.inv(C)
    b = b0 + a * (S_mat.T @ (Ci @ res))
    cov = a * np.eye(K) - a * a * (S_mat.T @ Ci @ S_mat)
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return b, sd, a, r


def _pred_L(e_gr, h_gap, d_gr, d_h):
    """Error in e-folds after a move predicted to change ln GR by d_gr[T] and ln h by
    d_h[T] at each temperature, at a node that needs ln GR to change by e_gr[T]
    (= ln target/GR there) and whose film sits h_gap[T] = ln(h_min/h) below the floor
    (negative = above it). The SAME form as the objective in _evaluate: the RMS over
    temperatures of the rate miss and the thickness shortfall."""
    tot, n = 0.0, 0
    for T, e in e_gr.items():
        hg = h_gap.get(T, -1e300)
        r = e - d_gr.get(T, 0.0)
        f = max(0.0, hg - d_h.get(T, 0.0)) if hg > -1e299 else 0.0
        tot += r * r + f * f
        n += 1
    return math.sqrt(tot / n) if n else float("inf")


def _damped(J, y, cap):
    """Least squares, damped just enough to fit inside the trust region.

    On a ladder the lever effects are nearly parallel from one temperature to the
    next, so J is ill-conditioned: the exact least-squares answer can demand tens of
    e-folds on a lever, and scaling that whole vector back to the trust region leaves
    a step that does nothing. Instead solve

        s(lam) = (J'J + lam I)^-1 J'y

    and take the smallest lam whose step already fits the region (max |s| <= cap).
    lam = 0 whenever the plain solution fits, so nothing changes where it was fine;
    where it does not, the damping drops the directions the data cannot resolve and
    keeps the ones it can. Nothing is chosen: cap IS the trust region.
    """
    JT = J.T
    A = JT @ J
    b = JT @ y
    n = A.shape[0]
    s0 = np.linalg.lstsq(J, y, rcond=None)[0]
    if not np.isfinite(s0).all():
        s0 = np.zeros(n)
    if np.max(np.abs(s0)) <= cap:
        return s0
    lo, hi = 0.0, max(1.0, float(np.trace(A)))
    for _ in range(60):
        while True:
            s = np.linalg.solve(A + hi * np.eye(n), b)
            if np.max(np.abs(s)) <= cap or hi > 1e12:
                break
            hi *= 4.0
        mid = 0.5 * (lo + hi)
        s = np.linalg.solve(A + mid * np.eye(n), b)
        if np.max(np.abs(s)) > cap:
            lo = mid
        else:
            hi = mid
    return np.linalg.solve(A + hi * np.eye(n), b)


def _apply(eff, steps):
    """{T: {lever: effect}} x {lever: step} -> {T: predicted change}."""
    return {T: sum(eff[T][k] * v for k, v in steps.items()) for T in eff}


def _auto_C(gr_root, target, tau, drc_D, step, n_children,
            h_gap=None, g_rate=None, g_h=None):
    """Exploration constant from the DRC, not from a guess.

    A child moves one lever by `step` e-folds of ln k, so its error becomes
    L_root - |D_i|*step and its reward follows. The spread of reward across
    the levers (plus a wrong-signed move) is the real signal UCB has to
    resolve; C is set so the first-visit exploration bonus C*sqrt(ln N) is
    worth about that much -- large enough that one unlucky evaluation cannot
    bury a sibling, small enough that a genuinely better lever still wins.

    Returns (C, diagnostics) and never returns 0."""
    import math as _m
    if gr_root <= 0 or not drc_D:
        return 0.1, {"note": "no root GR or no DRC magnitudes; fell back to 0.1"}
    Q = lambda L: _reward(L, tau)      # the SAME reward the tree scores on
    if g_rate and h_gap is not None:
        # RUN 4: each lever's step is scored with the two-gap error, rate AND
        # thickness, through the same _pred_L the moves are planned with.
        e = _m.log(target / gr_root)
        L0 = _pred_L(e, h_gap, 0.0, 0.0)
        qs = [Q(L0)]
        worst = None
        for k, g in g_rate.items():
            sg = step if g * e > 0 else -step          # the helpful direction
            qs.append(Q(_pred_L(e, h_gap, g * sg, g_h.get(k, 0.0) * sg)))
            if worst is None or abs(g) > abs(g_rate[worst]):
                worst = k
        sg = -step if g_rate[worst] * e > 0 else step  # wrong-signed step
        qs.append(Q(_pred_L(e, h_gap, g_rate[worst] * sg, g_h.get(worst, 0.0) * sg)))
        spread = max(qs) - min(qs)
        C = spread / _m.sqrt(_m.log(max(n_children, 3)))
        return max(C, 1e-3), {"L_root": L0, "h_gap_root": h_gap, "tau": tau,
                              "spread": spread, "N": n_children, "C": C}
    L0 = abs(_m.log(gr_root / target))
    qs = [Q(L0)]
    for D in drc_D.values():
        qs.append(Q(L0 - D * step))                 # correctly-signed step
    qs.append(Q(L0 + max(drc_D.values()) * step))   # wrong-signed step
    spread = max(qs) - min(qs)
    C = spread / _m.sqrt(_m.log(max(n_children, 3)))
    return max(C, 1e-3), {"L_root": L0, "tau": tau, "spread": spread,
                          "N": n_children, "C": C}


def _eval_grs(target_temps, params, lattice_file, n_rep, n_workers, base_seed,
              target_growth, max_iter, log_ctx=None, reward_ckpt=100000):
    """Dispatch: srun fan-out on a multi-node allocation, else the single-node
    multiprocessing.Pool. MCTS_PARALLEL=srun|pool|auto forces the choice
    (auto => srun iff SLURM_JOB_NUM_NODES>1). Backward compatible: a 1-node job
    is byte-for-byte the previous pool behaviour. `log_ctx` carries the read-only
    logging config/paths; both paths produce identical CSV rows. `reward_ckpt` is
    the MEAN-height checkpoint cadence both paths use to build the reward GR."""
    mode = os.environ.get("MCTS_PARALLEL", "auto").lower()
    if mode == "auto":
        mode = "srun" if int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) > 1 else "pool"
    if mode in ("srun", "fanout"):
        return _eval_grs_srun(target_temps, params, lattice_file, n_rep,
                              base_seed, target_growth, max_iter, log_ctx=log_ctx,
                              reward_ckpt=reward_ckpt)
    return _eval_grs_parallel(target_temps, params, lattice_file, n_rep,
                              n_workers, base_seed, target_growth, max_iter,
                              log_ctx=log_ctx, reward_ckpt=reward_ckpt)


# =============================================================================
# Load config
# =============================================================================
def _load_config(path: str) -> dict:
    try:
        import tomllib
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except ImportError:
        import tomli  # type: ignore
        with open(path, "rb") as fh:
            return tomli.load(fh)


def _build_settings(cfg: dict, c_override: float | None,
                    tight_pct_override: float | None = None):
    mcts_cfg = cfg.get("mcts", {})

    target_curve_raw = cfg.get("target_curve", {})
    target_curve     = {int(k): float(v) for k, v in target_curve_raw.items()}
    if not target_curve:
        raise ValueError("config.toml [target_curve] is empty — nothing to optimise against.")

    search_raw = cfg.get("search_params", {})
    if not search_raw and not cfg.get("tight_params"):
        raise ValueError("config.toml has no [search_params] or [tight_params].")

    # --- pinned background: [params] are NOT searched but MUST be applied to
    # every node evaluation. KMC.run_kmc_trajectory falls back to CATALOG
    # defaults for any key missing from the dict it receives, so omitting these
    # would silently evaluate the whole tree in the wrong background (this
    # exact bug invalidated the first MCTS#02 run: pinned Ea_E33/Ea_E12/S0_M
    # reverted to catalog and the root measured -44% instead of -11%).
    fixed_params = {str(k): float(v) for k, v in cfg.get("params", {}).items()}

    param_bounds  = {}
    initial_guess = {}

    # --- wide search parameters: [initial, low, high] ---
    for name, triplet in search_raw.items():
        if len(triplet) != 3:
            raise ValueError(
                f"[search_params] '{name}' must be [initial, low, high], got {triplet}")
        init, lo, hi = float(triplet[0]), float(triplet[1]), float(triplet[2])
        if not (lo <= init <= hi):
            raise ValueError(
                f"[search_params] '{name}': initial={init} outside [{lo}, {hi}]")
        param_bounds[name]  = (lo, hi)
        initial_guess[name] = init

    # --- tight parameters: param = center; bounds = center * (1 +- tight_pct/100) ---
    # These are the 1125 K (C=2) params, allowed only a small drift so the LT fit
    # cannot wreck the peak/HT fit. tight_pct is the swept "percentage bound".
    tight_pct = float(tight_pct_override if tight_pct_override is not None
                      else mcts_cfg.get("tight_pct", 3.0))
    for name, center in cfg.get("tight_params", {}).items():
        c = float(center)
        lo, hi = c * (1.0 - tight_pct / 100.0), c * (1.0 + tight_pct / 100.0)
        if hi < lo:
            lo, hi = hi, lo
        param_bounds[name]  = (lo, hi)
        initial_guess[name] = c

    if float(cfg.get("objective", {}).get("h_min_A", 0.0)) > 0.0:
        for blk in ("drc_rate", "drc_height"):
            for T in target_curve:
                got = cfg.get(blk, {}).get("T%d" % T)
                if got is None:
                    raise ValueError("[%s] has no block T%d" % (blk, T))
                miss = set(param_bounds) - set(got)
                if miss:
                    raise ValueError("[%s.T%d] lacks %s" % (blk, T, sorted(miss)))

    old = [k for k in ("stop_mean_height", "stop_plateau_idle_factor",
                       "stop_plateau_min_events", "stop_plateau_rel_tol",
                       "stop_plateau_abs_tol_A") if k in cfg.get("kmc", {})]
    if old:
        raise ValueError(
            f"[kmc] sets {old} -- keys of the plateau read, retired 2026-09-27. The rate "
            f"is read through gr_window_A = [lo, hi] and a run stops at hi; the stall "
            f"rule is stall_idle_factor, stall_min_events, stall_rel_tol, stall_abs_tol_A.")

    if "n_rep_wiggle_pct" in cfg.get("kmc", {}):
        raise ValueError(
            "[kmc] n_rep_wiggle_pct was retired 2026-09-28: a fixed +-1 % tolerance on the "
            "running mean returned n_flat = 1 whenever the films scatter by less than 1 %, and "
            "scored nodes on one film. The flat region is now kmc/flat_region.py (full mean "
            "+- 1 SEM, and the middle of that band reached) and a node's rate is the mean of all "
            "its replicates. Delete the key.")

    overlap = set(fixed_params) & set(param_bounds)
    if overlap:
        raise ValueError(
            f"[params] and the search space both define {sorted(overlap)} — "
            f"a parameter is either pinned or searched, never both.")

    return {
        "fixed_params":   fixed_params,
        "target_curve":   target_curve,
        "param_bounds":   param_bounds,
        "initial_guess":  initial_guess,
        "max_experiments": int(mcts_cfg.get("max_experiments", 500)),
        "max_depth":       int(mcts_cfg.get("max_depth",       7)),
        "max_children":    int(mcts_cfg.get("max_children",    4)),
        # Levers whose parameter is a BARRIER IN KELVIN: a step in e-folds of
        # ln k is step*T kelvin, not a multiplication. Everything else is the
        # rate knob itself and is multiplied. _expand needs this to be right;
        # treating a 5500 K barrier as multiplicative moves it to 14950 K in
        # one step.
        "additive_levers": [str(x) for x in mcts_cfg.get("additive_levers", [])],
        # C_ucb may be the string "auto" -> derived from the DRC after the root
        # is evaluated (see _auto_C). A number here or --C overrides.
        "C_ucb":           (float(c_override) if c_override else
                            (mcts_cfg.get("C_ucb", 2.0)
                             if isinstance(mcts_cfg.get("C_ucb", 2.0), str)
                             else float(mcts_cfg.get("C_ucb", 2.0)))),
        # |D| per lever, measured by the DRC -- used only to derive C.
        "drc_D":           {str(k): abs(float(v))
                            for k, v in cfg.get("drc_magnitude", {}).items()},
        # RUN 4: SIGNED effect of a +1 e-fold step of each searched lever on
        # ln GR and on ln h, the final mean height (DRC D and D_height, mapped in
        # build_search.py).
        # LT+1125: one block per temperature. [drc_rate.T900] etc. A step s of a
        # lever is defined ONCE for the whole ladder: multiplicative levers move by
        # exp(s); a barrier moves by s * step_ref_T kelvin, so its effect on ln k at
        # temperature T carries the factor step_ref_T / T. make_config bakes that in.
        "drc_rate":        {int(t[1:]): {str(k): float(v) for k, v in blk.items()}
                            for t, blk in cfg.get("drc_rate", {}).items()},
        "drc_height":      {int(t[1:]): {str(k): float(v) for k, v in blk.items()}
                            for t, blk in cfg.get("drc_height", {}).items()},
        "step_ref_T":      float(mcts_cfg.get("step_ref_T", 1125.0)),
        # Thickness floor [A]: a node is a result only if its films end above it at
        # every temperature. Set it to the window top, so every film must grow through
        # the whole window. 0 disables it (single temperature only).
        "h_min_A":         float(cfg.get("objective", {}).get("h_min_A", 0.0)),
        "max_step_efolds": float(mcts_cfg.get("max_step_efolds", 1.0)),
        # RUN 5: "drc" plans with the DRC effects throughout (run 4); "learned"
        # updates them from the tree's own moves after every node (_fit_effects).
        "planner_map":     str(mcts_cfg.get("planner_map", "drc")),
        "lever_map_csv":   str(cfg.get("logging", {}).get("lever_map_csv", "lever_map.csv")),
        "error_threshold": float(mcts_cfg.get("error_threshold", 5.0)),
        "traj_freq":       int(mcts_cfg.get("traj_freq", 0)) or None,
        "history_file":    mcts_cfg.get("history_file",     "mcts_history.txt"),
        "convergence_csv": mcts_cfg.get("convergence_csv",  "mcts_convergence.csv"),
        "best_params_json":mcts_cfg.get("best_params_json", "best_params.json"),
        "lattice_file":    cfg.get("kmc", {}).get("lattice_file", "Lattice.xyz"),
        # --- REPLICATES ARE NOT FIXED -----------------------------------
        # A node runs waves of n_rep_wave trajectories at ITS OWN parameters:
        # at least n_rep_total, then more while its running mean GR has not gone
        # flat (kmc/flat_region.py) or its SEM is above sem_target_pct, up to
        # n_rep_max. The node's rate is the mean of ALL its trajectories.
        # Seeds the CONTROLLER as well as the trajectories: with this fixed,
        # a rerun of the same config reproduces the same tree.
        "seed_base":        int(cfg.get("kmc", {}).get("seed_base", 90200000)),
        "n_rep_total":      int(cfg.get("kmc", {}).get("n_rep_total", 200)),
        "n_rep_wave":       int(cfg.get("kmc", {}).get("n_rep_wave", 48)),
        # ADAPTIVE. n_rep_total is the floor every node is measured to;
        # n_rep_max is the ceiling it may climb to while any temperature's
        # running mean is still moving at the end of the curve.
        "n_rep_max":        int(cfg.get("kmc", {}).get("n_rep_max", 0))
                            or int(cfg.get("kmc", {}).get("n_rep_total", 200)),
        # A node keeps sampling while CV/sqrt(n) at any temperature is above this.
        # 0 disables it and leaves only the running-mean test.
        "sem_target_pct":   float(cfg.get("kmc", {}).get("sem_target_pct", 0.0)),
        "save_interval":   int(cfg.get("logging", {}).get("save_interval", 1)),
        "node_waves_csv":  str(cfg.get("logging", {}).get("node_waves_csv",
                                                         "node_waves.csv")),
        # Trajectory termination: the window top (stop_mean_height), the stall rule,
        # or max_iterations events. The growth FRONT must never end a run, so
        # target_growth defaults out of reach; a front stop at 150 A would end every
        # film below a 200 A window top and score it 0.
        "target_growth":   float(cfg.get("kmc", {}).get("target_growth_angstrom", 1.0e6)),
        "max_iterations":  int(cfg.get("kmc", {}).get("max_iterations", 0)) or None,
        # Mean-height checkpoint cadence; the window crossings are interpolated between
        # checkpoints. Forced to divide max_iterations (reward_metric.reward_ckpt_every).
        "reward_ckpt_every": int(cfg.get("kmc", {}).get("reward_checkpoint_events",
                                                        20000)),
        # THE GROWTH RATE: the mean height's rate through [lo, hi] (reward_metric.py).
        "gr_window_A": check_window(*cfg.get("kmc", {}).get("gr_window_A", WINDOW_A)),
        # The stall rule: end a trajectory whose mean height has not advanced for
        # max(min_events, idle_factor x events at its last advance). Without it a dead
        # film burns its whole cap producing nothing. 0 turns it off.
        "stall_fac": float(cfg.get("kmc", {}).get("stall_idle_factor", 0.0)),
        "stall_min": int(cfg.get("kmc", {}).get("stall_min_events", 0)),
        "stall_rel": float(cfg.get("kmc", {}).get("stall_rel_tol", 0.0)),
        "stall_abs": float(cfg.get("kmc", {}).get("stall_abs_tol_A", 0.0)),
        # --- corrected error / reward + directed-search settings ---
        # reward = exp(-L/tau), L = RMS ln(GR/target) in e-folds (see _reward)
        # tau may be "auto": set to L_root/3 from the MEASURED root, like C.
        "tau":             (mcts_cfg.get("tau") if isinstance(mcts_cfg.get("tau"), str)
                            else float(mcts_cfg.get("tau", 0.1))),
        # fraction of expansion moves steered by sign(error) x sign(DRC); rest random
        "expand_bias":     float(mcts_cfg.get("expand_bias", 0.7)),
        # {param: +1/-1} sign of each parameter's effect on GR (from DRC).
        # Missing/zero entries fall back to undirected random perturbation.
        "drc_signs":       {str(k): int(v) for k, v in cfg.get("drc_signs", {}).items()},
        # percentage half-width of the tight (C=2) parameter bounds (swept variable)
        "tight_pct":       tight_pct,
        # parallel workers for the per-node KMC trajectories. Default = the SLURM
        # allocation (SLURM_CPUS_PER_TASK), else config [kmc] n_workers, else 20.
        "n_workers":       int(os.environ.get("SLURM_CPUS_PER_TASK",
                               cfg.get("kmc", {}).get("n_workers", 20))),
        # --- lightweight simulation logging (read-only; see [logging] in config) ---
        # Per-trajectory summary rows always; growth checkpoints gated on
        # log_checkpoints. Filenames are b<pct>_C<C>-tagged in __main__ so parallel
        # C-grid jobs in one dir never collide.
        "log_enabled":       bool(cfg.get("logging", {}).get("enabled", True)),
        "log_checkpoints":   bool(cfg.get("logging", {}).get("log_checkpoints", True)),
        "checkpoint_events": int(cfg.get("logging", {}).get("checkpoint_events", 50000)),
        "sim_log_csv":       cfg.get("logging", {}).get("sim_log_csv", "sim_log.csv"),
        "sim_ckpt_csv":      cfg.get("logging", {}).get("sim_ckpt_csv",
                                                        "sim_checkpoints.csv"),
    }


# =============================================================================
# Tree node
# =============================================================================
class MCTSNode:
    def __init__(self, parameters: dict, depth: int, parent=None):
        self.parameters = parameters
        self.depth      = depth
        self.parent     = parent
        self.children:  list[MCTSNode] = []
        self.visits:    int   = 0
        # max_reward is BACK-PROPAGATED (max over this node's whole subtree) —
        # after backprop the root always holds the global max, so it must never
        # be used to identify which node was best. own_reward is this node's
        # own evaluation only; _best_node() compares that.
        self.max_reward: float = -float("inf")
        self.own_reward: float = -float("inf")
        # mean signed relative error (sim-exp)/exp at this node; used to steer
        # the direction of expansion. <0 = undershoot (need more growth).
        self.signed_err: float = 0.0
        self.L: float = float('inf')   # error in e-folds, set by _evaluate
        # the two gaps _expand aims at, PER TEMPERATURE. e_gr[T] = ln(target/GR),
        # the change in ln GR needed there; h_gap[T] = ln(h_min/h), > 0 when that
        # film is too thin.
        self.e_gr: dict = {}
        self.h_gap: dict = {}
        self.rate_err: dict = {}
        self.film_err: dict = {}
        self.h_all: dict = {}
        self.var_lngr: dict = {}
        self.var_lnh: dict = {}
        self.lngr: dict = {}
        self.lnh: dict = {}


# =============================================================================
# Logger
# =============================================================================
class Logger:
    def __init__(self, txt_path: str, csv_path: str, settings: dict):
        self.txt_path  = txt_path
        self.csv_path  = csv_path
        self.settings  = settings
        self.count     = 0
        # Checkpoint the best params to disk every N experiments so a SLURM
        # wall-time kill (job dies mid-loop) can never lose the best-so-far.
        self.best_params_json = settings.get("best_params_json", "best_params.json")
        self.save_interval    = int(settings.get('save_interval', 1))
        self.best_reward    = -float("inf")
        self.best_params    = None
        self.best_rel_sse   = None
        self.best_max_err   = None
        self.best_exp_idx   = None
        self.best_film_ok   = False
        self.best_h         = (float("nan"), float("nan"))
        self.best_L_rate    = float("nan")

        with open(txt_path, "w") as f:
            f.write("MCTS KMC EXPLORATION HISTORY\n")
            f.write(f"Started      : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"C (UCB)      : {settings['C_ucb']}\n")
            f.write(f"seed_base    : {settings['seed_base']} (run is reproducible)\n")
            f.write(f"Temperatures : {sorted(settings['target_curve'])}\n")
            f.write(f"Parameters   : {list(settings['param_bounds'])}\n")
            fixed = settings.get("fixed_params", {})
            if fixed:
                f.write("Pinned [params] (merged into every evaluation):\n")
                for k, v in fixed.items():
                    f.write(f"  {k:<28}: {v:.6g}\n")
            f.write("=" * 70 + "\n\n")

        with open(csv_path, "w") as f:
            f.write("Experiment,Label,Depth,Rel_RMSE,Rel_SSE,Max_Pct_Error,Reward,"
                    "Per_T_Pct_Errors,L_rate,L_h,h_min_final_A,h_thinnest_A,film_above_floor\n")

    def log(self, node: MCTSNode, rel_sse: float, max_err: float,
            reward: float, label: str, per_T=None, L_rmse=float("nan"),
            L_rate=float("nan"), L_h=float("nan"), h_A=float("nan"),
            film_ok=True, h_all=float("nan"), per_T_h=None):
        self.count += 1
        # RUN 7: a node whose film is above the floor always outranks one that is
        # not; among nodes on the same side, the higher reward wins. best_params
        # therefore only ever holds a below-floor node while none is above it,
        # and says so in its metrics.
        better = (film_ok and not self.best_film_ok) or \
                 (film_ok == self.best_film_ok and reward > self.best_reward)
        if self.best_params is None or better:
            self.best_reward  = reward
            self.best_params  = node.parameters.copy()
            self.best_rel_sse = rel_sse
            self.best_max_err = max_err
            self.best_exp_idx = self.count
            self.best_film_ok = bool(film_ok)
            self.best_h       = (h_A, h_all)
            self.best_L_rate  = L_rate

        # compact per-temperature error string, e.g. "900:+5.1;925:-3.2;..."
        per_T = per_T or []
        per_T_str = ";".join(f"{int(T)}:{pct:+.2f}" for T, _s, _e, pct in per_T)

        with open(self.txt_path, "a") as f:
            f.write(f"[Experiment {self.count}]  {label}\n")
            f.write(f"  Depth : {node.depth}\n")
            for k, v in node.parameters.items():
                f.write(f"  {k:<28}: {v:.6g}\n")
            if per_T:
                f.write("  Per-temperature (Sim / Exp / Err):\n")
                for T, sim, exp, pct in per_T:
                    f.write(f"     {int(T)} K : {sim:.4f} / {exp:.4f} / {pct:+.2f}%\n")
            if per_T_h:
                f.write("  Film per temperature (mean / all) [A]:\n")
                for T, hf, ha in per_T_h:
                    f.write(f"     {int(T)} K : {hf:.2f} / {ha:.2f}\n")
            f.write(f"  {'thinnest film [A]':<28}: {h_all:.4f}\n")
            f.write(f"  {'film above floor':<28}: {'YES' if film_ok else 'NO'}\n")
            f.write(f"  {'L_rate [e-folds]':<28}: {L_rate:.8f}\n")
            f.write(f"  {'L_h [e-folds]':<28}: {L_h:.8f}\n")
            f.write(f"  {'Rel RMSE':<28}: {L_rmse:.8f}\n")
            f.write(f"  {'Rel SSE':<28}: {rel_sse:.8f}\n")
            f.write(f"  {'Max % Error':<28}: {max_err:+.4f}%\n")
            f.write(f"  {'Reward':<28}: {reward:.6f}\n")
            f.write("-" * 70 + "\n\n")

        with open(self.csv_path, "a") as f:
            f.write(f"{self.count},{label},{node.depth},"
                    f"{L_rmse:.8f},{rel_sse:.8f},{max_err:.4f},{reward:.6f},"
                    f"{per_T_str},{L_rate:.8f},{L_h:.8f},{h_A:.4f},{h_all:.4f},{int(bool(film_ok))}\n")

        # Periodic checkpoint: rewrite best_params.json every save_interval
        # experiments so the best result survives a wall-time / node failure.
        if self.count % self.save_interval == 0:
            self.save_best_json(verbose=False)

    def finalize(self):
        if self.best_params is None:
            return
        with open(self.txt_path, "a") as f:
            f.write("=" * 70 + "\n")
            f.write("BEST RESULT\n")
            f.write(f"  Experiment  : {self.best_exp_idx}\n")
            f.write(f"  Reward      : {self.best_reward:.6f}\n")
            f.write(f"  Rel SSE     : {self.best_rel_sse:.8f}\n")
            f.write(f"  Max % Error : {self.best_max_err:.4f}%\n")
            f.write(f"  Film        : {self.best_h[0]:.2f} A mean, {self.best_h[1]:.2f} A over all "
                    f"trajectories -> {'ABOVE the floor' if self.best_film_ok else 'NO NODE ABOVE THE FLOOR -- NOT A RESULT'}\n")
            for k, v in self.best_params.items():
                f.write(f"  {k:<28}: {v:.6g}\n")
            f.write("=" * 70 + "\n")
            f.write(f"Finished : {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n")

    def save_best_json(self, path: str | None = None, verbose: bool = True):
        if self.best_params is None:
            return
        path = path or self.best_params_json
        # "params" is the FULL evaluated parameter set (pinned [params] merged
        # under the searched values) so the file is self-contained: loading it
        # with KMC.py --params reproduces exactly what the MCTS evaluated.
        fixed = self.settings.get("fixed_params", {})
        out = {
            "params": {**fixed, **self.best_params},
            "searched_params": list(self.best_params),
            "pinned_params":   list(fixed),
            "metrics": {
                "rel_sse":       self.best_rel_sse,
                "max_pct_error": self.best_max_err,
                "reward":        self.best_reward,
                "experiment":    self.best_exp_idx,
                "film_above_floor": self.best_film_ok,
                "h_final_A":     self.best_h[0],
                "h_all_A":       self.best_h[1],
                "h_floor_A":     self.settings.get("h_min_A", 0.0),
                "L_rate":        self.best_L_rate,
            },
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        if verbose:
            print(f"[MCTS] Best parameters saved to: {path}")
        else:
            print(f"[MCTS] checkpoint @ exp {self.count}: best_params → {path} "
                  f"(reward={self.best_reward:.6f}, max_err={self.best_max_err:.2f}%)")


# =============================================================================
# Optimizer
# =============================================================================
class MCTSOptimizer:
    def __init__(self, settings: dict, logger: Logger):
        self.S         = settings
        self.logger    = logger
        if hasattr(_ENG, "check_constraints"):     # a root outside the constraints is an error
            _ENG.check_constraints({**settings["fixed_params"], **settings["initial_guess"]})
        self.root      = MCTSNode(settings["initial_guess"].copy(), depth=0)
        self.C         = settings["C_ucb"]      # may still be the string "auto"
        # REPRODUCIBILITY: every expansion draw comes from this module's
        # `random`, and every trajectory seed is derived from seed_base by
        # the evaluation index below. Same config + same code => same tree.
        random.seed(int(settings["seed_base"]))
        global _WIN, _STALL
        _WIN = tuple(settings["gr_window_A"])
        _STALL = (float(settings.get("stall_fac", 0.0)),
                  int(settings.get("stall_min", 0)),
                  float(settings.get("stall_rel", 0.0)),
                  float(settings.get("stall_abs", 0.0)))
        self.total_v   = 0
        self.moves     = []        # RUN 5: (steps, d ln GR, d ln h, var_gr, var_h, label)
        self._map_n    = -1
        self._map      = None

    # ---- tree helpers ----

    def _branch_full(self, node: MCTSNode) -> bool:
        if node.depth >= self.S["max_depth"]: return True
        if len(node.children) < self.S["max_children"]: return False
        return all(self._branch_full(c) for c in node.children)

    # ---- evaluate one node ----

    def _evaluate(self, node: MCTSNode, label: str):
        print(f"\n[Depth {node.depth}] {label}")
        for k, v in node.parameters.items():
            print(f"  {k}: {v:.6g}")

        rel_sse    = 0.0      # relative, kept for the log only
        ef_sse     = 0.0      # e-folds, the actual objective
        max_err    = 0.0
        signed_sum = 0.0
        n_pts      = 0
        stalled    = False
        per_T      = []   # (T, sim, exp, signed_pct_err) for logging

        # All (temperature x replicate) trajectories for this node run in parallel
        # across n_workers processes. node cost ~ ceil(n_temps*n_rep / n_workers).
        # Pinned [params] MUST be merged under the node's searched values here —
        # KMC falls back to catalog defaults for any key it doesn't receive.
        temps = sorted(self.S["target_curve"].keys())
        eval_params = {**self.S["fixed_params"], **node.parameters}
        log_ctx = {
            "enabled":           self.S["log_enabled"],
            "log_checkpoints":   self.S["log_checkpoints"],
            "checkpoint_events": self.S["checkpoint_events"],
            "experiment_label":  label,
            "sim_log_path":      self.S["sim_log_csv"],
            "sim_ckpt_path":     self.S["sim_ckpt_csv"],
        }
        # ---- EVERY NODE IS MEASURED TO n_rep_total, THEN READ BACK --------
        # No early stop. Each node runs n_rep_total trajectories at ITS OWN
        # parameters -- in waves only because the pool holds n_workers at a
        # time -- and the replicate count is then read off the running-mean
        # curve by _n_flat, the DRC viewer's rule. Rate AND height are reported
        # at that n. Both are properties of this node's parameter set.
        pool_gr = {T: [] for T in temps}
        pool_h  = {T: [] for T in temps}
        pool_pl = {T: [] for T in temps}
        # Derived, never drawn: evaluation index k gets its own seed block, so
        # node k's trajectories are the same on a rerun no matter what the
        # tree did before it. Stride 10000 >> n_rep_total * n_temps, so two
        # nodes can never share a seed.
        node_seed = int(self.S['seed_base']) + 10000 * int(self.logger.count)
        n_done = 0
        n_floor = int(self.S['n_rep_total'])
        n_ceil  = max(n_floor, int(self.S.get('n_rep_max', n_floor)))

        def _still_moving():
            """True if this node is not yet measured well enough, on EITHER test.

            1. RETROSPECTIVE -- the running mean only went flat (kmc/flat_region.py)
               in the last 20 % of the trajectories run: n_flat > FLAT_FRAC x n.
            2. PREDICTIVE -- the standard error of the mean, CV/sqrt(n) from this
               node's OWN scatter, is still above sem_target_pct. The first test can
               only certify the data it has: a curve can look flat to n = 96 and
               move at n = 100 if the distribution has a tail. The SEM test does not
               depend on the curve's shape, and a tail raises CV, so a node that
               develops one keeps sampling.

            Both are read per temperature; the worst temperature decides."""
            sem_t = float(self.S.get('sem_target_pct', 0.0)) / 100.0
            for Tk in temps:
                gk = np.asarray(pool_gr[Tk], dtype=float)
                if gk.size == 0:
                    return True
                if _n_flat(gk) > FLAT_FRAC * gk.size:
                    return True
                if sem_t > 0.0 and gk.mean() > 0:
                    sem = (gk.std(ddof=1) / gk.mean()) / np.sqrt(gk.size) if gk.size > 1 else 1.0
                    if sem > sem_t:
                        return True
            return False

        while n_done < n_floor or (n_done < n_ceil and _still_moving()):
            take = min(self.S['n_rep_wave'], n_ceil - n_done)
            if take <= 0:
                break
            if n_done >= n_floor:
                print('    [replicates] means still moving at n=%d -- adding %d more (ceiling %d)'
                      % (n_done, take, n_ceil))
            wave_ctx = dict(log_ctx)
            # tag the wave, so two waves of one node cannot write rows that
            # look like the same replicate index
            wave_ctx['experiment_label'] = '%s#n%d' % (label, n_done + take)
            part = _eval_grs(
                temps, eval_params, self.S['lattice_file'],
                take, self.S['n_workers'],
                base_seed=node_seed + n_done * max(1, len(temps)),
                target_growth=self.S['target_growth'],
                max_iter=self.S['max_iterations'],
                log_ctx=wave_ctx,
                reward_ckpt=self.S['reward_ckpt_every'])
            for T in temps:
                pool_gr[T].extend(part[T]['grs'])
                pool_h[T].extend(part[T]['hs'])
                pool_pl[T].extend(part[T]['reached'])
            n_done += take

        # --- the node's rate: the mean of ALL its trajectories -----------------
        # Every trajectory, in replicate order; one that did not grow through the
        # window counts as 0.0. n_flat (kmc/flat_region.py) is recorded with it: where
        # the running mean settled, and the check that the replicates were enough.
        gr_map   = {}
        h_map    = {}
        reach_map = {}
        n_by_T   = {}
        curves   = {}
        for T in temps:
            g_all = np.asarray(pool_gr[T], dtype=float)
            h_all = np.asarray(pool_h[T],  dtype=float)
            q_all = np.asarray(pool_pl[T], dtype=float)
            reach_map[T] = float(q_all.mean()) if q_all.size else 0.0
            # EVERY trajectory is scored (run 6: filtering on a stop flag scored
            # nodes on a few trajectories). A film that did not grow through the
            # window contributes 0.0 to the rate and its final height to h.
            g, h = g_all, h_all
            rg = _running_mean(g)
            nf = _n_flat(g)
            n_by_T[T]  = nf
            curves[T]  = (rg, _running_mean(np.nan_to_num(h, nan=0.0)), g.size)
            gr_map[T]  = (float(g.mean()), float(g.std())) if g.size else (0.0, 0.0)
            h_map[T]   = float(np.nanmean(h)) if h.size else float('nan')
            # thickness over ALL trajectories at this temperature (the floor test reads
            # it), and the variance of ln(mean) = CV^2 / n from this node's own replicates
            node.h_all[T] = h_map[T]
            if g.size and g.mean() > 0 and np.nanmean(h) > 0:
                node.var_lngr[T] = float((g.std() / g.mean()) ** 2 / g.size)
                node.var_lnh[T] = float((np.nanstd(h) / np.nanmean(h)) ** 2 / h.size)

        n_used = max(n_by_T.values()) if n_by_T else 0
        for T in temps:
            nf, (rg, rh, n_ok) = n_by_T[T], curves[T]
            print('    rate = mean of all %d run (%.0f %% grew through the window); running mean flat from n=%d;   mean final height %.2f A'
                  % (n_ok, 100.0 * reach_map[T], nf, h_map[T]))
            if n_ok and nf > FLAT_FRAC * n_ok:
                print('    WARNING: flat only from n=%d of %d -- the running mean was still moving' % (nf, n_ok))
                print('             near the end, even at the n_rep_max ceiling of %d.' % n_ceil)
        node.n_used    = n_used
        node.h_final = h_map
        node.frac_reached = reach_map

        # The whole curve goes to disk, one row per n, so the flat region and
        # the marked n can be looked at rather than taken on trust.
        if self.S.get('log_enabled', False):
            rows = []
            for T in temps:
                rg, rh, _n_ok = curves[T]
                for k in range(rg.size):
                    rows.append('%s,%d,%d,%.8f,%.4f,%d'
                                % (label, T, k + 1, rg[k], rh[k],
                                   1 if (k + 1) == n_by_T[T] else 0))
            _append_lines(self.S['node_waves_csv'],
                          'label,T_K,n,running_mean_GR_umin,'
                          'running_mean_final_height_A,is_n_flat' + chr(10), rows)

        for T in temps:
            target_gr = self.S["target_curve"][T]
            gr, gr_std = gr_map[T]
            if gr <= 0.0:
                print(f"  WARNING: KMC stalled at T={T} K.")
                stalled = True; break

            rel_err    = (gr - target_gr) / target_gr   # signed relative error
            rel_sse   += rel_err ** 2   # log only
            # THE OBJECTIVE IS IN E-FOLDS. A relative error cannot carry this
            # fit: the root sits 68x above the target, so exp(-68/tau)
            # underflows to 0 and every early node scores an identical 0 --
            # the tree would have no signal to select on at all.
            ef          = _m_log(gr / target_gr)
            # thickness floor, in the same e-folds. Zero at or above h_min.
            h_min = self.S.get("h_min_A", 0.0)
            hT = h_map[T]
            if h_min > 0.0:
                h_gap = (_m_log(h_min / hT) if (hT == hT and hT > 0.0) else float("inf"))
            else:
                h_gap = -float("inf")
            Lh = max(0.0, h_gap)
            node.e_gr[T], node.h_gap[T] = -ef, h_gap      # per temperature
            node.rate_err[T], node.film_err[T] = abs(ef), Lh
            ef_sse     += ef ** 2 + Lh ** 2
            signed_sum += ef
            n_pts     += 1
            pct        = abs(rel_err) * 100.0
            if pct > max_err: max_err = pct
            per_T.append((T, gr, target_gr, rel_err * 100.0))
            node.gr = gr        # stashed so _auto_C can read the root's rate
            print(f"  T={T} K  Exp={target_gr:.4f}  "
                  f"Sim={gr:.4f}±{gr_std / max(1.0, math.sqrt(n_done)):.4f} (mean of {n_done}, SEM; flat from n={n_by_T[T]})  "
                  f"h={h_map[T]:.2f} A  reached={100.0*reach_map[T]:.0f}%  "
                  f"Err={pct:+.2f}%  L_rate={abs(ef):.4f}  L_h={Lh:.4f}")

        # ---- objective + bounded reward, both in E-FOLDS ----
        # L = RMS of ln(GR/target) over temperatures; single-T => |ln(GR/target)|.
        # reward = exp(-L/tau) via _reward, the same function _auto_C scales C by.
        if stalled or n_pts == 0:
            rel_sse = float("inf"); max_err = float("inf")
            L = float("inf"); reward = 0.0
            node.signed_err = -1.0           # sim~0 < exp -> undershoot
            # a dead film has no gap of its own to aim at; a child of it re-plans
            # from the parent's gaps.
            if node.parent is not None:
                node.e_gr = dict(node.parent.e_gr)
                node.h_gap = dict(node.parent.h_gap)
        else:
            L = math.sqrt(ef_sse / n_pts)          # RMS error in e-folds
            if isinstance(self.S["tau"], str):
                # The first node evaluated is the root. tau = L_root/3 keeps the
                # root near reward e^-3 whatever the metric makes its error.
                self.S["tau"] = max(L / 3.0, 1e-3)
                print("[MCTS] tau derived from the measured root: L_root %.4f -> tau %.4f"
                      % (L, self.S["tau"]))
                try:
                    with open(self.logger.txt_path, "a") as _f:
                        _f.write("tau derived after the root evaluation: L_root %.6f -> tau %.6f" % (L, self.S["tau"]) + chr(10) + chr(10))
                except Exception:
                    pass
            reward = _reward(L, self.S["tau"])
            node.signed_err = signed_sum / n_pts   # mean signed error (direction)

        node.own_reward = reward
        self._record_move(node, gr_map, h_map)
        # Does this node's film clear the floor AT EVERY TEMPERATURE? A node that
        # does not can guide the search but can never be the result. The mean final
        # height over all trajectories is what counts (h_map and h_all are both that).
        _floor = float(self.S.get("h_min_A", 0.0))
        worst, worstT = float("inf"), None
        for T in temps:
            hs = [float(h_map.get(T, float("nan"))), float(node.h_all.get(T, float("nan")))]
            if any(h != h for h in hs):
                worst, worstT = float("nan"), T
                break
            if min(hs) < worst:
                worst, worstT = min(hs), T
        node.h_floor_check = worst
        node.film_ok = (_floor <= 0.0) or (worst == worst and worst > _floor)
        node.L_rate = math.sqrt(sum(v * v for v in node.rate_err.values()) / max(1, len(node.rate_err)))
        node.L_h = math.sqrt(sum(v * v for v in node.film_err.values()) / max(1, len(node.film_err)))
        print("  film check: thinnest film %s A (at %s K) -> %s the %g A floor"
              % ("%.2f" % worst if worst == worst else "n/a", worstT,
                 "ABOVE" if node.film_ok else "BELOW", _floor))
        node.L = L        # e-folds; _expand sizes its steps by this
        L_rmse = L if math.isfinite(L) else float("inf")
        print(f"  → L(e-folds)={L:.6f}  Max Err={max_err:.2f}%  Reward={reward:.6f}  "
              f"signed_err={node.signed_err:+.4f}")
        self.logger.log(node, rel_sse, max_err, reward, label, per_T=per_T, L_rmse=L_rmse,
                        film_ok=node.film_ok,
                        h_all=float(node.h_floor_check),
                        L_rate=getattr(node, "L_rate", float("nan")),
                        L_h=getattr(node, "L_h", float("nan")),
                        h_A=float(min((h_map[T] for T in temps if h_map[T] == h_map[T]),
                                      default=float("nan"))),
                        per_T_h=[(T, h_map[T], node.h_all.get(T, float("nan"))) for T in temps])
        # Second value is L in E-FOLDS: error_threshold is compared against
        # it, and the config states that threshold in e-folds.
        return reward, L

    # ---- UCB selection ----

    def _select(self, node: MCTSNode) -> MCTSNode:
        while (len(node.children) >= self.S["max_children"]
               or node.depth >= self.S["max_depth"]):
            if not node.children: break
            best_ucb, best_child = -float("inf"), None
            for child in node.children:
                if self._branch_full(child): continue
                if child.visits == 0:
                    ucb = float("inf")
                else:
                    ucb = (child.max_reward +
                           self.C * math.sqrt(math.log(self.total_v) / child.visits))
                if ucb > best_ucb:
                    best_ucb, best_child = ucb, child
            if best_child is None: break
            node = best_child
        return node

    # ---- expand ----

    def _expand(self, node: MCTSNode) -> MCTSNode | None:
        if node.depth >= self.S["max_depth"]: return None
        if self.S.get("h_min_A", 0.0) > 0.0:
            return self._expand_two_gap(node)
        return self._expand_rate_only(node)

    def _steps_between(self, par, child):
        """The step vector actually taken, in the ladder's own units: e-folds for a
        multiplicative lever, kelvin / step_ref_T for a barrier. A lever clipped at
        its bound is recorded as the step it really took."""
        add = set(self.S.get("additive_levers", []))
        Tr = float(self.S["step_ref_T"])
        return [((child.parameters[k] - par.parameters[k]) / Tr) if k in add
                else math.log(child.parameters[k] / par.parameters[k])
                for k in self.S["param_bounds"]]

    def _record_move(self, node, gr_map, h_map):
        """Store the parent -> child move this evaluation measured, one record per
        temperature where both ends grew."""
        par = node.parent
        for T in sorted(self.S["target_curve"]):
            gr, h = gr_map.get(T, (0.0, 0.0))[0], h_map.get(T, float("nan"))
            node.lngr[T] = math.log(gr) if gr > 0 else float("nan")
            node.lnh[T] = math.log(h) if (h == h and h > 0) else float("nan")
        if par is None:
            return
        st = self._steps_between(par, node)
        for T in sorted(self.S["target_curve"]):
            vals = (node.lngr.get(T, float("nan")), node.lnh.get(T, float("nan")),
                    par.lngr.get(T, float("nan")), par.lnh.get(T, float("nan")))
            if not all(math.isfinite(x) for x in vals):
                continue
            self.moves.append((st, vals[0] - vals[2], vals[1] - vals[3],
                               node.var_lngr.get(T, 0.0) + par.var_lngr.get(T, 0.0),
                               node.var_lnh.get(T, 0.0) + par.var_lnh.get(T, 0.0),
                               self.logger.count, T))

    def _effects(self):
        """Lever effects on ln GR and ln h AT EACH TEMPERATURE, used to plan the next
        move: the DRC's (planner_map = "drc"), or those learned from the moves measured
        so far at that temperature (planner_map = "learned"). Refitted only when a new
        move has arrived, and written to the history and lever_map_*.csv each time."""
        keys = list(self.S["param_bounds"])
        temps = sorted(self.S["target_curve"])
        g0 = {T: [self.S["drc_rate"][T][k] for k in keys] for T in temps}
        h0 = {T: [self.S["drc_height"][T][k] for k in keys] for T in temps}
        if self.S.get("planner_map", "drc") != "learned":
            return ({T: dict(zip(keys, g0[T])) for T in temps},
                    {T: dict(zip(keys, h0[T])) for T in temps})
        if self._map_n == len(self.moves):
            return self._map
        G, H, lines, rows = {}, {}, [], []
        for T in temps:
            mv = [m for m in self.moves if m[6] == T]
            Sm = [m[0] for m in mv]
            bg, sg, ag, rg = _fit_effects(Sm, [m[1] for m in mv], [m[3] for m in mv], g0[T])
            bh, sh, ah, rh = _fit_effects(Sm, [m[2] for m in mv], [m[4] for m in mv], h0[T])
            G[T] = dict(zip(keys, map(float, bg)))
            H[T] = dict(zip(keys, map(float, bh)))
            lines.append("  %4d K  from %d moves   rate: a %.3g r %.3g   height: a %.3g r %.3g"
                         % (T, len(mv), ag, rg, ah, rh))
            for i, k in enumerate(keys):
                lines.append("      %-12s rate %+.4f -> %+.4f (sd %.4f)   height %+.4f -> %+.4f (sd %.4f)"
                             % (k, g0[T][i], bg[i], sg[i], h0[T][i], bh[i], sh[i]))
                rows.append("%d,%d,%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6g,%.6g,%.6g,%.6g"
                            % (self.logger.count, T, len(mv), k, g0[T][i], bg[i], sg[i],
                               h0[T][i], bh[i], sh[i], ag, rg, ah, rh))
        self._map = (G, H)
        self._map_n = len(self.moves)
        head = "[lever map] refitted after %d measured moves" % len(self.moves)
        print(head + "\n" + "\n".join(lines), flush=True)
        try:
            with open(self.logger.txt_path, "a") as fh:
                fh.write(head + "\n" + "\n".join(lines) + "\n\n")
        except Exception:
            pass
        _append_lines(self.S["lever_map_csv"],
                      "after_experiment,T_K,n_moves,lever,rate_drc,rate_learned,rate_sd,"
                      "height_drc,height_learned,height_sd,a_rate,r_rate,a_height,r_height\n",
                      rows)
        return self._map

    def _plan_move(self, e_gr, h_gap):
        """The DIRECTED move for a node: the smallest step vector (least squares in
        step space) that the current lever effects predict will

          * change ln GR at EVERY temperature by that temperature's own gap e_gr[T],
          * and, at any temperature where that alone would leave the film below the
            floor, also lift ln h there to the floor.

        With more temperatures than levers the rate rows cannot all be met exactly;
        numpy's least-squares solution is the minimum-norm best compromise, which is
        what the ladder needs -- no temperature is privileged. Returns
        ({lever: step}, which) where which reports how many height rows were added."""
        keys = list(self.S["param_bounds"])
        g_map, h_map = self._effects()
        temps = sorted(e_gr)
        Gm = np.array([[g_map[T][k] for k in keys] for T in temps], float)
        Hm = np.array([[h_map[T][k] for k in keys] for T in temps], float)
        y = np.array([e_gr[T] for T in temps], float)
        s = _damped(Gm, y, float(self.S.get("max_step_efolds", 1.0)))
        pred_h = Hm @ s
        bind = [i for i, T in enumerate(temps)
                if h_gap.get(T, -1e300) > -1e299 and pred_h[i] < h_gap[T]]
        which = "rate"
        if bind:
            J = np.vstack([Gm, Hm[bind]])
            yy = np.concatenate([y, [h_gap[temps[i]] for i in bind]])
            s = _damped(J, yy, float(self.S.get("max_step_efolds", 1.0)))
            which = "rate+height(%d T)" % len(bind)
        return {k: float(v) for k, v in zip(keys, s)}, which

    def _expand_two_gap(self, node: MCTSNode) -> MCTSNode | None:
        """RUN 4 expansion: the child aims at the rate gap AND the thickness gap.

        1. Plan. d = _plan_move(node): the smallest move the DRC's linear model
           says closes both gaps.
        2. Trust region. d is scaled AS A WHOLE so that no lever moves more than
           max_step_efolds -- scaling the vector rather than clipping lever by
           lever keeps the balance between the levers that hold the thickness.
        3. How far. One random fraction u in [0, 1) per child, along the plan.
           Along the plan the predicted error falls monotonically, so the
           directed part never predicts a worse child than its parent.
        4. Exploration. A random step on every lever, scaled so that the whole
           child is still predicted to be no worse than its parent: the kick
           may spend at most what the directed part gained (triangle inequality
           on _pred_L). No fixed share of random moves; a child that goes further
           along its plan earns more room to wander.
        """
        depth  = node.depth + 1
        Tref   = float(self.S["step_ref_T"])
        add    = set(self.S.get("additive_levers", []))
        max_ef = float(self.S.get("max_step_efolds", 1.0))
        g_r, g_h = self._effects()      # DRC, or learned from the tree
        temps = sorted(self.S["target_curve"])
        e_gr = {T: float(node.e_gr.get(T, 0.0)) for T in temps}
        # h_gap is +inf only for a film with no height at all; such a node also
        # has no rate and scores 0, so aim it at the floor from just below.
        hg = {}
        for T in temps:
            v = float(node.h_gap.get(T, -1e300))
            hg[T] = (0.0 if v > 0 else -1e300) if not math.isfinite(v) else v
        zero = {T: 0.0 for T in temps}
        L_node = _pred_L(e_gr, hg, zero, zero)

        plan, which = self._plan_move(e_gr, hg)
        big = max(abs(v) for v in plan.values()) if plan else 0.0
        c = min(1.0, max_ef / big) if big > 0 else 0.0
        u = random.random()
        kick = {k: random.uniform(-1.0, 1.0) for k in self.S["param_bounds"]}

        directed = {k: u * c * plan[k] for k in self.S["param_bounds"]}
        L_dir = _pred_L(e_gr, hg, _apply(g_r, directed), _apply(g_h, directed))
        budget = max(0.0, L_node - L_dir)
        # worst case the kick can undo, by the triangle inequality on _pred_L,
        # measured with the largest per-temperature effect of each lever
        worst = sum(abs(v) * max(math.hypot(g_r[T][k], g_h[T][k]) for T in temps)
                    for k, v in kick.items())
        scale = (budget / worst) if worst > 0 else 0.0
        # Same per-lever trust region for the whole step. |directed| <= max_ef,
        # so clamping only shortens the kick and the bound above still holds.
        steps = {k: max(-max_ef, min(max_ef, directed[k] + kick[k] * scale))
                 for k in directed}

        new_p = {}
        for key, (lo, hi) in self.S["param_bounds"].items():
            v = node.parameters[key]
            if key in add:
                v = v + steps[key] * Tref       # barrier in kelvin
            else:
                v = v * _m_exp(steps[key])      # the rate knob itself
            new_p[key] = max(lo, min(hi, v))
        new_p = self._constrain(new_p)

        dg, dh = _apply(g_r, steps), _apply(g_h, steps)
        L_child = _pred_L(e_gr, hg, dg, dh)
        print("  [expand] plan=%s  u=%.3f  trust=%.3f  kick budget %.3f  L %.3f -> %.3f"
              % (which, u, c, budget, L_node, L_child))
        for T in temps:
            print("      %4d K  need d lnGR %+.3f, predicted %+.3f%s"
                  % (T, e_gr[T], dg[T],
                     "   floor gap %+.3f, predicted d ln h %+.3f" % (hg[T], dh[T])
                     if hg[T] > -1e299 else ""))
        child = MCTSNode(new_p, depth=depth, parent=node)
        child.planned = {"which": which, "u": u, "trust": c, "budget": budget,
                         "steps": steps, "L_pred": L_child}
        node.children.append(child)
        return child

    def _expand_rate_only(self, node: MCTSNode) -> MCTSNode | None:
        """Run-3 expansion, verbatim. Used when no thickness floor is set."""
        depth = node.depth + 1
        signs = self.S["drc_signs"]
        bias  = self.S["expand_bias"]

        # Direction the growth rate needs to move at this node:
        #   signed_err < 0 (undershoot)  -> need MORE growth  -> need = +1
        #   signed_err > 0 (overshoot)   -> need LESS growth   -> need = -1
        need = 0.0
        if node.signed_err < 0:   need = +1.0
        elif node.signed_err > 0: need = -1.0

        # A move is a step in E-FOLDS OF ln k, not in raw parameter units, so
        # one step means the same thing on a sticking coefficient and on a
        # barrier in kelvin -- and means what max_step_efolds says it means.
        #
        # The step is capped by what the lever needs to close THIS node's own
        # error: s_max = min(max_step_efolds, L_node / |D_lever|). The root,
        # 4.2 e-folds out, takes the full trust region; a node 0.19 out takes
        # 0.13 e-folds. Run 1 halved a parameter window with depth instead,
        # which let a depth-3 child overshoot its parent's error twelvefold.
        # SINGLE TEMPERATURE ONLY. This path converts a step to kelvin with the
        # stage's one temperature; on a ladder that silently uses the coldest
        # rung for every rung. _expand routes here only when h_min_A <= 0, which
        # the LT+1125 stage never does -- but fail loudly rather than quietly.
        if len(self.S['target_curve']) > 1:
            raise RuntimeError("_expand_rate_only is single-temperature; a ladder "
                               "must use _expand_two_gap (set objective.h_min_A > 0)")
        T       = float(sorted(self.S['target_curve'])[0])
        add     = set(self.S.get('additive_levers', []))
        Dmag    = self.S.get('drc_D', {})
        max_ef  = float(self.S.get('max_step_efolds', 1.0))
        L_node  = getattr(node, 'L', float('inf'))
        if not (L_node == L_node) or L_node == float('inf'):
            L_node = max_ef * 10.0          # stalled or unmeasured: no useful cap

        new_p = {}
        steps = {}
        for key, (lo, hi) in self.S['param_bounds'].items():
            D_k   = float(Dmag.get(key, 0.0))
            reach = (L_node / D_k) if D_k > 0 else max_ef
            s_max = min(max_ef, reach)
            sgn   = signs.get(key, 0)
            if sgn != 0 and need != 0.0 and random.random() < bias:
                # directed: parameter direction = need x sign(dGR/dparam)
                direction = need * (1.0 if sgn > 0 else -1.0)
                step = direction * random.uniform(0.0, s_max)
            else:
                step = random.uniform(-s_max, s_max)

            steps[key] = step

        # TRUST REGION ON THE WHOLE MOVE, not just lever by lever. Five levers
        # move at once and a directed expansion pushes them the same way, so
        # their predicted effects ADD. Project the move back so that the
        # predicted |d ln GR| never exceeds the parent's own error: a child
        # cannot be thrown further than the distance it is meant to close.
        pred = sum(abs(steps[k]) * float(Dmag.get(k, 0.0)) for k in steps)
        if pred > L_node > 0.0:
            scale = L_node / pred
            for k in steps:
                steps[k] *= scale

        for key, (lo, hi) in self.S['param_bounds'].items():
            step = steps[key]
            v = node.parameters[key]
            if key in add:
                # a barrier in kelvin: d ln k = -dEa/T, so `step` e-folds of the
                # PARAMETER's own effect is step*T kelvin
                v = v + step * T
            else:
                # multiplicative lever: the parameter IS the rate knob
                v = v * _m_exp(step)
            new_p[key] = max(lo, min(hi, v))
        new_p = self._constrain(new_p)

        child = MCTSNode(new_p, depth=depth, parent=node)
        node.children.append(child)
        return child

    def _constrain(self, new_p: dict) -> dict:
        """Clip a child's parameters back inside the engine's ordering constraints
        (KMC._CONSTRAINTS: S0 terrace <= step, step barrier < terrace barrier, Ga with
        two bonds leaves before one with three), holding the pinned [params]. The engine
        would refuse the set otherwise, and the node would be wasted."""
        if not hasattr(_ENG, "enforce_constraints"):
            return new_p
        out, clipped = _ENG.enforce_constraints(new_p, fixed=self.S["fixed_params"])
        if clipped:
            print("  [expand] ordering constraint: clipped %s" % ", ".join(clipped))
        return out

    # ---- backpropagate ----

    def _backprop(self, node: MCTSNode, reward: float):
        while node is not None:
            node.visits += 1
            if reward > node.max_reward: node.max_reward = reward
            node = node.parent

    # ---- best node in tree ----

    def _best_node(self) -> MCTSNode:
        best = self.root
        stack = [self.root]
        while stack:
            n = stack.pop()
            ok_n, ok_b = getattr(n, "film_ok", False), getattr(best, "film_ok", False)
            if (ok_n and not ok_b) or (ok_n == ok_b and n.own_reward > best.own_reward):
                best = n
            stack.extend(n.children)
        return best

    # ---- main loop ----

    def run(self):
        print("=" * 70)
        print(f"MCTS Optimizer | C={self.C} | {len(self.S['param_bounds'])} parameters")
        print(f"Temperatures   : {sorted(self.S['target_curve'])}")
        print(f"Max experiments: {self.S['max_experiments']}")
        if self.S["fixed_params"]:
            print(f"Pinned [params]: {self.S['fixed_params']}")
        print("=" * 70)

        root_reward, root_err = self._evaluate(self.root, "Root")
        self.total_v          += 1
        self.root.visits      += 1
        self.root.max_reward   = root_reward

        # --- C from the DRC, now that the root GR is measured ------------
        if isinstance(self.C, str):
            tgt = list(self.S["target_curve"].values())[0]
            self.C, diag = self._auto_C_ladder()
            print("\n[MCTS] C derived from the DRC, not set by hand:")
            for k, v in diag.items():
                print("        %-8s %s" % (k, ("%.4f" % v) if isinstance(v, float) else v))
            print("        C = spread / sqrt(ln N) = %.4f\n" % self.C)
            # Provenance on disk, not only on stdout: the history header was
            # written before the root existed and therefore says "auto".
            try:
                with open(self.logger.txt_path, "a") as _f:
                    _f.write("C (UCB) derived after the root evaluation" + chr(10))
                    for _k, _v in diag.items():
                        _f.write("  %-8s %s" % (_k, _v) + chr(10))
                    _f.write("  C        %.6f" % self.C + chr(10) + chr(10))
            except Exception:
                pass

        if self._converged(self.root):
            print(f"\nEARLY STOP: Root rate within {self.S['error_threshold']} e-folds and film above the floor.")
            self.logger.save_best_json()
            return self._best_node()

        iteration = 0
        while True:
            if self._branch_full(self.root):
                print("\nSTOP: Tree fully expanded.")
                break
            if self.logger.count >= self.S["max_experiments"]:
                print(f"\nSTOP: Reached {self.S['max_experiments']} evaluations.")
                break

            iteration += 1
            print(f"\n--- Round {iteration} ---")
            leaf = self._select(self.root)

            if leaf.depth >= self.S["max_depth"]:
                self.total_v += 1
                self._backprop(leaf, leaf.max_reward)
                continue

            n_expand = self.S["max_children"] - len(leaf.children)
            for c_idx in range(n_expand):
                if self.logger.count >= self.S["max_experiments"]: break
                child = self._expand(leaf)
                if child is None: break
                label   = f"Exp{self.logger.count + 1}_R{iteration}_C{c_idx + 1}"
                reward, err = self._evaluate(child, label)
                self.total_v += 1
                self._backprop(child, reward)
                if self._converged(child):
                    print(f"\nEARLY STOP: rate within {child.L_rate:.4f} <= {self.S['error_threshold']} e-folds "
                          f"AND film {child.h_floor_check:.2f} A > {self.S.get('h_min_A', 0.0):g} A")
                    self.logger.save_best_json()
                    return self._best_node()

        if not self.logger.best_film_ok and self.S.get("h_min_A", 0.0) > 0.0:
            print("\n" + "!" * 70 + "\nNO NODE HAS A FILM ABOVE THE %g A FLOOR. best_params is marked "
                  "film_above_floor = false and is NOT a result.\n" % self.S["h_min_A"] + "!" * 70)
        return self._best_node()

    def _auto_C_ladder(self):
        """C from the root, on the ladder. Each lever is stepped by max_step_efolds
        in whichever direction helps, its predicted error computed with the same
        _pred_L the moves are planned with, and C set so the first-visit bonus
        C*sqrt(ln N) matches the spread of the resulting rewards -- the signal UCB
        has to resolve. A wrong-signed step on the strongest lever sets the low end."""
        temps = sorted(self.S["target_curve"])
        g, h = self._effects()
        step = float(self.S.get("max_step_efolds", 1.0))
        e_gr = {T: float(self.root.e_gr.get(T, 0.0)) for T in temps}
        hg = {}
        for T in temps:
            v = float(self.root.h_gap.get(T, -1e300))
            hg[T] = (0.0 if v > 0 else -1e300) if not math.isfinite(v) else v
        zero = {T: 0.0 for T in temps}
        L0 = _pred_L(e_gr, hg, zero, zero)
        Q = lambda L: _reward(L, self.S["tau"])
        qs, strongest, best = [Q(L0)], None, -1.0
        for k in self.S["param_bounds"]:
            pull = sum(g[T][k] * e_gr[T] for T in temps)          # helpful direction
            sg = step if pull > 0 else -step
            one = {kk: (sg if kk == k else 0.0) for kk in self.S["param_bounds"]}
            qs.append(Q(_pred_L(e_gr, hg, _apply(g, one), _apply(h, one))))
            if abs(pull) > best:
                best, strongest = abs(pull), (k, -sg)
        if strongest:
            k, sg = strongest
            one = {kk: (sg if kk == k else 0.0) for kk in self.S["param_bounds"]}
            qs.append(Q(_pred_L(e_gr, hg, _apply(g, one), _apply(h, one))))
        spread = max(qs) - min(qs)
        C = max(spread / math.sqrt(math.log(max(self.S["max_children"], 3))), 1e-3)
        return C, {"L_root": L0, "tau": self.S["tau"], "spread": spread,
                   "N": self.S["max_children"], "C": C}

    def _converged(self, node) -> bool:
        """RUN 7 stop rule. BOTH must hold, separately:
          the rate is within error_threshold e-folds of the target, and
          the film is above the floor (the mean final height over all trajectories).
        Runs 4-6 stopped on the combined error instead, which let run 6 stop at a
        film of 18.6 A because its rate was close."""
        if self.S.get("h_min_A", 0.0) <= 0.0:
            return getattr(node, "L", float("inf")) <= self.S["error_threshold"]
        # every temperature must be inside the threshold on its own, and every
        # film above the floor -- an RMS would let one temperature hide behind five
        errs = getattr(node, "rate_err", {}) or {}
        return (bool(errs)
                and max(errs.values()) <= self.S["error_threshold"]
                and bool(getattr(node, "film_ok", False)))


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="MCTS optimizer for KMC parameters. Reads settings from config.toml."
    )
    ap.add_argument("--config", default="config.toml",
                    help="Path to TOML config file (default: config.toml)")
    ap.add_argument("--C", type=float, default=None,
                    help="Override UCB exploration constant C")
    ap.add_argument("--tight-pct", type=float, default=None,
                    help="Override percentage half-width of the [tight_params] bounds "
                         "(the swept 'percentage bound' around the 1125 K params)")
    args = ap.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: config file '{args.config}' not found.")
        sys.exit(1)

    cfg      = _load_config(args.config)
    settings = _build_settings(cfg, args.C, args.tight_pct)

    # Tag every output file with BOTH the bound % and the C value, so a 2-D grid
    # of jobs launched in the SAME directory never overwrites another's results.
    # C may still be the string "auto" here -- it is derived only after the
    # root has been evaluated, which is long after the output names are fixed.
    # Tag with "auto"; the derived value is printed and stored in the manifest.
    _c = settings['C_ucb']
    _cs = _c if isinstance(_c, str) else f"{_c:g}"
    c_tag = f"b{settings['tight_pct']:g}_C{_cs}"
    def _tag(path):
        root, ext = os.path.splitext(path)
        return f"{root}_{c_tag}{ext}"
    settings["history_file"]     = _tag(settings["history_file"])
    settings["convergence_csv"]  = _tag(settings["convergence_csv"])
    settings["best_params_json"] = _tag(settings["best_params_json"])
    settings["sim_log_csv"]      = _tag(settings["sim_log_csv"])
    settings["sim_ckpt_csv"]     = _tag(settings["sim_ckpt_csv"])
    settings["node_waves_csv"]   = _tag(settings["node_waves_csv"])
    settings["lever_map_csv"]    = _tag(settings["lever_map_csv"])
    print(f"[MCTS] tight_pct=±{settings['tight_pct']:g}%  C={_cs}  "
          f"→  outputs tagged '{c_tag}'")

    logger = Logger(
        txt_path  = settings["history_file"],
        csv_path  = settings["convergence_csv"],
        settings  = settings,
    )

    optimizer = MCTSOptimizer(settings, logger)
    best_node = optimizer.run()

    logger.finalize()
    logger.save_best_json(settings["best_params_json"])

    print("\n" + "=" * 70)
    print("OPTIMISATION COMPLETE")
    print(f"  History : {settings['history_file']}")
    print(f"  CSV     : {settings['convergence_csv']}")
    print(f"  Params  : {settings['best_params_json']}")
    print("=" * 70)
    print("\nBest parameters found:")
    for k, v in best_node.parameters.items():
        print(f"  {k:<28}: {v:.6g}")
    print(f"\nTo run KMC with these parameters:")
    print(f"  python KMC.py --params {settings['best_params_json']}")
