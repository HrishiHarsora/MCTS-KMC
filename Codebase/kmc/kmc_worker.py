"""
kmc_worker.py -- ONE KMC trajectory, launched as one srun task of an MCTS-node
evaluation fan-out.

The MCTS controller (MCTS.py, running once on the batch node) writes a manifest
listing every (temperature, seed) trajectory for the node it is evaluating, then
launches this script across the whole multi-node allocation via

    srun --ntasks=N --ntasks-per-node=48 python kmc_worker.py <manifest> <outdir> <offset>

SLURM gives each task a unique SLURM_PROCID in [0, N). This task's GLOBAL task
index is  offset + SLURM_PROCID . It runs exactly that one trajectory with the
seed the controller assigned it and writes the growth rate to
<outdir>/gr_<gidx>.txt (atomic rename). The controller gathers all gr_*.txt.

CRITICAL: this changes only WHERE a trajectory runs, not WHAT it computes. The
seed and the KMC call are byte-identical to the single-node multiprocessing.Pool
worker (_kmc_task in MCTS.py), so the GR — and therefore the reward and the whole
tree — is identical to the single-node version.

Engine-agnostic: calls the same public interface KMC.run_kmc_trajectory(...) that
both the current and Agent B's future faster KMC.py expose.

LIGHTWEIGHT LOGGING (read-only; never perturbs the trajectory)
  In addition to gr_<gidx>.txt, when the manifest's "log" block is enabled this
  worker also writes:
    sum_<gidx>.txt  — ONE CSV row: per-trajectory summary (events + per-reaction
                      breakdown + thickness/GR/wall). No header (controller adds it).
    ckpt_<gidx>.csv — growth checkpoints (cumulative_events, thickness_A) sampled
                      every `checkpoint_events` events. Only if log_checkpoints.
  The instrumentation is delivered through KMC module globals set at the end of
  run_kmc_trajectory (_LAST_EVENT_COUNTS, _LAST_ITERATIONS, _LAST_THICKNESS_A,
  _LAST_CHECKPOINTS) and the pre-call knob KMC._CHECKPOINT_EVERY. All reads use
  getattr fallbacks so a future engine that omits them logs blanks, never crashes.
"""
import os
# Force single-threaded numpy BEFORE importing it — each node packs 48 of these
# single-core tasks; per-process BLAS threads would oversubscribe and crawl.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import time

import numpy as np

# The bucket engine, same public interface and the same chemistry (it imports KMC.py).
import KMC_bucket as KMC
from reward_metric import WINDOW_A, reward_ckpt_every, window_from_sink

# per-reaction columns n_rid_01 .. n_rid_(NUM_RID-1); RID 0 is unused in the engine.
_NUM_RID = int(getattr(KMC, "_NUM_RID", 34))
# then one column per numbered reaction, n_R1 .. (the sum of its engine IDs)
_ENG = getattr(KMC, "K", KMC)                  # KMC_bucket exposes the engine module as K
_RXN_COUNTS = getattr(_ENG, "_reaction_counts", None)
_RXN_NCOL = len(_ENG._reaction_count_columns()) if _RXN_COUNTS else 0


def _atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def main():
    if len(sys.argv) < 4:
        sys.stderr.write("usage: kmc_worker.py <manifest.json> <outdir> <offset>\n")
        sys.exit(2)
    manifest_path, outdir, offset = sys.argv[1], sys.argv[2], int(sys.argv[3])

    # Task identity: srun sets SLURM_PROCID; MCTS_PROCID is the local-test fallback.
    procid = int(os.environ.get("SLURM_PROCID",
                                os.environ.get("MCTS_PROCID", "0")))
    gidx = offset + procid

    with open(manifest_path) as fh:
        man = json.load(fh)
    tasks = man["tasks"]

    # Overcommit guard: if the wave launched more tasks than trajectories (never
    # happens for the sized waves, but harmless), idle tasks just exit cleanly.
    if gidx >= len(tasks):
        return

    T, seed = tasks[gidx]

    # --- logging config (all optional; absent => logging off) ---
    log = man.get("log", {}) or {}
    log_on   = bool(log.get("enabled", False))
    ckpt_on  = bool(log.get("log_checkpoints", False))
    ckpt_ev  = int(log.get("checkpoint_events", 50000))
    exp_lbl  = str(log.get("experiment_label", "node"))
    n_rep    = int(man.get("n_rep", 1)) or 1
    rep_idx  = gidx % n_rep            # within-temperature replicate index (T-major tasks)

    # Legacy passive sampler of the OLD engine (base KMC.py) is left armed off; the
    # fast2 engine ignores this global and uses the checkpoint_sink args below.
    if hasattr(KMC, "_CHECKPOINT_EVERY"):
        KMC._CHECKPOINT_EVERY = ckpt_ev if (log_on and ckpt_on) else 0

    # THE GROWTH RATE = the mean height's rate through the window [lo, hi]
    # (reward_metric.py). The engine's checkpoint sink records the mean height; the run
    # stops at the first checkpoint at or above hi. The engine's max-height return value
    # is not used. reward_ckpt_every divides max_iter so a correct-baseline checkpoint
    # lands exactly at max_iter (see reward_metric.reward_ckpt_every).
    rce = reward_ckpt_every(man["max_iter"],
                            int(man.get("reward_ckpt_every", 20000)))
    lo, hi = (float(x) for x in man.get("gr_window_A", WINDOW_A))
    sink = []                # (events, elapsed_s, max_height_A, mean_height_A)

    np.random.seed(int(seed))                 # independent stream per trajectory
    t0 = time.perf_counter()
    gr_max = KMC.run_kmc_trajectory(
        target_T=T,
        params=man["params"],
        node_label="w",
        traj_freq=None,
        lattice_file=man["lattice_file"],
        target_growth=man["target_growth"],
        max_iter=man["max_iter"],
        checkpoint_every=rce,
        checkpoint_sink=sink,
        stop_mean_height=hi,
        # the stall rule (ends films that stopped growing; 0 = off)
        stop_plateau_idle_factor=float(man.get('stall_idle_factor', 0.0)),
        stop_plateau_min_events=int(man.get('stall_min_events', 0)),
        stop_plateau_rel_tol=float(man.get('stall_rel_tol', 0.0)),
        stop_plateau_abs_tol=float(man.get('stall_abs_tol_A', 0.0)),
    )
    wall_s = time.perf_counter() - t0

    # The window growth rate is the value fed to the MCTS reward, read in exactly one
    # place (reward_metric.window_from_sink). A film that did not grow through the
    # window scores 0.0: it did not grow at the rate being fitted.
    _gr, _reached, _tlo, _thi, _hfin = window_from_sink(sink, lo, hi)
    gr = _gr if _reached else 0.0

    # Atomic write: full-precision repr. The controller's GR gather reads exactly
    # this file. "gr h reached": the rate, the mean height the film ended at, and
    # whether it grew through the whole window.
    _atomic_write(os.path.join(outdir, f"gr_{gidx}.txt"),
                  "%s %s %s" % (repr(float(gr)), repr(float(_hfin)),
                                repr(1.0 if _reached else 0.0)))

    if not log_on:
        return

    # ---- per-trajectory summary row (read the engine's read-only log globals) ----
    total_events = int(getattr(KMC, "_LAST_ITERATIONS", 0))
    thickness_A  = getattr(KMC, "_LAST_THICKNESS_A", None)
    counts       = getattr(KMC, "_LAST_EVENT_COUNTS", None)
    total_fired  = getattr(KMC, "_LAST_TOTAL_FIRED", None)

    if counts is not None:
        rid_cells = [str(int(counts[rid])) for rid in range(1, _NUM_RID)]
        rid_cells += [str(n) for n in _RXN_COUNTS(counts)] if _RXN_COUNTS else []
    else:                                   # engine did not expose counts: leave blank
        rid_cells = ["" for _ in range(1, _NUM_RID)] + [""] * _RXN_NCOL
    thick_cell = f"{float(thickness_A):.6f}" if thickness_A is not None else ""
    fired_cell = str(int(total_fired)) if total_fired is not None else ""

    row = ",".join([
        exp_lbl, str(rep_idx), str(int(T)), str(int(seed)),
        str(total_events), fired_cell, thick_cell,
        repr(float(gr)), f"{wall_s:.3f}",
    ] + rid_cells)
    _atomic_write(os.path.join(outdir, f"sum_{gidx}.txt"), row + "\n")

    # ---- growth checkpoints (compact long-format; one small file per trajectory) ----
    if ckpt_on:
        cps = getattr(KMC, "_LAST_CHECKPOINTS", None) or []
        lines = [f"{exp_lbl},{rep_idx},{int(T)},{int(seed)},{int(ce)},{float(th):.6f}"
                 for (ce, th) in cps]
        _atomic_write(os.path.join(outdir, f"ckpt_{gidx}.csv"),
                      ("\n".join(lines) + "\n") if lines else "")


if __name__ == "__main__":
    main()
