"""roughness_study.py -- surface roughness against film thickness and temperature, at the LT+MT+HT fit.

    python roughness_study.py --params best_params.json --chunk C --nchunks N [--workers 48] [--out results]

The study: the LT+MT+HT search's best parameter set (best_params_b3_Cauto.json, lowest loss),
900-1230 K in steps of 10 K (34 temperatures), 240 films each (n chosen afterwards by the flat-region rule), the
roughness read every 10 A of mean film height from 50 A to 200 A.

Each film is grown as the searches grow one (kmc_worker.py: the same engine call, 10 M-event cap, 50-200 A window
and stop at 200 A, stall rule) with one difference: the engine checks the film every 2,000 events instead of
20,000, so a 10 A mark is read within ~1 A (a 900 K film grows ~8 A per 20,000 events). The checks are passive
(read-only, no random numbers): the film's events are the same; it only stops at the first check past 200 A,
which is now closer to 200 A. The engine is not changed.

    W    standard deviation, over the lattice columns, of the top real-atom height (the engine's own column
         definition, the one behind the mean film height), in A: the RMS roughness / interface width
    P-V  peak-to-valley range of the same column heights

Reading at mark m = the first check whose mean height is >= m (the height actually read is recorded).

Seeds: 97250000 + 1000 x (temperature index) + replicate. Films are dealt to chunks by global index mod N, so every
chunk carries every temperature. Writes, per chunk, rough_marks_c<C>.csv (one row per film per mark) and
rough_films_c<C>.csv (one row per film).
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import time

import codebase

TEMPS = list(range(900, 1231, 10))                 # 34 temperatures, 900 ... 1230 K
FILMS = 240
SEED_BASE = 97250000
MARKS = [float(m) for m in range(50, 201, 10)]     # 50, 60, ..., 200 A
RUN = dict(max_iter=10_000_000, target_growth=1e6, ckpt=2000, window=(50.0, 200.0),
           stall_idle_factor=2.0, stall_min_events=500000, stall_rel_tol=0.005, stall_abs_tol_A=0.01)
PARAMS = None


def load_params(path):
    d = json.load(open(path))
    p = d["params"] if "params" in d else d
    p = {k: float(v) for k, v in p.items() if not k.startswith("_")}
    if len(p) != 54:
        raise SystemExit("[roughness] %s: expected 54 parameters, found %d" % (path, len(p)))
    return p


def one(job):
    T, rep, seed, max_iter = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    codebase.use()
    import numpy as np
    import KMC_bucket as KB
    from reward_metric import reward_ckpt_every, window_from_sink
    K = KB.K
    live = {}
    make_state = KB.LatticeState

    def LatticeState(*a, **kw):                    # the engine's usual state, with a read-only handle kept
        live["st"] = make_state(*a, **kw)
        return live["st"]
    KB.LatticeState = LatticeState

    marks = list(MARKS)
    rows = []

    class Sink(list):
        """The engine's checkpoint sink, unchanged, plus a read of the surface at each 10 A mark."""
        def append(self, item):
            super().append(item)
            h = float(item[3])
            if not marks or h < marks[0]:
                return
            st = live["st"]
            if "cid" not in live:
                live["cid"], live["n"] = KB._column_ids(st)
            real = K._IS_REAL[st.species]
            top = np.full(live["n"], -np.inf)
            np.maximum.at(top, live["cid"][real], st.coords[real, 2])
            top = top[np.isfinite(top)]
            W = float(top.std()) if top.size else float("nan")
            pv = float(top.max() - top.min()) if top.size else float("nan")
            while marks and h >= marks[0]:         # one read can serve several marks if the film jumped past them
                rows.append((marks.pop(0), h, int(item[0]), float(item[1]), W, pv))

    sink = Sink()
    lo, hi = RUN["window"]
    np.random.seed(int(seed))
    t0 = time.perf_counter()
    KB.run_kmc_trajectory(
        target_T=float(T), params=PARAMS, node_label="rough", traj_freq=None,
        lattice_file=codebase.LATTICE, target_growth=RUN["target_growth"], max_iter=max_iter,
        checkpoint_every=reward_ckpt_every(max_iter, RUN["ckpt"]), checkpoint_sink=sink,
        stop_mean_height=hi,
        stop_plateau_idle_factor=RUN["stall_idle_factor"], stop_plateau_min_events=RUN["stall_min_events"],
        stop_plateau_rel_tol=RUN["stall_rel_tol"], stop_plateau_abs_tol=RUN["stall_abs_tol_A"])
    wall = time.perf_counter() - t0
    gr, reached, _, _, h_end = window_from_sink(sink, lo, hi)
    last = list(sink)[-1] if len(sink) else (0, 0.0, 0.0, 0.0)
    return dict(T=T, rep=rep, seed=seed, gr=gr if reached else None, reached=int(bool(reached)), h_end=h_end,
                events=int(last[0]), wall_s=wall, rows=rows)


def init(params):
    global PARAMS
    PARAMS = params


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", required=True)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--nchunks", type=int, required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 48)))
    ap.add_argument("--out", default="results")
    ap.add_argument("--films", type=int, default=FILMS)
    ap.add_argument("--temps", default="", help="comma list to override (smoke test only)")
    ap.add_argument("--max-iter", type=int, default=RUN["max_iter"], help="lower only for a smoke test")
    a = ap.parse_args()
    params = load_params(a.params)
    temps = [int(x) for x in a.temps.split(",")] if a.temps else TEMPS
    os.makedirs(a.out, exist_ok=True)
    allj = [(T, k, SEED_BASE + 1000 * TEMPS.index(T) + k, a.max_iter) for T in temps for k in range(a.films)]
    jobs = [j for g, j in enumerate(allj) if g % a.nchunks == a.chunk]
    print("[roughness] chunk %d/%d: %d of %d films, %d temperatures, %d workers, cap %d, check every %d events"
          % (a.chunk, a.nchunks, len(jobs), len(allj), len(temps), a.workers, a.max_iter, RUN["ckpt"]), flush=True)
    t0 = time.time()
    with mp.Pool(min(a.workers, max(1, len(jobs))), initializer=init, initargs=(params,), maxtasksperchild=4) as pool:
        res = pool.map(one, jobs, chunksize=1)
    with open(os.path.join(a.out, "rough_marks_c%d.csv" % a.chunk), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T_K", "rep", "seed", "mark_A", "h_read_A", "events", "time_s", "W_A", "PV_A"])
        for r in res:
            for m, h, e, t, W, pv in r["rows"]:
                w.writerow([r["T"], r["rep"], r["seed"], "%g" % m, "%.4f" % h, e, "%.6e" % t, "%.5f" % W, "%.4f" % pv])
    with open(os.path.join(a.out, "rough_films_c%d.csv" % a.chunk), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["T_K", "rep", "seed", "reached_200A", "GR_umin", "h_end_A", "events", "n_marks", "wall_s"])
        for r in res:
            w.writerow([r["T"], r["rep"], r["seed"], r["reached"], "" if r["gr"] is None else repr(r["gr"]),
                        "%.3f" % r["h_end"], r["events"], len(r["rows"]), "%.1f" % r["wall_s"]])
    print("[roughness] chunk %d done: %d films in %.0f s; reached 200 A: %d" % (
        a.chunk, len(res), time.time() - t0, sum(r["reached"] for r in res)), flush=True)


if __name__ == "__main__":
    main()
