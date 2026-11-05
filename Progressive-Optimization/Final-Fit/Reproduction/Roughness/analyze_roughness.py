"""analyze_roughness.py -- roughness at the LT+MT+HT fit (node 31) over every measured temperature: merge, choose n
per (T, thickness) by the flat-region rule, and plot.

    python analyze_roughness.py  -> ../Results/{roughness_summary.csv, roughness_map.png, roughness_vs_thickness.png}

Two runs of the same parameter set feed it:
  ../Results/Raw/results                   900-1230 K in 10 K steps (34 T), 240 films each (this stage)
  ../../DRC_FIT/Results/Raw/rough          Lin's 15 temperatures, 144 films each (the DRC baseline's films);
                                           only the 7 not on the 10 K grid are taken (925, 975, ... 1225 K)
41 temperatures in all. Per temperature and 10 A mark: the films that reached the mark, mean W and SEM, CV, and
n_flat -- the flat-region rule (kmc/flat_region.py): the first n from which the running mean of W (films in
replicate order) stays inside mean +- 1 SEM and has reached its middle; "enough" if n_flat <= 0.8 N.
"""
import argparse
import csv
import glob
import math
import os
import statistics as S
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.normpath(os.path.join(HERE, *[os.pardir] * 4))
# the engine's kmc/ folder: $KMC_CODE if set, else this repository's Codebase/ (../../../../Codebase from this folder)
sys.path.insert(0, os.path.join(os.environ.get("KMC_CODE") or os.path.join(HERE, *[os.pardir] * 4, "Codebase"), "kmc"))
sys.path.insert(0, os.path.join(STUDY, "Figures"))
from flat_region import flat_point  # noqa: E402
import house_style as hs  # noqa: E402

LIN = [900, 925, 950, 975, 1000, 1025, 1050, 1075, 1100, 1125, 1150, 1175, 1200, 1225, 1230]


def read(raw, keep=None):
    marks, films = defaultdict(list), defaultdict(list)
    for f in glob.glob(os.path.join(raw, "rough_marks_c*.csv")):
        for r in csv.DictReader(open(f)):
            T = int(r["T_K"])
            if keep is None or keep(T):
                marks[(T, float(r["mark_A"]))].append(
                    (int(r["rep"]), float(r["W_A"]), float(r["PV_A"]), float(r["h_read_A"])))
    for f in glob.glob(os.path.join(raw, "rough_films_c*.csv")):
        for r in csv.DictReader(open(f)):
            T = int(r["T_K"])
            if keep is None or keep(T):
                films[T].append(r)
    return marks, films


def edges(v, half):
    """Cell edges at the midpoints between neighbours; the outer cells are `half` wide on their open side."""
    mid = [(a + b) / 2 for a, b in zip(v, v[1:])]
    return [v[0] - half] + mid + [v[-1] + half]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=os.path.join(HERE, os.pardir, "Results", "Raw", "results"))
    ap.add_argument("--lin", default=os.path.join(HERE, os.pardir, os.pardir, "DRC_FIT", "Results", "Raw", "rough"))
    ap.add_argument("--out", default=os.path.join(HERE, os.pardir, "Results"))
    a = ap.parse_args()
    marks, films = read(a.grid)
    grid = set(films)
    m2, f2 = read(a.lin, keep=lambda T: T not in grid)
    marks.update(m2); films.update(f2)
    temps = sorted(films)
    mk = sorted({m for _, m in marks})
    print("%d temperatures (%d on the 10 K grid + %s from the Lin-temperature run), %d films"
          % (len(temps), len(grid), sorted(f2), sum(len(v) for v in films.values())))
    rows = []
    for T in temps:
        N = len(films[T])
        reached = sum(int(r["reached_200A"]) for r in films[T])
        for m in mk:
            v = sorted(marks.get((T, m), []))
            if not v:
                rows.append(dict(T_K=T, mark_A=m, n_films=N, n_read=0)); continue
            W = [x[1] for x in v]
            mean = S.mean(W); sd = S.stdev(W) if len(W) > 1 else 0.0
            nf, _, _ = flat_point(W)
            rows.append(dict(T_K=T, mark_A=m, n_films=N, n_read=len(W), reached_200A=reached,
                             W_mean_A=round(mean, 4), W_sem_A=round(sd / math.sqrt(len(W)), 4),
                             W_cv_pct=round(100 * sd / mean, 2) if mean else "", n_flat=nf,
                             enough=int(nf <= 0.8 * len(W)), PV_mean_A=round(S.mean(x[2] for x in v), 3),
                             h_read_mean_A=round(S.mean(x[3] for x in v), 3)))
    os.makedirs(a.out, exist_ok=True)
    keys = ["T_K", "mark_A", "n_films", "n_read", "reached_200A", "W_mean_A", "W_sem_A", "W_cv_pct", "n_flat",
            "enough", "PV_mean_A", "h_read_mean_A"]
    with open(os.path.join(a.out, "roughness_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    full = [r for r in rows if r.get("n_read")]
    print("n_flat: max %d, median %d over %d (T, mark) cells; cells not flat within 0.8 N: %d"
          % (max(r["n_flat"] for r in full), S.median(r["n_flat"] for r in full), len(full),
             sum(1 for r in full if not r["enough"])))

    hs.apply()
    Z = np.full((len(mk), len(temps)), np.nan)
    for r in full:
        Z[mk.index(r["mark_A"]), temps.index(r["T_K"])] = r["W_mean_A"]
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.pcolormesh(edges(temps, 5), edges(mk, 5), Z, cmap="viridis", shading="flat")
    cb = fig.colorbar(im, ax=ax); cb.set_label("RMS roughness W (Å)", fontsize=18); cb.ax.tick_params(labelsize=15)
    ax.set_xlabel("Temperature (K)", fontsize=20); ax.set_ylabel("Film thickness (Å)", fontsize=20)
    ax.tick_params(labelsize=16)
    hs.save(fig, os.path.join(a.out, "roughness_map.png"))

    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.get_cmap("plasma")
    for i, T in enumerate(LIN):
        rr = [r for r in full if r["T_K"] == T]
        ax.plot([r["mark_A"] for r in rr], [r["W_mean_A"] for r in rr], "-o",
                color=cmap(0.9 * i / (len(LIN) - 1)), ms=4, lw=1.8, label="%d K" % T)
    ax.set_xlabel("Film thickness (Å)", fontsize=20); ax.set_ylabel("RMS roughness W (Å)", fontsize=20)
    ax.tick_params(labelsize=16)
    ax.legend(frameon=False, fontsize=13, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5))
    hs.thicken_spines(ax)
    hs.save(fig, os.path.join(a.out, "roughness_vs_thickness.png"))
    print("wrote", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
