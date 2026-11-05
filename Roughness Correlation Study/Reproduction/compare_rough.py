"""compare_rough.py -- the rerun must reproduce the ROUGHNESS study: same seeds, same films, read-only snapshots.

    python compare_rough.py <this run's results folder>      (exit 1 on any difference)

Every (T, rep, mark) row of rough_marks_c*.csv and every film row of rough_films_c*.csv (wall time aside) must equal
the first study's (../../ROUGHNESS/Results/Raw/results), and every film must have its snapshot file with one row per mark.
"""
import csv, glob, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, os.pardir, os.pardir, "ROUGHNESS", "Results", "Raw", "results")
new = sys.argv[1]


def rows(d, pat, drop=()):
    out = {}
    for f in glob.glob(os.path.join(d, pat)):
        for r in csv.DictReader(open(f)):
            k = (r["T_K"], r["rep"], r.get("mark_A", ""))
            out[k] = {c: v for c, v in r.items() if c not in drop}
    return out


bad = 0
for pat, drop in (("rough_marks_c*.csv", ()), ("rough_films_c*.csv", ("wall_s",))):
    a, b = rows(OLD, pat, drop), rows(new, pat, drop)
    diff = [k for k in a if a[k] != b.get(k)] + [k for k in b if k not in a]
    print("%-20s first study %6d rows, rerun %6d, differing %d" % (pat, len(a), len(b), len(diff)))
    bad += len(diff) + (len(a) != len(b))
films = rows(new, "rough_films_c*.csv")
miss = 0
for (T, rep, _), r in films.items():
    f = os.path.join(new, "snaps", "T%s_r%03d.npz" % (T, int(rep)))
    if int(r["n_marks"]) == 0:
        continue
    if not os.path.exists(f) or np.load(f)["species"].shape[0] != int(r["n_marks"]):
        miss += 1
print("snapshots: %d films, %d missing or short" % (len(films), miss))
print("PASS" if not (bad or miss) else "FAIL")
sys.exit(1 if bad or miss else 0)
