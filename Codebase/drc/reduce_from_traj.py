#!/usr/bin/env python3
"""reduce_from_traj.py -- the LT+1125 DRC reduction, one row per (direction, T).

    python reduce_from_traj.py --results <merged results dir>

At each temperature separately: the growth rate of every replicate read through the
thickness window (run_kmc.py's GR_window_umin, the mean height's rate from 50 to
200 A), CRN-paired on replicate index, divisor = the step ACTUALLY APPLIED.

    D        = mean over pairs of ln(GR_plus / GR_minus) / step
               (pairs in which BOTH films grew through the window)
    D_height = mean over pairs of ln(h_plus / h_minus) / step
               (h = the mean height the film ended at; ~0 wherever both films reached
               the window top, and the lever's pull on growth where they did not)

BASELINE
    drc_baseline_T<T>_traj.csv, run with the same seeds as every arm at that
    temperature: growth rate, final height and the fraction that grew through the
    window.

TWO SLOPE COLUMNS -- the point of the low-temperature rung
    The measured apparent activation energy over 900-1000 K is ~0.50 eV. A lever
    changes it only if it moves 900 K and 1000 K differently, and HOW depends on
    how the lever enters the rate constant:

    prefactor-type (A, S0) -- a step of s e-folds in ln k
    at every temperature:
        dEa_app per e-fold  = -k_B [D(900) - D(1000)] / (1/900 - 1/1000)     [eV]

    barrier-type (Ea, Ea_chem) -- a step of dEa kelvin gives d ln k = -dEa/T, so
    even a lever with the SAME D at both temperatures tilts the slope:
        dEa_app / dEa       = [D(900)/900 - D(1000)/1000] / (1/900 - 1/1000)  [K/K]

    The searched barriers are Ea_chem_M, Ea_chem_As and Ea_E32, read from the
    stick_M, stick_As and E32 rows respectively.
"""
import argparse, csv, glob, math, os, re, statistics, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPS = [900, 925, 950, 975, 1000, 1125]
KB = 8.617333e-5
EXP = {900: 0.53, 925: 0.70, 950: 0.87, 975: 0.95, 1000: 1.00376, 1125: 1.16653}


def load(path):
    """{rep: (GR or None, h_final, seed)} for EVERY replicate, and the count run.
    GR is None for a film that did not grow through the window. load.flagged counts
    the films that did, for the report."""
    seen, out = set(), {}
    load.flagged = getattr(load, "flagged", {})
    nflag = 0
    with open(path) as fh:
        for r in csv.DictReader(fh):
            k = (int(r["seed"]), int(r["rep"]))
            if k in seen:
                continue
            seen.add(k)
            ok = int(r["reached"]) == 1 and r["GR_window_umin"] and float(r["GR_window_umin"]) > 0
            nflag += int(bool(ok))
            out[int(r["rep"])] = (float(r["GR_window_umin"]) if ok else None,
                                  float(r["mean_height_A_final"]), int(r["seed"]))
    load.flagged[path] = nflag
    return out, len(seen)


def main():
    global TEMPS, EXP
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "results"))
    ap.add_argument("--manifest", default=os.path.join(HERE, "drc_manifest.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "drc_from_traj.csv"))
    ap.add_argument("--temps", help="comma/space list; default is the published "
                                    "ladder " + ",".join(str(t) for t in TEMPS))
    ap.add_argument("--target-csv", dest="target_csv",
                    help="CSV with columns T_K, GR_exp_umin -- the measured curve "
                         "the 'x exp' column is reported against; default is the "
                         "Lin 1977 values built in below")
    ap.add_argument("--n-expected", type=int, default=240, dest="n_expected",
                    help="replicates per element, for the progress line only")
    a = ap.parse_args()

    # The stage's own ladder and target curve are the defaults; both are here so
    # that a screen at other temperatures, or against another measured curve,
    # needs no edit to this file.
    if a.temps:
        TEMPS = [int(float(x)) for x in a.temps.replace(",", " ").split()]
    if a.target_csv:
        with open(a.target_csv) as fh:
            EXP = {int(float(r["T_K"])): float(r["GR_exp_umin"])
                   for r in csv.DictReader(fh)}
    missing = [T for T in TEMPS if T not in EXP]
    if missing:
        sys.exit("no target growth rate at %s -- pass --target-csv" % missing)
    NEXP = max(1, a.n_expected)

    arms, ns = {}, {}
    for f in sorted(glob.glob(os.path.join(a.results, "drc_*_T*_traj.csv"))):
        m = re.match(r"drc_(.+)_T(\d+)_traj\.csv$", os.path.basename(f))
        if m:
            key = (m.group(1), int(m.group(2)))
            arms[key], ns[key] = load(f)
    if not arms:
        sys.exit("no drc_<arm>_T<T>_traj.csv in %s" % a.results)
    tot = sum(ns.values())
    n_arms = len({k[0] for k in arms})
    planned = max(1, n_arms * len(TEMPS) * NEXP)
    print("elements %d | at n=%d: %d | replicates %d/%d (%.0f %%)"
          % (len(arms), NEXP, sum(1 for v in ns.values() if v >= NEXP),
             tot, planned, 100 * tot / planned))

    # ---- baseline = the verification at these six temperatures ----------------
    print("\nBASELINE (the root itself, at every temperature)")
    print("     T   run  grew through window   GR um/min   x exp   h_final A")
    for T in TEMPS:
        b = arms.get(("baseline", T))
        if not b:
            print("   %4d   --" % T)
            continue
        g = [v[0] for v in b.values() if v[0] is not None]
        h = [v[1] for v in b.values()]
        n = ns[("baseline", T)]
        if len(g) > 1:
            print("   %4d  %3d   %3d (%3.0f %%)          %8.4f   %5.2f   %6.2f"
                  % (T, n, len(g), 100.0 * len(g) / n, statistics.mean(g),
                     statistics.mean(g) / EXP[T], statistics.mean(h)))
        else:
            print("   %4d  %3d   %3d -- too few films grew through the window   h_final %.2f"
                  % (T, n, len(g), statistics.mean(h)))
    chk = bad = 0
    for (arm, T), d in arms.items():
        b = arms.get(("baseline", T), {})
        for rep, v in d.items():
            if rep in b:
                chk += 1
                bad += v[2] != b[rep][2]
    print("   seed check vs baseline: %d replicates compared, %d mismatched" % (chk, bad))

    # ---- per (direction, T) ---------------------------------------------------
    man = defaultdict(dict)
    for r in csv.DictReader(open(a.manifest)):
        if r["arm"] in ("plus", "minus"):
            man[r["direction"]][r["arm"]] = r
    rows = []
    for d, pair in man.items():
        step = float(pair["plus"]["delta_applied"]) - float(pair["minus"]["delta_applied"])
        for T in TEMPS:
            A, B = arms.get(("%s_plus" % d, T)), arms.get(("%s_minus" % d, T))
            if not A or not B or step == 0:
                continue
            both = sorted(r for r in set(A) & set(B)
                          if A[r][0] is not None and B[r][0] is not None)
            hboth = sorted(r for r in set(A) & set(B) if A[r][1] > 0 and B[r][1] > 0)
            if len(both) < 2:
                continue
            rr = [math.log(A[r][0] / B[r][0]) / step for r in both]
            hr = [math.log(A[r][1] / B[r][1]) / step for r in hboth]
            D = statistics.mean(rr)
            Dh = statistics.mean(hr) if hr else 0.0
            se = statistics.stdev(rr) / math.sqrt(len(rr))
            seh = statistics.stdev(hr) / math.sqrt(len(hr)) if len(hr) > 1 else 0.0
            npl, nmi = ns[("%s_plus" % d, T)], ns[("%s_minus" % d, T)]
            gp = [v[0] for v in A.values() if v[0] is not None]
            gm = [v[0] for v in B.values() if v[0] is not None]
            rows.append(dict(direction=d, key=pair["plus"]["key"], T_K=T,
                             GR_plus=statistics.mean(gp), GR_minus=statistics.mean(gm),
                             n_plus=npl, n_minus=nmi,
                             scored_plus=len(gp), scored_minus=len(gm),
                             delta_applied=step, D=D, SEM=se, D_height=Dh, SEM_height=seh,
                             SNR=(abs(D) / se if se else float("inf")), n_pairs=len(both)))
    if not rows:
        sys.exit("\nno complete +/- pairs yet")
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    byd = defaultdict(dict)
    for r in rows:
        byd[r["direction"]][r["T_K"]] = r["D"]
    inv = 1.0 / 900 - 1.0 / 1000
    wide = []
    for d, m in byd.items():
        if 900 in m and 1000 in m:
            pre = -KB * (m[900] - m[1000]) / inv
            bar = (m[900] / 900 - m[1000] / 1000) / inv
        else:
            pre = bar = float("nan")
        wide.append((d, m, pre, bar))
    wide.sort(key=lambda x: -(abs(x[2]) if x[2] == x[2] else -1))

    print("\n%-12s" % "direction" + "".join("%9s" % ("D@%d" % T) for T in TEMPS)
          + "   dEa_app/e-fold(eV)   dEa_app/dEa")
    print("-" * (12 + 9 * len(TEMPS) + 36))
    for d, m, pre, bar in wide:
        cells = "".join("%9s" % ("%.4f" % m[T] if T in m else "--") for T in TEMPS)
        tail = ("   %+12.4f        %+9.4f" % (pre, bar)) if pre == pre else "   --"
        print("%-12s%s%s" % (d, cells, tail))
    print("\nwrote %s  (%d rows = directions x temperatures)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
