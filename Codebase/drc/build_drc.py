"""
build_drc.py -- build a direct rate-coefficient screen (DRC) from one JSON file.

    python build_drc.py examples/lt1125_drc.json

WHAT A DRC IS HERE
    One arm per direction per sign: take the root parameter set, multiply one rate
    coefficient by exp(+delta) or exp(-delta), and run the same KMC sweep as the
    root. The ratio of growth rates gives

        D(T) = d ln GR(T) / d ln k                (per direction, per temperature)

    measured, not linearised. Every arm at a given temperature shares its seeds
    with every other arm at that temperature (common random numbers), so the
    difference between two arms is the parameter's effect and not sampling noise.

    A single barrier's effect follows by the chain rule, per temperature:

        d ln GR(T) / d Ea = -D(T) / T             (Ea in kelvin)

    which is why one screen in the 20 default directions (16 Arrhenius reactions and the 4
    sticking coefficients) covers all 54 parameters: Ea_chem_As / Ea_chem_M only multiply
    the sticking coefficients. E3 is one direction: it moves engine IDs 3, 4 and 5 together;
    each incorporation reaction is one direction too (tied IDs).

WHAT THIS SCRIPT WRITES
    <out_dir>/            one JSON per arm, plus baseline.json
    drc_manifest.csv      direction, key, arm, value before/after, delta_applied,
                          and whether the step was clipped at a bound
    drc_tasklist.txt      one line per (arm, temperature): "<file> <T> <seed_base>"
    drc_root.json         the root, written out flat, 45 keys
    drc_stamp.json        md5s of everything above, for the record

    delta_applied is the log ratio ACTUALLY applied, which differs from the
    nominal delta whenever a bounded parameter clips (a sticking coefficient of
    0.9746 cannot take a +0.9 e-fold step; it stops at 1.0, a step of +0.0257).
    Always divide by delta_applied, never by the nominal -- reduce_from_traj.py
    does.

RUNNING THE ARMS
    Each line of drc_tasklist.txt is one KMC sweep at one temperature. Feed them
    to ../kmc/run_kmc.py one array element at a time (the stage folder supplies the
    sbatch script). Then reduce with reduce_from_traj.py.

SCHEMA -- see README.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(HERE, os.pardir, "kmc", "params_catalog_defaults.json")

TOP = {"root_parameters", "delta_efolds", "reactions", "extra_directions",
       "bounds", "temperatures", "replicates", "seed_base", "priority",
       "out_dir", "description", "root_md5"}

# The defaults are the published screen's own settings.
# E3 also covers engine IDs 4, 5; 14, 17, 20, 23, 26, 29 cover their incorporation reactions
DEFAULT_REACTIONS = [1, 2, 3, 12, 13, 14, 17, 20, 23, 26, 29, 32, 33,
                     37, 38, 41]        # 37-41: R18, R19, R20
# (27 Sep: incorporation onto an adsorbed precursor and the extra desorption cases are cases
# of existing reactions, tied to their owners, so they add no direction. The manifest column
# 'pushed' records a dependent moved along, e.g. R18's barrier when R10's is lowered.)
DEFAULT_EXTRA = [["stick_A_terr", "S0_A_terrace"], ["stick_A_step", "S0_A_step"],
                 ["stick_M_terr", "S0_M_terrace"], ["stick_M_step", "S0_M_step"]]
DEFAULT_BOUNDS = {"S0_A_terrace": [0.0, 1.0], "S0_A_step": [0.0, 1.0],
                  "S0_M_terrace": [0.0, 1.0], "S0_M_step": [0.0, 1.0]}
# The ordering constraints (S0_terrace <= S0_step, barrier pairs) live in the engine;
# an arm that would break one is clipped to the limit and marked CLIPPED.
sys.path.insert(0, os.path.join(HERE, os.pardir, "kmc"))
import KMC as _K


def die(msg):
    sys.exit(f"[build_drc] {msg}")


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("drc_json")
    ap.add_argument("--outdir", help="where to write; default is the JSON's folder")
    args = ap.parse_args()

    if not os.path.exists(args.drc_json):
        die(f"no such file: {args.drc_json}")
    with open(args.drc_json) as f:
        j = json.load(f)
    src_dir = os.path.dirname(os.path.abspath(args.drc_json))
    work = os.path.abspath(args.outdir or src_dir)
    os.makedirs(work, exist_ok=True)

    bad = sorted(set(j) - TOP)
    if bad:
        die(f"unknown key(s) in the DRC JSON: {', '.join(bad)}")
    for req in ("root_parameters", "temperatures"):
        if req not in j:
            die(f"the DRC JSON must set '{req}'")

    # --- the root -----------------------------------------------------------
    rp = j["root_parameters"]
    root_path = rp if os.path.isabs(rp) else os.path.join(src_dir, rp)
    if not os.path.exists(root_path):
        die(f"root parameter file not found: {root_path}")
    if j.get("root_md5") and md5(root_path) != j["root_md5"]:
        die(f"root_md5 does not match {root_path}\n"
            f"       expected {j['root_md5']}\n"
            f"       found    {md5(root_path)}")
    with open(root_path) as f:
        raw = json.load(f)
    if "params" in raw and isinstance(raw["params"], dict):
        raw = raw["params"]                      # an MCTS winner json
    root = {str(k): float(v) for k, v in raw.items() if not str(k).startswith("_")}

    with open(CATALOG) as f:
        known = set(json.load(f).keys())
    if set(root) != known:
        missing, extra = sorted(known - set(root)), sorted(set(root) - known)
        die(f"the root must carry all {len(known)} parameter names.\n"
            f"       missing: {missing[:5]}\n       unknown: {extra[:5]}")
    if not all(math.isfinite(v) for v in root.values()):
        die("the root contains a non-finite value")

    # --- the directions -----------------------------------------------------
    rxns = j.get("reactions", DEFAULT_REACTIONS)
    extra = [tuple(x) for x in j.get("extra_directions", DEFAULT_EXTRA)]
    directions = [(f"E{r}", f"A_E{r}") for r in rxns] + extra
    unknown = [k for _, k in directions if k not in known]
    if unknown:
        die(f"direction(s) name a parameter that does not exist: {unknown}")
    if len(set(n for n, _ in directions)) != len(directions):
        die("two directions share a name")

    bounds = {k: tuple(v) for k, v in j.get("bounds", DEFAULT_BOUNDS).items()}
    delta = float(j.get("delta_efolds", 0.9))
    if delta <= 0:
        die("delta_efolds must be positive")
    e = math.exp(delta)

    outdir = os.path.join(work, j.get("out_dir", "drc_params"))
    os.makedirs(outdir, exist_ok=True)

    man, files = [], []
    for name, key in directions:
        for sign, tag in ((+1, "plus"), (-1, "minus")):
            p = dict(root)
            v0 = root[key]
            v = v0 * (e if sign > 0 else 1.0 / e)
            lo, hi = bounds.get(key, (None, None))
            clipped = False
            if hi is not None and v > hi:
                v, clipped = hi, True
            if lo is not None and v < lo:
                v, clipped = lo, True
            fixed = {k: x for k, x in root.items() if k != key}
            try:
                cl, what = _K.enforce_constraints({key: v}, fixed=fixed)
            except ValueError as err:
                die(f"the root breaks an ordering constraint: {err}")
            if key in what:
                v, clipped = cl[key], True
            if v <= 0 or v == v0:
                die(f"direction {name} ({key}) cannot step: value {v0!r} -> {v!r}")
            p[key] = v
            pushed = {k: x for k, x in cl.items() if k != key}     # copy: dependents moved along
            p.update(pushed)
            fn = f"{name}_{tag}.json"
            with open(os.path.join(outdir, fn), "w") as f:
                json.dump(p, f, indent=2, sort_keys=True)
            files.append(fn)
            man.append(dict(direction=name, key=key, arm=tag, value_0=repr(v0),
                            value=repr(v), delta_nominal=sign * delta,
                            delta_applied=round(math.log(v / v0), 12),
                            clipped="YES" if clipped else "",
                            pushed=";".join(f"{k}={x!r}" for k, x in sorted(pushed.items()))))

    with open(os.path.join(outdir, "baseline.json"), "w") as f:
        json.dump(root, f, indent=2, sort_keys=True)
    files.append("baseline.json")
    man.append(dict(direction="baseline", key="", arm="base", value_0="",
                    value="", delta_nominal=0.0, delta_applied=0.0, clipped="", pushed=""))

    man_path = os.path.join(work, "drc_manifest.csv")
    with open(man_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0]))
        w.writeheader()
        w.writerows(man)

    # --- the task list ------------------------------------------------------
    # Ordered so that the arms you most want to see first are the first array
    # elements: baseline, then the directions named in "priority", then the rest.
    temps = [int(t) for t in j["temperatures"]]
    nmax = int(j.get("replicates", 240))
    seed_base = int(j.get("seed_base", 93000000))
    priority = list(j.get("priority", []))
    unknown = [d for d in priority if f"{d}_plus.json" not in files]
    if unknown:
        die(f"'priority' names a direction that does not exist: {unknown}")

    head = ["baseline.json"] + [f"{d}_{t}.json" for d in priority
                                for t in ("plus", "minus")]
    ordered = head + [f for f in files if f not in head]
    assert len(ordered) == len(files) == len(set(ordered))

    lines = [f"{fn} {T} {seed_base + t * nmax}"
             for fn in ordered for t, T in enumerate(temps)]
    task_path = os.path.join(work, "drc_tasklist.txt")
    with open(task_path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    root_path_out = os.path.join(work, "drc_root.json")
    with open(root_path_out, "w") as f:
        json.dump(root, f, indent=2, sort_keys=True)

    stamp = {"drc_json": os.path.basename(args.drc_json),
             "root_source": os.path.basename(root_path),
             "root_source_md5": md5(root_path),
             "drc_root_md5": md5(root_path_out),
             "manifest_md5": md5(man_path),
             "tasklist_md5": md5(task_path),
             "n_directions": len(directions), "n_arms": len(files),
             "delta_efolds": delta, "temperatures": temps,
             "replicates": nmax, "seed_base": seed_base}
    with open(os.path.join(work, "drc_stamp.json"), "w") as f:
        json.dump(stamp, f, indent=2, sort_keys=True)

    print(f"[build_drc] root        : {root_path}  (md5 {stamp['root_source_md5']})")
    print(f"[build_drc] directions  : {len(directions)} -> "
          f"{2 * len(directions)} arms + baseline = {len(files)}")
    print(f"[build_drc] temperatures: {len(temps)} -> "
          f"{', '.join(str(t) for t in temps)}")
    print(f"[build_drc] array size  : {len(lines)} elements, "
          f"{nmax} replicates each")
    for t, T in enumerate(temps):
        print(f"             {T:4d} K -> seed_base {seed_base + t * nmax}")
    for m in man:
        if m["clipped"]:
            print(f"[build_drc] CLIPPED     : {m['direction']:12s} {m['arm']:5s} "
                  f"{m['key']:11s} applied {m['delta_applied']:+.6f} "
                  f"of nominal {m['delta_nominal']:+.1f}")
    print(f"[build_drc] wrote {outdir}/, drc_manifest.csv, drc_tasklist.txt, "
          f"drc_root.json, drc_stamp.json")


if __name__ == "__main__":
    main()
