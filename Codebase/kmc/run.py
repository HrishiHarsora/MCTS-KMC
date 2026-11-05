"""
run.py -- run a KMC sweep from one JSON file.

    python run.py examples/baseline_1125K.json
    python run.py my_run.json --dry-run
    python run.py my_run.json --shard 3 --nshards 8 --workers 48

WHAT IT DOES
    Nothing scientific. It reads the JSON, checks it, writes the config.toml that
    run_kmc.py already knows how to read, and hands over. run_kmc.py is the driver
    that produced every number in the paper and is copied here unmodified; this
    file exists so that a run is one readable file instead of a TOML plus six
    command-line flags.

    The generated config.toml is written next to the output CSV, so a finished run
    carries the exact settings it ran under.

WHY NOT KMC.py's OWN CLI
    KMC.py has a __main__ block, but it ignores the event cap, the growth-rate
    window and its stop, checkpointing and seeding. It is a smoke test, not a runner.
    run_kmc.py is the entry point.

THE ONE THING THIS ADDS
    A parameter-name whitelist. The engine silently falls back to the catalog
    value for any key it does not recognise, so a typo like "Ea_chem_as" runs
    happily and reports a result for the wrong parameter set. Here an unknown key
    is a hard error.

SCHEMA -- see README.md for the annotated version.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(HERE, "params_catalog_defaults.json")

# Keys of the run JSON. Anything else is a typo and is refused.
TOP = {"parameters", "lattice", "temperatures", "replicates", "seed_base",
       "max_events", "checkpoint_events", "window_A", "stall", "workers", "out",
       "adaptive", "description"}
STALL = {"idle_factor", "min_events", "rel_tol", "abs_tol_A"}
ADAPTIVE = {"wave", "n_rep_max", "target_sem_pct"}


def die(msg):
    sys.exit(f"[run] {msg}")


def load_catalog_names():
    with open(CATALOG) as f:
        return set(json.load(f).keys())


# Keys of tied engine IDs, which take another ID's parameters (KMC._TIED_RIDS): R3 = IDs
# 3-5, and the incorporation reactions R12-R17 (2026-09-27). Checked against the engine in
# tests/test_rate_map.py.
TIED_TO = {4: 3, 5: 3, 15: 14, 16: 14, 18: 17, 19: 17, 21: 20, 22: 20, 24: 23, 25: 23,
           27: 26, 28: 26, 30: 29, 31: 29,
           # 36-43 the DDD_Ga route; 44-59 incorporation onto an adsorbed precursor, 60-65 film
           # atoms with two / three bonds desorbing, 66-77 a free bond bonding down (cases)
           36: 12, 39: 38, 40: 38, 42: 41, 43: 41,
           44: 14, 45: 14, 52: 17, 53: 17, 46: 26, 47: 26, 54: 20, 55: 20,
           48: 29, 49: 29, 56: 23, 57: 23, 50: 38, 51: 38, 58: 41, 59: 41,
           60: 32, 61: 33, 62: 32, 63: 33, 64: 32, 65: 33,
           66: 14, 67: 17, 68: 38, 69: 41, 70: 38, 71: 41,
           72: 26, 73: 20, 74: 29, 75: 23, 76: 29, 77: 23}
TIED_KEYS = {f"{p}{r}": f"{p}{s}" for p in ("A_E", "Ea_E", "beta_E") for r, s in TIED_TO.items()}


def resolve_params(spec, base_dir):
    """spec is a path to a JSON of parameters, or an inline object. Returns
    (dict, source string) with every key checked against the catalog's names."""
    if spec is None:
        return {}, "CATALOG DEFAULTS"
    if isinstance(spec, str):
        path = spec if os.path.isabs(spec) else os.path.join(base_dir, spec)
        if not os.path.exists(path):
            die(f"parameters file not found: {path}")
        with open(path) as f:
            params = json.load(f)
        src = path
    elif isinstance(spec, dict):
        params, src = dict(spec), "inline"
    else:
        die("'parameters' must be a path or an object")

    # A parameter file may carry bookkeeping alongside the numbers.
    params = {k: v for k, v in params.items() if not k.startswith("_")}

    known = load_catalog_names()
    unknown = sorted(set(params) - known)
    if unknown:
        tied = [k for k in unknown if k in TIED_KEYS]
        old = [k for k in unknown if k in ("S0_A", "S0_M", "f_terr_A", "f_terr_M")]
        die("unknown parameter name(s) in {}: {}\n"
            "       the engine would silently use the catalog default for these.\n"
            "       Valid names are the {} keys of params_catalog_defaults.json.{}{}"
            .format(src, ", ".join(unknown), len(known),
                    "\n       tied engine IDs share one parameter set; set {} and drop {}."
                    .format(", ".join(sorted({TIED_KEYS[k] for k in tied})), ", ".join(tied))
                    if tied else "",
                    "\n       {} were replaced (2026-09-27) by S0_A_terrace = f_terr_A x S0_A, "
                    "S0_A_step = S0_A, and the same for M.".format(", ".join(old))
                    if old else ""))
    bad = [k for k, v in params.items() if not isinstance(v, (int, float))]
    if bad:
        die(f"non-numeric parameter value(s) in {src}: {', '.join(sorted(bad))}")
    # the ordering constraints (S0 terrace <= step, barrier pairs): refuse before launching
    sys.path.insert(0, HERE)
    import KMC
    broken = KMC.constraint_violations(params)
    if broken:
        die(f"{src} breaks the ordering constraint(s):\n       " + "\n       ".join(broken))
    return {k: float(v) for k, v in params.items()}, src


def toml_path(p):
    """A path safe inside a TOML basic string. Backslashes start an escape
    sequence in TOML, so a Windows path written raw makes the config
    unparseable; forward slashes work on every platform."""
    return str(p).replace("\\", "/")


def write_config(cfg_path, j, params, params_src):
    stall = j.get("stall", {})
    ad = j.get("adaptive", {})
    lattice = j.get("lattice", "lattice/Lattice_10nm_h232.xyz")
    lo, hi = j.get("window_A", [50.0, 200.0])
    temps = ", ".join(f"{float(t):g}" for t in j["temperatures"])

    L = []
    a = L.append
    a("# Generated by run.py -- do not hand-edit; edit the run JSON and re-run.")
    a(f"# source: {os.path.basename(j['_src'])}")
    if j.get("description"):
        a(f"# {j['description']}")
    a("")
    a("[run]")
    a(f"temperatures = [{temps}]")
    a('output_csv   = "{}"'.format(toml_path(j["out"])))
    a("")
    a("[kmc]")
    a('lattice_file             = "{}"'.format(toml_path(lattice)))
    a("target_growth_angstrom   = 1000000.0")
    a("gr_window_A              = [{:g}, {:g}]".format(float(lo), float(hi)))
    a("stall_idle_factor        = {:g}".format(float(stall.get("idle_factor", 0.0))))
    a("stall_min_events         = {}".format(int(stall.get("min_events", 0))))
    a("stall_rel_tol            = {:g}".format(float(stall.get("rel_tol", 0.0))))
    a("stall_abs_tol_A          = {:g}".format(float(stall.get("abs_tol_A", 0.0))))
    a("max_iterations           = {}".format(int(j["max_events"])))
    a("reward_checkpoint_events = {}".format(int(j.get("checkpoint_events", 20000))))
    a("traj_freq     = 0")
    a("log_events    = false")
    a("n_replicates   = {}".format(int(j.get("replicates", 1))))
    a("n_rep_wave     = {}".format(int(ad.get("wave", 48))))
    a("n_rep_max      = {}".format(int(ad.get("n_rep_max", j.get("replicates", 1)))))
    a("target_sem_pct = {:g}".format(float(ad.get("target_sem_pct", 0.0))))
    a("seed_base      = {}".format(int(j.get("seed_base", 1000))))
    a("n_workers      = {}".format(int(j.get("workers", 1))))
    a("")
    a(f"[params]   # {len(params)} overrides from {params_src}")
    for k in sorted(params):
        a(f"{k} = {params[k]!r}")
    a("")
    with open(cfg_path, "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_json")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and the cost, run nothing")
    ap.add_argument("--shard", type=int,
                    help="SLURM array: run the tasks with (k mod nshards) == shard")
    ap.add_argument("--nshards", type=int)
    ap.add_argument("--workers", type=int, help="override the JSON's worker count")
    args = ap.parse_args()

    if not os.path.exists(args.run_json):
        die(f"no such file: {args.run_json}")
    with open(args.run_json) as f:
        j = json.load(f)
    base = os.path.dirname(os.path.abspath(args.run_json))
    j["_src"] = os.path.abspath(args.run_json)

    if "stop" in j:
        die("'stop' (the plateau rule) was retired on 2026-09-27. The growth rate is read "
            "through 'window_A': [50, 200] and a run stops at the window top; the stall "
            "rule is 'stall': {idle_factor, min_events, rel_tol, abs_tol_A}. See README.md.")
    unknown = sorted(set(j) - TOP - {"_src"})
    if unknown:
        die(f"unknown key(s) in the run JSON: {', '.join(unknown)}")
    for req in ("temperatures", "max_events", "out"):
        if req not in j:
            die(f"the run JSON must set '{req}'")
    w = j.get("window_A", [50.0, 200.0])
    if (not isinstance(w, list) or len(w) != 2
            or not all(isinstance(x, (int, float)) for x in w) or not 0 < w[0] < w[1]):
        die(f"'window_A' must be [lo, hi] in angstrom with 0 < lo < hi, got {w!r}")
    for group, allowed in (("stall", STALL), ("adaptive", ADAPTIVE)):
        bad = sorted(set(j.get(group, {})) - allowed)
        if bad:
            die(f"unknown key(s) in '{group}': {', '.join(bad)}")

    cap, ck = int(j["max_events"]), int(j.get("checkpoint_events", 20000))
    if ck <= 0 or cap % ck:
        die(f"checkpoint_events ({ck}) must divide max_events ({cap}) exactly")

    params, params_src = resolve_params(j.get("parameters"), base)

    out = j["out"] if os.path.isabs(j["out"]) else os.path.join(os.getcwd(), j["out"])
    if not out.endswith(".csv"):
        out += ".csv"
    j["out"] = out.replace("\\", "/")
    os.makedirs(os.path.dirname(j["out"]) or ".", exist_ok=True)

    cfg_path = j["out"][:-4] + "_config.toml"
    write_config(cfg_path, j, params, params_src)
    print(f"[run] {args.run_json} -> {cfg_path}")
    print(f"[run] {len(params)} parameter overrides from {params_src}")

    argv = ["run_kmc.py", "--config", cfg_path]
    if args.dry_run:
        argv.append("--dry-run")
    for flag, val in (("--shard", args.shard), ("--nshards", args.nshards),
                      ("--workers", args.workers)):
        if val is not None:
            argv += [flag, str(val)]

    sys.path.insert(0, HERE)
    import run_kmc
    sys.argv = argv
    run_kmc.main()


if __name__ == "__main__":
    main()
