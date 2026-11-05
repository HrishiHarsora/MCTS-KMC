"""
build_search.py -- set up an MCTS search from one JSON file.

    python build_search.py examples/lt1125_four_lever.json
    python build_search.py my_search.json --outdir runs/my_search

It writes, next to the search JSON (or into --outdir):

    config.toml        everything MCTS.py reads
    params_root.json   the root parameter set, flat, all 45 keys
    ROOT_SOURCE.json   provenance: what the root was and what the DRC table was

Then:  python MCTS.py --config config.toml

WHAT THE SEARCH DOES
    It moves a handful of parameters -- the levers -- to fit a measured growth
    rate at several temperatures at once. The error being minimised is, in
    e-folds,

        L = sqrt( mean_T [ ln(GR_T / GR_exp_T)^2 + max(0, ln(h_min / h_T))^2 ] )

    GR_T is read through the thickness window (the mean height's rate from 50 to
    200 A, kmc/reward_metric.py) and h_T is the mean height the films end at. The
    second term is a one-sided penalty: with h_min_A at the window top it is zero
    exactly when every film grew through the whole window, so a set of parameters
    whose films stop growing is not a fit. Reward is R = exp(-L / tau).

WHERE THE LEVER EFFECTS COME FROM
    The DRC table. For each direction it measured D(T) = d ln GR(T) / d ln k and
    the same for the final film height. This script converts those into the units the
    search actually steps in and writes them into config.toml, so MCTS.py never
    converts anything:

        multiplicative lever (a rate knob)   one step s multiplies it by exp(s)
                                             -> effect factor  f = +1
        additive lever (a barrier, kelvin)   one step s adds s * step_ref_T kelvin
                                             -> effect factor  f = -step_ref_T / T

    The minus sign is the whole point: raising a barrier lowers the rate, and it
    lowers it more at low temperature. That is what lets one lever set fit a
    curve rather than a point.

CHOOSING LEVERS
    Either name them ("levers": [...]) or let the script rank the DRC directions
    by their largest combined pull, max_T sqrt(D^2 + D_height^2), and keep the top
    few. Either way a lever the DRC never resolved (SNR below min_snr everywhere)
    is refused -- searching a direction you cannot measure wastes the whole run.

SCHEMA -- see README.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CATALOG = os.path.join(HERE, os.pardir, "kmc", "params_catalog_defaults.json")
NL = chr(10)

TOP = {"description", "root_parameters", "root_md5", "drc_table", "levers",
       "lever_selection", "lever_map", "lever_bounds", "barrier_span_efolds",
       "target_curve", "target_curve_csv", "reference_temperature_K",
       "objective", "search", "kmc", "outputs"}
OBJ = {"h_min_A", "error_threshold"}
SEARCH = {"max_experiments", "max_depth", "max_children", "max_step_efolds",
          "C_ucb", "tau", "planner_map", "expand_bias"}
KMC = {"lattice", "max_events", "checkpoint_events", "window_A", "stall", "workers",
       "seed_base", "n_rep_total", "n_rep_max", "n_rep_wave",
       "sem_target_pct"}

# Default DRC direction -> (searched parameter, kind). "add" = a barrier in
# kelvin; "mul" = the rate knob itself. Override per direction with "lever_map".
DEFAULT_MAP = {"stick_M": ["Ea_chem_M", "add"],
               "stick_As": ["Ea_chem_As", "add"],
               "stick_A_terr": ["S0_A_terrace", "mul"],
               "stick_A_step": ["S0_A_step", "mul"],
               "stick_M_terr": ["S0_M_terrace", "mul"],
               "stick_M_step": ["S0_M_step", "mul"]}
# sticking coefficients stay in (0, 1]; the floor keeps a multiplicative lever off zero. The
# ordering constraints (S0_terrace <= S0_step, barrier pairs) are enforced by MCTS.py.
DEFAULT_BOUNDS = {"S0_A_terrace": [1e-6, 1.0], "S0_A_step": [1e-6, 1.0],
                  "S0_M_terrace": [1e-6, 1.0], "S0_M_step": [1e-6, 1.0]}


def num(x, default=None):
    """Format a number for TOML the way it was written in the JSON: an integer
    stays an integer, a float keeps its decimal point. TOML needs the decimal
    point to read a value as a float, and a stage's config should look the same
    whether it was written by hand or generated."""
    if x is None:
        x = default
    if isinstance(x, bool):
        raise TypeError("expected a number")
    if isinstance(x, int):
        return str(x)
    return repr(float(x))


def die(msg):
    sys.exit(f"[build_search] {msg}")


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def lever_of(direction, key, custom):
    """Which parameter a DRC direction corresponds to, and how it is stepped."""
    if direction in custom:
        name, kind = custom[direction]
        return str(name), str(kind)
    if direction in DEFAULT_MAP:
        return tuple(DEFAULT_MAP[direction])
    if direction.startswith("E") and key.startswith("A_E"):
        return "Ea_" + key[2:], "add"           # the reaction's barrier
    die(f"unmapped DRC direction '{direction}' (key '{key}'). "
        f"Add it to 'lever_map' in the search JSON.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("search_json")
    ap.add_argument("--outdir", help="where to write; default is the JSON's folder")
    a = ap.parse_args()

    if not os.path.exists(a.search_json):
        die(f"no such file: {a.search_json}")
    with open(a.search_json) as f:
        j = json.load(f)
    src_dir = os.path.dirname(os.path.abspath(a.search_json))
    work = os.path.abspath(a.outdir or src_dir)
    os.makedirs(work, exist_ok=True)

    def rel(p):
        return p if os.path.isabs(p) else os.path.join(src_dir, p)

    bad = sorted(set(j) - TOP)
    if bad:
        die(f"unknown key(s) in the search JSON: {', '.join(bad)}")
    if "n_rep_wiggle_pct" in j.get("kmc", {}):
        die("kmc.n_rep_wiggle_pct was retired 2026-09-28: a fixed +-1 % tolerance returned "
            "n_flat = 1 whenever the films scatter by less than 1 %. The flat region is now "
            "kmc/flat_region.py (mean +- 1 SEM) and a node's rate is the mean of all its "
            "replicates. Delete the key.")
    for group, allowed in (("objective", OBJ), ("search", SEARCH), ("kmc", KMC)):
        bad = sorted(set(j.get(group, {})) - allowed)
        if bad:
            die(f"unknown key(s) in '{group}': {', '.join(bad)}")
    for req in ("root_parameters", "drc_table"):
        if req not in j:
            die(f"the search JSON must set '{req}'")

    # ---- the root ----------------------------------------------------------
    root_path = rel(j["root_parameters"])
    if not os.path.exists(root_path):
        die(f"root parameter file not found: {root_path}")
    if j.get("root_md5") and md5(root_path) != j["root_md5"]:
        die(f"root_md5 does not match {root_path}: expected {j['root_md5']}, "
            f"found {md5(root_path)}")
    with open(root_path) as f:
        raw = json.load(f)
    if "params" in raw and isinstance(raw["params"], dict):
        raw = raw["params"]
    root = {str(k): float(v) for k, v in raw.items() if not str(k).startswith("_")}
    with open(CATALOG) as f:
        known = set(json.load(f).keys())
    if set(root) != known:
        die(f"the root must carry all {len(known)} parameter names "
            f"(missing {sorted(known - set(root))[:5]}, "
            f"unknown {sorted(set(root) - known)[:5]})")

    # ---- the DRC table -----------------------------------------------------
    drc_path = rel(j["drc_table"])
    if not os.path.exists(drc_path):
        die(f"DRC table not found: {drc_path}")
    with open(drc_path) as f:
        rows = list(csv.DictReader(f))
    need = {"direction", "key", "T_K", "D", "D_height", "SNR"}
    if not rows or not need <= set(rows[0]):
        die(f"{drc_path} must have columns {sorted(need)}")
    temps = sorted({int(float(r["T_K"])) for r in rows})
    D, Dh, SNR, key_of = defaultdict(dict), defaultdict(dict), defaultdict(dict), {}
    for r in rows:
        d, T = r["direction"], int(float(r["T_K"]))
        D[d][T], Dh[d][T] = float(r["D"]), float(r["D_height"])
        SNR[d][T] = float(r["SNR"])
        key_of[d] = r["key"]
    for d in D:
        if sorted(D[d]) != temps:
            die(f"direction '{d}' is not measured at every temperature")

    # ---- the target curve --------------------------------------------------
    if j.get("target_curve"):
        exp = {int(float(k)): float(v) for k, v in j["target_curve"].items()}
    elif j.get("target_curve_csv"):
        with open(rel(j["target_curve_csv"])) as f:
            exp = {int(float(r["T_K"])): float(r["GR_exp_umin"])
                   for r in csv.DictReader(f)}
    else:
        die("set either 'target_curve' or 'target_curve_csv'")
    missing = [T for T in temps if T not in exp]
    if missing:
        die(f"the target curve has no value at {missing}")

    # ---- choose the levers -------------------------------------------------
    custom = {k: list(v) for k, v in j.get("lever_map", {}).items()}
    sel = j.get("lever_selection", {})
    min_snr = float(sel.get("min_snr", 3.0))
    resolved = [d for d in D if max(SNR[d].values()) >= min_snr]
    score = {d: max(math.hypot(D[d][T], Dh[d][T]) for T in temps) for d in D}

    if j.get("levers"):
        want = list(j["levers"])
        back = {lever_of(d, key_of[d], custom)[0]: d for d in D}
        unknown = [n for n in want if n not in back]
        if unknown:
            die(f"not reachable from any DRC direction: {unknown}")
        weak = [n for n in want if back[n] not in resolved]
        if weak:
            die(f"asked for {weak}, which the DRC never resolved anywhere "
                f"(SNR < {min_snr:g}). Searching an unmeasurable direction "
                f"burns the run; lower 'lever_selection.min_snr' only if you "
                f"mean it.")
        chosen = [back[n] for n in want]
    else:
        chosen = sorted(resolved, key=lambda d: -score[d])[:int(sel.get("top", 8))]
    if not chosen:
        die("no direction passed the SNR filter")

    tref = float(j.get("reference_temperature_K", max(temps)))
    span = float(j.get("barrier_span_efolds", 8.0))
    user_bounds = dict(DEFAULT_BOUNDS)
    user_bounds.update(j.get("lever_bounds", {}))

    LEV = {}
    for d in chosen:
        name, kind = lever_of(d, key_of[d], custom)
        if name not in root:
            die(f"lever '{name}' is not one of the {len(known)} parameters")
        v0 = root[name]
        if name in user_bounds:
            lo, hi = float(user_bounds[name][0]), float(user_bounds[name][1])
        elif kind == "add":
            lo, hi = max(0.0, v0 - span * tref), v0 + span * tref
        else:
            lo, hi = v0 / math.exp(span), v0 * math.exp(span)
        if not lo < v0 < hi:
            die(f"lever '{name}' starts at {v0!r}, outside its bounds "
                f"[{lo!r}, {hi!r}]")
        LEV[name] = dict(direction=d, key=key_of[d], kind=kind, lo=lo, hi=hi, v0=v0)

    # effect of ONE step of each lever, per temperature
    g, h = defaultdict(dict), defaultdict(dict)
    for name, L in LEV.items():
        for T in temps:
            f = 1.0 if L["kind"] == "mul" else -(tref / T)
            g[T][name] = f * D[L["direction"]][T]
            h[T][name] = f * Dh[L["direction"]][T]

    # ---- write config.toml -------------------------------------------------
    S = j.get("search", {})
    O = j.get("objective", {})
    K = j.get("kmc", {})
    if "stop" in K:
        die("kmc.stop (the plateau rule) was retired on 2026-09-27: set kmc.window_A "
            "[lo, hi] and kmc.stall {idle_factor, min_events, rel_tol, abs_tol_A}")
    stall = K.get("stall", {})
    win = K.get("window_A", [50.0, 200.0])
    if (not isinstance(win, list) or len(win) != 2
            or not all(isinstance(x, (int, float)) for x in win) or not 0 < win[0] < win[1]):
        die(f"kmc.window_A must be [lo, hi] in angstrom with 0 < lo < hi, got {win!r}")
    out = j.get("outputs", {})
    pinned = {k: v for k, v in root.items() if k not in LEV}
    add = [n for n, L in LEV.items() if L["kind"] == "add"]
    src_md5 = md5(root_path)

    W = []
    w = W.append
    w("# =============================================================================")
    w("# config.toml -- generated by build_search.py. Nothing here is typed in;")
    w(f"# every number comes from {os.path.basename(a.search_json)} or from the DRC.")
    if j.get("description"):
        w("#")
        for line in str(j["description"]).splitlines():
            w(f"# {line}")
    w("#")
    w(f"# Root    {os.path.basename(root_path)}  md5 {src_md5}")
    w(f"# Levers  {os.path.basename(drc_path)}, {len(chosen)} of {len(D)} directions")
    name_of = {L["direction"]: n for n, L in LEV.items()}
    for d in chosen:
        w("#           %-12s -> %-12s  |D| %.3f..%.3f  |Dh| %.3f..%.3f  best SNR %.0f"
          % (d, name_of[d],
             min(abs(D[d][T]) for T in temps), max(abs(D[d][T]) for T in temps),
             min(abs(Dh[d][T]) for T in temps), max(abs(Dh[d][T]) for T in temps),
             max(SNR[d].values())))
    w("#")
    w("# ERROR. Over every temperature at once:")
    w("#   L = sqrt( mean_T [ ln(GR_T/GR_exp_T)^2 + max(0, ln(h_min/h_T))^2 ] )")
    w("# The search stops only when EVERY temperature is inside error_threshold on")
    w("# the rate and EVERY film is above h_min_A; best_params holds the best node")
    w("# that is.")
    w("# =============================================================================")
    w("")
    w("[run]")
    w('mode         = "ladder"' if len(temps) > 1 else 'mode         = "single"')
    w("temperatures = [%s]" % ", ".join(str(T) for T in temps))
    w("")
    w("[target_curve]")
    for T in temps:
        w("%-5d = %.5f" % (T, exp[T]))
    w("")
    w("[kmc]")
    w('lattice_file             = "%s"' % K.get("lattice", "Lattice_10nm_h232.xyz"))
    w("")
    w("# THE GROWTH RATE: the mean height's rate through [lo, hi] A; a trajectory")
    w("# stops at the first checkpoint at or above hi (kmc/reward_metric.py).")
    w("gr_window_A              = [%s, %s]" % (num(win[0], 50.0), num(win[1], 200.0)))
    w("# STALL: end a film whose mean height has stopped advancing.")
    w("#   advance <=> h_mean > best + max(rel_tol*|best|, abs_tol)")
    w("#   stop    <=> idle >= max(min_events, idle_factor x events at last advance)")
    w("target_growth_angstrom   = 1000000.0")
    w("stall_idle_factor        = %s" % num(stall.get("idle_factor"), 2.0))
    w("stall_min_events         = %d" % int(stall.get("min_events", 500000)))
    w("stall_rel_tol            = %s" % num(stall.get("rel_tol"), 0.005))
    w("stall_abs_tol_A          = %s" % num(stall.get("abs_tol_A"), 0.01))
    w("")
    cap = int(K.get("max_events", 10000000))
    ck = int(K.get("checkpoint_events", 20000))
    if ck <= 0 or cap % ck:
        die(f"checkpoint_events ({ck}) must divide max_events ({cap}) exactly")
    w("# The backstop, not the intended stop: at most %d / %d = %d checkpoints."
      % (cap, ck, cap // ck))
    w("max_iterations           = %d" % cap)
    w("reward_checkpoint_events = %d" % ck)
    w("")
    w("pressure_AsH  = 608      # NOT read: KMC.py hardcodes PA, PM = 608, 65.86")
    w("pressure_MMGa = 65.86")
    w("")
    w("# --- replicates: every node measured to n_rep_total at EACH temperature,")
    w("# then extended a wave at a time while any temperature's running mean has")
    w("# not gone flat (kmc/flat_region.py: inside mean +- 1 SEM and at the middle of")
    w("# that band, before the last 20 % of the replicates), or its SEM is above")
    w("# sem_target_pct, up to n_rep_max. The rate is the mean of ALL replicates.")
    w("n_rep_total      = %d" % int(K.get("n_rep_total", 96)))
    w("n_rep_max        = %d" % int(K.get("n_rep_max", 240)))
    w("n_rep_wave       = %d" % int(K.get("n_rep_wave", 48)))
    w("sem_target_pct   = %s" % num(K.get("sem_target_pct"), 2.0))
    w("n_workers        = %d" % int(K.get("workers", 48)))
    w("seed_base        = %d" % int(K.get("seed_base", 90800000)))
    w("")
    w("[objective]")
    w("# Every film must end above this at EVERY temperature for a node to be a")
    w("# result. At the window top it means every film grew through the whole window.")
    w("h_min_A = %s" % num(O.get("h_min_A"), float(win[1])))
    w("")
    w("[mcts]")
    cu, ta = S.get("C_ucb", "auto"), S.get("tau", "auto")
    w('C_ucb           = %s' % (f'"{cu}"' if isinstance(cu, str) else f"{float(cu):g}"))
    w('tau             = %s' % (f'"{ta}"' if isinstance(ta, str) else f"{float(ta):g}"))
    w("# 'auto': C = spread of the DRC's predicted rewards / sqrt(ln max_experiments),")
    w("# tau = L_root / 3. Both are read off the root, so neither is a free knob.")
    w("error_threshold = %s        # e-folds, per temperature, on the RATE"
      % num(O.get("error_threshold"), 0.10))
    w("max_experiments = %d" % int(S.get("max_experiments", 80)))
    w("max_depth       = %d" % int(S.get("max_depth", 12)))
    w("max_children    = %d" % int(S.get("max_children", 5)))
    w("expand_bias     = %s         # rate-only path only" % num(S.get("expand_bias"), 0.7))
    w("# A child = u x (the smallest step vector that closes every temperature's")
    w("# rate gap and lifts any film back to the floor, least squares) + a random")
    w("# kick sized so the child is never predicted worse than its parent.")
    w("max_step_efolds = %s" % num(S.get("max_step_efolds"), 1.0))
    w("step_ref_T      = %s         # a barrier step s means s * this many kelvin"
      % num(j.get("reference_temperature_K"), tref))
    w('planner_map     = "%s"' % str(S.get("planner_map", "learned")))
    w("additive_levers = [%s]" % ", ".join('"%s"' % n for n in add))
    w("")
    w('history_file     = "%s"' % out.get("history", "mcts_history.txt"))
    w('convergence_csv  = "%s"' % out.get("convergence", "mcts_convergence.csv"))
    w('best_params_json = "%s"' % out.get("best_params", "best_params.json"))
    w("")
    w("# -----------------------------------------------------------------------------")
    w("# [search_params] name = [initial, low, high]. Initial = the root's value.")
    w("# -----------------------------------------------------------------------------")
    w("[search_params]")
    for n, L in LEV.items():
        w("%-12s = [%r, %r, %r]" % (n, L["v0"], L["lo"], L["hi"]))
    w("")
    w("# Signed effect of ONE step of each lever on ln GR, per temperature.")
    for T in temps:
        w("[drc_rate.T%d]" % T)
        for n in LEV:
            w("%-12s = %+.5f" % (n, g[T][n]))
        w("")
    w("# ... and on ln h, the final mean height.")
    for T in temps:
        w("[drc_height.T%d]" % T)
        for n in LEV:
            w("%-12s = %+.5f" % (n, h[T][n]))
        w("")
    w("# -----------------------------------------------------------------------------")
    w("# [params] -- the pinned parameters (all of them minus the levers).")
    w("# KMC falls back to CATALOG for any key it is not given, so this is complete.")
    w("# -----------------------------------------------------------------------------")
    w("[params]")
    for k in sorted(pinned):
        w("%-12s = %r" % (k, pinned[k]))
    w("")
    w("[logging]")
    w("enabled         = true")
    w('sim_log_csv     = "sim_log.csv"')
    w('node_waves_csv  = "node_waves.csv"')
    w('lever_map_csv   = "lever_map.csv"')
    w("save_interval   = 1")
    w("log_checkpoints = false")

    cfg_path = os.path.join(work, "config.toml")
    io.open(cfg_path, "w", encoding="utf-8", newline="").write(NL.join(W) + NL)
    with open(os.path.join(work, "params_root.json"), "w") as f:
        json.dump(root, f, indent=2, sort_keys=True)
    with open(os.path.join(work, "ROOT_SOURCE.json"), "w") as f:
        json.dump({"search_json": os.path.basename(a.search_json),
                   "source_file": os.path.basename(root_path),
                   "source_md5": src_md5,
                   "drc_file": os.path.basename(drc_path),
                   "drc_md5": md5(drc_path),
                   "levers": sorted(LEV),
                   "config_md5": md5(cfg_path)}, f, indent=2, sort_keys=True)

    print(f"[build_search] root   {root_path} (md5 {src_md5})")
    print(f"[build_search] levers ({len(LEV)}): {', '.join(sorted(LEV))}")
    for n in sorted(LEV):
        L = LEV[n]
        print("   %-12s %-3s start %-12.6g bounds [%.6g, %.6g]  rate effect "
              "%+.3f (%g K) .. %+.3f (%g K)"
              % (n, L["kind"], L["v0"], L["lo"], L["hi"],
                 g[temps[0]][n], temps[0], g[temps[-1]][n], temps[-1]))
    print(f"[build_search] -> {cfg_path}, params_root.json, ROOT_SOURCE.json")
    print("[build_search] next: python MCTS.py --config config.toml")


if __name__ == "__main__":
    main()
