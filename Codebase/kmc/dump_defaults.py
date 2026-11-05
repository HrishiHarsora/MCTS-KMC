"""
dump_defaults.py -- write the engine's catalog defaults out as an explicit JSON
parameter set.

    python dump_defaults.py                       # -> params_catalog_defaults.json
    python dump_defaults.py --verify              # prove the JSON is a no-op
    python dump_defaults.py --out other.json

WHY THIS EXISTS
    `[params]` empty (or an empty dict) means "use catalog defaults", which is
    correct but opaque: nothing on disk says WHAT the defaults are. This writes
    them explicitly so a default-parameter run is self-documenting, diffable
    against any fitted set, and usable as an MCTS root.

    The two forms are interchangeable by construction: the JSON contains exactly
    the values the engine would have used, so passing it changes nothing.
    --verify checks that claim rather than asserting it.

KEY NAMES (this is the whole parameter namespace)
    A_E<r>, Ea_E<r>, beta_E<r>   for r in 1, 2, 3, 12, 13, 14, 17, 20, 23, 26, 29, 32, 33
    S0_A_terrace, S0_A_step, Ea_chem_As   AsH sticking, terrace (R1) and step (R2)
    S0_M_terrace, S0_M_step, Ea_chem_M    MMGa sticking, terrace (R6) and step (R7)

    Tied engine IDs (KMC._TIED_RIDS: 4, 5 take 3's parameters, and the other IDs of
    each incorporation reaction take its first ID's) are excluded for the same
    reason as 6-11.

    Reactions 6-11 are impingement-driven: their rate is set by the
    Hertz-Knudsen formula, so A/Ea/beta for those IDs are silently ignored. They
    are excluded here for exactly that reason -- writing them out would suggest
    they are tunable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import KMC

# Adsorption defaults, read from the engine's own literals.
SCALARS = dict(KMC._STICK_DEFAULTS)


def arrhenius_rids():
    """The reaction IDs that own Arrhenius parameters: KMC._ARRHENIUS_RIDS.
    6..11 are impingement-driven; 4 and 5 are tied to 3."""
    return list(KMC._ARRHENIUS_RIDS)


def rate_rids():
    """Every reaction ID, tied and adsorption ones included (checked in verify)."""
    return list(range(1, KMC._NUM_RID))


def build():
    p = {}
    for r in arrhenius_rids():
        p[f"A_E{r}"] = float(KMC._BASE_A[r])
        p[f"Ea_E{r}"] = float(KMC._BASE_EA[r])
        p[f"beta_E{r}"] = float(KMC._BASE_BETA[r])
    p.update(SCALARS)
    return p


def verify(p):
    """The JSON must be a NO-OP: feeding it back must reproduce the engine's own
    rate table bit for bit, and the adsorption scalars must match the
    literals the engine falls back to."""
    import numpy as np
    ok = True
    for r in arrhenius_rids():
        for key, arr in ((f"A_E{r}", KMC._BASE_A), (f"Ea_E{r}", KMC._BASE_EA),
                         (f"beta_E{r}", KMC._BASE_BETA)):
            if p[key] != float(arr[r]):
                print(f"  MISMATCH {key}: json={p[key]} engine={float(arr[r])}")
                ok = False

    # Round-trip the full rate vector through _compute_rates both ways.
    rids = np.array(rate_rids(), dtype=np.int32)
    for T in (900.0, 1125.0, 1230.0):
        a = KMC._compute_rates(rids, T, 1.0, 1.0, 1.0, None)
        b = KMC._compute_rates(rids, T, 1.0, 1.0, 1.0, p)
        d = float(np.max(np.abs(a - b)))
        print(f"  T={T:>6.0f} K  max |rate(defaults) - rate(json)| = {d:.3e}")
        if d != 0.0:
            ok = False

    for k, v in SCALARS.items():
        if p[k] != v:
            print(f"  MISMATCH {k}: json={p[k]} expected={v}")
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="params_catalog_defaults.json")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    p = build()
    n_arr = len(arrhenius_rids())
    print(f"[dump_defaults] {n_arr} Arrhenius reactions x 3 = {n_arr * 3} "
          f"+ {len(SCALARS)} scalars = {len(p)} parameters")

    if args.verify:
        print("[dump_defaults] verifying the JSON is a no-op:")
        ok = verify(p)
        print("[dump_defaults] " + ("PASS -- identical to catalog defaults"
                                    if ok else "FAIL -- see mismatches above"))
        sys.exit(0 if ok else 1)

    with open(args.out, "w") as f:
        json.dump(p, f, indent=2, sort_keys=True)
    print(f"[dump_defaults] wrote {args.out}")


if __name__ == "__main__":
    main()
