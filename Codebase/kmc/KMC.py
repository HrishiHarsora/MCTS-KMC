"""
KMC.py — Kinetic Monte Carlo engine for GaAs MOCVD growth.

20 surface reactions (REACTIONS.md), 77 engine IDs, 54 parameters, 8 ordering constraints.
  * the Ga route is as general as the As route: DD_MMGa loses its methyl and becomes DDD_Ga
    (ID 36, part of R8), which desorbs (R18, ID 37) or incorporates on a terrace (R19, IDs
    38-40) or at a step (R20, IDs 41-43), as DDD_As does (R5, R16, R17);
  * the partner below an incorporating precursor can also be an adsorbed precursor of the
    other element (IDs 44-59), cases of R12-R17, R19, R20 at their rates;
  * every film atom with two / three bonds desorbs at R10 / R11's rate: DD_Ga / D_Ga, DD_As /
    D_As (IDs 60, 61), D_MMGa / MMGa (62, 63, leaving as MMGa), D_AsH / AsH (64, 65, leaving
    as AsH);
  * no bond forms automatically: every bond forms by a rated incorporation event. A film atom
    with a free bond bonds down to the atom below it that has a free bond (IDs 66-77, cases of
    R12-R17, R19, R20 by the upper atom's kind); an H or a methyl rotates out of the way.
Compiled numba kernels (nb_bucket.py) run the same events in the same order as enum_scalar and
give identical films; KMC_bucket.py uses them whenever numba imports.

Physics: BKL algorithm on a GaAs(001) slab, 77 engine IDs (34 and 35 are E4.2 and E6.2
with both upper sites free, see _RID_E42_SPLIT), Arrhenius rate law with optional
impingement formula for adsorption events.

PUBLIC API (used by MCTS.py):
    run_kmc_trajectory(target_T, params, node_label, traj_freq, lattice_file)
      -> growth rate in µm/min

STANDALONE USAGE (reads config.toml):
    python KMC.py                        # uses [run] settings from config.toml
    python KMC.py --temperature 1000     # override temperature
    python KMC.py --mode sweep           # override mode
    python KMC.py --params my_params.json  # load params from JSON instead of config.toml

CONFIG FILE: config.toml (must be in the working directory)
    [params]  section overrides default rate catalog values.
    [run]     section controls mode, temperatures, output file.
    [kmc]     section controls lattice file, supercell, max iterations, etc.
    If config.toml is missing or [params] is empty, all catalog defaults are used.

PARAMETER OVERRIDES (key names for the `params` dict):
    Arrhenius per-reaction : A_EN, Ea_EN, beta_EN  (N = _ARRHENIUS_RIDS: 1, 2, 3, 12, 13, 14, 17, 20, 23, 26,
                             29, 32, 33, 37, 38, 41; tied IDs in _TIED_RIDS)
    AsH adsorption         : S0_A_terrace (R1, IDs 6, 7, 34), S0_A_step (R2, ID 8), Ea_chem_As
    MMGa adsorption        : S0_M_terrace (R6, IDs 9, 10, 35), S0_M_step (R7, ID 11), Ea_chem_M
    Ordering constraints   : _CONSTRAINTS (checked on every run; see check_constraints)

DEPENDENCIES: numpy, scipy (cKDTree, to build the neighbour tables)
OPTIONAL   : tomllib (Python >=3.11 built-in) or tomli (pip install tomli)
"""
from __future__ import annotations

import math
import os
import re
import sys

import numpy as np

# Force UTF-8 stdout/stderr so non-ASCII chars (±, →) in print() don't crash on
# Windows' cp1252 console. No-op on systems already using UTF-8 (e.g. HPC Linux),
# so this is safe both locally and on the cluster.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# TOML loader — stdlib tomllib (Py 3.11+) or pip tomli fallback
# =============================================================================
def _load_toml(path: str) -> dict:
    try:
        import tomllib  # type: ignore   # Python 3.11+
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except ImportError:
        pass
    try:
        import tomli  # type: ignore   # pip install tomli
        with open(path, "rb") as fh:
            return tomli.load(fh)
    except ImportError:
        raise ImportError(
            "TOML support requires Python >=3.11 (built-in tomllib) "
            "or 'pip install tomli' for older Python."
        )


def _read_config(path: str = "config.toml") -> dict:
    if not os.path.exists(path):
        return {}
    return _load_toml(path)


# =============================================================================
# Species integer codes
# =============================================================================
V_AS, V_GA, S_AS, S_GA           = 0, 1, 2, 3
DD_GA, D_GA, GA_                  = 4, 5, 6
MMGA, DD_MMGA, D_MMGA             = 7, 8, 9
DD_AS, D_AS, AS_                  = 10, 11, 12
DD_ASH, D_ASH, ASH, DDD_AS        = 13, 14, 15, 16
DDD_GA                            = 17      # adsorbed Ga, one bond (copy: DD_MMGa minus methyl)

# Boolean category lookup tables indexed by species code. The hot loop asks the
# same category questions ("is this a real/deposited atom?", "is this vacant?")
# over the whole lattice every iteration; a single gather into these tables is far
# cheaper than np.isin (which sorts + searchsorts) and is bit-for-bit identical.
_N_SPECIES = 18
_IS_REAL = np.zeros(_N_SPECIES, dtype=bool)   # deposited/adsorbed: not substrate, not vacant
_IS_VAC  = np.zeros(_N_SPECIES, dtype=bool)   # vacant scaffold sites
for _c in (DD_GA, D_GA, GA_, MMGA, DD_MMGA, D_MMGA, DD_AS, D_AS, AS_,
           DD_ASH, D_ASH, ASH, DDD_AS, DDD_GA):
    _IS_REAL[_c] = True
_IS_VAC[V_AS] = True
_IS_VAC[V_GA] = True

# Species that can be the SOURCE of a reaction (everything the enumerator scans for).
# This excludes vacant/substrate AND the fully-incorporated GA_/AS_ bulk, so it selects
# only the thin reactive surface band — the enumerator works over just these sites
# instead of the whole lattice.
_IS_REACTIVE = np.zeros(_N_SPECIES, dtype=bool)
for _c in (DD_GA, D_GA, MMGA, DD_MMGA, D_MMGA, DD_AS, D_AS,
           DD_ASH, D_ASH, ASH, DDD_AS, DDD_GA):
    _IS_REACTIVE[_c] = True

_LAST_ITERATIONS = 0    # iterations taken by the most recent run_kmc_trajectory (diagnostics)
_LAST_THICKNESS_A = 0.0  # final film thickness [A] of the most recent run_kmc_trajectory (diagnostics)
_LAST_EVENT_COUNTS = None  # events fired per engine ID in the most recent trajectory (logs)
_LAST_TOTAL_FIRED = None

# Human-readable description for each reaction ID — used in event-frequency logs
_RID_DESCRIPTION = {
    1:  "DD_AsH desorption",
    2:  "DDD_As desorption",
    3:  "H abstraction from DD_AsH",
    4:  "H abstraction from D_AsH",
    5:  "H abstraction from AsH",
    6:  "AsH terrace adsorption (DD_Ga; D_MMGa with one upper site free)",
    7:  "AsH terrace adsorption (D_Ga, reactive)",
    8:  "AsH step adsorption",
    9:  "MMGa terrace adsorption (DD_As; D_AsH with one upper site free)",
    10: "MMGa terrace adsorption (D_As beside an adsorbed MMGa)",
    11: "MMGa step adsorption",
    12: "CH3 (methyl) desorption",
    13: "MMGa desorption",
    14: "DD_MMGa terrace incorp., partner DD_As",
    15: "DD_MMGa terrace incorp., partner D_AsH",
    16: "DD_MMGa terrace incorp., partner D_As",
    17: "DD_MMGa step incorp.,    partner DD_As",
    18: "DD_MMGa step incorp.,    partner D_AsH",
    19: "DD_MMGa step incorp.,    partner D_As",
    20: "DD_AsH step incorp.,     partner DD_Ga",
    21: "DD_AsH step incorp.,     partner D_MMGa",
    22: "DD_AsH step incorp.,     partner D_Ga",
    23: "DDD_As step incorp.,     partner DD_Ga",
    24: "DDD_As step incorp.,     partner D_MMGa",
    25: "DDD_As step incorp.,     partner D_Ga",
    26: "DD_AsH terrace incorp.,  partner DD_Ga",
    27: "DD_AsH terrace incorp.,  partner D_MMGa",
    28: "DD_AsH terrace incorp.,  partner D_Ga",
    29: "DDD_As terrace incorp.,  partner DD_Ga",
    30: "DDD_As terrace incorp.,  partner D_MMGa",
    31: "DDD_As terrace incorp.,  partner D_Ga",
    32: "DD_Ga desorption",
    33: "D_Ga  desorption",
    34: "AsH terrace adsorption (D_MMGa with both upper sites free: half rate each)",
    35: "MMGa terrace adsorption (D_AsH with both upper sites free: half rate each)",
    36: "CH3 (methyl) removal from DD_MMGa -> DDD_Ga",
    37: "DDD_Ga desorption",
    38: "DDD_Ga terrace incorp.,  partner DD_As",
    39: "DDD_Ga terrace incorp.,  partner D_AsH",
    40: "DDD_Ga terrace incorp.,  partner D_As",
    41: "DDD_Ga step incorp.,     partner DD_As",
    42: "DDD_Ga step incorp.,     partner D_AsH",
    43: "DDD_Ga step incorp.,     partner D_As",
    44: "DD_MMGa terrace incorp., partner adsorbed DD_AsH",
    45: "DD_MMGa terrace incorp., partner adsorbed DDD_As",
    46: "DD_AsH terrace incorp.,  partner adsorbed DD_MMGa",
    47: "DD_AsH terrace incorp.,  partner adsorbed DDD_Ga",
    48: "DDD_As terrace incorp.,  partner adsorbed DD_MMGa",
    49: "DDD_As terrace incorp.,  partner adsorbed DDD_Ga",
    50: "DDD_Ga terrace incorp.,  partner adsorbed DD_AsH",
    51: "DDD_Ga terrace incorp.,  partner adsorbed DDD_As",
    52: "DD_MMGa step incorp.,    partner adsorbed DD_AsH",
    53: "DD_MMGa step incorp.,    partner adsorbed DDD_As",
    54: "DD_AsH step incorp.,     partner adsorbed DD_MMGa",
    55: "DD_AsH step incorp.,     partner adsorbed DDD_Ga",
    56: "DDD_As step incorp.,     partner adsorbed DD_MMGa",
    57: "DDD_As step incorp.,     partner adsorbed DDD_Ga",
    58: "DDD_Ga step incorp.,     partner adsorbed DD_AsH",
    59: "DDD_Ga step incorp.,     partner adsorbed DDD_As",
    60: "DD_As desorption (two bonds; R10's rate)",
    61: "D_As  desorption (three bonds; R11's rate)",
    62: "D_MMGa desorption (two bonds; R10's rate)",
    63: "MMGa  desorption (three bonds; R11's rate)",
    64: "D_AsH desorption (two bonds; R10's rate)",
    65: "AsH   desorption (three bonds; R11's rate)",
    66: "D_MMGa bonds down (film, terrace; R12)",
    67: "D_MMGa bonds down (film, step; R13)",
    68: "DD_Ga bonds down (film, terrace; R19)",
    69: "DD_Ga bonds down (film, step; R20)",
    70: "D_Ga  bonds down (film, terrace; R19)",
    71: "D_Ga  bonds down (film, step; R20)",
    72: "D_AsH bonds down (film, terrace; R14)",
    73: "D_AsH bonds down (film, step; R15)",
    74: "DD_As bonds down (film, terrace; R16)",
    75: "DD_As bonds down (film, step; R17)",
    76: "D_As  bonds down (film, terrace; R16)",
    77: "D_As  bonds down (film, step; R17)",
}

# Reactions as the notes and paper number them (decisions 2026-09-26/27). One reaction can
# span several engine IDs; every engine ID belongs to exactly one reaction. Engine IDs,
# parameter names and the n_rid_XX log columns stay on the engine numbering; every output
# that a reader sees first (event log, n_R<k> log columns, REACTIONS.md) is by reaction.
_REACTIONS = [                         # (label, name, engine IDs)
    ("R1", "AsH terrace adsorption", (6, 34, 7)),
    ("R2", "AsH step adsorption", (8,)),
    ("R3", "Hydrogen removal from AsH", (3, 4, 5)),
    ("R4", "Desorption of AsH", (1,)),
    ("R5", "Desorption of As", (2,)),
    ("R6", "MMGa terrace adsorption", (9, 35, 10)),
    ("R7", "MMGa step adsorption", (11,)),
    ("R8", "Methyl removal from MMGa", (12, 36)),
    ("R9", "Desorption of MMGa", (13,)),
    ("R10", "Desorption of an atom with two bonds", (32, 62, 60, 64)),
    ("R11", "Desorption of an atom with three bonds", (33, 63, 61, 65)),
    ("R12", "MMGa terrace incorporation", (14, 15, 16, 44, 45, 66)),
    ("R13", "MMGa step incorporation", (17, 18, 19, 52, 53, 67)),
    ("R14", "AsH terrace incorporation", (26, 27, 28, 46, 47, 72)),
    ("R15", "AsH step incorporation", (20, 21, 22, 54, 55, 73)),
    ("R16", "As terrace incorporation", (29, 30, 31, 48, 49, 74, 76)),
    ("R17", "As step incorporation", (23, 24, 25, 56, 57, 75, 77)),
    # The Ga route, as general as the As route. Incorporation onto an adsorbed precursor
    # (IDs 44-59) and As desorption with two / three bonds (60, 61) are cases of the
    # reactions above.
    ("R18", "Desorption of Ga", (37,)),
    ("R19", "Ga terrace incorporation", (38, 39, 40, 50, 51, 68, 70)),
    ("R20", "Ga step incorporation", (41, 42, 43, 58, 59, 69, 71)),
]
_RID_REACTION = {r: int(lab[1:]) for lab, _, ids in _REACTIONS for r in ids}
_RID_CASE = {6: "E1.1/E1.2", 34: "E1.2", 7: "E1.3/E1.4", 3: "E3.1", 4: "E3.2", 5: "E3.3",
             9: "E6.1/E6.2", 35: "E6.2", 10: "E6.3",
             14: "E12.1", 15: "E12.2", 16: "E12.3",
             17: "E13.1", 18: "E13.2", 19: "E13.3/E13.4",
             26: "E14.1", 27: "E14.2", 28: "E14.3",
             20: "E15.1", 21: "E15.2", 22: "E15.3/E15.4",
             29: "E16.1", 30: "E16.2", 31: "E16.3",
             23: "E17.1", 24: "E17.2", 25: "E17.3/E17.4",
             12: "E8.1/E8.2", 36: "E8.3",
             38: "E19.1", 39: "E19.2", 40: "E19.3", 41: "E20.1", 42: "E20.2", 43: "E20.3/E20.4",
             44: "E12.4", 45: "E12.5", 52: "E13.5", 53: "E13.6",
             46: "E14.4", 47: "E14.5", 54: "E15.5", 55: "E15.6",
             48: "E16.4", 49: "E16.5", 56: "E17.5", 57: "E17.6",
             50: "E19.4", 51: "E19.5", 58: "E20.5", 59: "E20.6",
             32: "E10.1", 62: "E10.2", 60: "E10.3", 64: "E10.4",
             33: "E11.1-E11.3", 63: "E11.4", 61: "E11.5", 65: "E11.6",
             66: "E12.6", 67: "E13.7", 72: "E14.6", 73: "E15.7",
             74: "E16.6", 76: "E16.7", 75: "E17.7", 77: "E17.8",
             68: "E19.6", 70: "E19.7", 69: "E20.7", 71: "E20.8"}


def _rid_label(rid):
    """'R3 E3.2' for engine ID 4, 'R2' for ID 8, 'ID n' for an ID in no reaction (none now)."""
    if rid not in _RID_REACTION:
        return f"ID {rid}"
    lab = f"R{_RID_REACTION[rid]}"
    return f"{lab} {_RID_CASE[rid]}" if rid in _RID_CASE else lab


def _reaction_groups():
    """[(label, name, engine IDs)]: the numbered reactions, then every engine ID not in
    one yet, under its own ID."""
    done = {r for _, _, ids in _REACTIONS for r in ids}
    rest = [(f"ID {r}", _RID_DESCRIPTION.get(r, ""), (r,))
            for r in range(1, _NUM_RID) if r not in done]
    return list(_REACTIONS) + rest


def _reaction_count_columns():
    """Per-reaction count columns for the trajectory logs, next to n_rid_XX."""
    return [f"n_{lab}" for lab, _, _ in _REACTIONS]


def _reaction_counts(counts):
    return [int(sum(int(counts[r]) for r in ids)) for _, _, ids in _REACTIONS]


_CODE_TO_NAME = {
    V_AS: "V_As",   V_GA: "V_Ga",   S_AS: "S_As",   S_GA: "S_Ga",
    DD_GA: "DD_Ga", D_GA: "D_Ga",   GA_: "Ga",
    MMGA: "MMGa",   DD_MMGA: "DD_MMGa", D_MMGA: "D_MMGa",
    DD_AS: "DD_As", D_AS: "D_As",   AS_: "As",
    DD_ASH: "DD_AsH", D_ASH: "D_AsH", ASH: "AsH", DDD_AS: "DDD_As",
    DDD_GA: "DDD_Ga",
}


# =============================================================================
# Default Arrhenius rate catalog  (reaction IDs 1..33)
# Row: (prefactor A, activation energy Ea/k_B [K], temperature exponent beta)
# =============================================================================
_NUM_RID = 78                  # IDs 1..77 (0 unused); 34, 35 are the E4.2 / E6.2 splits; 36-77 added 27 Sep
_BASE_A   = np.zeros(_NUM_RID, dtype=np.float64)
_BASE_EA   = np.zeros(_NUM_RID, dtype=np.float64)
_BASE_BETA = np.zeros(_NUM_RID, dtype=np.float64)

for _rid, (_A, _Ea, _beta) in {
    1:  (1e14,    20128.6,  0.0),   # DD_AsH desorption
    2:  (1e14,    20128.6,  0.0),   # DDD_As desorption
    3:  (1.3e12,   5032.2,  0.5),   # H abstraction (E3: DD_AsH, and D_AsH / AsH via 4, 5)
    # rids 4, 5 (H abstraction from D_AsH, AsH) take rid 3's values: see _TIED_RIDS.
    # rids 6-11 (adsorption) — rates computed from impingement formula, not Arrhenius.
    # Tunable via: S0_A_terrace, S0_A_step, Ea_chem_As (rids 6-8, 34) and S0_M_terrace,
    # S0_M_step, Ea_chem_M (rids 9-11, 35).
    12: (2.6e13,  19072.0,  0.0),   # CH3 desorption
    13: (1e14,    22644.7,  0.0),   # MMGa desorption
    # Incorporation, one parameter set per reaction (2026-09-27, see _TIED_RIDS). Until then
    # the D_AsH partner of an MMGa had its own barrier: ID 15 12278 K, ID 18 10436 K.
    14: (4.5e17,  10064.0,  0.0),   # R12 DD_MMGa terrace inc (15, 16 tied)
    17: (4.5e17,   8554.0,  0.0),   # R13 DD_MMGa step inc (18, 19 tied)
    20: (4.5e17,  10436.0,  0.0),   # R15 DD_AsH step inc (21, 22 tied)
    23: (4.5e17,   8554.0,  0.0),   # R17 DDD_As step inc (24, 25 tied)
    26: (4.5e17,  12278.0,  0.0),   # R14 DD_AsH terrace inc (27, 28 tied)
    29: (4.5e17,  10064.0,  0.0),   # R16 DDD_As terrace inc (30, 31 tied)
    32: (1e14,    29000.0,  0.0),   # DD_Ga desorption
    33: (1e14,    36000.0,  0.0),   # D_Ga desorption
    # Starting values, not literature: each mirrors its As-side or existing counterpart. 36 (methyl off DD_MMGa) is tied to 12.
    37: (1e14,    22644.7,  0.0),   # R18 DDD_Ga desorption = MMGa desorption (as R5 = R4)
    38: (4.5e17,  10064.0,  0.0),   # R19 DDD_Ga terrace inc = MMGa terrace (39, 40 tied)
    41: (4.5e17,   8554.0,  0.0),   # R20 DDD_Ga step inc = 0.85 x terrace (42, 43 tied)
}.items():
    _BASE_A[_rid]    = _A
    _BASE_EA[_rid]   = _Ea
    _BASE_BETA[_rid] = _beta

# R3 (H abstraction) is engine IDs 3, 4 and 5: the same As-H bond breaks, so all three share
# ONE parameter set, A_E3 / Ea_E3 / beta_E3 (decision 2026-09-26). IDs 4 and 5 stay separate
# events only because their source species differ (D_AsH, AsH).
# Incorporation (decisions 2026-09-27): the IDs of one reaction differ only in a spectator --
# the partner's species (the H of a D_AsH partner, the methyl of a D_MMGa partner) -- so they
# share one parameter set: R12 = 14-16 (A_E14), R13 = 17-19 (A_E17), R14 = 26-28 (A_E26),
# R15 = 20-22 (A_E20), R16 = 29-31 (A_E29), R17 = 23-25 (A_E23). DD_AsH (R14, R15) and DDD_As
# (R16, R17) are different precursors and keep separate rates. The IDs stay separate so the
# event log still shows the partner.
_TIED_RIDS = {4: 3, 5: 3,                       # engine ID -> ID whose parameters it uses
              15: 14, 16: 14, 18: 17, 19: 17,
              21: 20, 22: 20, 24: 23, 25: 23,
              27: 26, 28: 26, 30: 29, 31: 29,
              # 36 is R8 on a DD_MMGa (the same Ga-CH3 bond breaks); 39-59 differ only in
              # the partner species (44-59: an adsorbed precursor, at the reaction's rate),
              # as above; 60-65 are the other film atoms with two / three bonds leaving at
              # R10 / R11's rate
              36: 12, 39: 38, 40: 38, 42: 41, 43: 41,
              44: 14, 45: 14, 52: 17, 53: 17, 46: 26, 47: 26, 54: 20, 55: 20,
              48: 29, 49: 29, 56: 23, 57: 23, 50: 38, 51: 38, 58: 41, 59: 41,
              60: 32, 61: 33, 62: 32, 63: 33, 64: 32, 65: 33,
              # 66-77: a film atom's free bond bonds down (no automatic bonding, 27 Sep)
              66: 14, 67: 17, 68: 38, 69: 41, 70: 38, 71: 41,
              72: 26, 73: 20, 74: 29, 75: 23, 76: 29, 77: 23}
for _rid, _src in _TIED_RIDS.items():
    _BASE_A[_rid] = _BASE_A[_src]; _BASE_EA[_rid] = _BASE_EA[_src]
    _BASE_BETA[_rid] = _BASE_BETA[_src]

# IDs that own an (A, Ea, beta) triple: 6-11, 34 and 35 are impingement-driven, the others
# in _TIED_RIDS take their source's.
_ARRHENIUS_RIDS = [1, 2, 3, 12, 13, 14, 17, 20, 23, 26, 29, 32, 33,
                   37, 38, 41]                          # 37-41: R18, R19, R20

# Ordering constraints (decision 2026-09-27). Each (a, op, b): the value of a must be below
# that of b -- strictly for barriers, or equal allowed for sticking coefficients. The engine
# refuses a parameter set that breaks one; the DRC builder and the MCTS clip their moves.
#   R10 < R11   a Ga with two bonds leaves more easily than one with three (IDs 32, 33)
#   step < terrace barrier: MMGa R13 < R12, AsH R15 < R14, As R17 < R16
#   S0 terrace <= S0 step, AsH and MMGa (a precursor never sticks better on a terrace)
_CONSTRAINTS = [("Ea_E32", "<", "Ea_E33"), ("Ea_E17", "<", "Ea_E14"),
                ("Ea_E20", "<", "Ea_E26"), ("Ea_E23", "<", "Ea_E29"),
                ("S0_A_terrace", "<=", "S0_A_step"), ("S0_M_terrace", "<=", "S0_M_step")]
#   R18 < R10   a Ga with one bond (DDD_Ga) leaves more easily than one with two
#   R20 < R19   step < terrace barrier for DDD_Ga, as for the other three precursors
# (Until 27 Sep also R21-R24 slower than the standard incorporations; now they are cases of
# them, so there is nothing to order.)
_CONSTRAINTS += [("Ea_E37", "<", "Ea_E32"), ("Ea_E41", "<", "Ea_E38")]


def param_value(params, key):
    """The value the engine uses for `key`: the override in params, else the catalog."""
    p = params or {}
    if key in p:
        return float(p[key])
    if key in _STICK_DEFAULTS:
        return float(_STICK_DEFAULTS[key])
    m = re.match(r"(A|Ea|beta)_E(\d+)$", key)
    if m:
        r = _TIED_RIDS.get(int(m.group(2)), int(m.group(2)))
        return float({"A": _BASE_A, "Ea": _BASE_EA, "beta": _BASE_BETA}[m.group(1)][r])
    raise KeyError(key)


def constraint_violations(params):
    """[message] for every constraint that `params` (catalog values filled in) breaks."""
    out = []
    for a, op, b in _CONSTRAINTS:
        va, vb = param_value(params, a), param_value(params, b)
        if not (va < vb if op == "<" else va <= vb):
            out.append(f"{a} = {va:g} must be {op} {b} = {vb:g}")
    return out


def check_constraints(params):
    bad = constraint_violations(params)
    if bad:
        raise ValueError("parameter ordering constraint(s) broken: " + "; ".join(bad))


# Dependent parameters (copy). The DDD_Ga desorption barrier only says that a one-bond Ga
# leaves more easily than a two-bond one; when the search lowers R10's barrier past it,
# enforce_constraints PUSHES it along instead of blocking the move. A parameter FILE that
# breaks a constraint is still refused; only the search tools push. (The precursor-precursor
# sets were dependents too until they became cases of the incorporation reactions, 27 Sep.)
_DEPENDENT = frozenset(["Ea_E37"])


def _clip_levers(new, val, edges, margin_K):
    """Exact clip of the levers `new` (dict, changed in place) into the constraints `edges`
    (a, b, gap) over values `val` (every other node held). Returns [clipped keys]."""
    nodes = list(val)
    succ = {n: [] for n in nodes}; pred = {n: [] for n in nodes}
    for a, b, g in edges:
        succ[a].append((b, g)); pred[b].append((a, g))
    order, indeg = [], {n: len(pred[n]) for n in nodes}      # topological order (Kahn)
    ready = [n for n in nodes if indeg[n] == 0]
    while ready:
        n = ready.pop(); order.append(n)
        for b, _ in succ[n]:
            indeg[b] -= 1
            if indeg[b] == 0:
                ready.append(b)
    if len(order) != len(nodes):
        raise ValueError("_CONSTRAINTS contain a cycle")
    lev = lambda n: n in new
    lo = {n: (-math.inf if lev(n) else val[n]) for n in nodes}
    for n in order:
        for a, g in pred[n]:
            if lev(n):
                lo[n] = max(lo[n], lo[a] + g)
    hi = {n: (math.inf if lev(n) else val[n]) for n in nodes}
    for n in reversed(order):
        for b, g in succ[n]:
            if lev(n):
                hi[n] = min(hi[n], hi[b] - g)
    for a, b, g in edges:
        if not lev(a) and not lev(b) and not val[a] + g <= val[b]:
            raise ValueError(f"the fixed parameters break {a} < {b}")
    for n in nodes:
        if lev(n):
            if lo[n] > hi[n]:
                raise ValueError(f"the fixed parameters leave no room for {n}: "
                                 f"it must lie in [{lo[n]:g}, {hi[n]:g}]")
            val[n] = min(max(val[n], lo[n]), hi[n])
    for n in reversed(order):                         # from the top of each chain down
        if lev(n):
            for b, g in succ[n]:
                val[n] = min(val[n], val[b] - g)
    clipped = [n for n in nodes if lev(n) and val[n] != float(new[n])]
    for n in clipped:
        new[n] = val[n]
    return clipped


def enforce_constraints(levers, fixed=None, margin_K=1.0):
    """Clip searched parameters (`levers`) back inside _CONSTRAINTS, holding `fixed` ones
    (catalog values for the rest). The one that must be smaller moves to its partner: a
    barrier pair ends `margin_K` kelvin apart, a sticking pair may end equal. Returns
    (new levers, [clipped keys]). Used by the MCTS after every move.
    Copy: (1) constraints chain (Ea_E17 < Ea_E14 < Ea_E44), so pair-by-pair clipping can
    oscillate. Exact instead: every lever gets its feasible interval from the non-levers
    (longest-path bounds along the chains), is clipped into it, and one sweep from the top
    of each chain down lowers any lever still above its partner. On a single pair this is
    the main engine's rule. (2) _DEPENDENT parameters that are not levers do not bound the
    levers; afterwards each is pushed to the nearest value that keeps it inside, and is
    returned in `new` (and listed in clipped) so the caller runs with it."""
    new = dict(levers)
    full = {**(fixed or {}), **new}
    dep = {k for k in _DEPENDENT if k not in new}
    val, edges, dep_edges = {}, [], []                # edge (a, b, gap): a + gap <= b
    for a, op, b in _CONSTRAINTS:
        e = (a, b, margin_K if op == "<" else 0.0)
        (dep_edges if (a in dep or b in dep) else edges).append(e)
        for k in (a, b):
            if k not in dep:
                val.setdefault(k, param_value(full, k))
    clipped = _clip_levers(new, val, edges, margin_K)
    for d in sorted(dep):                             # push the dependents (chain ends)
        v0 = v = param_value(full, d)
        for a, b, g in dep_edges:
            if b == d:
                v = max(v, val.get(a, param_value(full, a)) + g)
            elif a == d:
                v = min(v, val.get(b, param_value(full, b)) - g)
        if v != v0:
            new[d] = v; clipped.append(d)
    check_constraints({**(fixed or {}), **new})
    return new, clipped

# E4.2 (AsH on a D_MMGa) and E6.2 (MMGa on a D_AsH). The host has ONE dangling bond, but
# its methyl / H is small enough to move out of the way, so the adsorbate can land on
# either upper site. With both free, each landing is a split-ID event at half the site
# rate, so the host adsorbs at the rate of one site; with one free, that landing is an
# ordinary ID-6 / ID-9 event at the full rate (2026-09-26). Separate IDs because every
# event of one ID must have the same rate.
_RID_E42_SPLIT = 34
_RID_E62_SPLIT = 35
_E42_SHARE = 0.5                    # one bond shared by two landing sites (E4.2 and E6.2)

# Impingement-driven IDs. Their events are owned by the host b (the menu re-enumerates an
# event when its owner is dirty); every other event is owned by its source a.
_ASH_ADS_RIDS = (6, 7, 8, _RID_E42_SPLIT)
_MMGA_ADS_RIDS = (9, 10, 11, _RID_E62_SPLIT)
_IS_ADS_RID = np.zeros(_NUM_RID, dtype=bool)
_IS_ADS_RID[list(_ASH_ADS_RIDS + _MMGA_ADS_RIDS)] = True

# Adsorption (decision 2026-09-27): terrace and step have their own sticking coefficients,
# S0_A_terrace (R1: IDs 6, 7, 34), S0_A_step (R2: ID 8), S0_M_terrace (R6: IDs 9, 10, 35)
# and S0_M_step (R7: ID 11). They replace S0_A / S0_M x f_terr_A / f_terr_M (same number of
# parameters; S0_terrace = f_terr x S0 and S0_step = S0, so the catalog rates are unchanged).
# S0_terrace <= S0_step is enforced (_CONSTRAINTS).
_TERRACE_ASH_RIDS = (6, 7, _RID_E42_SPLIT)
_TERRACE_MMGA_RIDS = (9, 10, _RID_E62_SPLIT)
_STICK_DEFAULTS = {"S0_A_terrace": 0.04, "S0_A_step": 0.08, "Ea_chem_As": 5500.0,
                   "S0_M_terrace": 0.04, "S0_M_step": 0.08, "Ea_chem_M": 5500.0}
# Arrival flux (Hertz-Knudsen) of the species that land: AsH and MMGa (GaCH3); the arsine
# and TMGa inlet pressures stand for them. Masses 75.93 u and 84.76 u (2026-09-27; until then
# AsH3's 77.95 u and 1.39e-25 kg for MMGa).
_P_ASH, _P_MMGA = 608.0, 65.86          # Pa
_M_ASH, _M_MMGA = 1.2608e-25, 1.4074e-25  # kg
_A_SITE = 1.6e-19                       # m^2
_K_B = 1.38e-23                         # J/K
_OLD_STICK_KEYS = {"S0_A": "S0_A_step (= S0_A) and S0_A_terrace (= f_terr_A x S0_A)",
                   "f_terr_A": "S0_A_terrace (= f_terr_A x S0_A)",
                   "S0_M": "S0_M_step (= S0_M) and S0_M_terrace (= f_terr_M x S0_M)",
                   "f_terr_M": "S0_M_terrace (= f_terr_M x S0_M)"}


def _sticking(params):
    """(S0_A_terrace, S0_A_step, Ea_chem_As, S0_M_terrace, S0_M_step, Ea_chem_M), checked."""
    p = params or {}
    old = [k for k in _OLD_STICK_KEYS if k in p]
    if old:
        raise ValueError("parameter(s) %s no longer exist: use %s." % (
            ", ".join(old), "; ".join(_OLD_STICK_KEYS[k] for k in old)))
    v = {k: float(p.get(k, d)) for k, d in _STICK_DEFAULTS.items()}
    for k in ("S0_A_terrace", "S0_A_step", "S0_M_terrace", "S0_M_step"):
        if not 0.0 < v[k] <= 1.0:
            raise ValueError(f"{k} = {v[k]} is outside (0, 1].")
    check_constraints(params)
    return (v["S0_A_terrace"], v["S0_A_step"], v["Ea_chem_As"],
            v["S0_M_terrace"], v["S0_M_step"], v["Ea_chem_M"])


# =============================================================================
# XYZ I/O
# =============================================================================
def extract_coords_with_species_labels(filepath):
    atoms = []
    species_count = {}
    with open(filepath, "r") as fh:
        lines = fh.readlines()
    header = lines[1]
    origin = (np.array(list(map(float, header.split('Origin="')[1].split('"')[0].split())))
              if 'Origin="' in header else np.zeros(3))
    unit_cell = np.array(
        list(map(float, header.split('Lattice="')[1].split('"')[0].split()))
    ).reshape(3, 3)
    for line in lines[2:]:
        parts = line.split()
        if not parts:
            continue
        element = "".join(filter(str.isalpha, parts[0]))
        species_count[element] = species_count.get(element, 0) + 1
        atoms.append((np.array(list(map(float, parts[1:4]))),
                      f"{element}_{species_count[element]}"))
    return atoms, origin, unit_cell


def label_lattice_sites(atoms, bottom_layers=3):
    z_levels = sorted({round(c[2], 6) for c, _ in atoms})
    labeled = []
    top_idx = bottom_layers - 1
    for coord, _ in atoms:
        layer = z_levels.index(round(coord[2], 6))
        if layer < bottom_layers:
            dist = top_idx - layer
            site = "S_As" if dist % 2 == 0 else "S_Ga"
        elif layer == bottom_layers:
            site = "DD_Ga"
        else:
            n = layer - (bottom_layers + 1)
            site = "V_As" if n % 2 == 0 else "V_Ga"
        labeled.append((coord, site))
    return labeled


def _append_xyz_frame(f, coords, species, origin, unit_cell, frame_info=""):
    N = coords.shape[0]
    f.write(f"{N}\n")
    lat = " ".join(map(str, unit_cell.flatten()))
    ori = " ".join(map(str, origin))
    f.write(f'Lattice="{lat}" Origin="{ori}" '
            f"Properties=species:S:1:pos:R:3:id:I:1 {frame_info}\n")
    names = [_CODE_TO_NAME.get(int(s), "V") for s in species]
    for i in range(N):
        x, y, z = coords[i]
        f.write(f"{names[i]} {x:.8f} {y:.8f} {z:.8f} {i+1}\n")


# =============================================================================
# LatticeState
# =============================================================================
class LatticeState:
    MAX_BONDS = 4

    def __init__(self, atoms_init, origin, unit_cell):
        from scipy.spatial import cKDTree

        self.N         = len(atoms_init)
        self.origin    = origin
        self.unit_cell = unit_cell
        self.Lx        = float(np.linalg.norm(unit_cell[0]))
        self.Ly        = float(np.linalg.norm(unit_cell[1]))

        self.coords  = np.ascontiguousarray(
            np.array([c for c, _ in atoms_init], dtype=np.float64))
        self.species = np.empty(self.N, dtype=np.int16)
        _label_map = {
            "V_As": V_AS, "V_Ga": V_GA, "S_As": S_AS, "S_Ga": S_GA,
            "DD_Ga": DD_GA, "D_Ga": D_GA, "Ga": GA_,
            "MMGa": MMGA, "DD_MMGa": DD_MMGA, "D_MMGa": D_MMGA,
            "DD_As": DD_AS, "D_As": D_AS, "As": AS_,
            "DD_AsH": DD_ASH, "D_AsH": D_ASH, "AsH": ASH, "DDD_As": DDD_AS,
            "DDD_Ga": DDD_GA,
        }
        for i, (_, lab) in enumerate(atoms_init):
            self.species[i] = _label_map.get(lab, V_AS)

        self.H         = np.zeros(self.N, dtype=np.int8)
        self.DB        = np.zeros(self.N, dtype=np.int8)
        self.has_entry = np.zeros(self.N, dtype=bool)
        self.bonds     = np.full((self.N, self.MAX_BONDS), -1, dtype=np.int32)
        self.n_bonds   = np.zeros(self.N, dtype=np.int8)

        self._kdtree = cKDTree(self.coords)
        self._idx_map = {
            (round(float(self.coords[i, 0]), 3),
             round(float(self.coords[i, 1]), 3),
             round(float(self.coords[i, 2]), 3)): i
            for i in range(self.N)
        }

        self.vectors_As = np.array([
            [ 1.99888,  0.0,     -1.41342],
            [-1.99888,  0.0,     -1.41342],
            [ 0.0,     -1.99889,  1.41343],
            [ 0.0,      1.99889,  1.41343],
        ], dtype=np.float64)
        self.vectors_Ga = np.array([
            [ 0.0,     -1.99888, -1.41343],
            [ 0.0,      1.99896, -1.41343],
            [ 1.99887,  0.0,      1.41342],
            [-1.99890,  0.0,      1.41342],
        ], dtype=np.float64)

        self._compute_edge_masks()
        self._precompute_neighbor_tables()

    def idx_of(self, coord):
        k = (round(float(coord[0]), 3), round(float(coord[1]), 3), round(float(coord[2]), 3))
        j = self._idx_map.get(k, -1)
        if j >= 0:
            return j
        d, idx = self._kdtree.query(np.asarray(coord, dtype=np.float64),
                                     k=1, distance_upper_bound=0.05)
        return int(idx) if np.isfinite(d) and idx < self.N else -1

    def _compute_edge_masks(self):
        s   = self.species
        xs  = self.coords[:, 0]
        ys  = self.coords[:, 1]
        tol = 1e-6

        is_As = np.isin(s, [V_AS, S_AS, DD_AS, D_AS, AS_, DD_ASH, D_ASH, ASH, DDD_AS])
        is_Ga = np.isin(s, [V_GA, S_GA, DD_GA, D_GA, GA_, MMGA, DD_MMGA, D_MMGA, DDD_GA])

        def _edge_masks(mask, corner_pair):
            if not mask.any():
                xmx = xmn = ymx = ymn = 0.0
            else:
                xmx = float(xs[mask].max()); xmn = float(xs[mask].min())
                ymx = float(ys[mask].max()); ymn = float(ys[mask].min())
            a_mx = mask & (np.abs(xs - xmx) < tol)
            a_mn = mask & (np.abs(xs - xmn) < tol)
            b_mx = mask & (np.abs(ys - ymx) < tol)
            b_mn = mask & (np.abs(ys - ymn) < tol)
            c1 = a_mn & b_mx if corner_pair == "min_x_max_y" else a_mx & b_mx
            c2 = a_mx & b_mn if corner_pair == "min_x_max_y" else a_mn & b_mn
            corners = c1 | c2
            return (a_mx & ~corners, a_mn & ~corners,
                    b_mx & ~corners, b_mn & ~corners, c1, c2)

        (self.is_max_x_As, self.is_min_x_As,
         self.is_max_y_As, self.is_min_y_As,
         self.is_min_x_max_y_As, self.is_max_x_min_y_As) = _edge_masks(is_As, "min_x_max_y")

        (self.is_max_x_Ga, self.is_min_x_Ga,
         self.is_max_y_Ga, self.is_min_y_Ga,
         self.is_max_x_max_y_Ga, self.is_min_x_min_y_Ga) = _edge_masks(is_Ga, "max_xy")

    def _precompute_neighbor_tables(self):
        N = self.N; Lx = self.Lx; Ly = self.Ly
        tree = self._kdtree; tol = 0.05

        self.nbr_As = np.full((N, 4), -1, dtype=np.int32)
        self.nbr_Ga = np.full((N, 4), -1, dtype=np.int32)

        for k in range(4):
            tgt = self.coords + self.vectors_As[k][None, :]
            if k == 2: tgt[self.is_min_y_As, 1] += Ly
            elif k == 3: tgt[self.is_max_y_As, 1] -= Ly
            elif k == 1: tgt[self.is_min_x_As, 0] += Lx
            elif k == 0: tgt[self.is_max_x_As, 0] -= Lx
            if k == 1: tgt[self.is_min_x_max_y_As, 0] += Lx
            elif k == 3: tgt[self.is_min_x_max_y_As, 1] -= Ly
            if k == 0: tgt[self.is_max_x_min_y_As, 0] -= Lx
            elif k == 2: tgt[self.is_max_x_min_y_As, 1] += Ly
            ds, idxs = tree.query(tgt, k=1, distance_upper_bound=tol)
            ok = np.isfinite(ds) & (idxs < N)
            self.nbr_As[ok, k] = idxs[ok].astype(np.int32)

        for k in range(4):
            tgt = self.coords + self.vectors_Ga[k][None, :]
            if k == 0: tgt[self.is_min_y_Ga, 1] += Ly
            elif k == 1: tgt[self.is_max_y_Ga, 1] -= Ly
            elif k == 3: tgt[self.is_min_x_Ga, 0] += Lx
            elif k == 2: tgt[self.is_max_x_Ga, 0] -= Lx
            if k == 2: tgt[self.is_max_x_max_y_Ga, 0] -= Lx
            elif k == 1: tgt[self.is_max_x_max_y_Ga, 1] -= Ly
            if k == 3: tgt[self.is_min_x_min_y_Ga, 0] += Lx
            elif k == 0: tgt[self.is_min_x_min_y_Ga, 1] += Ly
            ds, idxs = tree.query(tgt, k=1, distance_upper_bound=tol)
            ok = np.isfinite(ds) & (idxs < N)
            self.nbr_Ga[ok, k] = idxs[ok].astype(np.int32)

    # ---- bond helpers ----
    def _add_bond(self, i, j):
        if i < 0 or j < 0: return
        row = self.bonds[i]; n = int(self.n_bonds[i])
        for k in range(n):
            if row[k] == j: return
        if n < self.MAX_BONDS:
            row[n] = j; self.n_bonds[i] = n + 1

    def _remove_bond(self, i, j):
        if i < 0 or j < 0: return
        row = self.bonds[i]; n = int(self.n_bonds[i])
        for k in range(n):
            if row[k] == j:
                row[k] = row[n - 1]; row[n - 1] = -1
                self.n_bonds[i] = n - 1; return

    def is_bonded(self, i, j):
        row = self.bonds[i]
        for k in range(int(self.n_bonds[i])):
            if row[k] == j:
                return True
        return False

    def _clear_entry(self, i):
        self.bonds[i, :] = -1; self.n_bonds[i] = 0
        self.has_entry[i] = False; self.DB[i] = 0; self.H[i] = 0

    def _set_entry(self, i, DB, H, bonded_to=()):
        self.DB[i] = DB; self.H[i] = H; self.bonds[i, :] = -1
        n = 0
        for j in bonded_to:
            if j >= 0 and n < self.MAX_BONDS:
                self.bonds[i, n] = j; n += 1
        self.n_bonds[i] = n; self.has_entry[i] = True


# =============================================================================
# Initialization
# =============================================================================
def _initialize_bonding(st):
    for i in range(st.N):
        if st.species[i] != DD_GA: continue
        pos = st.coords[i]
        t1 = pos + st.vectors_Ga[0]
        if st.is_min_y_Ga[i] or st.is_min_x_min_y_Ga[i]:
            t1 = t1 + np.array([0.0, st.Ly, 0.0])
        t2 = pos + st.vectors_Ga[1]
        j1, j2 = st.idx_of(t1), st.idx_of(t2)
        if j1 < 0 or j2 < 0:
            raise ValueError(f"DD_Ga at {pos} missing substrate bond.")
        st._set_entry(i, DB=2, H=0, bonded_to=(j1, j2))


def _purge(st):
    orig = st.species.copy()
    for i in range(st.N):
        if orig[i] != DD_GA: continue
        pos = st.coords[i]
        for k_loc, vec in enumerate((st.vectors_Ga[2], st.vectors_Ga[3])):
            tgt = pos + vec
            if k_loc == 1 and (st.is_min_x_Ga[i] or st.is_min_x_min_y_Ga[i]):
                tgt = tgt + np.array([st.Lx, 0.0, 0.0])
            j = st.idx_of(tgt)
            if j < 0 or orig[j] != V_AS: continue
            st.species[j] = DD_ASH
            st._set_entry(j, DB=2, H=1, bonded_to=(i,))
            st.species[i] = D_GA; st.DB[i] = 1
            st._add_bond(i, j); st.has_entry[i] = True
            break


# =============================================================================
# Event enumeration (vectorized)
# =============================================================================
def _upper_step(st, i, up, src, ads):
    """Terrace vs step incorporation (sections 3/4, 7/8). True if atom i is BONDED to one of
    its upper neighbours `up` other than the incorporating precursor `src`, and that
    neighbour is not an adsorbate (`ads`): its other upper bond holds an incorporated atom.
    Read from the bonds themselves, not from i's label, so an atom that has lost a LOWER
    bond (its Ga / As below desorbed) is still judged by what sits in its upper slot."""
    row = st.bonds[i, :int(st.n_bonds[i])].tolist()
    for u in up:
        u = int(u)
        if u >= 0 and u != src and u in row and st.species[u] not in ads:
            return True
    return False


# Adsorbed precursors of each element (one bond, waiting to
# incorporate): they count as adsorbates in the terrace / step test, and they are the lower
# partners of precursor-precursor incorporation.
_GA_ADS = (DD_MMGA, DDD_GA)
_AS_ADS = (DD_ASH, DDD_AS)
# Incorporation onto an adsorbed precursor: terrace engine ID by (upper precursor, lower
# precursor it bonds down to); the step ID is + _PP_STEP. Cases of R12-R17, R19, R20.
_PP_STEP = 8
_PP_RID = {(DD_MMGA, DD_ASH): 44, (DD_MMGA, DDD_AS): 45,
           (DD_ASH, DD_MMGA): 46, (DD_ASH, DDD_GA): 47,
           (DDD_AS, DD_MMGA): 48, (DDD_AS, DDD_GA): 49,
           (DDD_GA, DD_ASH): 50, (DDD_GA, DDD_AS): 51}
# Film atoms with two / three bonds that desorb at R10 / R11's rate, besides DD_Ga (32) and
# D_Ga (33): engine ID by species (copy, 27 Sep)
_DES_TWO = {DD_AS: 60, D_MMGA: 62, D_ASH: 64}
_DES_THREE = {D_AS: 61, MMGA: 63, ASH: 65}
# A film atom (two or three bonds) with a free bond that bonds down to the atom below it with a
# free bond: (terrace ID, step ID) by the upper atom's kind (27 Sep). MMGa-type -> R12 / R13,
# Ga -> R19 / R20, AsH-type -> R14 / R15, As -> R16 / R17.
_FILM_BOND = {D_MMGA: (66, 67), DD_GA: (68, 69), D_GA: (70, 71),
              D_ASH: (72, 73), DD_AS: (74, 75), D_AS: (76, 77)}
_PP_TABLE = np.zeros((_N_SPECIES, _N_SPECIES), dtype=np.int16)
for (_u, _l), _r in _PP_RID.items():
    _PP_TABLE[_u, _l] = _r


def _host_ok(st, h, ga_precursor):
    """Vectorised: is h (bonds[precursor, 0]) a host the precursor may incorporate from? The
    rule of sections 3/4 and 7/8: an As host of a Ga precursor is D_As, AsH, or As holding
    at least two bonds; a Ga host of an As precursor is D_Ga, MMGa, or Ga likewise."""
    s = st.species
    valid = h >= 0; hs = np.maximum(h, 0); sp = s[hs]
    if ga_precursor:
        full = (sp == D_AS) | (sp == ASH); bulk = sp == AS_
    else:
        full = (sp == D_GA) | (sp == MMGA); bulk = sp == GA_
    return valid & (full | (bulk & st.has_entry[hs] & (st.n_bonds[hs] >= 2)))


def _enumerate_events(st, react_idx):
    s = st.species; N = st.N
    nbr_As = st.nbr_As; nbr_Ga = st.nbr_Ga
    ridx = react_idx                    # global indices of reactive (event-source) sites
    sr = s[ridx]                        # their species (small: only the reactive band)
    he = st.has_entry[ridx]             # subset views reused by the desorption sections
    nb = st.n_bonds[ridx]

    # Source masks are now small (over the reactive band). np.where returns ascending
    # indices and ridx is ascending, so ridx[is_X] reproduces the original
    # np.where((s==X))[0] global-index order exactly -> bit-identical event ordering.
    is_D_AsH   = (sr == D_ASH)
    is_DD_As   = (sr == DD_AS)
    is_D_As    = (sr == D_AS)
    is_DD_Ga   = (sr == DD_GA)
    is_D_MMGa  = (sr == D_MMGA)
    is_D_Ga    = (sr == D_GA)
    is_DD_MMGa = (sr == DD_MMGA)
    is_DD_AsH  = (sr == DD_ASH)
    is_DDD_As  = (sr == DDD_AS)
    is_MMGa    = (sr == MMGA)
    is_AsH     = (sr == ASH)
    is_DDD_Ga  = (sr == DDD_GA)

    rid_list, a_list, b_list = [], [], []

    def _emit(rid, a, b):
        if a.size == 0: return
        rid_list.append(np.full(a.size, rid, dtype=np.int16))
        a_list.append(a.astype(np.int32, copy=False))
        b_list.append(b.astype(np.int32, copy=False))

    def _sp(nbr):
        safe = np.maximum(nbr, 0)          # clamp -1 sentinels to 0 for safe gather
        return s[safe], nbr >= 0

    # 1 — MMGa step adsorption (rid 11). Copy: an adsorbed DDD_Ga in the other upper slot
    # makes it terrace (rid 10), as an adsorbed DD_MMGa does -- as DDD_As does for rid 7.
    if is_D_As.any():
        src = ridx[is_D_As]
        v3 = nbr_As[src, 2]; v4 = nbr_As[src, 3]
        sv3, ok3 = _sp(v3);  sv4, ok4 = _sp(v4)
        g3 = ok3 & ((sv3 == DD_MMGA) | (sv3 == DDD_GA))
        g4 = ok4 & ((sv4 == DD_MMGA) | (sv4 == DDD_GA))
        _emit(11, v3[ok3 & (sv3 == V_GA) & ~g4], src[ok3 & (sv3 == V_GA) & ~g4])
        _emit(11, v4[ok4 & (sv4 == V_GA) & ~g3], src[ok4 & (sv4 == V_GA) & ~g3])

    # 2 — MMGa terrace adsorption, R6 (rids 9, 10, 35; sticking S0_M_terrace)
    mask_src = is_D_AsH | is_DD_As
    if mask_src.any():
        src = ridx[mask_src]
        v3 = nbr_As[src, 2]; v4 = nbr_As[src, 3]
        sv3, ok3 = _sp(v3);  sv4, ok4 = _sp(v4)
        f3 = ok3 & (sv3 == V_GA); f4 = ok4 & (sv4 == V_GA)
        both = (s[src] == D_ASH) & f3 & f4           # E6.2, one dangling bond, two sites
        _emit(9, v3[f3 & ~both], src[f3 & ~both])
        _emit(9, v4[f4 & ~both], src[f4 & ~both])
        _emit(_RID_E62_SPLIT, v3[both], src[both])
        _emit(_RID_E62_SPLIT, v4[both], src[both])
    if is_D_As.any():
        src = ridx[is_D_As]
        v3 = nbr_As[src, 2]; v4 = nbr_As[src, 3]
        sv3, ok3 = _sp(v3);  sv4, ok4 = _sp(v4)
        ca = ok3 & ok4 & (sv3 == V_GA) & ((sv4 == DD_MMGA) | (sv4 == DDD_GA))
        cb = ok3 & ok4 & (sv4 == V_GA) & ((sv3 == DD_MMGA) | (sv3 == DDD_GA))
        _emit(10, v3[ca], src[ca]); _emit(10, v4[cb], src[cb])

    # 3/4 — DD_MMGa incorporation: terrace (rids 14/15/16) or step (rids 17/18/19).
    # "Either side" rule (2026-09-27). The DD_MMGa is bonded to its host As and forms a
    # bond to a partner As; each As has one other upper slot. Terrace only if BOTH slots
    # hold an H, a dangling bond or a DD_MMGa; step if EITHER holds an incorporated Ga.
    # Read from the bonds (_upper_step), not the labels: for a normally bonded As the two
    # agree (host D_As / AsH, partner DD_As / D_AsH terrace-like; an As or D_As whose
    # other upper Ga is not a DD_MMGa step-like), and an As that has lost a lower bond is
    # still judged by its upper slot. One event per DD_MMGa, rid by partner. Until
    # 2026-09-27 only the host decided, and before that the E6.3 pair got a terrace AND a
    # step event.
    # Copy: DDD_Ga incorporates by the same rules, terrace rids 38-40 (R19), step 41-43
    # (R20); an adsorbed DDD_Ga in an upper slot counts as an adsorbate, like a DD_MMGa.
    for is_src, base_t, base_s in ((is_DD_MMGa, 14, 17), (is_DDD_Ga, 38, 41)):
        if not is_src.any():
            continue
        src = ridx[is_src]
        ba = st.bonds[src, 0]; valid = ba >= 0
        bas = np.where(valid, ba, 0); asp = s[bas]
        ok = valid & ((asp == D_AS) | (asp == ASH))
        for ii in np.where(valid & (asp == AS_))[0]:
            ai = int(ba[ii])
            ok[ii] = bool(st.has_entry[ai]) and st.n_bonds[ai] >= 2
        h_step = np.zeros_like(valid)
        for ii in np.where(ok)[0]:
            h_step[ii] = _upper_step(st, int(ba[ii]), nbr_As[ba[ii], 2:], int(src[ii]), _GA_ADS)
        so = src[ok]; bao = bas[ok]; hs = h_step[ok]
        v1 = nbr_Ga[so, 0]; v2 = nbr_Ga[so, 1]
        sv1, ok1 = _sp(v1); sv2, ok2 = _sp(v2)
        # The partner can also be an adsorbed As precursor (DD_AsH / DDD_As)
        # on a proper host of its own (so two floating adsorbates never bond): IDs 44/45
        # (terrace, R12 / R19 cases) or 52/53 (step), by the same either-side rule
        def _pm(vv, vok, vs):
            std = vok & ((vs == DD_AS) | (vs == D_ASH) | ((vs == D_AS) & (vv != bao)))
            pp = vok & ((vs == DD_ASH) | (vs == DDD_AS)) & (vv != bao)
            hl = np.where(pp, st.bonds[np.maximum(vv, 0), 0], -1)
            return std | (pp & (hl != so) & _host_ok(st, hl, False))
        m1 = _pm(v1, ok1, sv1); m2 = _pm(v2, ok2, sv2) & ~m1
        def _eb(mask, vv, vs):
            if not mask.any(): return
            s_ = so[mask]; p = vv[mask]; ps = vs[mask]
            sk = s_.tolist()
            step = hs[mask] | np.array([_upper_step(st, int(p[k]), nbr_As[p[k], 2:], sk[k], _GA_ADS)
                                        for k in range(p.size)], dtype=bool)
            off = np.where(ps == DD_AS, 0, np.where(ps == D_ASH, 1, 2))
            ra = np.where(step, base_s, base_t) + off
            ads = (ps == DD_ASH) | (ps == DDD_AS)
            ra = np.where(ads, _PP_TABLE[s[s_], ps].astype(np.int64) + np.where(step, _PP_STEP, 0), ra)
            for r in np.unique(ra):
                mm = ra == r
                _emit(int(r), s_[mm], p[mm])
        _eb(m1, v1, sv1); _eb(m2, v2, sv2)

    # 5 — AsH terrace adsorption, R1 (rids 6, 7, 34; sticking S0_A_terrace)
    m1 = is_DD_Ga | is_D_MMGa
    if m1.any():
        src = ridx[m1]
        v3 = nbr_Ga[src, 2]; v4 = nbr_Ga[src, 3]
        sv3, ok3 = _sp(v3); sv4, ok4 = _sp(v4)
        f3 = ok3 & (sv3 == V_AS); f4 = ok4 & (sv4 == V_AS)
        both = (s[src] == D_MMGA) & f3 & f4          # E4.2, one dangling bond, two sites
        _emit(6, v3[f3 & ~both], src[f3 & ~both])
        _emit(6, v4[f4 & ~both], src[f4 & ~both])
        _emit(_RID_E42_SPLIT, v3[both], src[both])
        _emit(_RID_E42_SPLIT, v4[both], src[both])
    if is_D_Ga.any():
        src = ridx[is_D_Ga]
        v3 = nbr_Ga[src, 2]; v4 = nbr_Ga[src, 3]
        sv3, ok3 = _sp(v3); sv4, ok4 = _sp(v4)
        po3 = ok4 & ((sv4 == DD_ASH) | (sv4 == DDD_AS))
        po4 = ok3 & ((sv3 == DD_ASH) | (sv3 == DDD_AS))
        _emit(7, v3[ok3 & (sv3 == V_AS) & po3], src[ok3 & (sv3 == V_AS) & po3])
        _emit(7, v4[ok4 & (sv4 == V_AS) & po4], src[ok4 & (sv4 == V_AS) & po4])

    # 6 — AsH step adsorption, R2 (rid 8)
    if is_D_Ga.any():
        src = ridx[is_D_Ga]
        v3 = nbr_Ga[src, 2]; v4 = nbr_Ga[src, 3]
        sv3, ok3 = _sp(v3); sv4, ok4 = _sp(v4)
        bl4 = ok4 & ((sv4 == DD_ASH) | (sv4 == DDD_AS))
        bl3 = ok3 & ((sv3 == DD_ASH) | (sv3 == DDD_AS))
        _emit(8, v3[ok3 & (sv3 == V_AS) & ~bl4], src[ok3 & (sv3 == V_AS) & ~bl4])
        _emit(8, v4[ok4 & (sv4 == V_AS) & ~bl3], src[ok4 & (sv4 == V_AS) & ~bl3])

    # 7/8 — DD_AsH/DDD_As incorporation: step (rids 20-25) or terrace (rids 26-31).
    # Same "either side" rule as section 3/4, Ga and As swapped (2026-09-27): terrace only
    # if the host's and the partner's other upper slots both hold a dangling bond, a methyl
    # or an adsorbed As (DD_AsH / DDD_As); step if either holds an incorporated As. Read
    # from the bonds (_upper_step). One event per DD_AsH / DDD_As. Until 2026-09-27 only
    # the host decided.
    mask_asl = is_DD_AsH | is_DDD_As
    if mask_asl.any():
        src = ridx[mask_asl]
        bg = st.bonds[src, 0]; valid = bg >= 0
        bgs = np.where(valid, bg, 0); gsp = s[bgs]
        ok = valid & ((gsp == D_GA) | (gsp == MMGA))
        for ii in np.where(valid & (gsp == GA_))[0]:
            gi = int(bg[ii])
            ok[ii] = bool(st.has_entry[gi]) and st.n_bonds[gi] >= 2
        h_step = np.zeros_like(valid)
        for ii in np.where(ok)[0]:
            h_step[ii] = _upper_step(st, int(bg[ii]), nbr_Ga[bg[ii], 2:], int(src[ii]), (DD_ASH, DDD_AS))
        so = src[ok]; bgo = bgs[ok]; slab = s[so]; hs = h_step[ok]
        v1 = st.nbr_As[so, 0]; v2 = st.nbr_As[so, 1]
        sv1, ok1 = _sp(v1); sv2, ok2 = _sp(v2)
        # The partner can also be an adsorbed Ga precursor (DD_MMGa / DDD_Ga)
        # on a proper host of its own: IDs 46-49 (terrace, R14 / R16) or 54-57 (step)
        def _pms(vv, vok, vs):
            std = vok & ((vs == DD_GA) | (vs == D_MMGA) | ((vs == D_GA) & (vv != bgo)))
            pp = vok & ((vs == DD_MMGA) | (vs == DDD_GA)) & (vv != bgo)
            hl = np.where(pp, st.bonds[np.maximum(vv, 0), 0], -1)
            return std | (pp & (hl != so) & _host_ok(st, hl, True))
        m1 = _pms(v1, ok1, sv1); m2 = _pms(v2, ok2, sv2) & ~m1
        def _ems(mask, vv, vs):
            if not mask.any(): return
            s_ = so[mask]; sl = slab[mask]; p = vv[mask]; ps = vs[mask]
            sk = s_.tolist()
            step = hs[mask] | np.array([_upper_step(st, int(p[k]), nbr_Ga[p[k], 2:], sk[k], (DD_ASH, DDD_AS))
                                        for k in range(p.size)], dtype=bool)
            off = np.where(ps == DD_GA, 0, np.where(ps == D_MMGA, 1, 2))
            base = np.where(sl == DD_ASH, np.where(step, 20, 26), np.where(step, 23, 29))
            ra = base + off
            ads = (ps == DD_MMGA) | (ps == DDD_GA)
            ra = np.where(ads, _PP_TABLE[sl, ps].astype(np.int64) + np.where(step, _PP_STEP, 0), ra)
            for r in np.unique(ra):
                mm = ra == r
                _emit(int(r), s_[mm], p[mm])
        _ems(m1, v1, sv1); _ems(m2, v2, sv2)

    # 9 — H abstraction, R3 (rids 3, 4, 5; one parameter set)
    for mask, rid in ((is_DD_AsH, 3), (is_D_AsH, 4), (is_AsH, 5)):
        if mask.any():
            idx = ridx[mask]
            _emit(rid, idx, np.full(idx.size, -1, dtype=np.int32))

    # 10 — methyl desorption (rid 12; copy: rid 36 from DD_MMGa, tied to 12)
    m12 = is_D_MMGa | is_MMGa
    if m12.any():
        idx = ridx[m12]
        _emit(12, idx, np.full(idx.size, -1, dtype=np.int32))
    if is_DD_MMGa.any():
        idx = ridx[is_DD_MMGa]
        _emit(36, idx, np.full(idx.size, -1, dtype=np.int32))

    # 11 — MMGa desorption (rid 13)
    if is_DD_MMGa.any():
        idx = ridx[is_DD_MMGa & he & (nb >= 1)]
        b = st.bonds[idx, 0]; m = b >= 0
        _emit(13, idx[m], b[m])

    # 12 — DD_AsH/DDD_As desorption (rids 1, 2)
    for mask, rid in ((is_DD_AsH & he & (nb >= 1), 1),
                      (is_DDD_As & he & (nb >= 1), 2)):
        if mask.any():
            idx = ridx[mask]; n = st.n_bonds[idx]
            lb = st.bonds[idx, n - 1]; m = lb >= 0
            _emit(rid, idx[m], lb[m])

    # 13 — DD_Ga/D_Ga desorption (rids 32, 33)
    # FIX v2: not for a Ga bonded to the substrate (a first-layer hole cannot refill)
    def _off_substrate(idx):
        bb = st.bonds[idx]; sb = s[np.maximum(bb, 0)]
        return ~((bb >= 0) & ((sb == S_AS) | (sb == S_GA))).any(axis=1)
    if is_DD_Ga.any():
        idx = ridx[is_DD_Ga & he & (nb >= 1)]
        idx = idx[_off_substrate(idx)]
        b = st.bonds[idx, 0]; m = b >= 0
        _emit(32, idx[m], b[m])
    if is_D_Ga.any():
        idx = ridx[is_D_Ga & he & (nb >= 3)]
        idx = idx[_off_substrate(idx)]
        n = st.n_bonds[idx]; b = st.bonds[idx, n - 1]; m = b >= 0
        _emit(33, idx[m], b[m])
    # Every other film atom with two / three bonds desorbs at exactly R10 /
    # R11's rate (IDs 60-65), by the rules of DD_Ga / D_Ga (not when bonded to the substrate)
    for sp_, rid_ in _DES_TWO.items():
        msk = (sr == sp_) & he & (nb >= 1)
        if msk.any():
            idx = ridx[msk]; idx = idx[_off_substrate(idx)]
            b = st.bonds[idx, 0]; m = b >= 0
            _emit(rid_, idx[m], b[m])
    for sp_, rid_ in _DES_THREE.items():
        msk = (sr == sp_) & he & (nb >= 3)
        if msk.any():
            idx = ridx[msk]; idx = idx[_off_substrate(idx)]
            n = st.n_bonds[idx]; b = st.bonds[idx, n - 1]; m = b >= 0
            _emit(rid_, idx[m], b[m])
    # copy: DDD_Ga desorption, R18 (rid 37), as DDD_As (rid 2)
    if is_DDD_Ga.any():
        idx = ridx[is_DDD_Ga & he & (nb >= 1)]
        idx = idx[_off_substrate(idx)]
        n = st.n_bonds[idx]; b = st.bonds[idx, n - 1]; m = b >= 0
        _emit(37, idx[m], b[m])

    # 14 — a film atom's free bond bonds down (IDs 66-77; 27 Sep 2026). A film atom (two or
    # three bonds) with a free bond bonds to each lower neighbour that has a free bond and is
    # not bonded to it yet -- film atom or adsorbed precursor; an H / methyl rotates out of
    # the way. A case of the incorporation reaction of the upper atom's kind (_FILM_BOND),
    # terrace or step by the either-side rule: step if one of the upper atom's bonded lower
    # neighbours, or the lower atom, holds a non-adsorbate in its other upper slot. No bond
    # forms any other way.
    for sp_, (r_t, r_s) in _FILM_BOND.items():
        msk = sr == sp_
        if not msk.any():
            continue
        src = ridx[msk]
        ga = sp_ in (D_MMGA, DD_GA, D_GA)
        lo = nbr_Ga if ga else nbr_As          # a Ga's lower neighbours are As, and back
        up = nbr_As if ga else nbr_Ga          # upper neighbours of those lower atoms
        ads = _GA_ADS if ga else _AS_ADS
        for k in (0, 1):
            L = lo[src, k]; Ls = np.maximum(L, 0)
            ok = (L >= 0) & _IS_REAL[s[Ls]] & (st.DB[Ls] > 0)
            ok &= ~(st.bonds[src] == L[:, None]).any(axis=1)
            if not ok.any():
                continue
            so = src[ok]; lo_ = L[ok]
            step = np.zeros(so.size, dtype=bool)
            for q, (u_, l_) in enumerate(zip(so.tolist(), lo_.tolist())):
                row = st.bonds[u_, :int(st.n_bonds[u_])].tolist()
                step[q] = _upper_step(st, l_, up[l_, 2:], u_, ads) or any(
                    _upper_step(st, int(h), up[int(h), 2:], u_, ads)
                    for h in lo[u_, :2] if h >= 0 and int(h) != l_ and int(h) in row)
            _emit(r_s, so[step], lo_[step]); _emit(r_t, so[~step], lo_[~step])

    if not rid_list:
        return (np.empty(0, dtype=np.int16),
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.int32))
    return np.concatenate(rid_list), np.concatenate(a_list), np.concatenate(b_list)


# =============================================================================
# Rate computation (vectorized, generalized parameter overrides)
# =============================================================================
def _arrhenius_arrays(params):
    """Per-ID (A, Ea, beta) arrays with `params` applied. IDs 6-11 are skipped: their
    rates are always set by the impingement formula, so A/Ea/beta for them have no
    effect. Tied IDs (_TIED_RIDS) copy their source. A key for a tied ID is accepted only
    if it equals the source's value -- older parameter files carry A_E4 == A_E3 -- and
    anything else is an error, since the engine could not honour it."""
    bA = _BASE_A.copy(); bEa = _BASE_EA.copy(); bB = _BASE_BETA.copy()
    if params:
        for r in _ARRHENIUS_RIDS:
            ak = f"A_E{r}"; ek = f"Ea_E{r}"; bk = f"beta_E{r}"
            if ak in params: bA[r]  = float(params[ak])
            if ek in params: bEa[r] = float(params[ek])
            if bk in params: bB[r]  = float(params[bk])
    for r, src in _TIED_RIDS.items():
        for pre, arr in (("A_E", bA), ("Ea_E", bEa), ("beta_E", bB)):
            if params and f"{pre}{r}" in params and float(params[f"{pre}{r}"]) != arr[src]:
                raise ValueError(
                    f"{pre}{r} = {params[f'{pre}{r}']} differs from {pre}{src} = {arr[src]}. "
                    f"Engine ID {r} is tied to ID {src} and shares its parameter set: "
                    f"set {pre}{src} and drop {pre}{r}.")
            arr[r] = arr[src]
    return bA, bEa, bB


def _compute_rates(rid_arr, T, imp_AsH, imp_MMGa, A_param, params):
    SA_t, SA_s, Ea_chem_As, SM_t, SM_s, Ea_chem_M = _sticking(params)

    bA, bEa, bB = _arrhenius_arrays(params)

    rid_arr = rid_arr.astype(np.int32, copy=False)
    T = float(T)
    rate = bA[rid_arr] * np.exp(-bEa[rid_arr] / T) * (T ** bB[rid_arr])

    sel_AsH = np.isin(rid_arr, _ASH_ADS_RIDS)
    if sel_AsH.any():
        rate[sel_AsH] = imp_AsH * A_param * (SA_s * math.exp(-Ea_chem_As / T))       # R2 (ID 8)
        sel_terr = np.isin(rid_arr, _TERRACE_ASH_RIDS)
        rate[sel_terr] = imp_AsH * A_param * (SA_t * math.exp(-Ea_chem_As / T))      # R1
        rate[rid_arr == _RID_E42_SPLIT] *= _E42_SHARE         # E1.2: two sites, one bond

    sel_MMGa = np.isin(rid_arr, _MMGA_ADS_RIDS)
    if sel_MMGa.any():
        rate[sel_MMGa] = imp_MMGa * A_param * (SM_s * math.exp(-Ea_chem_M / T))      # R7 (ID 11)
        sel_terr = np.isin(rid_arr, _TERRACE_MMGA_RIDS)
        rate[sel_terr] = imp_MMGa * A_param * (SM_t * math.exp(-Ea_chem_M / T))      # R6
        rate[rid_arr == _RID_E62_SPLIT] *= _E42_SHARE         # E6.2: two sites, one bond

    return rate


def _precompute_rate_tables(T, params, imp_AsH, imp_MMGa, A_param):
    """Per-trajectory CONSTANT rate factors.

    Params and T are fixed for the whole trajectory, so the rate of every reaction is
    a constant. Computing it ONCE here — instead of copying the base arrays and
    rebuilding the parameter overrides on every iteration inside _compute_rates —
    is bit-for-bit identical and removes that per-step work.

    Returns
      RATE0     : array[_NUM_RID]  rate of ONE event of each reaction id: Arrhenius
                                   A*exp(-Ea/T)*T^b, or the adsorption rate for 6-11,
                                   34, 35 (terrace IDs with S0_*_terrace, step IDs with
                                   S0_*_step; the split IDs 34, 35 at half)
      base_AsH  : float            imp_AsH  * A_param * S0_A_step*exp(-Ea_chem_As/T)  (= ID 8)
      base_MMGa : float            imp_MMGa * A_param * S0_M_step*exp(-Ea_chem_M /T)  (= ID 11)
    Raises ValueError if the parameters break a _CONSTRAINTS ordering.
    """
    SA_t, SA_s, Ea_chem_As, SM_t, SM_s, Ea_chem_M = _sticking(params)

    bA, bEa, bB = _arrhenius_arrays(params)

    T = float(T)
    RATE0    = bA * np.exp(-bEa / T) * (T ** bB)
    base_AsH  = imp_AsH  * A_param * (SA_s * math.exp(-Ea_chem_As / T))
    base_MMGa = imp_MMGa * A_param * (SM_s * math.exp(-Ea_chem_M  / T))
    RATE0[list(_ASH_ADS_RIDS)] = base_AsH                     # R2 (8)
    RATE0[list(_TERRACE_ASH_RIDS)] = imp_AsH * A_param * (SA_t * math.exp(-Ea_chem_As / T))  # R1
    RATE0[_RID_E42_SPLIT] *= _E42_SHARE                        # E1.2: two sites, one bond
    RATE0[list(_MMGA_ADS_RIDS)] = base_MMGa                   # R7 (11)
    RATE0[list(_TERRACE_MMGA_RIDS)] = imp_MMGa * A_param * (SM_t * math.exp(-Ea_chem_M / T))  # R6
    RATE0[_RID_E62_SPLIT] *= _E42_SHARE                        # E6.2: two sites, one bond
    return RATE0, base_AsH, base_MMGa


def _compute_rates_fast(rid_arr, RATE0, base_AsH=None, base_MMGa=None):
    """Per-event rates from the precomputed table. RATE0 already holds every reaction
    id's rate, adsorption included, so this is one gather; base_AsH / base_MMGa are
    accepted for older callers and not used. Integer-array indexing returns a fresh
    array, so callers may modify the result without touching RATE0."""
    return RATE0[rid_arr]


def landing_rate_errors(st, events, RATE0):
    """Check E4.1/E4.2 and E6.1/E6.2: an adsorbate lands at the rate of one site per
    dangling bond of the host. A DD_Ga's AsH landings must add up to k4 x (its free upper
    sites); a D_MMGa's, one dangling bond with a methyl that moves out of the way, to k4 if
    any upper site is free and 0 if none is. Likewise DD_As / D_AsH for MMGa with k6.
    k4 = RATE0[6] and k6 = RATE0[9] are the rates of one site. `events` is an iterable of
    (rid, a, b). Returns [(host, species, expected, got)] for every host that breaks the
    rule; empty means pass. Run by the tests and by KMC_bucket's Gate 1."""
    events = list(events)               # read twice below: a one-pass iterator would run dry
    cases = ((( 6, _RID_E42_SPLIT), DD_GA, D_MMGA, st.nbr_Ga, V_AS),
             (( 9, _RID_E62_SPLIT), DD_AS, D_ASH,  st.nbr_As, V_GA))
    bad = []
    for rids, two_db, one_db, nbr, vac in cases:
        got = {}
        for r, a, b in events:
            if r in rids:
                got[b] = got.get(b, 0.0) + float(RATE0[r])
        k = float(RATE0[rids[0]])
        for h in np.where((st.species == two_db) | (st.species == one_db))[0]:
            free = int(sum(1 for u in nbr[h, 2:4] if u >= 0 and st.species[u] == vac))
            want = k * (free if st.species[h] == two_db else min(free, 1))
            have = got.get(int(h), 0.0)
            if abs(have - want) > 1e-9 * max(k, 1e-300):
                bad.append((int(h), _CODE_TO_NAME[int(st.species[h])], want, have))
    return bad


# =============================================================================
# Event execution
# =============================================================================
# =============================================================================
# FIX v2 (2026-09-26): bond bookkeeping when atoms leave and when holes refill
# =============================================================================
# An earlier engine updated at most ONE neighbour when an atom left (rid 33: its last bond;
# rid 32: none). The others kept a bond to the empty site, their labels said the bond
# existed, no rule could refill the site, and the growth front froze. v2:
#   * every removal (rids 1, 2, 13, 32, 33) releases EVERY bond of the leaving atom;
#   * a neighbour left with no bond at all leaves too (on any side);
#   * an atom placed into a hole bonds at once to the incorporated atoms above it that
#     face it with a dangling bond, and then to its second lower neighbour;
#   * a Ga bonded to the substrate cannot desorb (enumeration), because no rule can
#     refill a hole in the first Ga layer.
import collections as _collections

# same element and ligand, one dangling bond more / fewer: (label, DB, H)
# Copy: DD_Ga <-> DDD_Ga, as DD_As <-> DDD_As. A DD_Ga that loses a bond is now a DDD_Ga
# (the engine removed it before, having no label for it: FIX v3).
_DB_UP = {D_GA: (DD_GA, 2, 0), GA_: (D_GA, 1, 0), D_MMGA: (DD_MMGA, 2, 1),
          MMGA: (D_MMGA, 1, 1), DD_AS: (DDD_AS, 3, 0), D_AS: (DD_AS, 2, 0),
          AS_: (D_AS, 1, 0), D_ASH: (DD_ASH, 2, 1), ASH: (D_ASH, 1, 1),
          DD_GA: (DDD_GA, 3, 0)}
_DB_DOWN = {DD_GA: (D_GA, 1, 0), D_GA: (GA_, 0, 0), DD_MMGA: (D_MMGA, 1, 1),
            D_MMGA: (MMGA, 0, 1), DDD_AS: (DD_AS, 2, 0), DD_AS: (D_AS, 1, 0),
            D_AS: (AS_, 0, 0), DD_ASH: (D_ASH, 1, 1), D_ASH: (ASH, 0, 1),
            DDD_GA: (DD_GA, 2, 0)}
_GA_FAMILY = (DD_GA, D_GA, GA_, MMGA, DD_MMGA, D_MMGA, DDD_GA)
_ADSORBED = (DD_ASH, DDD_AS, DD_MMGA, DDD_GA)   # one bond, waiting to incorporate (rids 14-31, 38-51)
# Sites other than the event's (a, b) that an event changed / emptied. The engines use
# them to repair their dirty set and growth front.
_TOUCHED: list = []
_VACATED: list = []
# How often each v2 path fired (diagnostics for the tests; never read by the engine).
FIX_COUNTS = _collections.Counter()


def _relabel(st, i, table):
    t = table.get(int(st.species[i]))
    if t is not None:
        st.species[i] = t[0]; st.DB[i] = t[1]; st.H[i] = t[2]
    st.has_entry[i] = True
    return t is not None


def _vacate(st, j):
    st.species[j] = V_GA if st.species[j] in _GA_FAMILY else V_AS
    st._clear_entry(j)


# labels the earlier engine relabelled for the event's own partner b (anything else it left alone)
_MAIN_PARTNER_LABELS = {1: (GA_, MMGA, D_GA), 2: (GA_, MMGA, D_GA), 13: (AS_, ASH, D_AS),
                        32: (), 33: (AS_, ASH, D_AS, D_ASH),
                        37: (AS_, ASH, D_AS),          # DDD_Ga, as MMGa (13)
                        60: (), 61: (), 62: (), 63: (), 64: (), 65: ()}   # copy


def _remove_atom(st, a, b, vac, rid):
    """FIX v2. Remove the atom at a and release every bond it held. Each partner gains
    one dangling bond; a partner left with no bond leaves too. (Copy: a Ga left with one
    bond is now a DDD_Ga and stays, as an As left with one bond stays as a DDD_As; the
    engine used to remove it, FIX v3.) b is the event's own partner: the engine already
    accounts for it, so it is not reported in _TOUCHED / _VACATED."""
    s = st.species
    partners = [int(st.bonds[a, k]) for k in range(int(st.n_bonds[a]))]
    s[a] = vac; st._clear_entry(a)
    for j in partners:
        if j < 0 or not _IS_REAL[s[j]]:
            continue                                    # substrate keeps no record
        sp_j = int(s[j])
        if j != b:
            FIX_COUNTS["extra_partner_released"] += 1
        elif sp_j not in _MAIN_PARTNER_LABELS[rid]:
            FIX_COUNTS["partner_label_main_missed_rid%d" % rid] += 1
        st._remove_bond(j, a)
        if not _relabel(st, j, _DB_UP):
            FIX_COUNTS["no_label_for_one_more_db"] += 1
        if sp_j == DD_GA and st.n_bonds[j] > 0:
            FIX_COUNTS["dd_ga_to_ddd_ga"] += 1              # copy: kept as a DDD_Ga
        if st.n_bonds[j] == 0:
            below = st.coords[j, 2] < st.coords[a, 2]
            _vacate(st, j)
            FIX_COUNTS["orphan_removed_below" if below else "orphan_removed_above"] += 1
            if j != b:
                _VACATED.append(j)
        if j != b:
            _TOUCHED.append(j)


def _bond(st, i, j):
    st._add_bond(i, j); st._add_bond(j, i)
    _relabel(st, i, _DB_DOWN); _relabel(st, j, _DB_DOWN)


# No automatic bonding (27 Sep 2026). Until then an atom that landed in a hole bonded at once
# to the film atoms above it that faced it with a free bond, and then to its second lower
# neighbour (_close_refilled, FIX v2), and the copy also bonded any two facing film atoms after
# every event (_close_facing). Both are gone: such a bond is now an incorporation event with a
# rate (IDs 66-77, section 14 of _enumerate_events), and the H / methyl rotates out of the way.


def _execute_event(st, rid, a, b):
    s = st.species

    # ---- the Ga route (R8 case E8.3, R18) -------------------------------------------
    if rid == 36:                                          # R8 on DD_MMGa: methyl off
        s[a] = DDD_GA; st.DB[a] = 3; st.H[a] = 0; st.has_entry[a] = True
        return

    if rid == 37:                                          # R18: DDD_Ga desorbs
        _remove_atom(st, int(a), int(b), V_GA, 37)
        return

    if 38 <= rid <= 43:                                    # R19 / R20: DDD_Ga incorporates
        s[a] = DD_GA; st.DB[a] = 2; st.H[a] = 0
        st._add_bond(a, int(b)); st.has_entry[a] = True
        if rid in (38, 41):   s[b] = D_AS;  st.DB[b] = 1; st.H[b] = 0
        elif rid in (39, 42): s[b] = ASH;   st.DB[b] = 0; st.H[b] = 1
        else:                 s[b] = AS_;   st.DB[b] = 0; st.H[b] = 0
        st._add_bond(b, int(a)); st.has_entry[b] = True
        return

    if 44 <= rid <= 59 or 66 <= rid <= 77:                 # onto an adsorbed precursor; a
        _bond(st, int(a), int(b))                          # film atom's free bond: both one
        return                                             # dangling bond fewer

    if 60 <= rid <= 65:                                    # film atom, two / three bonds, leaves
        _remove_atom(st, int(a), int(b), V_GA if rid in (62, 63) else V_AS, rid)
        return

    if rid in (9, 10, 11, _RID_E62_SPLIT):
        ads_sp = int(s[b]); s[a] = DD_MMGA
        st._set_entry(a, DB=2, H=1, bonded_to=(int(b),))
        if rid in (9, _RID_E62_SPLIT):
            if ads_sp == D_ASH:
                s[b] = ASH; st.DB[b] = 0; st.H[b] = 1
                st._add_bond(b, int(a)); st.has_entry[b] = True
            elif ads_sp == DD_AS:
                s[b] = D_AS; st.DB[b] = 1; st.H[b] = 0
                st._add_bond(b, int(a)); st.has_entry[b] = True
        else:
            s[b] = AS_; st.DB[b] = 0; st.H[b] = 0
            st._add_bond(b, int(a)); st.has_entry[b] = True
        return

    if rid == 13:
        _remove_atom(st, int(a), int(b), V_GA, 13)         # FIX v2: every bond
        return

    if rid == 12:
        sp = int(s[a])
        if sp == D_MMGA:   s[a] = DD_GA; st.DB[a] = 2; st.H[a] = 0; st.has_entry[a] = True
        elif sp == MMGA:   s[a] = D_GA;  st.DB[a] = 1; st.H[a] = 0; st.has_entry[a] = True
        return

    if rid in (6, 7, 8, _RID_E42_SPLIT):
        ads_sp = int(s[b]); s[a] = DD_ASH
        st._set_entry(a, DB=2, H=1, bonded_to=(int(b),))
        if rid in (6, _RID_E42_SPLIT):
            if ads_sp == D_MMGA:
                s[b] = MMGA; st.DB[b] = 0; st.H[b] = 1
                st._add_bond(b, int(a)); st.has_entry[b] = True
            elif ads_sp == DD_GA:
                s[b] = D_GA; st.DB[b] = 1; st.H[b] = 0
                st._add_bond(b, int(a)); st.has_entry[b] = True
        else:
            s[b] = GA_; st.DB[b] = 0; st.H[b] = 0
            st._add_bond(b, int(a)); st.has_entry[b] = True
        return

    if rid in (1, 2):
        _remove_atom(st, int(a), int(b), V_AS, rid)        # FIX v2: every bond
        return

    if rid in (3, 4, 5):
        if rid == 3:   s[a] = DDD_AS; st.DB[a] = 3; st.H[a] = 0
        elif rid == 4: s[a] = DD_AS;  st.DB[a] = 2; st.H[a] = 0
        else:          s[a] = D_AS;   st.DB[a] = 1; st.H[a] = 0
        st.has_entry[a] = True
        return

    if rid in (14, 15, 16, 17, 18, 19):
        s[a] = D_MMGA; st.DB[a] = 1; st.H[a] = 1
        st._add_bond(a, int(b)); st.has_entry[a] = True
        if rid in (14, 17):   s[b] = D_AS;  st.DB[b] = 1; st.H[b] = 0
        elif rid in (15, 18): s[b] = ASH;   st.DB[b] = 0; st.H[b] = 1
        else:                 s[b] = AS_;   st.DB[b] = 0; st.H[b] = 0
        st._add_bond(b, int(a)); st.has_entry[b] = True
        return

    if rid in (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):
        if rid in (20, 21, 22, 26, 27, 28):
            s[a] = D_ASH; st.DB[a] = 1; st.H[a] = 1
        else:
            s[a] = DD_AS; st.DB[a] = 2; st.H[a] = 0
        st._add_bond(a, int(b)); st.has_entry[a] = True
        if rid in (20, 23, 26, 29):   s[b] = D_GA;  st.DB[b] = 1; st.H[b] = 0
        elif rid in (21, 24, 27, 30): s[b] = MMGA;  st.DB[b] = 0; st.H[b] = 1
        else:                         s[b] = GA_;   st.DB[b] = 0; st.H[b] = 0
        st._add_bond(b, int(a)); st.has_entry[b] = True
        return

    if rid in (32, 33):
        _remove_atom(st, int(a), int(b), V_GA, rid)        # FIX v2: every bond
        return


# =============================================================================
# Growth front helpers
# =============================================================================
def _growth_front_z(st):
    active = _IS_REAL[st.species]
    return float(st.coords[active, 2].max()) if active.any() else -np.inf


def _column_ids(st):
    """Static (x,y) lateral-column index for every atom. Atoms never move in KMC
    (only species labels change), so this is computed ONCE per trajectory.
    Returns (col_id[N] int64, n_cols). Passive: read-only, no RNG touch."""
    key = np.round(st.coords[:, :2], 2)
    _, col_id = np.unique(key, axis=0, return_inverse=True)
    return col_id.astype(np.int64), int(col_id.max()) + 1 if col_id.size else 0


def _mean_height_z(st, col_id, n_cols):
    """Mean over lateral columns of the per-column top REAL-atom z (same column
    definition as the roughness study: real = _IS_REAL species, per-(x,y) max z,
    averaged over all columns that contain a real atom). Columns with no real atom
    are skipped. O(N); called only at checkpoints. Passive: read-only, no RNG."""
    real = _IS_REAL[st.species]
    if not real.any():
        return -np.inf
    colmax = np.full(n_cols, -np.inf)
    np.maximum.at(colmax, col_id[real], st.coords[real, 2])
    finite = colmax[np.isfinite(colmax)]
    return float(finite.mean()) if finite.size else -np.inf


def _active_mask(st, layer_height=1.42, num_layers=5):
    is_vac = _IS_VAC[st.species]
    real = ~is_vac
    if not real.any(): return np.ones(st.N, dtype=bool)
    cutoff = float(st.coords[real, 2].max()) + num_layers * layer_height
    too_high = is_vac & (st.coords[:, 2] > cutoff)
    return ~too_high


# =============================================================================
# Supercell builder
# =============================================================================
def make_supercell(input_file: str, Nx: int, Ny: int, Nz: int,
                   output_file: str) -> str:
    with open(input_file) as fh:
        lines = fh.readlines()
    N_orig   = int(lines[0].strip())
    header   = lines[1]
    lattice  = np.array(
        list(map(float, header.split('Lattice="')[1].split('"')[0].split()))
    ).reshape(3, 3)
    origin   = (np.array(list(map(float, header.split('Origin="')[1].split('"')[0].split())))
                if 'Origin="' in header else np.zeros(3))
    atoms = []
    for line in lines[2:]:
        p = line.split()
        if p: atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    assert len(atoms) == N_orig

    a1, a2, a3 = lattice[0], lattice[1], lattice[2]
    new_atoms = []
    for iz in range(Nz):
        for iy in range(Ny):
            for ix in range(Nx):
                shift = ix * a1 + iy * a2 + iz * a3
                for sp, x, y, z in atoms:
                    new_atoms.append((sp, x + shift[0], y + shift[1], z + shift[2]))

    new_lat = lattice.copy()
    new_lat[0] = Nx * a1; new_lat[1] = Ny * a2; new_lat[2] = Nz * a3

    with open(output_file, "w") as fh:
        fh.write(f"{len(new_atoms)}\n")
        fh.write(f'Lattice="{" ".join(f"{v:.8f}" for v in new_lat.flatten())}" '
                 f'Origin="{" ".join(f"{v:.8f}" for v in origin)}" '
                 f"Properties=species:S:1:pos:R:3\n")
        for sp, x, y, z in new_atoms:
            fh.write(f"{sp} {x:.8f} {y:.8f} {z:.8f}\n")

    print(f"[supercell] {Nx}x{Ny}x{Nz}: {N_orig} → {len(new_atoms)} atoms → {output_file}")
    return output_file


def _parse_supercell(sc_str: str):
    for sep in ("x", "X", "×", ","):
        if sep in sc_str:
            parts = [int(p) for p in sc_str.split(sep)]
            return (parts[0], parts[1], parts[2] if len(parts) > 2 else 1)
    n = int(sc_str)
    return n, n, 1


# =============================================================================
# PUBLIC API
# =============================================================================
def _write_event_log(path: str, event_counts: np.ndarray, total_fired: int,
                     target_T: float, node_label: str):
    """Write event frequency per REACTION to a CSV file. One row per reaction (Case
    'all'); a reaction that spans several engine IDs is followed by one row per case, so
    the 'all' rows alone add up to the total. EngineID is a reference to the code."""
    import csv as _csv
    frac = lambda n: f"{(n / total_fired if total_fired > 0 else 0.0):.6f}"
    with open(path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["Reaction", "Case", "Description", "Count", "Fraction", "EngineID"])
        for lab, name, ids in _reaction_groups():
            n = int(sum(int(event_counts[r]) for r in ids))
            w.writerow([lab, "all", name, n, frac(n), ", ".join(str(r) for r in ids)])
            if len(ids) > 1:
                for r in ids:
                    n = int(event_counts[r])
                    w.writerow([lab, _RID_CASE.get(r, ""), _RID_DESCRIPTION.get(r, ""),
                                n, frac(n), r])
    print(f"  [Events] Log written to {path}  "
          f"(total fired: {total_fired}, T={target_T} K)")


# =============================================================================
# INCREMENTAL enumeration helpers (fast2). _execute_event changes state only at
# {a,b}, so after a fire only sites within k graph-hops of {a,b} can change their
# event menu -> re-enumerate ONLY those and splice the event list. Ownership makes
# drop-and-replace exact: source(owner) = b for adsorption rids 6..11, else a.
#
# fast2 additionally removes the two per-step O(N=lattice) rescans of KMC_base:
#   * growth front          -> O(1) upward tracking at {a,b}, re-seeded exactly
#     every _INCR_RESYNC steps.
#
# Validated: incremental event SET == full _enumerate_events every step, front never
# undershoots
# (560k lattice 600 steps; small lattice 2500 steps). Ensemble GR vs KMC_base: 0.24 sigma.
# =============================================================================
_INCR_KHOP = 3
_INCR_RESYNC = 256

def _incr_adjacency(st):
    cols = [t for t in (st.nbr_As, st.nbr_Ga) if t is not None]
    adj = np.concatenate(cols, axis=1) if cols else np.full((st.N, 1), -1)
    return adj.astype(np.int64, copy=False)

def _incr_khop(seeds, adj, k, bonds, nbonds):
    frontier = set(int(x) for x in seeds if x >= 0); seen = set(frontier)
    for _ in range(k):
        nxt = set()
        for i in frontier:
            for j in adj[i]:
                if j >= 0 and j not in seen: nxt.add(int(j))
            for kk in range(int(nbonds[i])):
                j = int(bonds[i, kk])
                if j >= 0 and j not in seen: nxt.add(j)
        seen |= nxt; frontier = nxt
        if not frontier: break
    return seen

def _incr_enum_batched(st, sites_arr):
    """ONE batched _enumerate_events over all (reactive) dirty sites; owner derived
    vectorially. sites_arr: int64 array. Returns (rid,a,b,own) int64 arrays."""
    if sites_arr.size == 0:
        z = np.empty(0, np.int64); return z, z.copy(), z.copy(), z.copy()
    rid, a, b = _enumerate_events(st, sites_arr)
    if rid.size == 0:
        z = np.empty(0, np.int64); return z, z.copy(), z.copy(), z.copy()
    ridL = rid.astype(np.int64, copy=False)
    aL = a.astype(np.int64, copy=False); bL = b.astype(np.int64, copy=False)
    own = np.where(_IS_ADS_RID[ridL], bL, aL)
    return ridL, aL, bL, own

def run_kmc_trajectory(target_T, params=None, node_label="node",
                       traj_freq=None, lattice_file="Lattice.xyz",
                       log_events=False, target_growth=150.0, max_iter=None,
                       checkpoint_every=None, checkpoint_sink=None,
                       stop_mean_height=0.0,
                       stop_plateau_idle_factor=0.0, stop_plateau_min_events=0,
                       stop_plateau_rel_tol=0.0, stop_plateau_abs_tol=0.0):
    """
    Run one KMC trajectory at temperature target_T [K].
    Returns simulated growth rate in µm/min.

    Parameters
    ----------
    target_T     : float   Temperature in Kelvin.
    params       : dict    Parameter overrides. None = use all catalog defaults.
                           Keys: A_EN, Ea_EN, beta_EN, S0_A_terrace, S0_A_step,
                                 Ea_chem_As, S0_M_terrace, S0_M_step, Ea_chem_M
    node_label   : str     Label used in output filenames.
    traj_freq    : int     Write XYZ every N iterations. None/0 = off.
    lattice_file : str     Path to Lattice.xyz (or supercell file).
    log_events   : bool    If True, write a CSV counting how many times each
                           reaction fired. File: events_{node_label}_T{T}K.csv
    checkpoint_every : int  If set (>0) and checkpoint_sink is a list, append a
                           PASSIVE diagnostic tuple every N events:
                               (cumulative_events, elapsed_sim_time_s,
                                max_height_A, mean_height_A)
                           max_height_A = growth front - z_init (= thickness);
                           mean_height_A = mean over lateral columns of the top
                           real-atom z, minus z_init. Read-only, never touches the
                           RNG or state, so the trajectory is bit-identical whether
                           or not checkpointing is on. A final checkpoint at exactly
                           the last iteration is always appended, so its max_height
                           / elapsed match the summary GR exactly.
    checkpoint_sink  : list Mutable list the checkpoints are appended to.
    """
    TARGET_GROWTH_A = float(target_growth)

    # OPTIONAL EARLY STOP AT A MEAN-HEIGHT TARGET.  0.0 = off.
    #
    # With it off this engine is bit-identical to the unpatched one: the flag only
    # ever adds a `break` at a checkpoint that was already being written, so no
    # rate, no draw and no RNG call changes.
    #
    # Why the MEAN and not the growth FRONT.  The adopted growth rate is
    # GR = h_mean / t, so the trajectory must stop when the MEAN film reaches the
    # measurement thickness.  The front runs far ahead of the mean -- on the
    # lambda 2.50 rung at 30,000 events the front is at 7.07 A while the mean is
    # at 1.24 A -- so `target_growth`, which watches the front, would cut the run
    # long before the measurement is due.  The two are not interchangeable.
    _STOP_MEAN_H = float(stop_mean_height or 0.0)
    # STALL STOP: 'the film has stopped' is 'the mean height has not advanced for a long
    # time' (below). The window is
    # scaled to the run's own history -- idle_factor x (events at the last
    # advance), floored at min_events -- so a fast film is judged quickly and
    # a slow one is given proportionally longer before it is called dead.
    # Like the mean-height stop, it only ever fires ON a periodic checkpoint,
    # so the trace is a strict PREFIX of the same run with the stop disabled.
    _PLAT_FAC  = float(stop_plateau_idle_factor or 0.0)
    _PLAT_MIN  = int(stop_plateau_min_events or 0)
    # The STALL rule, judged on the MEAN height (was: the front). A checkpoint is
    # an advance only if it beats the best mean by more than a tolerance; a run
    # whose mean height has not advanced for long enough has stopped growing and
    # ends (run_kmc.py: stall_idle_factor, stall_min_events, stall_rel_tol,
    # stall_abs_tol_A). The growth rate is read by reward_metric.window_from_sink.
    _PLAT_REL  = float(stop_plateau_rel_tol or 0.0)
    _PLAT_ABS  = float(stop_plateau_abs_tol or 0.0)
    _plat_best = None         # best MEAN height at any checkpoint
    _plat_since = 0           # iteration of the last mean-height advance
    GROWTH_CHECK    = 8          # recompute the growth front (O(N)) only every N steps
    PA, PM          = _P_ASH, _P_MMGA
    A_param         = _A_SITE
    k_B             = _K_B
    M_ash           = _M_ASH       # AsH, 75.93 u
    M_mmga          = _M_MMGA      # GaCH3, 84.76 u

    atoms_init, origin, unit_cell = extract_coords_with_species_labels(lattice_file)
    atoms_init = label_lattice_sites(atoms_init, bottom_layers=3)
    st = LatticeState(atoms_init, origin, unit_cell)
    _initialize_bonding(st)
    _purge(st)

    z_init  = _growth_front_z(st)
    elapsed = 0.0
    growth  = 0.0

    write_traj = (traj_freq is not None) and (traj_freq > 0)
    f_traj     = None
    traj_name  = f"KMC_Traj_{node_label}_T{int(target_T)}K.xyz"
    if write_traj:
        f_traj = open(traj_name, "w")
        _append_xyz_frame(f_traj, st.coords, st.species, st.origin, st.unit_cell,
                          f"Frame=0 Time=0.0 T={target_T}")
        f_traj.flush()
        print(f"  [Traj] Writing to {traj_name} every {traj_freq} iterations")

    imp_AsH  = PA / math.sqrt(2.0 * math.pi * M_ash  * k_B * float(target_T))
    imp_MMGa = PM / math.sqrt(2.0 * math.pi * M_mmga * k_B * float(target_T))
    rng = np.random

    # Event frequency counter — one slot per reaction ID (index 0 unused)
    event_counts  = np.zeros(_NUM_RID, dtype=np.int64)
    total_fired   = 0

    # Rate factors that are constant for the whole trajectory (params + T fixed) —
    # computed once instead of rebuilt every iteration.
    _RATE0, _base_AsH, _base_MMGa = _precompute_rate_tables(
        float(target_T), params, imp_AsH, imp_MMGa, A_param)

    iteration = 0
    _prog_every = int(os.environ.get("KMC_PROG_EVERY", "0"))   # live progress cadence
    import time as _prog_time
    _prog_wall0 = _prog_time.perf_counter()
    # INCREMENTAL(fast2) setup: static adjacency, owner-tagged menu, O(1) growth
    # front. Per-step cost is now independent of lattice size.
    _adj = _incr_adjacency(st)
    _dirty_flag = np.zeros(st.N, dtype=bool)
    _zc = st.coords[:, 2]
    _front = z_init
    _r0 = np.where(_IS_REACTIVE[st.species])[0]
    ev_rid, ev_a, ev_b, ev_own = _incr_enum_batched(st, _r0)

    # PASSIVE checkpoint instrumentation (mean/max height, sim time). Column ids are
    # static (atoms never move), built once. Enabled only when a sink is provided.
    _do_ckpt = bool(checkpoint_every) and checkpoint_every > 0 and checkpoint_sink is not None
    _col_id, _n_cols = (_column_ids(st) if _do_ckpt else (None, 0))
    # mean height has its OWN baseline (initial mean column-top), not the max front
    # z_init -- otherwise mean growth reads negative (mean top < max front).
    _z_init_mean = (_mean_height_z(st, _col_id, _n_cols) if _do_ckpt else 0.0)
    try:
        while growth < TARGET_GROWTH_A and (max_iter is None or iteration < max_iter):
            if ev_rid.size == 0:
                iteration += 1; continue

            rates = _compute_rates_fast(ev_rid, _RATE0, _base_AsH, _base_MMGa)
            total = float(rates.sum())
            if total <= 0.0:
                iteration += 1; continue

            cum = np.cumsum(rates)
            sel = int(np.searchsorted(cum, float(rng.rand()) * total))
            if sel >= ev_rid.size: sel = ev_rid.size - 1

            fired_rid = int(ev_rid[sel]); _fa = int(ev_a[sel]); _fb = int(ev_b[sel])

            _TOUCHED.clear(); _VACATED.clear()
            _execute_event(st, fired_rid, _fa, _fb)

            event_counts[fired_rid] += 1
            total_fired += 1
            elapsed   += -math.log(rng.rand()) / total
            iteration += 1

            # O(1) upward growth front (only a,b can rise it)
            if _IS_REAL[st.species[_fa]] and _zc[_fa] > _front: _front = float(_zc[_fa])
            if _fb >= 0 and _IS_REAL[st.species[_fb]] and _zc[_fb] > _front: _front = float(_zc[_fb])

            # INCREMENTAL menu update: drop+re-enumerate only sites near {_fa,_fb}.
            _dirty = _incr_khop((_fa, _fb) + tuple(set(_TOUCHED)), _adj, _INCR_KHOP,
                                st.bonds, st.n_bonds)          # FIX v2: + touched
            _da = np.fromiter(_dirty, np.int64, len(_dirty))
            _dirty_flag[_da] = True
            _keep = ~_dirty_flag[ev_own]
            _dirty_flag[_da] = False
            ev_rid, ev_a, ev_b, ev_own = ev_rid[_keep], ev_a[_keep], ev_b[_keep], ev_own[_keep]
            _rd = _da[_IS_REACTIVE[st.species[_da]]]
            _nr, _na, _nb, _no = _incr_enum_batched(st, _rd)
            if _nr.size:
                ev_rid = np.concatenate([ev_rid, _nr]); ev_a = np.concatenate([ev_a, _na])
                ev_b = np.concatenate([ev_b, _nb]); ev_own = np.concatenate([ev_own, _no])

            # exact resync every _INCR_RESYNC steps: corrects any front drop from a
            # top-of-film desorption.
            if iteration % _INCR_RESYNC == 0:
                _front = _growth_front_z(st)
                growth = _front - z_init
            elif iteration % GROWTH_CHECK == 0:
                growth = _front - z_init

            # PASSIVE checkpoint: exact O(N) front + mean height. Read-only.
            if _do_ckpt and iteration % checkpoint_every == 0:
                _cp_max_h  = _growth_front_z(st) - z_init
                _cp_mean_h = _mean_height_z(st, _col_id, _n_cols) - _z_init_mean
                checkpoint_sink.append((int(iteration), float(elapsed),
                                        float(_cp_max_h), float(_cp_mean_h)))
                # Stop ON a periodic checkpoint, never between them. That keeps the
                # final row on the correct mean baseline (_z_init_mean) -- the
                # guaranteed final append uses the max-front baseline and would bias
                # the mean -- and makes the whole trace a strict PREFIX of the run
                # this would have been with the stop off.
                if _STOP_MEAN_H > 0.0 and _cp_mean_h >= _STOP_MEAN_H:
                    break

                # --- stall stop ---------------------------------------------
                if (_plat_best is None or
                        _cp_mean_h > _plat_best + max(_PLAT_REL * abs(_plat_best),
                                                      _PLAT_ABS)):
                    _plat_best  = _cp_mean_h
                    _plat_since = iteration
                elif _PLAT_FAC > 0.0:
                    _idle = iteration - _plat_since
                    if _idle >= max(_PLAT_MIN, int(_PLAT_FAC * _plat_since)):
                        break

            if _prog_every and iteration % _prog_every == 0:
                _pw = _prog_time.perf_counter() - _prog_wall0
                print(f"  events={iteration:>11,}   thickness={growth:7.2f} A = {growth/10:5.2f} nm"
                      f"   wall={_pw:8.1f} s", flush=True)

            if write_traj and iteration % traj_freq == 0:
                _append_xyz_frame(f_traj, st.coords, st.species, st.origin, st.unit_cell,
                                  f"Frame={iteration} Time={elapsed:.6e} "
                                  f"T={target_T} Growth={growth:.3f}A")
                f_traj.flush()
    finally:
        if write_traj and f_traj is not None:
            _append_xyz_frame(f_traj, st.coords, st.species, st.origin, st.unit_cell,
                              f"Frame={iteration} Time={elapsed:.6e} FINAL")
            f_traj.flush(); f_traj.close()

        if log_events:
            event_log_path = f"events_{node_label}_T{int(target_T)}K.csv"
            _write_event_log(event_log_path, event_counts, total_fired,
                             target_T, node_label)

    growth = _growth_front_z(st) - z_init          # exact final thickness for the reported GR
    t_min = elapsed / 60.0
    gr    = (growth * 1e-4) / t_min if t_min > 0 else 0.0

    # PASSIVE final checkpoint: guaranteed last row uses the SAME growth+elapsed as
    # the summary GR, so GR_max at the last row == the reported final GR exactly,
    # regardless of whether the run stopped on target or on the event cap.
    if _do_ckpt and ((not checkpoint_sink) or checkpoint_sink[-1][0] != iteration):
        _fin_mean_h = _mean_height_z(st, _col_id, _n_cols) - z_init
        checkpoint_sink.append((int(iteration), float(elapsed),
                                float(growth), float(_fin_mean_h)))
    global _LAST_ITERATIONS, _LAST_THICKNESS_A, _LAST_EVENT_COUNTS, _LAST_TOTAL_FIRED
    _LAST_ITERATIONS = iteration          # exposed for benchmarking / diagnostics
    _LAST_THICKNESS_A = float(growth)     # exposed final film thickness [A] for logging
    # per-engine-ID counts, read by the MCTS / worker logs (n_rid_XX and n_R<k> columns)
    _LAST_EVENT_COUNTS = event_counts.copy()
    _LAST_TOTAL_FIRED = int(total_fired)
    return gr


# =============================================================================
# Averaged run — KMC is stochastic, so "one KMC run" = N replicate trajectories
# whose growth rates are averaged. The MEAN is what MCTS uses for the error and
# what the CLI reports.
# =============================================================================
DEFAULT_N_REPLICATES = 10


def run_kmc(target_T, params=None, node_label="node", traj_freq=None,
            lattice_file="Lattice.xyz", log_events=False,
            n_replicates=DEFAULT_N_REPLICATES):
    """
    Run n_replicates independent KMC trajectories at target_T and average the
    growth rate.

    Returns
    -------
    mean_gr : float          mean growth rate over the replicates [µm/min]
    std_gr  : float          standard deviation across replicates [µm/min]
    all_grs : list[float]    the individual replicate growth rates

    Notes
    -----
    Only the FIRST replicate writes a trajectory file / event log (when those
    are enabled), to avoid producing N redundant output files. The physics of
    each replicate is identical; only the random number stream differs.
    """
    n = max(1, int(n_replicates))
    all_grs = []
    for i in range(n):
        gr = run_kmc_trajectory(
            target_T     = target_T,
            params       = params,
            node_label   = node_label if i == 0 else f"{node_label}_rep{i + 1}",
            traj_freq    = traj_freq   if i == 0 else None,
            lattice_file = lattice_file,
            log_events   = log_events  if i == 0 else False,
        )
        all_grs.append(gr)
    arr = np.asarray(all_grs, dtype=float)
    return float(arr.mean()), float(arr.std()), all_grs


# =============================================================================
# PARALLEL sweep evaluation — same pattern as 09's MCTS parallel pool.
# Every (temperature x replicate) trajectory is an independent task; they all
# run across a multiprocessing.Pool of n_workers cores in the SAME job (no
# extra SLURM scheduling). Wall time ~ ceil(n_tasks / n_workers) waves.
# =============================================================================
def _sweep_task(task):
    """Worker: ONE KMC trajectory with its own RNG seed. Top-level so it is
    picklable for multiprocessing."""
    import time as _t
    T, params, lattice_file, seed = task
    np.random.seed(seed)                      # independent stream per trajectory
    t0 = _t.time()
    gr = run_kmc_trajectory(
        target_T=T, params=params, node_label="p",
        traj_freq=None, lattice_file=lattice_file)
    return (T, gr, _t.time() - t0)


def run_sweep_parallel(temperatures, params, lattice_file, n_replicates,
                       n_workers, base_seed=12345):
    """
    Run n_replicates trajectories at every temperature, ALL in parallel across
    n_workers processes. Returns {T: (mean_gr, std_gr, cpu_seconds, all_grs)}.
    cpu_seconds = summed worker time for that temperature (not wall time);
    all_grs = the individual replicate growth rates (kept for box plots).
    """
    import multiprocessing as mp
    tasks, k = [], 0
    for T in temperatures:
        for _ in range(int(n_replicates)):
            tasks.append((T, params, lattice_file, base_seed + k))
            k += 1
    nproc = max(1, min(int(n_workers), len(tasks)))
    print(f"[KMC] parallel sweep: {len(tasks)} trajectories over {nproc} workers "
          f"(~{-(-len(tasks) // nproc)} waves)")
    if nproc == 1:
        results = [_sweep_task(t) for t in tasks]
    else:
        with mp.Pool(processes=nproc) as pool:
            results = pool.map(_sweep_task, tasks)
    out = {}
    for T in temperatures:
        grs  = [g for (t, g, _w) in results if t == T]
        wall = sum(w for (t, _g, w) in results if t == T)
        arr  = np.asarray(grs, dtype=float)
        out[T] = (float(arr.mean()), float(arr.std()), wall, grs)
    return out


# =============================================================================
# CLI entry point
# =============================================================================
if __name__ == "__main__":
    import argparse
    import csv
    import json
    import time

    ap = argparse.ArgumentParser(
        description="Run KMC simulation(s). Settings are read from config.toml."
    )
    ap.add_argument("--config",      default="config.toml",
                    help="Path to TOML config file (default: config.toml)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Override: single temperature in K")
    ap.add_argument("--temperatures", default=None,
                    help="Override (sweep mode): comma-separated temperature list, "
                         "e.g. '900,925,950,975,1000'. Lets one job handle a subset.")
    ap.add_argument("--mode",        default=None,
                    choices=["single", "sweep"],
                    help="Override: 'single' or 'sweep'")
    ap.add_argument("--params",      default=None,
                    help="JSON file with parameter overrides (overrides config.toml [params])")
    ap.add_argument("--output",      default=None,
                    help="Override: output CSV file")
    ap.add_argument("--traj-freq",   type=int, default=None,
                    help="Override: XYZ trajectory write interval (0 = off)")
    ap.add_argument("--log-events",  action="store_true", default=None,
                    help="Override: write per-reaction event frequency CSV")
    ap.add_argument("--n-replicates", type=int, default=None,
                    help="Replicate trajectories per temperature; GR is their mean "
                         f"(default: config [kmc] n_replicates or {DEFAULT_N_REPLICATES})")
    args = ap.parse_args()

    cfg = _read_config(args.config)

    # --- resolve run settings ---
    run_cfg = cfg.get("run", {})
    kmc_cfg = cfg.get("kmc", {})

    mode        = args.mode or run_cfg.get("mode", "single")
    csv_path    = args.output or run_cfg.get("output_csv", "kmc_results.csv")
    lattice_f   = kmc_cfg.get("lattice_file", "Lattice.xyz")
    sc_str      = kmc_cfg.get("supercell", "1x1x1")
    traj_freq   = args.traj_freq if args.traj_freq is not None else kmc_cfg.get("traj_freq", 0)
    traj_freq   = traj_freq if traj_freq > 0 else None
    log_events  = args.log_events if args.log_events else kmc_cfg.get("log_events", False)
    n_replicates = (args.n_replicates if args.n_replicates is not None
                    else int(kmc_cfg.get("n_replicates", DEFAULT_N_REPLICATES)))

    # --- resolve parameter overrides ---
    if args.params:
        with open(args.params) as f:
            params = json.load(f)
        print(f"[KMC] Parameters loaded from {args.params}")
    else:
        params = {str(k): v for k, v in cfg.get("params", {}).items()} or None
        if params:
            print(f"[KMC] Using [params] from {args.config}: {params}")
        else:
            print(f"[KMC] No parameter overrides — using catalog defaults.")

    # --- resolve temperatures ---
    if mode == "single":
        T = args.temperature or float(run_cfg.get("temperature", 900))
        temperatures = [T]
    elif args.temperatures:
        # explicit subset (used by the 3-job split sweep)
        temperatures = [float(t) for t in args.temperatures.split(",") if t.strip()]
        print(f"[KMC] sweep over --temperatures: {temperatures}")
    else:
        temperatures = [float(t) for t in run_cfg.get(
            "temperatures", [900, 950, 1000, 1050, 1100, 1150, 1200, 1230])]
        if args.temperature:
            print(f"[KMC] --temperature ignored in sweep mode")

    # --- supercell ---
    Nx, Ny, Nz = _parse_supercell(sc_str)
    sc_label    = f"{Nx}x{Ny}x{Nz}"
    if Nx == 1 and Ny == 1 and Nz == 1:
        lattice_file = lattice_f
    else:
        lattice_file = f"Lattice_{sc_label}.xyz"
        make_supercell(lattice_f, Nx, Ny, Nz, lattice_file)

    # --- ensure CSV header ---
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["Supercell", "Temperature_K", "Growth_Rate_um_min",
                 "GR_std", "N_replicates", "Wall_s", "Status"])

    # --- parallel workers: SLURM allocation > config [kmc] n_workers > 40 ---
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK",
                                   kmc_cfg.get("n_workers", 40)))

    print(f"\n[KMC] mode={mode}  supercell={sc_label}  temperatures={temperatures}")
    print(f"[KMC] Replicates per temperature: {n_replicates} (GR = mean of these)")
    print(f"[KMC] Parallel workers: {n_workers}")
    print(f"[KMC] Results → {csv_path}\n")

    # Raw replicate GRs are written alongside the summary CSV so the final plot
    # can show the SPREAD (box plot) at each temperature, not just mean ± std.
    rep_csv = (csv_path.replace("kmc_results", "kmc_replicates")
               if "kmc_results" in csv_path
               else os.path.splitext(csv_path)[0] + "_replicates.csv")

    t0_all = time.time()
    try:
        gr_map = run_sweep_parallel(temperatures, params, lattice_file,
                                    n_replicates, n_workers)
        for T in temperatures:
            gr, gr_std, cpu_s, all_grs = gr_map[T]
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([sc_label, int(T), f"{gr:.6f}",
                                        f"{gr_std:.6f}", n_replicates,
                                        f"{cpu_s:.1f}", "ok"])
            if not os.path.exists(rep_csv):
                with open(rep_csv, "w", newline="") as f:
                    csv.writer(f).writerow(["Temperature_K", "Replicate",
                                            "GR_um_min"])
            with open(rep_csv, "a", newline="") as f:
                w = csv.writer(f)
                for i, g in enumerate(all_grs, 1):
                    w.writerow([int(T), i, f"{g:.6f}"])
            print(f"[KMC] T={T:.0f} K  →  GR={gr:.4f} ± {gr_std:.4f} µm/min  "
                  f"(mean of {n_replicates}; cpu: {cpu_s:.1f} s)")
        print(f"[KMC] Replicate GRs → {rep_csv}")
    except Exception as exc:
        # fall back: record the failure for every temperature in this job
        print(f"[KMC] ERROR in parallel sweep: {exc}")
        for T in temperatures:
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([sc_label, int(T), "nan", "nan",
                                        n_replicates, "0.0", f"error: {exc}"])
        raise

    print(f"\n[KMC] Done in {time.time() - t0_all:.1f} s wall. "
          f"Results saved to: {csv_path}")
