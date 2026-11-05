"""
KMC_bucket -- the production fast2 engine with an O(1) event menu.

Drop-in for KMC.run_kmc_trajectory. Imports the GaAs chemistry (species table,
_enumerate_events, _execute_event, rate tables) UNCHANGED from KMC.py and replaces
only the engine machinery around it:

  1. Selection.  Every event with the same reaction id has the same rate
     (see _compute_rates_fast: the rate depends on the rid only).
     So   R_total = sum_r  n_r * k_r   over the ~35 reaction ids.
     Pick r from the 35 bucket totals, then a uniform slot inside bucket r.
     Exact, not approximate. Replaces the O(M) cumsum over the whole menu.

  2. Menu storage.  One Python list per rid for a, b, owner. Removal is
     swap-with-last, O(1). Insertion is append, O(1). An owner index
     (site -> list of (rid, slot)) makes "drop every event owned by a dirty
     site" O(events dropped) instead of an O(M) boolean mask + compaction +
     concatenate.

  3. Dirty set.  Bonds are always geometric neighbours (verified: 0 exceptions
     over 4000 steps; both come from the same tetrahedral vectors), so the k-hop
     neighbourhood over adjacency+bonds equals the k-hop closure over static
     adjacency alone. That closure is cached per site; the per-step dirty set is
     two cache lookups and a set union instead of a Python BFS.

Everything else -- chemistry, growth-front tracking,
resync cadence, checkpoints, RNG draws per step (one for selection, one for the
clock) -- is identical to KMC.py.

Correctness gates (see gate1_menu_identity.py, gate2_ensemble.py):
  Gate 1  bucket menu == full _enumerate_events over all reactive sites, as a SET
          of (rid, a, b), at every step.
  Gate 2  ensemble distribution of GR / mean height matches KMC.py within
          sampling error. Not seed-for-seed: the random stream is consumed the
          same number of times per step, but a different event is selected for
          the same u because the menu is ordered differently.
"""
from __future__ import annotations

import math
import os
import time as _time

import numpy as np

import KMC as K
from KMC import (
    _IS_REACTIVE, _IS_REAL, _NUM_RID, _INCR_RESYNC, _IS_ADS_RID,
    LatticeState, extract_coords_with_species_labels, label_lattice_sites,
    _initialize_bonding, _purge, _growth_front_z, _column_ids, _mean_height_z,
    _precompute_rate_tables, _execute_event, _enumerate_events,
    _incr_adjacency,
    _write_event_log, _append_xyz_frame,
)

KHOP = int(os.environ.get("KMC_INCR_KHOP", str(K._INCR_KHOP)))
#: 1 = re-enumerate the per-step dirty set with enum_scalar (fast on ~40 sites),
#: 0 = with the vectorised KMC._enumerate_events (the reference).
SCALAR = int(os.environ.get("KMC_BUCKET_SCALAR", "1"))
if SCALAR:
    from enum_scalar import enumerate_scalar
#: Gate 1 harness: if > 0, compare the bucket menu against a full re-enumeration
#: at EVERY step for this many steps, and raise on the first mismatch.
CHECK_STEPS = int(os.environ.get("KMC_BUCKET_CHECK", "0"))
#: start of the Gate 1 window, in events (0 = from the first step). Lets the menu
#: be checked late in growth, not only on the bare substrate.
CHECK_FROM = int(os.environ.get("KMC_BUCKET_CHECK_FROM", "0"))
CHECK_RATE_DEV: list = []   # max relative total-rate deviation per checked step
CHECK_LOG: list = []
CHECK_RIDS: set = set()     # rids ever present in a checked menu
CHECK_FIRED: set = set()    # rids ever fired during checked steps


def _full_menu_set(st) -> set:
    r = np.where(_IS_REACTIVE[st.species])[0]
    rid, a, b = _enumerate_events(st, r)
    return set(zip(rid.tolist(), a.tolist(), b.tolist()))



def _owner_of(rid: int, a: int, b: int) -> int:
    """Same rule as _incr_enum_batched: impingement rids (6-11, 34) are owned by b."""
    return b if _IS_ADS_RID[rid] else a


class ClosureCache:
    """k-hop closure over static adjacency.

    Bonds are always geometric neighbours in this chemistry (checked: 0 exceptions
    over 4000 fired events, see docstring), so the adjacency closure IS the dirty
    set. Computed once per site: batched for the initial reactive band, lazily
    (vectorised, ~15 us) for sites that become reactive later. Each entry is
    (frozenset, int64 array)."""
    __slots__ = ("adj", "k", "cache")

    def __init__(self, adj: np.ndarray, k: int):
        self.adj = adj
        self.k = k
        self.cache: dict = {}

    def _rows_unique(self, block: np.ndarray):
        """block: (n, m) int64 with -1 padding. Yields one sorted unique array per row."""
        block = np.sort(block, axis=1)
        for row in block:
            row = row[row >= 0]
            if row.size:
                keep = np.empty(row.size, dtype=bool); keep[0] = True
                np.not_equal(row[1:], row[:-1], out=keep[1:])
                row = row[keep]
            yield row

    def prefill(self, sites: np.ndarray, batch: int = 4096) -> None:
        adj = self.adj
        for s0 in range(0, sites.size, batch):
            S = sites[s0:s0 + batch]
            cur = S[:, None]
            cols = [cur]
            for _ in range(self.k):
                nxt = adj[np.maximum(cur, 0)]          # (n, w, 8)
                nxt = np.where((cur >= 0)[..., None], nxt, -1).reshape(cur.shape[0], -1)
                cols.append(nxt)
                cur = nxt
            block = np.concatenate(cols, axis=1)
            for site, row in zip(S.tolist(), self._rows_unique(block)):
                self.cache[site] = (frozenset(row.tolist()), row.astype(np.int64))

    def get(self, i: int):
        c = self.cache.get(i)
        if c is not None:
            return c
        self.prefill(np.array([i], dtype=np.int64))
        return self.cache[i]


class BucketMenu:
    """Per-rid event buckets with O(1) insert / remove and O(#rid) selection."""

    def __init__(self, n_rid: int, n_sites: int):
        self.n_rid = n_rid
        self.A = [[] for _ in range(n_rid)]
        self.B = [[] for _ in range(n_rid)]
        self.O = [[] for _ in range(n_rid)]
        # owner index: site -> list of (rid, slot). Lazily created lists.
        self.owner_slots: list = [None] * n_sites
        self.n = np.zeros(n_rid, dtype=np.int64)

    # -- insertion ----------------------------------------------------------
    def insert_arrays(self, rid_arr, a_arr, b_arr, own_arr) -> None:
        if rid_arr.size == 0:
            return
        rid_l = rid_arr.tolist(); a_l = a_arr.tolist()
        b_l = b_arr.tolist(); o_l = own_arr.tolist()
        A, B, O, OS = self.A, self.B, self.O, self.owner_slots
        for r, a, b, o in zip(rid_l, a_l, b_l, o_l):
            slot = len(A[r])
            A[r].append(a); B[r].append(b); O[r].append(o)
            lst = OS[o]
            if lst is None:
                OS[o] = [(r, slot)]
            else:
                lst.append((r, slot))
        # bucket counts
        for r in set(rid_l):
            self.n[r] = len(A[r])

    def insert_lists(self, rid_l, a_l, b_l) -> None:
        """Owner rule inline: impingement rids (6-11, 34, 35) are owned by b, else by a."""
        if not rid_l:
            return
        A, B, O, OS = self.A, self.B, self.O, self.owner_slots
        touched = set()
        for r, a, b in zip(rid_l, a_l, b_l):
            o = b if _IS_ADS_RID[r] else a
            slot = len(A[r])
            A[r].append(a); B[r].append(b); O[r].append(o)
            lst = OS[o]
            if lst is None:
                OS[o] = [(r, slot)]
            else:
                lst.append((r, slot))
            touched.add(r)
        for r in touched:
            self.n[r] = len(A[r])

    # -- removal ------------------------------------------------------------
    def remove_owned_by(self, sites) -> None:
        A, B, O, OS = self.A, self.B, self.O, self.owner_slots
        touched = set()
        for o in sites:
            lst = OS[o]
            if not lst:
                continue
            while lst:
                r, s = lst.pop()
                Ar, Br, Or = A[r], B[r], O[r]
                last = len(Ar) - 1
                if s != last:
                    la, lb, lo = Ar[last], Br[last], Or[last]
                    Ar[s] = la; Br[s] = lb; Or[s] = lo
                    L = OS[lo]
                    L[L.index((r, last))] = (r, s)
                Ar.pop(); Br.pop(); Or.pop()
                touched.add(r)
        for r in touched:
            self.n[r] = len(A[r])

    # -- selection (pure Python over the live buckets) ---------------------
    def select_py(self, kbase, u: float):
        """kbase[r] = per-event rate of rid r. O(#live rids), no numpy. One uniform
        u picks both the bucket and the slot inside it."""
        A = self.A
        live = []            # (r, bucket_total)
        R = 0.0
        for r in range(self.n_rid):
            nr = len(A[r])
            if nr:
                kr = kbase[r]
                t = nr * kr
                live.append((r, kr, t))
                R += t
        if R <= 0.0:
            return None
        x = u * R
        acc = 0.0
        for r, kr, t in live:
            if x < acc + t:
                slot = int((x - acc) / kr)
                nr = len(A[r])
                if slot >= nr: slot = nr - 1
                return r, A[r][slot], self.B[r][slot], R
            acc += t
        r, kr, t = live[-1]                       # round-off at the very top
        return r, A[r][-1], self.B[r][-1], R

    def select(self, krate: np.ndarray, u: float):
        """krate[r] = rate of ONE event of rid r.
        u ~ U[0,1). Returns (rid, a, b, R_total) or None if nothing can fire."""
        tot_r = self.n * krate
        R = float(tot_r.sum())
        if R <= 0.0:
            return None
        x = u * R
        cum = np.cumsum(tot_r)
        r = int(np.searchsorted(cum, x, side="right"))
        if r >= self.n_rid:
            r = self.n_rid - 1
        # guard against landing in an empty bucket by round-off: walk to a live one
        while self.n[r] == 0:
            r -= 1
            if r < 0:
                r = int(np.nonzero(self.n)[0][-1])
                break
        prev = float(cum[r - 1]) if r > 0 else 0.0
        frac = (x - prev) / float(krate[r]) if krate[r] > 0 else 0.0
        slot = int(frac)
        nr = int(self.n[r])
        if slot >= nr:
            slot = nr - 1
        if slot < 0:
            slot = 0
        return r, self.A[r][slot], self.B[r][slot], R

    # -- introspection ------------------------------------------------------
    @property
    def size(self) -> int:
        return int(self.n.sum())

    def as_set(self) -> set:
        out = set()
        for r in range(self.n_rid):
            out.update(zip([r] * len(self.A[r]), self.A[r], self.B[r]))
        return out


def _per_rid_rates(RATE0: np.ndarray, base_AsH: float, base_MMGa: float) -> np.ndarray:
    """Rate of one event of each rid. Mirrors _compute_rates_fast: RATE0 already holds
    the adsorption rates (terrace IDs with S0_*_terrace, step IDs with S0_*_step, split
    IDs 34, 35 at half), so it is used as is."""
    return RATE0.copy()


def run_kmc_trajectory(target_T, params=None, node_label="node",
                       traj_freq=None, lattice_file="Lattice.xyz",
                       log_events=False, target_growth=150.0, max_iter=None,
                       checkpoint_every=None, checkpoint_sink=None,
                       stop_mean_height=0.0,
                       stop_plateau_idle_factor=0.0, stop_plateau_min_events=0,
                       stop_plateau_rel_tol=0.0, stop_plateau_abs_tol=0.0,
                       _profile=None):
    """Same signature and semantics as KMC.run_kmc_trajectory.

    `_profile`: optional dict; if given, per-section wall time is accumulated into
    it (keys: select, exec, dirty, remove, reenum, insert, resync) plus
    'menu' (summed menu size) and 'steps'.
    """
    TARGET_GROWTH_A = float(target_growth)
    # Two stops, both checked at checkpoints, ported verbatim from KMC.py so the two
    # engines stop on the same rules: the mean height reaching stop_mean_height (the
    # growth-rate window's top) and the stall rule (stop_plateau_*: the mean height has
    # not advanced for max(min_events, idle_factor x events at its last advance)).
    _STOP_MEAN_H = float(stop_mean_height or 0.0)
    _PLAT_FAC  = float(stop_plateau_idle_factor or 0.0)
    _PLAT_MIN  = int(stop_plateau_min_events or 0)
    _PLAT_REL  = float(stop_plateau_rel_tol or 0.0)
    _PLAT_ABS  = float(stop_plateau_abs_tol or 0.0)
    _plat_best = None
    _plat_since = 0
    GROWTH_CHECK = 8
    PA, PM = K._P_ASH, K._P_MMGA
    A_param = K._A_SITE
    k_B = K._K_B
    M_ash = K._M_ASH            # AsH, 75.93 u
    M_mmga = K._M_MMGA          # GaCH3, 84.76 u

    atoms_init, origin, unit_cell = extract_coords_with_species_labels(lattice_file)
    atoms_init = label_lattice_sites(atoms_init, bottom_layers=3)
    st = LatticeState(atoms_init, origin, unit_cell)
    _initialize_bonding(st)
    _purge(st)

    z_init = _growth_front_z(st)
    elapsed = 0.0
    growth = 0.0

    write_traj = (traj_freq is not None) and (traj_freq > 0)
    f_traj = None
    traj_name = f"KMC_Traj_{node_label}_T{int(target_T)}K.xyz"
    if write_traj:
        f_traj = open(traj_name, "w")
        _append_xyz_frame(f_traj, st.coords, st.species, st.origin, st.unit_cell,
                          f"Frame=0 Time=0.0 T={target_T}")
        f_traj.flush()

    imp_AsH = PA / math.sqrt(2.0 * math.pi * M_ash * k_B * float(target_T))
    imp_MMGa = PM / math.sqrt(2.0 * math.pi * M_mmga * k_B * float(target_T))
    rng = np.random

    event_counts = np.zeros(_NUM_RID, dtype=np.int64)
    total_fired = 0

    RATE0, base_AsH, base_MMGa = _precompute_rate_tables(
        float(target_T), params, imp_AsH, imp_MMGa, A_param)

    iteration = 0
    _prog_every = int(os.environ.get("KMC_PROG_EVERY", "0"))
    _prog_wall0 = _time.perf_counter()

    adj = _incr_adjacency(st)
    closure = ClosureCache(adj, KHOP)
    zc = st.coords[:, 2]

    # Exact O(1) growth front: count REAL atoms per discrete z level, keep a pointer
    # to the highest non-empty level. Handles the top-of-film desorption case that
    # the production engine only repairs at its 256-step resync.
    zlev_vals, zlev = np.unique(zc, return_inverse=True)   # planes are exact floats here
    zlev = zlev.astype(np.int64)
    zlev_l = zlev.tolist(); zlev_vals_l = zlev_vals.tolist()
    zcnt = np.bincount(zlev[_IS_REAL[st.species]], minlength=zlev_vals.size).tolist()
    top = zlev_vals.size - 1
    while top >= 0 and zcnt[top] == 0:
        top -= 1
    front = zlev_vals_l[top] if top >= 0 else -np.inf
    assert abs(front - z_init) < 1e-6, (front, z_init)
    RESYNC = int(os.environ.get("KMC_BUCKET_RESYNC", "0"))   # 0 = off (front proven exact)

    r_band = np.where(_IS_REACTIVE[st.species])[0]
    closure.prefill(r_band)
    menu = BucketMenu(_NUM_RID, st.N)
    # per-event rate of each rid
    kbase = _per_rid_rates(RATE0.astype(float), base_AsH, base_MMGa).tolist()
    r0 = np.where(_IS_REACTIVE[st.species])[0]
    rid0, a0, b0 = _enumerate_events(st, r0)
    if rid0.size:
        rid0 = rid0.astype(np.int64); a0 = a0.astype(np.int64); b0 = b0.astype(np.int64)
        own0 = np.where(_IS_ADS_RID[rid0], b0, a0)
        menu.insert_arrays(rid0, a0, b0, own0)

    _do_ckpt = bool(checkpoint_every) and checkpoint_every > 0 and checkpoint_sink is not None
    _col_id, _n_cols = (_column_ids(st) if _do_ckpt else (None, 0))
    _z_init_mean = (_mean_height_z(st, _col_id, _n_cols) if _do_ckpt else 0.0)

    P = _profile is not None
    if P:
        for k in ("select", "exec", "dirty", "remove", "reenum", "insert", "resync", "menu", "steps"):
            _profile.setdefault(k, 0.0)
    C = _time.perf_counter

    try:
        while growth < TARGET_GROWTH_A and (max_iter is None or iteration < max_iter):
            if menu.size == 0:
                iteration += 1
                continue
            if P:
                _profile["menu"] += menu.size; _profile["steps"] += 1

            s_ = C() if P else 0.0
            picked = menu.select_py(kbase, float(rng.rand()))
            if P: _profile["select"] += C() - s_
            if picked is None:
                iteration += 1
                continue
            fired_rid, _fa, _fb, total = picked

            s_ = C() if P else 0.0
            sp = st.species
            if _IS_REAL[sp[_fa]]: zcnt[zlev_l[_fa]] -= 1
            if _fb >= 0 and _fb != _fa:
                if _IS_REAL[sp[_fb]]: zcnt[zlev_l[_fb]] -= 1

            K._TOUCHED.clear(); K._VACATED.clear()
            _execute_event(st, fired_rid, _fa, _fb)
            for _j in K._VACATED:              # FIX v2: a third site emptied
                zcnt[zlev_l[_j]] -= 1

            if _IS_REAL[sp[_fa]]:
                la = zlev_l[_fa]; zcnt[la] += 1
                if la > top: top = la
            if _fb >= 0 and _fb != _fa:
                if _IS_REAL[sp[_fb]]:
                    lb_ = zlev_l[_fb]; zcnt[lb_] += 1
                    if lb_ > top: top = lb_
            while top >= 0 and zcnt[top] == 0:
                top -= 1
            front = zlev_vals_l[top]
            growth = front - z_init

            event_counts[fired_rid] += 1
            total_fired += 1
            elapsed += -math.log(rng.rand()) / total
            iteration += 1
            if P: _profile["exec"] += C() - s_

            # -- incremental menu repair, O(neighbourhood) ------------------
            s_ = C() if P else 0.0
            ca = closure.get(_fa)
            if _fb >= 0 and _fb != _fa:
                cb = closure.get(_fb)
                dirty = ca[0] | cb[0]
                da = np.fromiter(dirty, np.int64, len(dirty))
            else:
                dirty, da = ca
            if K._TOUCHED:                     # FIX v2: re-enumerate around them too
                dirty = set(dirty)
                for _j in set(K._TOUCHED):
                    dirty |= closure.get(_j)[0]
                da = np.fromiter(dirty, np.int64, len(dirty))
            if P: _profile["dirty"] += C() - s_

            s_ = C() if P else 0.0
            menu.remove_owned_by(dirty)
            if P: _profile["remove"] += C() - s_

            s_ = C() if P else 0.0
            rd = da[_IS_REACTIVE[st.species[da]]]
            if SCALAR:
                if rd.size:
                    lr, la, lb = enumerate_scalar(st, rd)
                else:
                    lr = la = lb = []
                if P: _profile["reenum"] += C() - s_
                s_ = C() if P else 0.0
                menu.insert_lists(lr, la, lb)
                if P: _profile["insert"] += C() - s_
            else:
                if rd.size:
                    nr, na, nb = _enumerate_events(st, rd)
                else:
                    nr = np.empty(0, np.int64); na = nb = nr
                if P: _profile["reenum"] += C() - s_
                s_ = C() if P else 0.0
                if nr.size:
                    nr = nr.astype(np.int64); na = na.astype(np.int64); nb = nb.astype(np.int64)
                    no = np.where(_IS_ADS_RID[nr], nb, na)
                    menu.insert_arrays(nr, na, nb, no)
                if P: _profile["insert"] += C() - s_

            if CHECK_STEPS and CHECK_FROM < iteration <= CHECK_FROM + CHECK_STEPS:
                full = _full_menu_set(st); got = menu.as_set()
                if full != got:
                    miss = sorted(full - got)[:5]; extra = sorted(got - full)[:5]
                    raise AssertionError(
                        f"Gate 1 FAILED at step {iteration}: menu has {len(got)} events, "
                        f"full enumeration {len(full)}. missing={miss} extra={extra}")
                # the TOTAL RATE must match the production rate function on the full
                # enumeration too: identical menus with a wrong per-rid rate would pass
                # the set test and still bias every selection.
                _rf = np.fromiter((r for r, _, _ in full), np.int64, len(full))
                _Rf = float(K._compute_rates_fast(_rf, RATE0, base_AsH, base_MMGa).sum()) if _rf.size else 0.0
                _Rb = 0.0
                for _r in range(_NUM_RID):
                    _nr = len(menu.A[_r])
                    if _nr:
                        _Rb += _nr * kbase[_r]
                _dev = abs(_Rf - _Rb) / max(1e-300, abs(_Rf))
                CHECK_RATE_DEV.append(_dev)
                if _dev > 1e-9:
                    raise AssertionError(f"Gate 1 FAILED at step {iteration}: total rate "
                                         f"bucket {_Rb!r} vs production on full menu {_Rf!r}")
                # E4.1/E4.2: one site's rate per dangling bond (a D_MMGa's two landing
                # sites share one), checked on the full menu with the production rates.
                _bad = K.landing_rate_errors(st, full, kbase)
                if _bad:
                    raise AssertionError(f"Gate 1 FAILED at step {iteration}: AsH landing "
                                         f"rate per host wrong (host, species, want, got) "
                                         f"{_bad[:3]}")
                CHECK_LOG.append((iteration, len(full)))
                CHECK_RIDS.update(r for r, _, _ in full)
                CHECK_FIRED.add(fired_rid)

            s_ = C() if P else 0.0
            if RESYNC and iteration % RESYNC == 0:
                # debug cross-check only: the front is maintained exactly above
                assert abs(_growth_front_z(st) - front) < 1e-6, ("front drifted", _growth_front_z(st), front)
            if P: _profile["resync"] += C() - s_

            if _do_ckpt and iteration % checkpoint_every == 0:
                _cp_max_h = _growth_front_z(st) - z_init
                _cp_mean_h = _mean_height_z(st, _col_id, _n_cols) - _z_init_mean
                checkpoint_sink.append((int(iteration), float(elapsed),
                                        float(_cp_max_h), float(_cp_mean_h)))
                if _STOP_MEAN_H > 0.0 and _cp_mean_h >= _STOP_MEAN_H:
                    break
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
                _pw = _time.perf_counter() - _prog_wall0
                print(f"  events={iteration:>11,}   thickness={growth:7.2f} A = {growth/10:5.2f} nm"
                      f"   wall={_pw:8.1f} s   menu={menu.size}", flush=True)

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
            _write_event_log(f"events_{node_label}_T{int(target_T)}K.csv",
                             event_counts, total_fired, target_T, node_label)

    growth = _growth_front_z(st) - z_init
    t_min = elapsed / 60.0
    gr = (growth * 1e-4) / t_min if t_min > 0 else 0.0

    if _do_ckpt and ((not checkpoint_sink) or checkpoint_sink[-1][0] != iteration):
        _fin_mean_h = _mean_height_z(st, _col_id, _n_cols) - z_init
        checkpoint_sink.append((int(iteration), float(elapsed),
                                float(growth), float(_fin_mean_h)))

    global _LAST_ITERATIONS, _LAST_THICKNESS_A, _LAST_STATE, _LAST_MENU
    global _LAST_EVENT_COUNTS, _LAST_TOTAL_FIRED
    _LAST_ITERATIONS = iteration
    _LAST_THICKNESS_A = float(growth)
    _LAST_STATE = st
    _LAST_MENU = menu
    # per-engine-ID counts, read by the MCTS / worker logs (n_rid_XX and n_R<k> columns);
    # those logs expected these names but no engine set them, so the columns were blank
    _LAST_EVENT_COUNTS = event_counts.copy()
    _LAST_TOTAL_FIRED = int(total_fired)
    return gr


_LAST_ITERATIONS = 0
_LAST_THICKNESS_A = 0.0
_LAST_STATE = None
_LAST_MENU = None
_LAST_EVENT_COUNTS = None
_LAST_TOTAL_FIRED = None

# Compiled kernels (nb_bucket.py, needs numba) swap the compiled event menu and scalar
# enumerator in for this process, 4.6x faster per event. Films are bit-identical for the
# same seed (tests/bench_numba.py); same events in the same order as enum_scalar
# (tests/walk_full.py with WALK_NUMBA=1). ON whenever numba imports (2026-09-27):
#   KMC_NUMBA unset / "auto"  numba if it imports, else the Python path
#   KMC_NUMBA=1               numba, and fail if it does not import
#   KMC_NUMBA=0               the Python path
# NUMBA_ON records which one this process runs; run_kmc.py prints it.
NUMBA_ON = False
_nb_mode = os.environ.get("KMC_NUMBA", "auto").strip().lower()
if _nb_mode not in ("0", "1", "auto", ""):
    raise ValueError(f"KMC_NUMBA must be 0, 1 or auto, got {_nb_mode!r}")
if _nb_mode != "0" and SCALAR:
    try:
        import nb_bucket as _nb
    except ImportError:
        if _nb_mode == "1":
            raise
    else:
        BucketMenu = _nb.NBMenu
        enumerate_scalar = _nb.enumerate_scalar
        NUMBA_ON = True
