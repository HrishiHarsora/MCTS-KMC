"""nb_bucket.py -- compiled (numba) kernels for KMC_bucket, 4.6x faster per event. KMC_bucket
uses them whenever numba imports (KMC_NUMBA=0 turns them off, KMC_NUMBA=1 makes them
required); the engine runs unchanged without them.

Written to be BIT-IDENTICAL to KMC_bucket for the same seed: the same enumeration and
emission order as enum_scalar.enumerate_scalar, the same swap-with-last removal and
owner-index bookkeeping as BucketMenu, and the same floating-point operations, in the
same order, as BucketMenu.select_py. tests/bench_numba.py and tests/walk_full.py (with
WALK_NUMBA=1) check it: identical final film (species + bonds) and checkpoint trace, and the
same events in the same order on every dirty set.

Covers every engine ID 1-77: the DDD_Ga route (IDs 36-43), incorporation onto an adsorbed
precursor (44-59), film atoms with two / three bonds desorbing (60-65) and a film atom's free
bond bonding down (66-77). use_numba() swaps the kernels in by hand.
"""
import numpy as np
from numba import njit

import KMC as K

V_AS, V_GA, S_AS, S_GA = K.V_AS, K.V_GA, K.S_AS, K.S_GA
DD_GA, D_GA, GA_ = K.DD_GA, K.D_GA, K.GA_
MMGA, DD_MMGA, D_MMGA = K.MMGA, K.DD_MMGA, K.D_MMGA
DD_AS, D_AS, AS_ = K.DD_AS, K.D_AS, K.AS_
DD_ASH, D_ASH, ASH, DDD_AS = K.DD_ASH, K.D_ASH, K.ASH, K.DDD_AS
DDD_GA = K.DDD_GA
E42, E62 = K._RID_E42_SPLIT, K._RID_E62_SPLIT
PP_STEP = K._PP_STEP
IS_ADS = K._IS_ADS_RID.astype(np.bool_)
IS_REAL = K._IS_REAL.astype(np.bool_)
PP = K._PP_TABLE.astype(np.int64)                    # [upper, lower] -> terrace ID (0: none)
FB_T = np.zeros(K._N_SPECIES, np.int64); FB_S = np.zeros(K._N_SPECIES, np.int64)
for _sp, (_t, _s) in K._FILM_BOND.items():
    FB_T[_sp] = _t; FB_S[_sp] = _s
DES2 = np.zeros(K._N_SPECIES, np.int64); DES3 = np.zeros(K._N_SPECIES, np.int64)
for _sp, _r in K._DES_TWO.items():
    DES2[_sp] = _r
for _sp, _r in K._DES_THREE.items():
    DES3[_sp] = _r


# ----------------------------------------------------------------------------
# enum_scalar.enumerate_scalar, same rules, same emission order
# ----------------------------------------------------------------------------
@njit(cache=True)
def _upper_step(i, nbr, src, a1, a2, s, bonds, n_bonds):
    """enum_scalar._upper_step: atom i bonded to an upper neighbour (nbr[i, 2:4]) other than
    src that is not an adsorbate (a1 / a2)?"""
    for k in range(2, 4):
        u = nbr[i, k]
        if u >= 0 and u != src and s[u] != a1 and s[u] != a2:
            for kk in range(n_bonds[i]):
                if bonds[i, kk] == u:
                    return True
    return False


@njit(cache=True)
def _host_ok(h, ga_precursor, s, n_bonds, has_entry):
    """enum_scalar._host_ok."""
    x = s[h]
    if ga_precursor:
        return x == D_AS or x == ASH or (x == AS_ and has_entry[h] and n_bonds[h] >= 2)
    return x == D_GA or x == MMGA or (x == GA_ and has_entry[h] and n_bonds[h] >= 2)


@njit(cache=True)
def _in_row(j, i, bonds, n_bonds):
    for kk in range(n_bonds[i]):
        if bonds[i, kk] == j:
            return True
    return False


@njit(cache=True)
def _on_substrate(i, s, bonds, n_bonds):
    for kk in range(n_bonds[i]):
        j = bonds[i, kk]
        if j >= 0 and (s[j] == S_AS or s[j] == S_GA):
            return True
    return False


@njit(cache=True)
def _enum_scalar(sites, s, nbr_As, nbr_Ga, bonds, n_bonds, has_entry, DB,
                 is_real, pp, fb_t, fb_s, des2, des3):
    n = sites.size
    cap = 16 * n + 16
    R = np.empty(cap, np.int64); A = np.empty(cap, np.int64); B = np.empty(cap, np.int64)
    c = 0
    for i in range(n):
        src = sites[i]; x = s[src]
        # ---- 9, 10, 11, 35 : MMGa adsorption
        if x == D_AS:
            v3 = nbr_As[src, 2]; v4 = nbr_As[src, 3]
            ok3 = v3 >= 0; ok4 = v4 >= 0
            s3 = s[v3] if ok3 else -1; s4 = s[v4] if ok4 else -1
            if ok3 and s3 == V_GA:
                R[c] = 10 if (ok4 and (s4 == DD_MMGA or s4 == DDD_GA)) else 11; A[c] = v3; B[c] = src; c += 1
            if ok4 and s4 == V_GA:
                R[c] = 10 if (ok3 and (s3 == DD_MMGA or s3 == DDD_GA)) else 11; A[c] = v4; B[c] = src; c += 1
        elif x == D_ASH or x == DD_AS:
            v3 = nbr_As[src, 2]; v4 = nbr_As[src, 3]
            f3 = v3 >= 0 and s[v3] == V_GA
            f4 = v4 >= 0 and s[v4] == V_GA
            r = E62 if (x == D_ASH and f3 and f4) else 9
            if f3:
                R[c] = r; A[c] = v3; B[c] = src; c += 1
            if f4:
                R[c] = r; A[c] = v4; B[c] = src; c += 1
        # ---- 14-19, 38-43, 44/45/50/51 (+8) : Ga precursor incorporation
        if x == DD_MMGA or x == DDD_GA:
            base_t = 14 if x == DD_MMGA else 38
            base_s = 17 if x == DD_MMGA else 41
            ba = bonds[src, 0]
            if ba >= 0:
                asp = s[ba]
                if (asp == D_AS) or (asp == ASH) or (asp == AS_ and has_entry[ba] and n_bonds[ba] >= 2):
                    p = -1; ps = -1
                    for k in range(2):
                        v = nbr_Ga[src, k]
                        if v < 0:
                            continue
                        sv = s[v]
                        if sv == DD_AS or sv == D_ASH or (sv == D_AS and v != ba):
                            p = v; ps = sv
                            break
                        if (sv == DD_ASH or sv == DDD_AS) and v != ba:
                            hl = bonds[v, 0]
                            if hl >= 0 and hl != src and _host_ok(hl, False, s, n_bonds, has_entry):
                                p = v; ps = sv
                                break
                    if p >= 0:
                        step = (_upper_step(ba, nbr_As, src, DD_MMGA, DDD_GA, s, bonds, n_bonds)
                                or _upper_step(p, nbr_As, src, DD_MMGA, DDD_GA, s, bonds, n_bonds))
                        if ps == DD_ASH or ps == DDD_AS:
                            R[c] = pp[x, ps] + (PP_STEP if step else 0)
                        else:
                            off = 0 if ps == DD_AS else (1 if ps == D_ASH else 2)
                            R[c] = (base_s if step else base_t) + off
                        A[c] = src; B[c] = p; c += 1
        # ---- 36, 13 : methyl off DD_MMGa; MMGa desorption
        if x == DD_MMGA:
            R[c] = 36; A[c] = src; B[c] = -1; c += 1
            if has_entry[src] and n_bonds[src] >= 1:
                b = bonds[src, 0]
                if b >= 0:
                    R[c] = 13; A[c] = src; B[c] = b; c += 1
        # ---- 6, 7, 8, 34 : AsH adsorption
        if x == DD_GA or x == D_MMGA:
            v3 = nbr_Ga[src, 2]; v4 = nbr_Ga[src, 3]
            f3 = v3 >= 0 and s[v3] == V_AS
            f4 = v4 >= 0 and s[v4] == V_AS
            r = E42 if (x == D_MMGA and f3 and f4) else 6
            if f3:
                R[c] = r; A[c] = v3; B[c] = src; c += 1
            if f4:
                R[c] = r; A[c] = v4; B[c] = src; c += 1
        elif x == D_GA:
            v3 = nbr_Ga[src, 2]; v4 = nbr_Ga[src, 3]
            ok3 = v3 >= 0; ok4 = v4 >= 0
            s3 = s[v3] if ok3 else -1; s4 = s[v4] if ok4 else -1
            po3 = ok4 and (s4 == DD_ASH or s4 == DDD_AS)
            po4 = ok3 and (s3 == DD_ASH or s3 == DDD_AS)
            if ok3 and s3 == V_AS:
                R[c] = 7 if po3 else 8; A[c] = v3; B[c] = src; c += 1
            if ok4 and s4 == V_AS:
                R[c] = 7 if po4 else 8; A[c] = v4; B[c] = src; c += 1
        # ---- 20-31, 46-49 (+8) : As precursor incorporation; 1, 2 : its desorption
        if x == DD_ASH or x == DDD_AS:
            bg = bonds[src, 0]
            if bg >= 0:
                gsp = s[bg]
                base_t = 26 if x == DD_ASH else 29
                base_s = 20 if x == DD_ASH else 23
                if (gsp == D_GA) or (gsp == MMGA) or (gsp == GA_ and has_entry[bg] and n_bonds[bg] >= 2):
                    p = -1; ps = -1
                    for k in range(2):
                        v = nbr_As[src, k]
                        if v < 0:
                            continue
                        sv = s[v]
                        if sv == DD_GA or sv == D_MMGA or (sv == D_GA and v != bg):
                            p = v; ps = sv
                            break
                        if (sv == DD_MMGA or sv == DDD_GA) and v != bg:
                            hl = bonds[v, 0]
                            if hl >= 0 and hl != src and _host_ok(hl, True, s, n_bonds, has_entry):
                                p = v; ps = sv
                                break
                    if p >= 0:
                        step = (_upper_step(bg, nbr_Ga, src, DD_ASH, DDD_AS, s, bonds, n_bonds)
                                or _upper_step(p, nbr_Ga, src, DD_ASH, DDD_AS, s, bonds, n_bonds))
                        if ps == DD_MMGA or ps == DDD_GA:
                            R[c] = pp[x, ps] + (PP_STEP if step else 0)
                        else:
                            off = 0 if ps == DD_GA else (1 if ps == D_MMGA else 2)
                            R[c] = (base_s if step else base_t) + off
                        A[c] = src; B[c] = p; c += 1
            if has_entry[src] and n_bonds[src] >= 1:
                lb = bonds[src, n_bonds[src] - 1]
                if lb >= 0:
                    R[c] = 1 if x == DD_ASH else 2; A[c] = src; B[c] = lb; c += 1
        # ---- 3, 4, 5 : H abstraction
        if x == DD_ASH:
            R[c] = 3; A[c] = src; B[c] = -1; c += 1
        elif x == D_ASH:
            R[c] = 4; A[c] = src; B[c] = -1; c += 1
        elif x == ASH:
            R[c] = 5; A[c] = src; B[c] = -1; c += 1
        # ---- 12 : methyl desorption
        if x == D_MMGA or x == MMGA:
            R[c] = 12; A[c] = src; B[c] = -1; c += 1
        # ---- 32, 33, 37 : Ga desorption, not if bonded to the substrate
        if x == DD_GA or x == D_GA or x == DDD_GA:
            if not _on_substrate(src, s, bonds, n_bonds):
                if x == DD_GA:
                    if has_entry[src] and n_bonds[src] >= 1:
                        b = bonds[src, 0]
                        if b >= 0:
                            R[c] = 32; A[c] = src; B[c] = b; c += 1
                elif x == D_GA:
                    if has_entry[src] and n_bonds[src] >= 3:
                        lb = bonds[src, n_bonds[src] - 1]
                        if lb >= 0:
                            R[c] = 33; A[c] = src; B[c] = lb; c += 1
                else:
                    if has_entry[src] and n_bonds[src] >= 1:
                        lb = bonds[src, n_bonds[src] - 1]
                        if lb >= 0:
                            R[c] = 37; A[c] = src; B[c] = lb; c += 1
        # ---- 66-77 : a film atom's free bond bonds down
        if fb_t[x] > 0:
            ga = x == D_MMGA or x == DD_GA or x == D_GA
            for k in range(2):
                L = nbr_Ga[src, k] if ga else nbr_As[src, k]
                if L < 0 or _in_row(L, src, bonds, n_bonds) or not is_real[s[L]] or DB[L] <= 0:
                    continue
                if ga:
                    step = _upper_step(L, nbr_As, src, DD_MMGA, DDD_GA, s, bonds, n_bonds)
                else:
                    step = _upper_step(L, nbr_Ga, src, DD_ASH, DDD_AS, s, bonds, n_bonds)
                if not step:
                    for kk in range(2):
                        h = nbr_Ga[src, kk] if ga else nbr_As[src, kk]
                        if h >= 0 and h != L and _in_row(h, src, bonds, n_bonds):
                            if ga:
                                st_ = _upper_step(h, nbr_As, src, DD_MMGA, DDD_GA, s, bonds, n_bonds)
                            else:
                                st_ = _upper_step(h, nbr_Ga, src, DD_ASH, DDD_AS, s, bonds, n_bonds)
                            if st_:
                                step = True
                                break
                R[c] = fb_s[x] if step else fb_t[x]; A[c] = src; B[c] = L; c += 1
        # ---- 60-65 : other film atoms with two / three bonds desorb
        if (des2[x] > 0 or des3[x] > 0) and not _on_substrate(src, s, bonds, n_bonds):
            if des2[x] > 0:
                if has_entry[src] and n_bonds[src] >= 1:
                    b = bonds[src, 0]
                    if b >= 0:
                        R[c] = des2[x]; A[c] = src; B[c] = b; c += 1
            elif has_entry[src] and n_bonds[src] >= 3:
                lb = bonds[src, n_bonds[src] - 1]
                if lb >= 0:
                    R[c] = des3[x]; A[c] = src; B[c] = lb; c += 1
    return R[:c], A[:c], B[:c]


def enumerate_scalar(st, sites):
    return _enum_scalar(np.asarray(sites, np.int64), st.species, st.nbr_As, st.nbr_Ga,
                        st.bonds, st.n_bonds, st.has_entry, st.DB,
                        IS_REAL, PP, FB_T, FB_S, DES2, DES3)


# ----------------------------------------------------------------------------
# BucketMenu, same slot semantics
# ----------------------------------------------------------------------------
@njit(cache=True)
def _insert(rid, a, b, is_ads, A, B, O, n, OSr, OSs, OSn):
    for i in range(rid.size):
        r = rid[i]; aa = a[i]; bb = b[i]
        o = bb if is_ads[r] else aa
        slot = n[r]
        if slot >= A.shape[1] or OSn[o] >= OSr.shape[1]:
            return -1
        A[r, slot] = aa; B[r, slot] = bb; O[r, slot] = o
        n[r] = slot + 1
        k = OSn[o]; OSr[o, k] = r; OSs[o, k] = slot; OSn[o] = k + 1
    return 0


@njit(cache=True)
def _remove(sites, A, B, O, n, OSr, OSs, OSn):
    for t in range(sites.size):
        o = sites[t]
        while OSn[o] > 0:
            k = OSn[o] - 1; r = OSr[o, k]; s = OSs[o, k]; OSn[o] = k      # lst.pop()
            last = n[r] - 1
            if s != last:
                la = A[r, last]; lb = B[r, last]; lo = O[r, last]
                A[r, s] = la; B[r, s] = lb; O[r, s] = lo
                for j in range(OSn[lo]):                                   # L.index((r, last))
                    if OSr[lo, j] == r and OSs[lo, j] == last:
                        OSs[lo, j] = s
                        break
            n[r] = last                                                    # pop()


@njit(cache=True)
def _select(n, kbase, u):
    """The same float operations, in the same order, as BucketMenu.select_py."""
    R = 0.0
    for r in range(n.size):
        nr = n[r]
        if nr:
            R += nr * kbase[r]
    if R <= 0.0:
        return -1, -1, R
    x = u * R
    acc = 0.0
    lastr = -1
    for r in range(n.size):
        nr = n[r]
        if nr:
            kr = kbase[r]
            t = nr * kr
            lastr = r
            if x < acc + t:
                slot = int((x - acc) / kr)
                if slot >= nr:
                    slot = nr - 1
                return r, slot, R
            acc += t
    return lastr, n[lastr] - 1, R


class NBMenu:
    """Drop-in for KMC_bucket.BucketMenu backed by fixed-size arrays."""
    CAP = 1 << 15          # events per engine-ID bucket
    OCAP = 16              # events owned by one site

    def __init__(self, n_rid, n_sites):
        assert n_rid <= 127, "owner index stores the engine ID as int8"
        self.n_rid = n_rid
        self.An = np.empty((n_rid, self.CAP), np.int32)
        self.Bn = np.empty((n_rid, self.CAP), np.int32)
        self.On = np.empty((n_rid, self.CAP), np.int32)
        self.n = np.zeros(n_rid, np.int64)
        self.OSr = np.empty((n_sites, self.OCAP), np.int8)
        self.OSs = np.empty((n_sites, self.OCAP), np.int32)
        self.OSn = np.zeros(n_sites, np.int8)
        self._kb = None

    def _ins(self, rid, a, b):
        if rid.size and _insert(rid, a, b, IS_ADS, self.An, self.Bn, self.On, self.n,
                                self.OSr, self.OSs, self.OSn) != 0:
            raise RuntimeError("NBMenu capacity exceeded")

    def insert_arrays(self, rid_arr, a_arr, b_arr, own_arr):
        self._ins(np.asarray(rid_arr, np.int64), np.asarray(a_arr, np.int64), np.asarray(b_arr, np.int64))

    def insert_lists(self, rid_l, a_l, b_l):
        self._ins(np.asarray(rid_l, np.int64), np.asarray(a_l, np.int64), np.asarray(b_l, np.int64))

    def remove_owned_by(self, sites):
        # iteration order of the set object == BucketMenu's `for o in sites`
        arr = np.fromiter(sites, np.int64, len(sites))
        _remove(arr, self.An, self.Bn, self.On, self.n, self.OSr, self.OSs, self.OSn)

    def select_py(self, kbase, u):
        if self._kb is None or self._kb[0] is not kbase:
            self._kb = (kbase, np.asarray(kbase, np.float64))
        r, slot, R = _select(self.n, self._kb[1], float(u))
        if r < 0:
            return None
        return int(r), int(self.An[r, slot]), int(self.Bn[r, slot]), float(R)

    @property
    def size(self):
        return int(self.n.sum())

    @property
    def A(self):
        return [self.An[r, :self.n[r]].tolist() for r in range(self.n_rid)]

    @property
    def B(self):
        return [self.Bn[r, :self.n[r]].tolist() for r in range(self.n_rid)]

    def as_set(self):
        out = set()
        for r in range(self.n_rid):
            k = int(self.n[r])
            out.update(zip([r] * k, self.An[r, :k].tolist(), self.Bn[r, :k].tolist()))
        return out


def use_numba():
    """Swap the compiled menu and scalar enumerator into KMC_bucket (this process)."""
    import KMC_bucket
    KMC_bucket.BucketMenu = NBMenu
    KMC_bucket.enumerate_scalar = enumerate_scalar
