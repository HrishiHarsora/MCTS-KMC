"""
Scalar re-enumeration for SMALL site sets.

KMC._enumerate_events is vectorised: ~160 numpy calls per invocation. On the whole
reactive band (hundreds to thousands of sites) that is the right shape. On the
per-step dirty set (~41 sites) it costs ~275 us of pure call overhead to produce
~80 events. This module produces the SAME events for a small site set with a
dozen batched gathers followed by plain Python over lists.

It is a second implementation of the same chemistry and is therefore validated
the hard way: Gate 1 compares its output, as a set, against KMC._enumerate_events
at EVERY step (see KMC_bucket.CHECK_STEPS). Any divergence raises.

Returns three Python lists (rid, a, b). The owner of every event is its source
site `src` (for impingement rids 6-11, 34 and 35 the vectorised code tags b, and b == src).

Covers every engine ID 1-77, including 36-43 (DDD_Ga route), 44-59 (partner an adsorbed
precursor), 60-65 (other film atoms with two / three bonds desorb at R10 / R11's rate) and
66-77 (a film atom's free bond bonds down; no bond forms automatically).
"""
from __future__ import annotations

import numpy as np
from KMC import (V_AS, V_GA, S_AS, S_GA, DD_GA, D_GA, GA_, MMGA, DD_MMGA, D_MMGA,
                 DD_AS, D_AS, AS_, DD_ASH, D_ASH, ASH, DDD_AS, DDD_GA, _PP_RID, _PP_STEP,
                 _DES_TWO, _DES_THREE, _FILM_BOND, _IS_REAL)

_ASL = (DD_ASH, DDD_AS)
from KMC import _RID_E42_SPLIT as _E42, _RID_E62_SPLIT as _E62


def _upper_step(st, s, i, nbr, src, a1, a2):
    """KMC._upper_step: atom i is bonded to an upper neighbour (nbr[i, 2:4]) other than src
    that is not an adsorbate (a1 / a2) -- its other upper slot holds an incorporated atom."""
    row = st.bonds[i, :int(st.n_bonds[i])].tolist()
    for u in (int(nbr[i, 2]), int(nbr[i, 3])):
        if u >= 0 and u != src and u in row and s[u] != a1 and s[u] != a2:
            return True
    return False


def _host_ok(st, s, h, ga_precursor):
    """KMC._host_ok for one host h of a Ga (ga_precursor) or As precursor."""
    x = s[h]
    if ga_precursor:
        return x == D_AS or x == ASH or (x == AS_ and st.has_entry[h] and st.n_bonds[h] >= 2)
    return x == D_GA or x == MMGA or (x == GA_ and st.has_entry[h] and st.n_bonds[h] >= 2)


def enumerate_scalar(st, sites: np.ndarray):
    """sites: int64 array of reactive source sites. Returns (rid, a, b) lists."""
    n = sites.size
    if n == 0:
        return [], [], []
    s = st.species
    nbA = st.nbr_As[sites]                 # (n,4)
    nbG = st.nbr_Ga[sites]                 # (n,4)
    bnd = st.bonds[sites]                  # (n,maxb)
    sp = s[sites].tolist()
    he = st.has_entry[sites].tolist()
    nb = st.n_bonds[sites].tolist()
    spA = s[np.maximum(nbA, 0)].tolist();  nbA_l = nbA.tolist()
    spG = s[np.maximum(nbG, 0)].tolist();  nbG_l = nbG.tolist()
    bnd_l = bnd.tolist()
    # first bonded partner and ITS neighbourhood (sections 3,4,7,8)
    b0 = bnd[:, 0]
    b0c = np.maximum(b0, 0)
    spb0 = s[b0c].tolist()
    he2 = st.has_entry[b0c].tolist()
    nb2 = st.n_bonds[b0c].tolist()
    site_l = sites.tolist()

    rid_o, a_o, b_o = [], [], []
    emit = lambda r, a, b: (rid_o.append(r), a_o.append(a), b_o.append(b))

    for i in range(n):
        src = site_l[i]; x = sp[i]
        nA = nbA_l[i]; sA = spA[i]
        nG = nbG_l[i]; sG = spG[i]

        # ---- rids 9, 10, 11 : MMGa adsorption onto V_GA beside an As-type source
        if x == D_AS:
            v3, v4 = nA[2], nA[3]
            ok3, ok4 = v3 >= 0, v4 >= 0
            s3 = sA[2] if ok3 else -1; s4 = sA[3] if ok4 else -1
            if ok3 and s3 == V_GA:
                if ok4 and (s4 == DD_MMGA or s4 == DDD_GA): emit(10, v3, src)
                else:                                       emit(11, v3, src)
            if ok4 and s4 == V_GA:
                if ok3 and (s3 == DD_MMGA or s3 == DDD_GA): emit(10, v4, src)
                else:                                       emit(11, v4, src)
        elif x == D_ASH or x == DD_AS:
            v3, v4 = nA[2], nA[3]
            f3 = v3 >= 0 and sA[2] == V_GA; f4 = v4 >= 0 and sA[3] == V_GA
            r = _E62 if (x == D_ASH and f3 and f4) else 9    # E6.2 split: half rate each
            if f3: emit(r, v3, src)
            if f4: emit(r, v4, src)

        # ---- rids 14-19 : DD_MMGa incorporation, terrace (14-16) or step (17-19).
        # "Either side" rule (2026-09-27): step if the host's OR the partner's other
        # upper slot holds an incorporated Ga, read from the bonds; see
        # KMC._enumerate_events section 3/4. Copy: DDD_Ga likewise, rids 38-40 / 41-43.
        if x == DD_MMGA or x == DDD_GA:
            base_t = 14 if x == DD_MMGA else 38
            base_s = 17 if x == DD_MMGA else 41
            ba = bnd_l[i][0]
            if ba >= 0:
                asp = spb0[i]
                if (asp == D_AS) or (asp == ASH) or (asp == AS_ and he2[i] and nb2[i] >= 2):
                    p, ps = -1, -1
                    for k in (0, 1):            # first lower neighbour that can be the partner
                        v = nG[k]; sv = sG[k] if v >= 0 else -1
                        if v < 0:
                            continue
                        if sv == DD_AS or sv == D_ASH or (sv == D_AS and v != ba):
                            p, ps = v, sv; break
                        if (sv == DD_ASH or sv == DDD_AS) and v != ba:      # copy: adsorbed
                            hl = int(st.bonds[v, 0])
                            if hl >= 0 and hl != src and _host_ok(st, s, hl, False):
                                p, ps = v, sv; break
                    if p >= 0:
                        step = (_upper_step(st, s, ba, st.nbr_As, src, DD_MMGA, DDD_GA)
                                or _upper_step(st, s, p, st.nbr_As, src, DD_MMGA, DDD_GA))
                        if ps == DD_ASH or ps == DDD_AS:
                            emit(_PP_RID[(x, ps)] + (_PP_STEP if step else 0), src, p)
                        else:
                            off = 0 if ps == DD_AS else 1 if ps == D_ASH else 2
                            emit((base_s if step else base_t) + off, src, p)
        # ---- rid 13 : MMGa desorption; rid 36 : its methyl off (copy)
        if x == DD_MMGA:
            emit(36, src, -1)
            if he[i] and nb[i] >= 1:
                b = bnd_l[i][0]
                if b >= 0: emit(13, src, b)

        # ---- rids 6, 7, 8 : AsH adsorption onto V_AS beside a Ga-type source
        if x == DD_GA or x == D_MMGA:
            v3, v4 = nG[2], nG[3]
            f3 = v3 >= 0 and sG[2] == V_AS; f4 = v4 >= 0 and sG[3] == V_AS
            r = _E42 if (x == D_MMGA and f3 and f4) else 6   # E4.2 split: half rate each
            if f3: emit(r, v3, src)
            if f4: emit(r, v4, src)
        elif x == D_GA:
            v3, v4 = nG[2], nG[3]
            ok3, ok4 = v3 >= 0, v4 >= 0
            s3 = sG[2] if ok3 else -1; s4 = sG[3] if ok4 else -1
            po3 = ok4 and (s4 == DD_ASH or s4 == DDD_AS)     # partner side occupied
            po4 = ok3 and (s3 == DD_ASH or s3 == DDD_AS)
            if ok3 and s3 == V_AS:
                emit(7 if po3 else 8, v3, src)
            if ok4 and s4 == V_AS:
                emit(7 if po4 else 8, v4, src)

        # ---- rids 20-31 : DD_AsH / DDD_As incorporation, step (20-25) or terrace (26-31).
        # Same "either side" rule, Ga and As swapped (2026-09-27).
        if x == DD_ASH or x == DDD_AS:
            bg = bnd_l[i][0]
            if bg >= 0:
                gsp = spb0[i]
                base_t = 26 if x == DD_ASH else 29      # terrace
                base_s = 20 if x == DD_ASH else 23      # step
                if (gsp == D_GA) or (gsp == MMGA) or (gsp == GA_ and he2[i] and nb2[i] >= 2):
                    p, ps = -1, -1
                    for k in (0, 1):            # first lower neighbour that can be the partner
                        v = nA[k]; sv = sA[k] if v >= 0 else -1
                        if v < 0:
                            continue
                        if sv == DD_GA or sv == D_MMGA or (sv == D_GA and v != bg):
                            p, ps = v, sv; break
                        if (sv == DD_MMGA or sv == DDD_GA) and v != bg:     # copy: adsorbed
                            hl = int(st.bonds[v, 0])
                            if hl >= 0 and hl != src and _host_ok(st, s, hl, True):
                                p, ps = v, sv; break
                    if p >= 0:
                        step = (_upper_step(st, s, bg, st.nbr_Ga, src, DD_ASH, DDD_AS)
                                or _upper_step(st, s, p, st.nbr_Ga, src, DD_ASH, DDD_AS))
                        if ps == DD_MMGA or ps == DDD_GA:
                            emit(_PP_RID[(x, ps)] + (_PP_STEP if step else 0), src, p)
                        else:
                            off = 0 if ps == DD_GA else 1 if ps == D_MMGA else 2
                            emit((base_s if step else base_t) + off, src, p)
            # rids 1, 2 : desorption of the last-bonded partner
            if he[i] and nb[i] >= 1:
                lb = bnd_l[i][nb[i] - 1]
                if lb >= 0: emit(1 if x == DD_ASH else 2, src, lb)

        # ---- rids 3, 4, 5 : H abstraction
        if x == DD_ASH:   emit(3, src, -1)
        elif x == D_ASH:  emit(4, src, -1)
        elif x == ASH:    emit(5, src, -1)

        # ---- rid 12 : methyl desorption
        if x == D_MMGA or x == MMGA:
            emit(12, src, -1)

        # ---- rids 32, 33, 37 : Ga desorption (FIX v2: not if bonded to the substrate)
        if (x == DD_GA or x == D_GA or x == DDD_GA) and any(
                j >= 0 and (s[j] == S_AS or s[j] == S_GA) for j in bnd_l[i][:nb[i]]):
            pass
        elif x == DD_GA:
            if he[i] and nb[i] >= 1:
                b = bnd_l[i][0]
                if b >= 0: emit(32, src, b)
        elif x == D_GA:
            if he[i] and nb[i] >= 3:
                lb = bnd_l[i][nb[i] - 1]
                if lb >= 0: emit(33, src, lb)
        elif x == DDD_GA:                                   # copy: R18
            if he[i] and nb[i] >= 1:
                lb = bnd_l[i][nb[i] - 1]
                if lb >= 0: emit(37, src, lb)

        # ---- rids 66-77 : a film atom's free bond bonds down (27 Sep; KMC section 14)
        fb = _FILM_BOND.get(x)
        if fb is not None:
            ga = x == D_MMGA or x == DD_GA or x == D_GA
            lo = nG if ga else nA
            up = st.nbr_As if ga else st.nbr_Ga
            a1, a2 = (DD_MMGA, DDD_GA) if ga else (DD_ASH, DDD_AS)
            row = bnd_l[i][:nb[i]]
            for k in (0, 1):
                L = lo[k]
                if L < 0 or L in row or not _IS_REAL[s[L]] or st.DB[L] <= 0:
                    continue
                step = _upper_step(st, s, L, up, src, a1, a2) or any(
                    _upper_step(st, s, lo[kk], up, src, a1, a2)
                    for kk in (0, 1) if lo[kk] >= 0 and lo[kk] != L and lo[kk] in row)
                emit(fb[1] if step else fb[0], src, L)

        # ---- rids 60-65 : other film atoms with two / three bonds desorb (copy; R10 / R11)
        if (x in _DES_TWO or x in _DES_THREE) and not any(
                j >= 0 and (s[j] == S_AS or s[j] == S_GA) for j in bnd_l[i][:nb[i]]):
            if x in _DES_TWO:
                if he[i] and nb[i] >= 1:
                    b = bnd_l[i][0]
                    if b >= 0: emit(_DES_TWO[x], src, b)
            elif he[i] and nb[i] >= 3:
                lb = bnd_l[i][nb[i] - 1]
                if lb >= 0: emit(_DES_THREE[x], src, lb)

    return rid_o, a_o, b_o
