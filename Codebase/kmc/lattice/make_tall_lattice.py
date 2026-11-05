"""make_tall_lattice.py -- a taller copy of Lattice_10nm_h232.xyz, for films thicker than 200 A.

    python make_tall_lattice.py --room 1032            # -> Lattice_10nm_h1032.xyz

The 10 nm lattice is a zincblende stack along z: 168 layers of 625 sites, 1.4134 A apart, the xy pattern and
the As / Ga alternation repeating every 4 layers. The engine (KMC.label_lattice_sites) takes the bottom 3 layers
as substrate, layer 3 as the DD_Ga surface and everything above as vacant sites, so the growth room is
(layers - 4) x 1.4134 A: 232 A in the h232 file. This script appends layers, continuing the 4-layer period, until
the room is at least --room angstrom, and writes the same extended-XYZ format with the cell's z extended to match.
The first 168 layers are copied as they are, so a film on the new lattice starts from the same surface.
Deterministic: the output's md5 is printed; the cluster copy must match the local one.
"""
import argparse, hashlib, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "Lattice_10nm_h232.xyz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", type=float, default=1032.0, help="growth room wanted, A (window top + 32 A margin)")
    a = ap.parse_args()
    lines = open(SRC).read().split("\n")
    n = int(lines[0]); head = lines[1]
    rows = [l.split() for l in lines[2:2 + n]]
    sp = np.array([r[0] for r in rows]); xyz = np.array([[float(v) for v in r[1:4]] for r in rows])
    z = xyz[:, 2]
    zl = np.unique(np.round(z, 4))
    d = (zl[-1] - zl[0]) / (len(zl) - 1)                 # the layer spacing, 1.4134 A
    layer = np.rint((z - zl[0]) / d).astype(int)
    assert len(zl) == 168 and np.all(np.bincount(layer) == 625), "not the 10 nm h232 lattice"
    need = int(np.ceil(a.room / d)) + 4                    # layers for the room asked, plus substrate + surface
    extra = max(0, need - 168)
    out_sp, out_xyz = [sp], [xyz]
    for k in range(168, 168 + extra):
        src = layer == (164 + (k - 168) % 4)               # the period: layers 164-167 repeated
        s = xyz[src].copy(); s[:, 2] = zl[0] + k * d
        out_sp.append(sp[src]); out_xyz.append(s)
    SP, XYZ = np.concatenate(out_sp), np.concatenate(out_xyz)
    total = 168 + extra
    room = (total - 4) * d
    # the header: the cell's z length grows with the stack (the h232 cell is 237.455 A for 168 layers)
    import re
    m = re.search(r'Lattice="([^"]+)"', head)
    cell = [float(v) for v in m.group(1).split()]
    cell[8] = cell[8] + extra * d
    head2 = head.replace(m.group(1), " ".join("%.8f" % v for v in cell))
    out = os.path.join(HERE, "Lattice_10nm_h%d.xyz" % int(round(room)))
    with open(out, "w", newline="\n") as f:
        f.write("%d\n%s\n" % (len(SP), head2))
        for s, (x, y, zz) in zip(SP, XYZ):
            f.write("%s %.8f %.8f %.8f\n" % (s, x, y, zz))
    md5 = hashlib.md5(open(out, "rb").read()).hexdigest()
    print("wrote %s: %d layers, %d sites, room %.1f A, z %.3f .. %.3f, md5 %s"
          % (os.path.basename(out), total, len(SP), room, XYZ[:, 2].min(), XYZ[:, 2].max(), md5))


if __name__ == "__main__":
    main()
