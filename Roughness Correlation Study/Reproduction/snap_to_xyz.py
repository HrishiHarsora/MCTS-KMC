"""snap_to_xyz.py -- turn snapshots from roughness_study.py into XYZ files (vacancies stripped).

    python snap_to_xyz.py <snaps/T1150_r000.npz> [more .npz ...] [--marks 50,100,200] [--movie] [--out xyz]

--movie writes one multi-frame XYZ per film, T<T>_r<rep>.xyz, one frame per mark (thickness 50 -> 200 A), for
OVITO / VMD; otherwise one file per mark, T<T>_r<rep>_<mark>A.xyz.

The site coordinates come from the lattice exactly as the engine builds them (extract_coords_with_species_labels,
then label_lattice_sites), so site i of a snapshot is site i here. Every occupied site is written (substrate, film,
adsorbates); V_As / V_Ga are left out. No KMC is run.
"""
import argparse, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))   # this repository's root
ap = argparse.ArgumentParser()
ap.add_argument("npz", nargs="+")
ap.add_argument("--marks", default="", help="comma list of marks (A); default all")
ap.add_argument("--movie", action="store_true", help="one multi-frame file per film")
ap.add_argument("--out", default="xyz")
ap.add_argument("--code", default=os.environ.get("KMC_CODE") or os.path.join(REPO, "Codebase"))
a = ap.parse_args()
sys.path.insert(0, os.path.join(a.code, "kmc"))
import KMC as K
import KMC_bucket as KB

lat = os.path.join(a.code, "kmc", "lattice", "Lattice_10nm_h232.xyz")
atoms, origin, cell = KB.extract_coords_with_species_labels(lat)
atoms = KB.label_lattice_sites(atoms, bottom_layers=3)
xyz = np.array([c for c, _ in atoms], dtype=float)
want = {float(x) for x in a.marks.split(",")} if a.marks else None
os.makedirs(a.out, exist_ok=True)
names = np.array([K._CODE_TO_NAME.get(c, "V") for c in range(K._N_SPECIES)], dtype=object)
lattice = " ".join(map(str, np.asarray(cell).flatten()))
pos = np.array(["%.4f %.4f %.4f" % tuple(r) for r in xyz], dtype=object)


def frame(f, d, k, m):
    sp = d["species"][k].astype(int)
    keep = np.nonzero(~K._IS_VAC[sp])[0]
    f.write("%d\n" % keep.size)
    f.write('Lattice="%s" Origin="%s" Properties=species:S:1:pos:R:3 T_K=%d rep=%d mark_A=%g h_read_A=%.4f '
            'events=%d time_s=%.6e\n' % (lattice, " ".join(map(str, origin)), int(d["T_K"]), int(d["rep"]), m,
                                        d["h_read_A"][k], d["events"][k], d["time_s"][k]))
    f.write("\n".join(names[sp[keep]] + " " + pos[keep]) + "\n")
    return keep.size


for path in a.npz:
    d = np.load(path)
    if d["species"].shape[1] != len(xyz):
        sys.exit("[snap_to_xyz] %d sites in the snapshot, %d in the lattice" % (d["species"].shape[1], len(xyz)))
    ks = [(k, m) for k, m in enumerate(d["mark_A"]) if want is None or float(m) in want]
    stem = "T%d_r%03d" % (int(d["T_K"]), int(d["rep"]))
    if a.movie:
        fn = os.path.join(a.out, stem + ".xyz")
        with open(fn, "w", newline="\n") as f:
            n = [frame(f, d, k, m) for k, m in ks]
        print(fn, len(n), "frames,", n[-1] if n else 0, "atoms in the last")
    else:
        for k, m in ks:
            fn = os.path.join(a.out, "%s_%dA.xyz" % (stem, int(m)))
            with open(fn, "w", newline="\n") as f:
                print(fn, frame(f, d, k, m), "atoms")
