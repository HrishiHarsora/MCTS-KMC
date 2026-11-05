"""Where THE codebase is. This stage folder holds only its own inputs and scripts; the engine
always runs from the one codebase:

    $KMC_CODE                      if set (the Codebase folder), else
    ../../Codebase                 this repository's Codebase/, relative to this folder

CODEBASE.md5 pins the exact files this stage was verified with; the job scripts refuse to run
on anything else (md5sum -c).
"""
import os, sys

CODE = os.environ.get("KMC_CODE") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "Codebase"))
KMC_DIR = os.path.join(CODE, "kmc")
CATALOG = os.path.join(KMC_DIR, "params_catalog_defaults.json")
LATTICE = os.path.join(KMC_DIR, "lattice", "Lattice_10nm_h232.xyz")


def use():
    """Put the codebase's kmc/ on the import path (after checking it is there)."""
    if not os.path.exists(os.path.join(KMC_DIR, "KMC.py")):
        sys.exit("[codebase] no KMC codebase at %s -- set KMC_CODE to the Codebase folder" % CODE)
    if KMC_DIR not in sys.path:
        sys.path.insert(0, KMC_DIR)
    return KMC_DIR
