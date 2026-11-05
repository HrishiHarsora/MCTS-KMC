"""warm_numba.py -- compile the engine's numba kernels once, before the worker pool starts.

The kernels are cached next to the codebase's nb_bucket.py (__pycache__), so the first process
compiles and every later one loads them in about a second. Run once per job, it keeps 48
workers from all compiling at the same moment. Prints which path the engine will take; never
fails the job.
"""
import time

import codebase

t0 = time.perf_counter()
try:
    codebase.use()
    import numpy as np
    import KMC_bucket as K
    if K.NUMBA_ON:
        np.random.seed(0)
        K.run_kmc_trajectory(1125.0, lattice_file=codebase.LATTICE, max_iter=300, target_growth=1e9)
        print("[warm_numba] numba kernels ON, compiled / loaded in %.1f s" % (time.perf_counter() - t0))
    else:
        print("[warm_numba] numba NOT available: the Python path, ~4.6x slower per event. "
              "pip install --user numba to speed this up.")
except SystemExit:
    raise
except Exception as e:                  # the run itself will report any real problem
    print("[warm_numba] warm-up skipped: %r" % (e,))
