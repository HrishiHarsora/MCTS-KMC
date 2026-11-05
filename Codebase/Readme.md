# Codebase

Three modules, used in this order:

1. **`kmc/`** simulates GaAs film growth for a given parameter set and reads its growth rate.
2. **`drc/`** measures which parameters control the growth rate and film thickness.
3. **`mcts/`** searches those parameters to fit the experimental growth-rate curve.

Requires Python 3.11+, NumPy and SciPy
(numba (optional) makes the engine 4.6x faster with identical results)

```
Codebase/
├── kmc/                              the growth engine
│   ├── run.py                        entry point: runs a JSON config
│   ├── run_kmc.py                    driver: seeding, parallel pool, output
│   ├── KMC_bucket.py                 the engine
│   ├── nb_bucket.py                  its compiled (numba) kernels
│   ├── KMC.py                        reference implementation the engine is validated against
│   ├── enum_scalar.py                site and event enumeration
│   ├── reward_metric.py              the growth rate: the mean height's rate through 50-200 A
│   ├── kmc_worker.py                 one trajectory as a subprocess (used by mcts/)
│   ├── lattice/                      the GaAs(001) starting slab, 10 nm x 10 nm, 231.8 A of room (Lattice_10nm_h232.xyz)
│   ├── REACTIONS.md                  every reaction, its rate law and parameters
│   ├── dump_reactions.py             regenerates REACTIONS.md if events/parameters are changed
│   ├── dump_defaults.py              checks parameter set used by KMC.py if it is NOT provided is same as params_catalog_defaults.json
│   ├── params_catalog_defaults.json  Default 54 parameter set
│   └── examples/                     ready-to-run configs and parameter sets
├── drc/                              the sensitivity screen
│   ├── build_drc.py                  writes the perturbed parameter sets and task list
│   ├── reduce_from_traj.py           turns finished runs into D = d ln GR / d ln k and D_h
│   └── examples/
└── mcts/                             the search
    ├── build_search.py               writes config.toml and the root parameter set
    ├── MCTS.py                       runs the tree search
    └── examples/                     target curve (Lin 1977), DRC table, four-lever setup
```
