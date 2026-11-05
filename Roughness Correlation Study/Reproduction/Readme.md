# Reproduction

Everything runs from the one codebase, [`../../Codebase`](../../Codebase) (`CODEBASE.md5` pins the files these results were made with; `run_rough.sh` refuses to start on any other).

```
Reproduction/
├── Readme.md
├── best_params_node31.json               the 54 parameters of the final fit (node 31) — the only input
├── roughness_study.py                    grows the films and reads W, P–V and a snapshot at every 10 Å mark
├── roughness_study_lin_temperatures.py   the same for Lin's seven off-grid temperatures, on the seeds of the
│                                         node-31 DRC baseline (reads drc_tasklist.txt next to it)
├── drc_tasklist.txt                      those seeds
├── run_rough.sh                          the SLURM array job (15 chunks of 48 cores)
├── codebase.py, warm_numba.py            find the codebase; compile the numba kernels once per job
├── CODEBASE.md5                          the pinned engine
├── analyze_roughness.py                  merges the films into roughness.csv (mean, SEM, n_flat) and plots
├── snap_to_xyz.py                        turns a film's snapshots (.npz) into an XYZ trajectory
└── compare_rough.py                      checks a rerun against the original film by film
```

## The films

```bash
export KMC_CODE=/path/to/Codebase            # the engine; CODEBASE.md5 must match it
# the 10 K grid, 34 temperatures x 240 films, split into 15 chunks (one 48-core node each):
python roughness_study.py --params best_params_node31.json --chunk 0 --nchunks 15 --workers 48 --out results
#   ... --chunk 1 ... up to --chunk 14  (run_rough.sh does this as a SLURM array: sbatch --array=0-14 run_rough.sh)
# Lin's seven off-grid temperatures, 144 films each, on the DRC baseline's seeds:
python roughness_study_lin_temperatures.py --params best_params_node31.json --chunk 0 --nchunks 15 --workers 48 --out rough
```

Seeds are fixed in the scripts (`97250000 + 1000 × temperature index + replicate` on the grid; the DRC tasklist's
seed base + replicate off it), so a rerun gives the same films bit for bit. Cost: about 205 core-hours for the grid
and 60 for the seven temperatures — roughly 45 min on three 240-core allocations. A smoke test:

```bash
python roughness_study.py --params best_params_node31.json --chunk 0 --nchunks 1 --films 2 --temps 900,1230 --max-iter 200000 --out smoke
```

Each chunk writes `rough_marks_c<N>.csv` (one row per film per 10 Å mark: height read, events, time, W, P–V),
`rough_films_c<N>.csv` (one row per film) and `snaps/T<T>_r<rep>.npz` (the lattice occupancy at every mark).

## The table and the figure

```bash
python analyze_roughness.py --grid results --lin rough --out .      # -> roughness_summary.csv, roughness_map.png
```

`roughness_summary.csv` must equal [`../roughness.csv`](../roughness.csv).

## The snapshots

```bash
python snap_to_xyz.py results/snaps/T1125_r000.npz --movie --out XYZ --code /path/to/Codebase
gzip -9 XYZ/T1125_r000.xyz                                           # -> ../Snapshots/T1125_r000.xyz.gz
```

`--movie` writes one 16-frame XYZ per film (50 → 200 Å); without it, one file per mark. Vacant lattice sites are
left out. `compare_rough.py <results folder>` confirms that a rerun reproduces every W row of the original study.
