# `kmc/` — the simulator

> **The working engine (adopted 2026-09-27).** Reactions R1-R20 as listed in
> [REACTIONS.md](REACTIONS.md): the Ga route as general as the As route (DDD_Ga: R8 case E8.3, R18-R20),
> incorporation onto an adsorbed precursor and of a film atom's free bond (cases of R12-R17,
> R19, R20), desorption of any film atom with two / three bonds (cases of R10 / R11). No bond
> forms automatically: every bond is a rated incorporation event. 54 parameters, 8 ordering
> constraints. Compiled numba kernels on whenever numba imports (4.6x faster, identical films).
> Verified by rule walks against an independent model of every event, by re-enumerating the
> event menu after every step, and against the reference engine (KMC.py).

A lattice kinetic Monte Carlo model of GaAs MOCVD growth: **20 surface reactions
(77 engine IDs)** on a GaAs(001) slab 10 nm x 10 nm wide with 231.8 Å of room above the
starting surface, rejection-free BKL with a bucketed rate list. One run measures a growth
rate at one or more temperatures.

**[REACTIONS.md](REACTIONS.md) lists all 20** — what each one is, its rate law,
its catalog values and which parameters tune it. That file is generated from the
engine itself (`python dump_reactions.py > REACTIONS.md`), so it cannot drift
from what is simulated.

> The paper's Figure 3 groups some of these and numbers them 1–23. Reconciling
> the two numberings is open; this folder currently uses the engine's.

```
python run.py examples/baseline_1125K.json
python run.py examples/ladder_six_temperatures.json --dry-run
```

`run.py` reads the JSON, checks it, writes a `config.toml` next to the output
CSV and hands over to `run_kmc.py`, the driver. You can also drive `run_kmc.py`
directly; `python run_kmc.py --help` lists its flags.

`KMC.py` has a `__main__` block. **Do not use it to reproduce anything** — it
ignores the event cap, the growth-rate window and its stop, checkpointing and
seeding. It is a smoke test.

## The run JSON

```json
{
  "description": "free text, copied into the generated config.toml",

  "parameters": "params/stage01_1125K_winner.json",
  "lattice":    "lattice/Lattice_10nm_h232.xyz",

  "temperatures": [900, 925, 950, 975, 1000, 1125],
  "replicates":   240,
  "seed_base":    93000000,

  "max_events":        10000000,
  "checkpoint_events": 20000,

  "window_A": [50.0, 200.0],
  "stall": {
    "idle_factor": 2.0,
    "min_events":  500000,
    "rel_tol":     0.005,
    "abs_tol_A":   0.01
  },

  "workers": 48,
  "out":     "out/my_run.csv"
}
```

| key | meaning |
|---|---|
| `parameters` | a path to a JSON of parameter values, or an inline object. Omit it to run at catalog values. Every key is checked against the 54 names in `params_catalog_defaults.json`; an unrecognised name is a hard error, because the engine would otherwise silently substitute the catalog value and report a result for a parameter set you did not ask for. |
| `temperatures` | kelvin. Replicate `k` of the `i`-th temperature always gets seed `seed_base + i*replicates + k`, so a given (T, k) is the same trajectory no matter how many replicates the run ends up doing. That is what makes two parameter sets comparable replicate for replicate. |
| `max_events` | the backstop event budget. `checkpoint_events` must divide it exactly. |
| `window_A` | the growth-rate window `[lo, hi]` in Å — see below. A run stops when the mean height reaches `hi`. |
| `stall` | ends a film that has stopped growing — see below. Omit it to turn it off. |
| `out` | output CSV. Two files are written: `<out>` with one row per checkpoint, and `<out:-4>_traj.csv` with one row per trajectory. |

The optional `adaptive` block (`wave`, `n_rep_max`, `target_sem_pct`) adds replicates a
wave at a time. It stops when, at every temperature, both conditions hold:

- the standard error of the growth rate meets the target, and
- the running mean has gone flat.

Flat is defined in `flat_region.py`, and the MCTS uses the same rule:

- The band is the full mean ± its standard error.
- `n_flat` is the first n from which the running mean stays inside the band **and**
  has reached its middle.
- The curve counts as flat if `n_flat` comes before the last 20 % of the replicates.

Leave the block out for a fixed number of replicates.

## The growth rate

The film's mean height — the mean over lateral columns of the top atom, measured
from the starting surface — is recorded every `checkpoint_events`. The growth rate
is read through a thickness window, `window_A = [lo, hi]`, by default 50–200 Å:

```
GR = (hi - lo) / (t(hi) - t(lo))        [µm/min]
```

where `t(h)` is the simulated time at which the mean height first reaches `h`,
interpolated between the two checkpoints around it. The first 50 Å, the start-up
transient, are left out. The window top sits 32 Å below the top of the lattice,
which is well clear of the growth front: the front leads the mean height by ~7–12 Å.

A trajectory ends at the first of:

- **window**: the mean height reaches `hi`. This is the intended stop, and a run
  costs only the events it needs to grow through the window.
- **stall**: the mean height has not advanced for
  `max(min_events, idle_factor × events at the last advance)`, where a checkpoint
  advances when `h_mean > best + max(rel_tol × |best|, abs_tol_A)`. This only
  saves the cost of a film that stopped growing.
- **cap**: `max_events`.

A trajectory that did not reach `hi` has no growth rate. It is reported as not
having grown through the window (`reached = 0`), never as a number. The
per-trajectory CSV records `reached`, `t_lo_s`, `t_hi_s`, `GR_window_umin` and
which stop ended it (`stop`).

## Files

| file | what it is |
|---|---|
| `run.py` | the JSON entry point. Marshals arguments; contains no physics. |
| `run_kmc.py` | the driver: task list, seeding, sharding, pooling, resume, output. |
| `KMC_bucket.py` | the engine. `import KMC_bucket as KMC` everywhere. |
| `KMC.py` | the reference implementation the bucket engine was validated against. |
| `enum_scalar.py` | the site/event enumeration. |
| `nb_bucket.py` | compiled (numba) event menu and enumeration for `KMC_bucket`, 4.6x faster. On whenever numba imports; `KMC_NUMBA=0` turns it off, `KMC_NUMBA=1` makes it required. Bit-identical films for the same seed (`tests/bench_numba.py`). |
| `reward_metric.py` | the growth-rate window, read off a checkpoint trace. |
| `kmc_worker.py` | one trajectory as a subprocess, used by the search. |
| `params_catalog_defaults.json` | all 54 parameters at their catalog values (the terrace sticking coefficients, 0.04 = 0.5 × the step value, are starting values, not literature). |
| `dump_defaults.py` | `python dump_defaults.py --verify` proves that feeding the catalog file back in is a bit-identical no-op. |

## Parameters

54 names in one flat namespace — 16 Arrhenius parameter sets × 3 + 6 — laid out
reaction by reaction in [REACTIONS.md](REACTIONS.md): `A_E<r>`, `Ea_E<r>`,
`beta_E<r>` for `r ∈ 1, 2, 3, 12, 13, 14, 17, 20, 23, 26, 29, 32, 33, 37, 38, 41`, plus `S0_A_terrace`, `S0_A_step`,
`Ea_chem_As`, `S0_M_terrace`, `S0_M_step`, `Ea_chem_M`.

- **R3 (H abstraction) is one reaction with three engine IDs**: 3 (DD_AsH), 4
  (D_AsH) and 5 (AsH) break the same As–H bond and share `A_E3`, `Ea_E3`,
  `beta_E3`. `A_E4`, `A_E5` and friends are rejected by `run.py`; the engine accepts
  them only when they equal the E3 value, as older parameter files have them.
- **Each incorporation reaction has one parameter set** (2026-09-27): R12 = IDs 14–16
  (`A_E14`), R13 = 17–19 (`A_E17`), R14 = 26–28 (`A_E26`), R15 = 20–22 (`A_E20`),
  R16 = 29–31 (`A_E29`), R17 = 23–25 (`A_E23`). The IDs of one reaction differ only in
  the partner, a spectator; DD_AsH (R14, R15) and DDD_As (R16, R17) keep separate
  rates. R19 = 38–40 (`A_E38`) and R20 = 41–43 (`A_E41`) do the same for an adsorbed Ga
  (DDD_Ga). Each also holds the cases whose partner below is an adsorbed precursor (IDs
  44–59) and those of a film atom's free bond bonding down (IDs 66–77), at its rate.
  Tied keys (`A_E15`, `Ea_E18`, ...) are rejected the same way as `A_E4`.
- **No automatic bonding** (2026-09-27): every bond forms by a rated incorporation event;
  an H or a methyl rotates out of the way. An atom that lands in a hole under film atoms
  is bonded to them by those atoms' incorporation events (IDs 66–77), not at once.
- **Desorption with two / three bonds** (R10 / R11) covers every film atom: DD_Ga,
  D_MMGa, DD_As, D_AsH (IDs 32, 62, 60, 64) and D_Ga, MMGa, D_As, AsH (33, 63, 61, 65),
  not when bonded to the substrate. R18 (ID 37) is desorption of an adsorbed Ga (DDD_Ga).
- **Terrace or step incorporation, "either side"** (2026-09-27): a precursor
  incorporates at a step if its host's or its partner's other upper bond holds an
  incorporated atom, and on a terrace only if both hold a dangling bond, an H /
  methyl or an adsorbate. Each adsorbed precursor has exactly one incorporation event;
  a film atom has one per free lower neighbour that has a free bond.
- **Sticking coefficients, terrace and step** (2026-09-27): `S0_A_terrace` (R1, IDs 6, 7,
  34), `S0_A_step` (R2, ID 8), `S0_M_terrace` (R6, IDs 9, 10, 35), `S0_M_step` (R7, ID
  11); catalog 0.04 / 0.08 each. They replace `S0_A`, `S0_M` × `f_terr_A`, `f_terr_M`
  (S0_terrace = f_terr × S0, S0_step = S0; same rates), and the old names are refused.
- **Ordering constraints** (`KMC._CONSTRAINTS`, 2026-09-27), checked on every run and by
  `run.py`; the DRC builder and the MCTS clip their moves to stay inside:
  S0_terrace ≤ S0_step for AsH and MMGa; `Ea_E32` < `Ea_E33` (R10 < R11); step
  barrier < terrace barrier, `Ea_E17` < `Ea_E14`, `Ea_E20` < `Ea_E26`, `Ea_E23` < `Ea_E29`,
  `Ea_E41` < `Ea_E38`; `Ea_E37` < `Ea_E32` (R18 < R10; pushed along by the search tools
  when R10's barrier is lowered past it).
- **Arrival flux** P / sqrt(2 pi m k_B T) × A_site of the species that land, AsH
  (75.93 u) and MMGa (GaCH3, 84.76 u) (2026-09-27; AsH3's mass and 1.39e-25 kg before).
- **Split IDs 34 and 35 (E4.2, E6.2).** A D_MMGa (for AsH) or a D_AsH (for MMGa) has
  one dangling bond, but its methyl / H moves out of the way, so the adsorbate can
  land on either upper site. With both free, each landing is a split-ID event at half
  the site rate, so the host adsorbs at the rate of one site; with one free, that
  landing is an ordinary ID-6 / ID-9 event. `KMC.landing_rate_errors` checks this and
  runs in the bucket engine's Gate 1.
- **Reaction numbers.** The notes number every engine ID into R1–R20: R1 = IDs 6, 34, 7,
  R2 = 8, R3 = 3–5, R4 = 1, R5 = 2, R6 = 9, 35, 10, R7 = 11, R8 = 12, 36, R9 = 13,
  R10 = 32, 62, 60, 64, R11 = 33, 63, 61, 65, R12 = 14–16, 44, 45, 66, R13 = 17–19, 52,
  53, 67, R14 = 26–28, 46, 47, 72, R15 = 20–22, 54, 55, 73, R16 = 29–31, 48, 49, 74, 76,
  R17 = 23–25, 56, 57, 75, 77, R18 = 37, R19 = 38–40, 50, 51, 68, 70, R20 = 41–43, 58,
  59, 69, 71 ([REACTIONS.md](REACTIONS.md)). Engine IDs,
  parameter names and `n_rid_XX` columns keep the engine numbering; the event log and
  the `n_R<k>` log columns are by reaction.
- **`Ea` is in kelvin**, not eV and not J/mol. `k = A · T^beta · exp(-Ea/T)`.
- Of the 77 engine IDs, **8 are not Arrhenius**: IDs 6–11, 34 and 35 are adsorption,
  rated from the impingement flux, and carry no prefactor or barrier. `A_E7` and
  friends exist in no file and are rejected.
- `pressure_MMGa × S0_M_*` is the only combination that enters an MMGa impingement
  rate (likewise `pressure_AsH × S0_A_*`). The pressures are fixed in the engine at
  the experiment's values, so `S0` carries that degree of freedom.

## Sharding

`--shard i --nshards N` runs the tasks with `k % N == i` — strided, not
contiguous, so a partially finished run has replicates spread over all
temperatures rather than all of the first and none of the rest. Output is
appended under an exclusive lock, so every shard can write the same CSV. A
replicate already present in `<out>_traj.csv` is skipped, which makes an
interrupted run cheap to resume and raising `replicates` cheap to extend.
