# `drc/` — the sensitivity screen

A direct rate-coefficient screen. For each direction it perturbs one rate
coefficient up and down by `delta` e-folds about a root parameter set, runs the
same KMC sweep as the root, and measures

```
D(T)        = d ln GR(T) / d ln k
D_height(T) = d ln h(T)  / d ln k
```

GR is the growth rate through the 50-200 Å window (`../kmc/README.md`), from the
pairs in which both films grew through it; h is the mean height the film ended at,
so `D_height` is ~0 wherever both films reached the window top and measures a
lever's pull on growth itself where they did not.

at every temperature. These are measured derivatives, not linearised ones, and
they are what tells you which parameters are worth searching — and, through the
temperature dependence, which ones can change the *slope* of the growth-rate
curve rather than just its height.

## Build

```
python build_drc.py examples/lt1125_drc.json
```

writes:

| | |
|---|---|
| `drc_params/` | one parameter JSON per arm (`<direction>_plus.json`, `<direction>_minus.json`) plus `baseline.json` |
| `drc_manifest.csv` | direction, key, arm, value before and after, `delta_applied`, clipped flag |
| `drc_tasklist.txt` | one line per (arm, temperature): `<file> <T> <seed_base>` |
| `drc_root.json` | the root, flat, all 54 keys |
| `drc_stamp.json` | md5s of all of the above |

## Run

Each line of the task list is one KMC sweep at one temperature, run through
`../kmc/run_kmc.py`. On a cluster that is one array element per line:

```bash
read -r ARM T SEED <<< "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" drc_tasklist.txt)"
python ../kmc/run_kmc.py --config config.toml \
       --params "drc_params/${ARM}" --temps "$T" --seed-base "$SEED" \
       --out "results/drc_${ARM%.json}_T${T}.csv" --workers 48
```

One element per (arm, temperature) rather than one per arm: at low temperature a
single trajectory runs for hours, and six temperatures in one element overruns a
wall clock limit.

## Reduce

```
python reduce_from_traj.py --results results/ --temps 900,925,950,975,1000,1125
```

One row per (direction, temperature), with `D`, `D_height`, their standard
errors, and SNR. Also reports the baseline arm — the root's own growth rate, final
height and the share of films that grew through the window at each temperature,
which is the verification of whatever fit produced that root.

Two things this does deliberately:

- **Common random numbers.** Every arm at a given temperature shares `seed_base`,
  so plus and minus are paired replicate by replicate and the difference between
  them is the parameter's effect rather than sampling noise. The reducer checks
  the pairing and reports any mismatch.
- **Divides by `delta_applied`, never the nominal `delta`.** A bounded parameter
  clips: a sticking coefficient already at 0.9746 cannot take a +0.9 e-fold step,
  it stops at 1.0, which is a step of +0.0257. Dividing by 0.9 there would
  understate that direction by a factor of 35.

A film that did not grow through the window has no growth rate; its pair is left
out of `D` and counted in `scored_plus` / `scored_minus`, so an arm that kills
films shows up as a low count rather than as a biased mean.

## The DRC JSON

```json
{
  "root_parameters": "params/stage01_1125K_winner.json",
  "root_md5":        "3058766269a32bbd2fb3cf5fb52aa0b8",

  "delta_efolds": 0.9,
  "reactions":    [1, 2, 3, 12, 13, 14, 17, 20, 23, 26, 29, 32, 33],
  "extra_directions": [["stick_A_terr", "S0_A_terrace"], ["stick_A_step", "S0_A_step"],
                       ["stick_M_terr", "S0_M_terrace"], ["stick_M_step", "S0_M_step"]],
  "bounds": {"S0_A_terrace": [0.0, 1.0], "S0_A_step": [0.0, 1.0],
             "S0_M_terrace": [0.0, 1.0], "S0_M_step": [0.0, 1.0]},

  "temperatures": [900, 925, 950, 975, 1000, 1125],
  "replicates":   240,
  "seed_base":    93000000,
  "priority":     ["stick_M_terr", "stick_M_step", "E32", "E33"],
  "out_dir":      "drc_params"
}
```

| key | meaning |
|---|---|
| `root_parameters` | a flat 45-key JSON, or a search winner JSON (its `params` block is used). |
| `root_md5` | optional. If set, the root file must hash to it — cheap insurance against screening one parameter set and reporting it as another. |
| `reactions` | one direction per reaction, stepping its prefactor `A_E<r>`. R3 (H abstraction) is `3` and moves engine IDs 3, 4 and 5 together; `4` and `5` are not directions. Likewise each incorporation reaction is one direction, named by the ID that owns its parameters: 14, 17, 20, 23, 26, 29. |
| `extra_directions` | `[name, parameter]` for directions that are not a reaction prefactor. The default adds the four sticking coefficients, `S0_A_terrace`, `S0_A_step`, `S0_M_terrace`, `S0_M_step`, bounded to [0, 1]. An arm that would break an ordering constraint (S0_terrace ≤ S0_step) is clipped to the limit and marked CLIPPED. |
| `bounds` | `[lo, hi]` per parameter; a step that would leave the range is clipped and recorded as such. |
| `priority` | directions to put first in the task list, so the arms you most want to see come back first. Ordering only. |

`delta` is a compromise: large enough that the effect clears the replicate noise,
small enough to still be a local derivative. 0.9 e-folds was used here, and the
resulting `D` values were checked against a half-step for linearity.

## Reading a row

A direction is worth searching if `|D|` is sizeable **and** `SNR ≥ 3`. `SNR`
below 3 means the screen could not resolve the effect at all, and searching such
a direction spends the whole run learning nothing.

A screen is only meaningful at a **growing** operating point. At GR ≈ 0 every
`D` is structurally zero and the screen tells you nothing about which parameters
matter.

Because a barrier enters as `d ln k = -dEa/T`, one direction's measured `D(T)`
gives the effect of the corresponding barrier at every temperature by the chain
rule, `d ln GR(T)/d Ea = -D(T)/T`. That is why 20 directions cover all 54
parameters, and it is what `../mcts/build_search.py` consumes.
