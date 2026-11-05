# `mcts/` — the search

Monte Carlo tree search over a handful of KMC parameters, fitting a measured
growth-rate-versus-temperature curve. Each node is a parameter set; evaluating a
node means running the KMC at every temperature and scoring the result.

```
python build_search.py examples/lt1125_four_lever.json
python MCTS.py --config config.toml
```

`build_search.py` writes `config.toml`, `params_root.json` and
`ROOT_SOURCE.json`. Nothing in the config is typed in: the root comes from a
parameter file, the lever effects from a DRC table, the targets from the measured
curve.

## The objective

Over all temperatures at once, in e-folds:

```
L = sqrt( mean_T [ ln(GR_T / GR_exp_T)² + max(0, ln(h_min / h_T))² ] )
R = exp(-L / tau)
```

Two things are deliberate.

**The error is logarithmic.** The measured rates span more than a factor of two
across the ladder; a squared *absolute* error would let the search buy a good
total by fitting the fastest temperature and abandoning the slowest. In e-folds
every temperature is worth the same, and `L = 0.1` means "within about 10 % —
everywhere".

**The film-thickness term.** GR_T is read through the 50-200 Å window and h_T is
the mean height the films end at. A film that stops growing before the window top
scores a rate of 0 and ends thin. With `h_min_A` at the window top, the one-sided
penalty costs nothing when every film grew through the whole window and grows
sharply otherwise, and a node is a result only if every film did.

`R = exp(-L/tau)` turns the error into a reward in `(0, 1]` for the tree to back
up. `tau` sets how much of an improvement counts as one: at `tau = L_root/3`, the
default, a node three times better than the root scores `e⁻¹`.

## Selection

```
U = max_reward(child) + C · sqrt( ln N_total / n_child )
```

with **max** backup, not mean. This is a search for one best parameter set, not
for a good policy in a stochastic game — a branch that contains an excellent node
is a good branch even if most of its siblings are poor, and averaging would hide
exactly the node being looked for.

`C_ucb = "auto"` sets `C` from the DRC itself: the spread of rewards the DRC
predicts for one step in each lever, divided by `sqrt(ln max_experiments)`. That
puts the exploration bonus on the same scale as a real move's effect, measured
rather than tuned. Set a number instead to override it.

## Expansion

A child is not a random jitter of its parent. `build_search.py` bakes the DRC's
measured effect of one step of each lever, per temperature, into
`[drc_rate.T*]` and `[drc_height.T*]`, and the expander:

1. solves, by least squares, for the smallest step vector that would close every
   temperature's rate gap **and** lift any film back above the floor;
2. clips it to the trust region `max_step_efolds` and the lever bounds;
3. takes a fraction `u ~ U[0,1)` of it, so children spread along the plan rather
   than piling up at its end;
4. adds a random kick sized so the child is never *predicted* to be worse than
   its parent.

With `planner_map = "learned"`, the effect matrix is refitted from the tree's own
evaluated moves as the search proceeds — Bayesian linear regression with the DRC
as the prior — so the plan improves where the DRC's local estimate turns out to
be wrong.

## Step units

A multiplicative lever (a prefactor or sticking coefficient) is multiplied by `exp(s)` for
a step `s`. An additive lever — a barrier, in kelvin — gains `s × step_ref_T`
kelvin. Since `d ln k = -dEa / T`, its effect on the rate at temperature `T`
carries the factor

```
f = -step_ref_T / T
```

which is stronger at low temperature. That asymmetry is what lets a few levers
fit a curve rather than a single point, and `build_search.py` bakes the factor
into the config so `MCTS.py` never converts anything.

## Replicates

Every node is measured to `n_rep_total` replicates at **every** temperature. It is
then extended a wave at a time while either of these holds, up to `n_rep_max`:

- a temperature's running mean growth rate has not gone flat, or
- its standard error `CV/√n` is still above `sem_target_pct`.

**The node's rate is the mean of all its replicates.**

**Flat** is defined in `kmc/flat_region.py`, and `run_kmc.py`'s adaptive mode uses the
same rule:

- The band is the full mean ± its standard error.
- `n_flat` is the first n from which the running mean stays inside that band **and**
  has reached the middle of it (within half a standard error).
- The curve counts as flat if `n_flat` comes before the last 20 % of the replicates.

The first test is retrospective, the second predictive. A mean that looks settled at
96 but has a large CV still gets more replicates. `CV` is physics and does not shrink
with `n`; only the standard error does. So never size `n` from a `CV` measured at some
other operating point.

Until 2026-09-28 the running mean was read at a fixed ±1 % tolerance
(`n_rep_wiggle_pct`, now refused). With films that scatter by less than 1 %, that
returned `n_flat = 1` and scored every node on one film.

## The search JSON

```json
{
  "root_parameters": "params/stage01_1125K_winner.json",
  "root_md5":        "3058766269a32bbd2fb3cf5fb52aa0b8",
  "drc_table":       "drc/drc_LT1125_v2.csv",
  "target_curve_csv": "curves/lin1977_gr_vs_T.csv",
  "reference_temperature_K": 1125,

  "levers": ["Ea_chem_M", "Ea_chem_As", "Ea_E32", "Ea_E33"],
  "barrier_span_efolds": 8.0,
  "lever_selection": {"min_snr": 3.0},

  "objective": {"h_min_A": 200.0, "error_threshold": 0.10},

  "search": {"max_experiments": 60, "max_depth": 20, "max_children": 5,
             "max_step_efolds": 2, "C_ucb": "auto", "tau": "auto",
             "planner_map": "learned", "expand_bias": 0.7},

  "kmc": { "...": "the same run settings as kmc/README.md, plus n_rep_* " }
}
```

| key | meaning |
|---|---|
| `drc_table` | the reduced DRC, with columns `direction, key, T_K, D, D_height, SNR`. The temperatures of the search are the temperatures in this table. |
| `levers` | parameter names. Omit to let `lever_selection` pick the top `top` directions by their largest combined pull `max_T sqrt(D² + D_height²)`. |
| `lever_selection.min_snr` | a lever the DRC never resolved anywhere is refused either way. |
| `barrier_span_efolds` | default bounds for a barrier: `±span × step_ref_T` kelvin, i.e. ±`span` e-folds of `ln k` at the reference temperature. |
| `reference_temperature_K` | `step_ref_T`. Usually the highest temperature, the one already fitted. |
| `objective.error_threshold` | the stopping criterion, in e-folds, **per temperature** on the rate. The search declares success only when every temperature is inside it and every film is above `h_min_A`. |

**Parameters already fitted at an earlier rung belong in `[params]`, not in
`[search_params]`.** Anything not named in `levers` is pinned automatically,
which is the point of naming them explicitly.

## Cost, before you launch

A node costs `n_replicates × n_temperatures` trajectories, and a low-temperature
trajectory can be hours. Measure the per-step cost at the *stage's own*
parameters — it differs by an order of magnitude between a dead surface and a
growing one — and multiply out before committing cluster time. `max_experiments`
nodes is the budget; the tree does not know how to stop early if the fit is
unreachable.

Which is worth checking first: bounded least squares on the DRC effect matrix
will tell you the smallest error the chosen levers can reach *at all*, within
their bounds, before a single trajectory is run. If that number is above
`error_threshold`, the lever set is the problem, not the search.
