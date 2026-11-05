# 1125 K

## File structure

```
1125K/
├── Results/                           Result files of the stage
│   ├── sensitivity.csv                Film height based sensitivity
│   ├── sensitivity_films.csv          KMC simulations for height sensitivity
│   ├── sens_ranking_1.png … _5.png
│   ├── 1125K_Height.png
│   ├── inflation.csv                  Inflated parameter values
│   ├── inflation_films.csv            KMC simulations at the inflated values
│   ├── inflation.png
│   ├── phase-1_gr_vs_T.csv            
│   ├── phase-1_gr_vs_T_films.csv      
│   ├── phase-1_gr_vs_T.png
│   ├── phase-1_parameters_start.json  Rate catalog before 1125K phase 1
│   ├── phase-1_parameters_end.json    Rate catalog after 1125K phase 1
│   ├── drc.csv                        Degree of rate control for rate catalog  
│   ├── drc_films.csv                  KMC simulations for DRC study
│   ├── drc.png
│   ├── mcts_nodes.csv                 MCTS search for rate parameters
│   ├── mcts_films.csv                 KMC simulations during MCTS search
│   ├── mcts.png
│   ├── fit_height_and_GR_1125K.png
│   ├── phase-2_gr_vs_T.csv            
│   ├── phase-2_gr_vs_T_films.csv      
│   ├── phase-2_gr_vs_T.png
│   ├── phase-2_parameters_start.json  Rate catalog before 1125K phase 2
│   └── phase-2_parameters_end.json    Rate catalog after 1125K phase 2
└── Reproduction/                      Reproduction of 1125K phase 
    ├── Phase-1_GR_vs_T_start/         config.toml, params_catalog_defaults.json
    ├── Phase-1_Sensitivity/           config.toml, sens_params/ (the parameter set of every arm, 109 files)
    ├── Phase-1_Inflation/             config.toml, lambda_params/ (the parameter set at every λ, 9 files)
    ├── Phase-1_GR_vs_T_end/           config.toml, root_lambda_1.00.json (also where Phase 2 starts)
    ├── Phase-2_DRC/                   config.toml, drc.json (the directions), root_lambda_1.00.json
    ├── Phase-2_MCTS/                  config.toml, root_lambda_1.00.json
    └── Phase-2_GR_vs_T_end/           config.toml, fitted_1125K_phase2.json
```

The experimental growth rate peaks at 1125 K, so this temperature is fitted first. This phase consists of two parts.

## Why two parts?

The growth rate (GR) is the rate at which the mean film height rises from 50 to 200 Å. With the catalog parameters the film does not grow at 1125 K (after 10<sup>6</sup> KMC events the mean film height is −0.14 Å and the growth front is 2 Å above the starting surface). Since no film reaches the window, there is not a GR to compare with experiment. A DRC is blind here too.

**Phase 1 — Sensitivity and Inflation** is not MCTS. The aim of this phase is to arrive at a parameter set at which the film grows. 

- *Sensitivity* moves every parameter up and down and measures its effect on the film height. The dominant "lever" parameters are then chosen and inflated in the step.
- *Inflation* moves the levers together, λ e-folds from the catalog in the direction that raises the film. The smallest λ at which every film grows through the window is chosen. This becomes the starting parameter set for MCTS stages.

**Phase 2 — Search** starts from the parameter set established above after inflation. GR is now measurable, so a DRC can rank the parameters that actually move it, and MCTS optimizes them until the simulated GR matches experiment at 1125 K.

## Sensitivity with respect to height, not GR

Without growth there is no GR, but the film height still exists. The screen therefore reads the mean film height $h$ after a fixed budget of 10<sup>6</sup> events (by this point every height has settled). For each parameter $p$:

$$S_h = \frac{\partial h}{\partial \ln k} \approx \frac{1}{N}\sum_{n=1}^{N} \frac{h_n^{+}-h_n^{-}}{\delta^{+}-\delta^{-}}$$

in Å per e-fold of the rate constant $k$ that $p$ enters. The difference of the heights is used, not the log of their ratio, because the heights sit near zero and can be negative.

- Each of the 54 parameters is moved on its own, both ways, by 0.9 e-folds of its rate constant at 1125 K: $A_i$ and $S_{0(j,s)}$ are multiplied by $e^{\pm 0.9}$, $E_{ai}$ and $E_{a,\text{chem}(j)}$ are shifted by $\mp 0.9 \times 1125$ K, and $\beta_i$ by $\pm 0.9/\ln 1125$.
- An arm clipped by an ordering constraint ($S_{0(j,\text{terrace})} \leq S_{0(j,\text{step})}$) is divided by the step it actually made, $\delta^{\pm} = \ln(k^{\pm}/k_0)$.
