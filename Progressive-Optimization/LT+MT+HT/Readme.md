# LT + MT + HT

## File structure

```
LT+MT+HT/
├── Results/                           Result files of the stage
│   ├── sensitivity.csv                Film height based sensitivity
│   ├── sensitivity_films.csv          KMC simulations for height sensitivity
│   ├── sensitivity.png
│   ├── ladder.csv                     Parameter values along the ladder
│   ├── ladder_films.csv               KMC simulations along the ladder
│   ├── ladder.png
│   ├── R10_045_gr_vs_T.png
│   ├── phase-1_gr_vs_T.csv
│   ├── phase-1_gr_vs_T_films.csv
│   ├── phase-1_gr_vs_T.png
│   ├── phase-1_parameters_start.json  Rate catalog before LT+MT+HT phase 1
│   ├── phase-1_parameters_end.json    Rate catalog after LT+MT+HT phase 1
│   ├── drc.csv                        Degree of rate control for rate catalog
│   ├── drc_films.csv                  KMC simulations for DRC study
│   ├── drc.png
│   ├── mcts_nodes.csv                 MCTS search for rate parameters
│   ├── mcts_films.csv                 KMC simulations during MCTS search
│   ├── mcts.png
│   ├── best_node_parameters.json      Rate catalog of the best MCTS node
│   ├── phase-2_gr_vs_T.csv
│   ├── phase-2_gr_vs_T_films.csv
│   ├── phase-2_gr_vs_T.png
│   ├── phase-2_parameters_start.json  Rate catalog before LT+MT+HT phase 2
│   └── phase-2_parameters_end.json    Rate catalog after LT+MT+HT phase 2
└── Reproduction/                      Reproduction of LT+MT+HT stage
    ├── Phase-1_GR_vs_T_start/         config.toml, root_fit_LTMT.json
    ├── Phase-1_Sensitivity/           config.toml, root_fit_LTMT.json, sens_params/
    ├── Phase-1_Ladder/                config.toml, ladder.toml, ladder_params/, root_fit_LTMT.json
    ├── Phase-1_GR_vs_T_end/           config.toml, root_R10_045.json
    ├── Phase-2_DRC/                   config.toml, drc.json, root_R10_045.json
    ├── Phase-2_MCTS/                  config.toml, drc_table.csv, drc_table_notes.txt, root_R10_045.json
    └── Phase-2_GR_vs_T_end/           config.toml, best_params_node31.json
```

The last stage fits all 15 of Lin's temperatures. At the LT + MT fit (node 52) no film grows at 1225 or 1230 K, so, as at 1125 K, it runs in two phases: Phase 1 finds a set at which every temperature grows (the height sensitivity at 1225–1230 K, then a ladder along R10, two-bond film-atom desorption); Phase 2 measures the DRC there and searches. The search's lowest-loss node, node 31, is the final fit: RMS error 6.7 %, worst 12.5 % (950 K).
