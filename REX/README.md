# REX: renewable siting

The PPO environment allocates county-level wind and PV additions over seven
2020–2050 planning stages. Its four reward terms are C_var (`CI1`), C_nl (`CI2`),
C_tau (`C3_kendall`) and C_CV (`C4_CV`).

Paths are in `config/paths.json`, and the current SI Table S23 configuration is
`REX/configs/ppo_table_S23.json`. From the repository root:

```bash
python -m REX.train_ppo_gd --province 44 --config REX/configs/ppo_table_S23.json
```

Add `--check-config` to validate configuration without training or loading weather.
The LCO and GRD baseline sources are in `RL_main/`; their model expressions are
retained and their paths adapted for this repository. See the
[main README](../README.md) for data requirements and the scientific description.
