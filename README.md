# REX-Grid

**R**einforcement learning for **R**enewable **EX**pansion with **Grid**-consistent system evaluation

This repository contains the code and model inputs used in:

> Zhou Z, Cai G, Chen Y, Nie S, Du W, He G. *Load-aligned renewable siting reduces net-load stress and downstream flexibility needs.* Energy (manuscript EGY-D-26-10722, under revision).

Release `v1.1.0` corresponds to the revised manuscript. The scientific description of the model, its assumptions and all parameter values are given in the paper and its Supplementary Information (SI); this README explains how the code maps onto that description.

---

## 1. What the framework does

REX-Grid links where wind and PV capacity is built to the storage, transmission and dispatch that the power system needs afterwards. It has two stages.

| Stage | Module | Role |
|---|---|---|
| 1. Siting | `REX/` | A proximal policy optimization (PPO) agent allocates county-level onshore wind, offshore wind and PV additions in one province over seven planning stages (2020–2050). |
| 2. System evaluation (PSTE) | `Grid/` | The power-system transition and expansion model (PSTE) takes each siting portfolio as given and optimizes thermal retirement and new build, CCS retrofits, storage, interprovincial corridors, the intra-provincial collection network and hourly dispatch. |

Storage and transmission are decided only in Stage 2 and are not terms in the Stage 1 reward, so their changes measure how the power system responds to the siting objective.

**Study region.** Five provinces of South China: Guangdong (code 44), Guangxi (45), Hainan (46), Guizhou (52) and Yunnan (53).

**Siting strategies compared in the paper.**

| Strategy | Code mode |
|---|---|
| CMP: complementarity-oriented (the PPO agent) | `RL` |
| LCO: least-cost, ranked by levelized cost of electricity | `LCOE` |
| GRD: greedy, ranked by resource potential | `Greedy` |

All three strategies add the same wind–PV capacity.

---

## 2. Stage 1: REX (siting)

- **Capacity additions per province:** Guangdong 500 GW, Guangxi 120 GW, Hainan 40 GW, Guizhou 80 GW and Yunnan 115 GW, released in seven equal stage quotas. Each action proposes additions in 100-MW increments, which are then scaled to the stage quota and capped by the remaining county potential.
- **Weather:** hourly historical MERRA-2 reanalysis. During training, each month is represented by a randomly drawn historical 7-day block from the same calendar month. No climate projections are used.
- **Reward (Eq. 9 of the paper):** four indicators are computed for each month, clipped as below, and averaged over the twelve months. Policies are trained separately for each province.

| Paper symbol | Code name | Definition (V = wind + PV output, L = load) | Weight and clipping |
|---|---|---|---|
| C<sub>var</sub> | `CI1` | 1 − Var(V) / (Var(wind) + Var(PV)) | 1200 if positive, 500 if negative; monthly clip to [−1, 1]; sign split applied to the annual mean |
| C<sub>nl</sub> | `CI2` | 1 − Var(L − V) / Var(L) | 50; monthly clip to [−1, 1] |
| C<sub>τ</sub> | `C3_kendall` | Kendall's τ between V and L | 100 |
| C<sub>CV</sub> | `C4_CV` | std(V) / mean(V) | contributes 30 × (2 − C<sub>CV</sub>); monthly clip to [0, 2], so C<sub>CV,max</sub> = 2 |

- **PPO settings:** see SI Table S23 (`REX/configs/ppo_table_S23.json`).

---

## 3. Stage 2: Grid (PSTE)

- **Problem class:** a linear program solved with Gurobi. There are no integer or binary variables.
- **Planning stages:** seven sequential five-year stages (2020, 2025, …, 2050). Each stage inherits the thermal and CCS stock, storage vintages and corridor capacities of the previous stage. There is no perfect foresight across stages.
- **Temporal resolution:** 2,016 hours per stage. For each month and province, candidate daily wind–PV profiles from 500 sampled historical weeks are clustered with k-medoids into seven representative days, and extreme-output candidates enter the selection (`Grid/typical_week_selector.py`). Hourly operating quantities are scaled to the year by 8760/2016.
- **Thermal commitment:** continuous aggregated (relaxed) unit commitment for provincial thermal technology groups. It covers online, startup and shutdown capacity, startup costs, minimum up/down times, output bounds and ramping limits.
- **Adequacy and reserves:** an hourly planning reserve margin on provincial net load, and a spinning-reserve requirement met by thermal and storage headroom.
- **Storage:** pumped hydro, lithium-ion batteries, compressed-air and vanadium redox flow batteries.
- **Networks:**
  - interprovincial corridor reinforcement;
  - county-to-city spur and city-to-city trunk links, charged at USD 1,181.18 per MW·km plus a substation term of USD 38,000 per MW.
- **Carbon:** annual provincial CO₂ caps under the NDC, GM2.0 and CN2050 pathways, with a penalized slack of USD 100 per tonne. CCS retrofits are limited to vintage-eligible thermal capacity; CCS costs follow SI Table S12.
- **Demand:** low, medium and high growth (L, M, H).
- **Costs:** in 2020 USD. Path costs reported in the paper discount each stage cost to 2020 at 7% per year, i.e. by (1.07)<sup>−5k</sup> for stage k = 0, …, 6. `Grid/main_multiyear.py` reports the same quantity.

---

## 4. Repository structure

```text
REX/        Stage 1 environment, reward, wind/PV generation models and PPO training
Grid/       Stage 2 PSTE model (dispatch_model.py), stage loop (main_multiyear.py),
            representative-day selection, intra-provincial network and portfolio loader
config/     paths.json: all input and output paths, relative to the repository root
data/       model inputs and the siting portfolios used in the paper (data/expansion/{RL,LCOE,Greedy}/)
```

`Grid/` is a cleaned implementation of the PSTE code used for the paper.

---

## 5. Installation and usage

**Requirements:**
- Python 3.10;
- `pip install -r requirements.txt`;
- a Gurobi licence for Stage 2 (free academic licences are available).

**Data:**
- Hourly MERRA-2 wind and radiation files are not included because of their size. Place them under `data/weather/` as listed in `config/paths.json`. Sources and processing are described in SI Section 2.
- All other inputs are included: decision-point table, load profiles, thermal fleet, existing transmission, hydro and nuclear factors.
- The siting portfolios evaluated in the paper are included under `data/expansion/`, so Stage 2 can be run without retraining.

**Commands (from the repository root):**

```bash
# Stage 1: train a provincial agent (example: Guangdong)
python -m REX.train_ppo_gd --province 44 --config REX/configs/ppo_table_S23.json

# Stage 2: evaluate a strategy under the M demand pathway
python -m Grid.main_multiyear --mode RL --carbon CN2050
```

The Stage 2 entry currently exposes the M demand pathway (the retained 4% then 2% annual load-growth implementation). Use `--help` to inspect available options. `python -m REX.train_ppo_gd --check-config` validates the configuration without training or loading weather.

**Outputs:**
- `multi_year_dispatch_results.xlsx`: hourly dispatch by stage;
- `summary_tables.xlsx`: capacities, network, emissions and cost components.

Representative days are selected for each run, so repeated runs can differ slightly in absolute values (SI Section 6.4).

---

## 6. Revision experiments

Code for the revision experiments is available from the authors on request.

---

## 7. Licence and citation

The code is released under the MIT Licence (see `LICENSE`). If you use REX-Grid, please cite the paper above (see `CITATION.cff`).
