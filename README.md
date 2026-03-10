# REX-Grid: Reinforcement Learning for Renewable Energy Siting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**REX-Grid** is an advanced modeling framework that couples Deep Reinforcement Learning (DRL) with a long-term power system dispatch model. It optimizes the spatial siting of renewable energy (wind and solar) across multiple provinces to enhance source-side flexibility, grid complementarity, and overall economic efficiency over a 30-year planning horizon (2020-2050).

---

## 🧠 Project Logic & Architecture

The framework is conceptually divided into two major interconnected pipelines:

**1. REX (Reinforcement Learning Siting Engine)**
* **Role**: Acts as the "Spatial Planner".
* **Mechanism**: An RL agent (PPO) interacts with a custom `Gymnasium` environment. It observes regional renewable potentials and evaluates system-level complementarity indices (Kendall's Tau, Coefficient of Variation).
* **Output**: Generates optimized capacity expansion trajectories for onshore wind, offshore wind, and solar PV across different regions and years.

**2. Grid (Power System Dispatch & Evaluation Engine)**
* **Role**: Acts as the "System Evaluator".
* **Mechanism**: A dual-layer mixed-integer linear programming (MILP) model powered by Gurobi. It takes the spatial deployment plans from REX, clusters 8760-hour meteorological data into representative weeks, and optimizes:
    * *Investment Layer*: Thermal power early retirement, CCS retrofitting, energy storage deployment, and inter-provincial transmission network expansion.
    * *Operational Layer*: Hourly unit commitment, power flow, and curtailment.
* **Output**: Comprehensive 30-year technical and economic evaluations (NPV, CO2 emissions, hourly dispatch profiles).

---

## 📂 Repository Structure

* **`data/`**: The central data hub. Contains all raw inputs, meteorological NetCDF files, load curves, and pre-calculated capacity expansion trajectories (in `data/expansion/`).
* **`REX/`**: The RL engine. Contains vectorized wind/solar generation models and the PPO training environment.
* **`Grid/`**: The Gurobi-based power system dispatch and evaluation engine.

---

## ⚙️ How to Run

### Step 1: Environment Setup
1. Clone the repository to your local machine.
2. Create a virtual environment and install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. **Gurobi License**: The `Grid` dispatch model requires [Gurobi](https://www.gurobi.com/). Please ensure you have a valid Academic or Commercial license configured in your environment.

### Step 2: Path Configuration (Critical!)
Before running any scripts, you **MUST** update the absolute data paths in the code to match your local repository directory.
1. Open `REX/compute_power_gap.py`: Update the paths in the `Config` class to point to your local `data/` folder.
2. Open `REX/train_ppo_gd.py`: Update `decision_points_excel` to point to your initial state table.
3. Open `Grid/main_multiyear.py`: Update `thermal_csv_path`, `storage_excel_path`, `transmission_csv_path`, etc., under the `run_case` function.
4. Open `Grid/utils_yearly.py`: Update `PROV_DIR_MAP_rl`, `PROV_DIR_MAP_greedy`, and `PROV_DIR_MAP_lcoe` to correctly point to the `data/expansion/` subdirectories.

### Step 3: Train the RL Agent (Optional)
If you want to re-train the RL agent to generate new siting strategies for a specific province (e.g., Guangdong, Province ID = 44), run:
```bash
python REX/train_ppo_gd.py
```
*Note: You can monitor the training progress via TensorBoard (`tensorboard --logdir=./logs/`).*

### Step 4: Run the Multi-Year Dispatch & Evaluation
To evaluate the long-term system performance and economic costs based on the expansion plans, run the core dispatch engine:
```bash
python Grid/main_multiyear.py
```
This script will loop through the specified expansion modes (RL, Greedy, LCOE) and carbon scenarios.

### Step 5: Check Outputs
Once Step 4 finishes, check your designated output directory. The model generates two primary reports:
* `multi_year_dispatch_results.xlsx`: Highly detailed hourly dispatch logs for every simulated year.
* `summary_tables.xlsx`: A macro-level summary of coal/gas fleet capacities, grid expansions, annual CO2 emissions, and detailed cost breakdowns (NPV).