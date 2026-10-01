import os
import time
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
from scipy.stats import kendalltau
from concurrent.futures import ThreadPoolExecutor

# 引入重构后的类
from compute_power_gap import PowerGapCalculator
from bridge_decision import read_decision_points


def _compute_monthly_CI(calculator: PowerGapCalculator, year: int, month: int,
                        decision_points: pd.DataFrame) -> pd.DataFrame:
    """计算单月互补性与系统指标：CI1、CI2'、C3(坎德尔系数)、C4(变异系数)"""
    # 修复点 1：通过传入的 calculator 实例调用 compute_gap
    gap_df, load_df, clean_df = calculator.compute_gap(year, month, decision_points)

    wind_arr = clean_df['Wind'].values
    solar_arr = clean_df['Solar'].values
    load_arr = load_df.values
    wind_solar_sum = wind_arr + solar_arr

    var_w = np.nanvar(wind_arr, axis=0)
    var_s = np.nanvar(solar_arr, axis=0)
    var_ws = np.nanvar(wind_solar_sum, axis=0)
    var_l = np.nanvar(load_arr, axis=0)
    var_lws = np.nanvar(load_arr - wind_solar_sum, axis=0)

    # CI1 = C_var in the paper.
    sum_var = var_w + var_s + 1e-6
    ci1_arr = np.clip((sum_var - var_ws) / sum_var, -1.0, 1.0)
    np.nan_to_num(ci1_arr, copy=False, nan=0.0)

    # CI2 = C_nl in the paper.
    ci2_prime_arr = np.clip((var_l - var_lws) / (var_l + 1e-6), -1.0, 1.0)
    np.nan_to_num(ci2_prime_arr, copy=False, nan=0.0)

    # C3_kendall = C_tau: Kendall's Tau
    provinces = clean_df['Wind'].columns
    num_provinces = len(provinces)
    c3_kendall_arr = np.zeros(num_provinces, dtype=np.float32)
    for idx in range(num_provinces):
        ws_series = wind_solar_sum[:, idx]
        load_series = load_arr[:, idx]
        if np.std(ws_series) < 1e-6 or np.std(load_series) < 1e-6:
            c3_kendall_arr[idx] = 0.0
        else:
            tau, _ = kendalltau(ws_series, load_series)
            c3_kendall_arr[idx] = tau if not np.isnan(tau) else 0.0

    # C4_CV = C_CV: coefficient of variation
    mean_ws = np.nanmean(wind_solar_sum, axis=0)
    std_ws = np.nanstd(wind_solar_sum, axis=0)
    c4_cv_arr = np.clip(std_ws / (mean_ws + 1e-6), 0.0, 2.0)
    np.nan_to_num(c4_cv_arr, copy=False, nan=0.0, posinf=2.0, neginf=0.0)

    data = np.stack([ci1_arr, ci2_prime_arr, c3_kendall_arr, c4_cv_arr], axis=1)
    return pd.DataFrame(data, index=provinces, columns=['CI1', 'CI2', 'C3_kendall', 'C4_CV'])


def compute_yearly_CI(calculator: PowerGapCalculator, decision_points_df: pd.DataFrame, year: int, *,
                      use_thread_pool: bool = False, max_workers: Optional[int] = None):
    """计算全年 CI（每月一次）"""
    months = range(1, 13)
    if use_thread_pool and (max_workers or os.cpu_count() > 2):
        if max_workers is None:
            max_workers = min(4, os.cpu_count() // 2)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            # 修复点 2：把 calculator 一起传进去
            dfs = list(pool.map(lambda m: _compute_monthly_CI(calculator, year, m, decision_points_df), months))
    else:
        # 修复点 2：把 calculator 一起传进去
        dfs = [_compute_monthly_CI(calculator, year, m, decision_points_df) for m in months]

    all_df = pd.concat(dfs, axis=0)
    grouped = all_df.groupby(all_df.index).mean()

    return {
        str(prov): [
            grouped.loc[prov, 'CI1'],
            grouped.loc[prov, 'CI2'],
            grouped.loc[prov, 'C3_kendall'],
            grouped.loc[prov, 'C4_CV']
        ]
        for prov in grouped.index
    }


class DispatchBridgeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, province_num, decision_points_excel, start_year=2020, terminal_year=2050, total_budget_mw=500000,
                 years_to_meet=7):
        super(DispatchBridgeEnv, self).__init__()
        self.decision_points_excel = decision_points_excel
        self.start_year = start_year
        self.terminal_year = terminal_year
        self.current_year = self.start_year

        self.TOTAL_BUDGET = float(total_budget_mw)
        self.YEARS_TO_MEET = int(years_to_meet)
        self.YEAR_QUOTA = self.TOTAL_BUDGET / self.YEARS_TO_MEET
        self.remaining_budget = self.TOTAL_BUDGET
        self.global_step = 0

        self.province_num = int(province_num)

        full_df = read_decision_points(self.decision_points_excel)
        mask = full_df["Provence_num"] == province_num
        if mask.sum() == 0:
            raise ValueError(f"province_num={province_num} 不在决策点表中！")
        self._prov_decision_points_template = full_df.loc[mask].reset_index(drop=True)

        self.decision_points_df = self._prov_decision_points_template.copy()
        self.n_regions = self.decision_points_df.shape[0]

        # 修复点 3：实例化全局唯一的计算器，预加载气象数据！
        self.calculator = PowerGapCalculator()

        # 初始化状态（传入 self.calculator）
        provincial_agg = compute_yearly_CI(self.calculator, decision_points_df=self.decision_points_df,
                                           year=self.current_year)
        self.state = self._build_state_from_df(self.decision_points_df, provincial_agg)

        # 状态空间定义
        per_region_low = np.array([0.0, 0.0, 0.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32)
        per_region_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0], dtype=np.float32)
        low0 = np.tile(per_region_low, self.n_regions)
        high0 = np.tile(per_region_high, self.n_regions)
        global_low = np.array([0.0, 0.0], dtype=np.float32)
        global_high = np.array([1.0, 1.0], dtype=np.float32)
        low = np.concatenate([low0, global_low]).astype(np.float32)
        high = np.concatenate([high0, global_high]).astype(np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # 动作空间：每个区域 3 维扩建决策
        action_dim = self.n_regions * 3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

    def _build_state_from_df(self, df: pd.DataFrame, provincial_agg: dict):
        inst = df[["Installed_onshorewind", "Installed_offshorewind", "Installed_pv"]].values
        pot = df[["Potential_onshorewind", "Potential_offshorewind", "Potential_pv"]].values + 1e-6
        remain = np.clip(1 - inst / pot, 0.0, 1.0)
        remain = np.nan_to_num(remain, nan=0.0, posinf=1.0, neginf=0.0)

        prov_ids = df["Provence_num"].astype(str).values
        prov_feats = np.vstack([provincial_agg.get(p, [0, 0, 0, 0]) for p in prov_ids])
        prov_feats = np.nan_to_num(prov_feats, nan=0.0, posinf=1e6, neginf=-1e6)

        region_states = np.hstack([remain, prov_feats])
        flat_regions = region_states.ravel()

        budget_ratio = self.remaining_budget / self.TOTAL_BUDGET
        years_left = (self.terminal_year - self.current_year) / self.YEARS_TO_MEET

        state = np.concatenate([flat_regions, [budget_ratio, years_left]])
        state = np.nan_to_num(state, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.clip(state, -1e6, 1e6).astype(np.float32)

    def compute_reward(self, provincial_agg: dict, decision_points_df: pd.DataFrame, weights: dict = None) -> dict:
        if weights is None:
            weights = {"w1_pos": 1200, "w1_neg": 500, "w2": 50.0, "w3": 100.0, "w4": 30.0}

        inst_cols = ["Installed_onshorewind", "Installed_offshorewind", "Installed_pv"]
        cap_series = decision_points_df.groupby("Provence_num")[inst_cols].sum().sum(axis=1)
        cap_total = cap_series.sum() + 1e-8
        provs_str = [str(p) for p in cap_series.index]

        share_array = np.array([cap_series.iloc[i] / cap_total for i in range(len(provs_str))], dtype=np.float32)

        c1 = np.array([provincial_agg.get(p, [0] * 4)[0] for p in provs_str], dtype=np.float32)
        c2 = np.array([provincial_agg.get(p, [0] * 4)[1] for p in provs_str], dtype=np.float32)
        c3 = np.array([provincial_agg.get(p, [0] * 4)[2] for p in provs_str], dtype=np.float32)
        c4 = np.array([provincial_agg.get(p, [0] * 4)[3] for p in provs_str], dtype=np.float32)

        R1_vec = weights["w1_pos"] * np.clip(c1, 0, None) - weights["w1_neg"] * np.clip(-c1, 0, None)
        R2_vec = weights["w2"] * c2
        R3_vec = weights["w3"] * c3
        R4_vec = weights["w4"] * (2.0 - c4)

        comp_terms = R1_vec + R2_vec + R3_vec + R4_vec
        comp_reward = float(np.dot(share_array, comp_terms))

        return {"comp": comp_reward, "R1": float(R1_vec[0]), "R2": float(R2_vec[0]), "R3": float(R3_vec[0]),
                "R4": float(R4_vec[0])}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.current_year = self.start_year
        self.remaining_budget = self.TOTAL_BUDGET
        self.decision_points_df = self._prov_decision_points_template.copy()

        # 修复点 4：重置时也要传 self.calculator
        provincial_agg = compute_yearly_CI(self.calculator, decision_points_df=self.decision_points_df,
                                           year=self.current_year)
        self.state = self._build_state_from_df(self.decision_points_df, provincial_agg)

        info = {"year": self.current_year, "remaining_budget": self.remaining_budget}
        return self.state.copy(), info

    def step(self, action):
        self.global_step += 1

        action = np.clip(action, -1.0, 1.0).reshape(self.n_regions, 3)
        df = self.decision_points_df
        inst_cols = ["Installed_onshorewind", "Installed_offshorewind", "Installed_pv"]
        pot_cols = ["Potential_onshorewind", "Potential_offshorewind", "Potential_pv"]

        installed = df[inst_cols].values.astype(np.float32)
        potential = df[pot_cols].values.astype(np.float32)
        remain = np.clip(potential - installed, 0.0, None)

        max_steps = (remain // 100).astype(np.int32)
        step_choice = np.floor(((action + 1.0) * 0.5) * max_steps)
        desired_add = np.clip(step_choice * 100.0, 0.0, remain)

        max_allow = min(self.remaining_budget, self.YEAR_QUOTA)
        real_scale = np.minimum(1., max_allow / (desired_add.sum() + 1e-6))
        real_add = np.clip(real_scale, 0.0, 1.0) * desired_add
        year_add = float(real_add.sum())

        self.remaining_budget -= year_add
        new_installed = np.minimum(installed + real_add, potential)
        for i, col in enumerate(inst_cols):
            self.decision_points_df[col] = new_installed[:, i]

        # 修复点 5：step 迭代时也要传 self.calculator
        provincial_agg = compute_yearly_CI(self.calculator, decision_points_df=self.decision_points_df,
                                           year=self.current_year)
        self.state = self._build_state_from_df(self.decision_points_df, provincial_agg)

        self.current_year += 5
        terminated = (self.current_year > self.terminal_year)
        truncated = False

        rew_dict = self.compute_reward(provincial_agg=provincial_agg, decision_points_df=self.decision_points_df)

        info = {
            "year": self.current_year - 5,
            "year_add": year_add,
            "remaining_budget": self.remaining_budget,
            "comp_reward": rew_dict["comp"],
            "R1": rew_dict["R1"], "R2": rew_dict["R2"], "R3": rew_dict["R3"], "R4": rew_dict["R4"],
            "decision_points": self.decision_points_df.copy()
        }

        return self.state.copy(), rew_dict["comp"], terminated, truncated, info

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]