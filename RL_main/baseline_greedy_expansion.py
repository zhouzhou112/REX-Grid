# baseline_greedy_expansion.py
"""
Greedy baseline: ignore wind–solar complementarity, expand purely
by capacity factor ranking.

Author : ChatGPT (baseline helper)
Date   : 2025-05-28
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "REX"))
from config.paths import input_path
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==== 项目内工具函数（按实际路径修改） ====
from bridge_decision import read_decision_points          # 读取决策点 Excel
from REX.compute_power_gap import compute_gap as compute_power_gap  # 评估风光出力
from WindSolarExpansionEnv import compute_yearly_CI
# ------------------------------------------------------------------------------
# 1. 计算每个栅格的年平均容量系数（CF）
# ------------------------------------------------------------------------------

def calc_capacity_factors(dp_df: pd.DataFrame,
                          year: int = 2020,
                          months: range = range(1, 13)) -> pd.DataFrame:
    """
    - 将 Installed_* 覆盖为 Potential_*（只在副本中改，不动原 df）
    - 调用 compute_power_gap 多次，得到 wind / solar 年发电量（MWh）
    - 计算 CF = 年发电量 / (Potential * 8760)
    返回: 新 DataFrame，仅含 idx + 三个 CF 列
    """
    df_tmp = dp_df.copy()
    for col_i, col_p in [
        ("Installed_onshorewind",  "Potential_onshorewind"),
        ("Installed_offshorewind", "Potential_offshorewind"),
        ("Installed_pv",           "Potential_pv")
    ]:
        df_tmp[col_i] = df_tmp[col_p]

    # 先准备累加器
    energy_wind, energy_pv = (
         np.zeros(len(df_tmp)), np.zeros(len(df_tmp))
    )

    for m in months:      # 逐月调度，获取 hourly generation
        gap_df, _, clean_df = compute_power_gap(year, m, df_tmp)
        # clean_df 假设返回一个 dict-like，多张表；按您实际返回结构调整
        energy_wind  += clean_df["Wind"].values.sum(axis=0)

        energy_pv  += clean_df["Solar"].values.sum(axis=0)

    hours = 168*12


    cf_pv  = energy_pv  / (df_tmp["Potential_pv"].values           * hours + 1e-6)
    cf_wind= energy_wind/(df_tmp["Potential_onshorewind"].values  * hours+df_tmp["Potential_offshorewind"].values * hours+ 1e-6)
    cf_df = pd.DataFrame({
        # "cf_onshore" : cf_on,
        # "cf_offshore": cf_off,
        "cf_wind": cf_wind,
        "cf_pv"      : cf_pv
    }, index=df_tmp.index)

    return cf_df


# ------------------------------------------------------------------------------
# 2. 贪婪扩张主流程
# ------------------------------------------------------------------------------

def greedy_expand(dp_df: pd.DataFrame,
                  cf_df: pd.DataFrame,
                  total_target_mw: float = 500_000.0,
                  wind_to_solar_ratio: float = 2.0,
                  start_year: int = 2020,
                  end_year: int = 2050,
                  step_years: int = 5) -> pd.DataFrame:
    years   = list(range(start_year, end_year + 1, step_years))
    periods = len(years)
    quota_per_period = total_target_mw / periods

    wind_ratio  = wind_to_solar_ratio / (wind_to_solar_ratio + 1)
    solar_ratio = 1 - wind_ratio

    # —— ① 根据 cf_wind 对“地点”整体排序（不拆陆/海） ——
    wind_rank = cf_df.sort_values("cf_wind", ascending=False).index.tolist()
    solar_rank = cf_df.sort_values("cf_pv",   ascending=False).index.tolist()

    dp = dp_df.copy()

    for yr in years:
        wind_quota  = quota_per_period * wind_ratio
        solar_quota = quota_per_period * solar_ratio

        # —— ② 扩建风电：对每个地点依次把陆风、海风都加进去 ——
        for idx in wind_rank:
            if wind_quota <= 0:
                break
            # 陆上
            avail_on  = dp.loc[idx, "Potential_onshorewind"]  - dp.loc[idx, "Installed_onshorewind"]
            add_on    = min(avail_on, wind_quota)
            dp.loc[idx, "Installed_onshorewind"] += add_on
            wind_quota -= add_on

            if wind_quota <= 0:
                break
            # 海上
            avail_off = dp.loc[idx, "Potential_offshorewind"] - dp.loc[idx, "Installed_offshorewind"]
            add_off   = min(avail_off, wind_quota)
            dp.loc[idx, "Installed_offshorewind"] += add_off
            wind_quota -= add_off

        # —— ③ 扩建光伏（与之前一致） ——
        for idx in solar_rank:
            if solar_quota <= 0:
                break
            avail_pv = dp.loc[idx, "Potential_pv"] - dp.loc[idx, "Installed_pv"]
            add_pv   = min(avail_pv, solar_quota)
            dp.loc[idx, "Installed_pv"] += add_pv
            solar_quota -= add_pv

        print(f"[{yr}] Wind added={quota_per_period*wind_ratio - wind_quota:.1f} MW , "
              f"Solar added={quota_per_period*solar_ratio - solar_quota:.1f} MW")

    return dp


# ------------------------------------------------------------------------------
# 3. 一键运行
# ------------------------------------------------------------------------------

def main(decision_points_path: Path,
         output_path: Path,
         total_target_mw: float = 500_000.0,
         wind_to_solar_ratio: float = 2.0):
    # 读取决策点
    PROVINCE_NUM=44
    full_df = read_decision_points(decision_points_path)
    dp_df = full_df[full_df["Provence_num"] == PROVINCE_NUM].reset_index(drop=True)
    # 1) 预评估容量系数
    print("Calculating capacity factors …")
    cf_df = calc_capacity_factors(dp_df)

    # 2) 贪婪扩张
    dp_expanded = greedy_expand(dp_df, cf_df,
                                total_target_mw=total_target_mw,
                                wind_to_solar_ratio=wind_to_solar_ratio)

    # 3) 保存结果（仍是同结构 Excel）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df=pd.read_excel(input_path("baseline_comparison_portfolio"))
    print(" 基于greedy策略的互补指数为")
    compute_yearly_CI(dp_expanded, year=2050)
    print(" 基于互补策略的互补指数为")
    compute_yearly_CI(df,year=2050)
    #dp_expanded.to_excel(output_path, index=False)
    print(f"✓ Baseline expansion saved to: {output_path}")


if __name__ == "__main__":
    # === 修改为您自己的路径 ===
    main(
        decision_points_path=input_path("decision_points"),
        output_path        =input_path("output_dir") / "baselines/Greedy/baseline500GW_greedy.xlsx"
    )