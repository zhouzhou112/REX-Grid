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
import random
import matplotlib.pyplot as plt
# ==== 项目内工具函数（按实际路径修改） ====
from bridge_decision import read_decision_points          # 读取决策点 Excel
from REX.compute_power_gap import compute_gap as compute_power_gap  # 评估风光出力
from WindSolarExpansionEnv import compute_yearly_CI
# ------------------------------------------------------------------------------
# 1. 计算每个栅格的年平均容量系数（CF）
# ------------------------------------------------------------------------------
from REX.compute_wind_solar  import compute_wind_solar    # ← 已在 utils_intra.py 中包装


# --- 在 baselin_greddy_expansion_v1.py 顶部 import 后加入（或新文件也行） ---
import math

# ========= 年度价格参数（$）=========
CAPEX_USD_PER_W = {
    "onshorewind": {2020:1.30, 2025:1.10, 2030:1.00, 2035:0.90, 2040:0.85, 2045:0.82, 2050:0.78},
    "offshorewind":{2020:2.00, 2025:1.80, 2030:1.70, 2035:1.60, 2040:1.53, 2045:1.46, 2050:1.40},
    "pv":          {2020:0.70, 2025:0.65, 2030:0.60, 2035:0.57, 2040:0.55, 2045:0.52, 2050:0.50},
}
OM_USD_PER_MW_YR = {
    "onshorewind": {2020:13.1, 2025:13.1, 2030:13.1, 2035:13.1, 2040:13.1, 2045:13.1, 2050:13.1},
    "offshorewind":{2020:21.3, 2025:21.3, 2030:21.3, 2035:21.3, 2040:21.3, 2045:21.3, 2050:21.3},
    "pv":          {2020: 6.0, 2025: 6.0, 2030: 6.0, 2035: 6.0, 2040: 6.0, 2045: 6.0, 2050: 6.0},
}
FINANCE = {
    "discount_rate": 0.07,  # r
    "lifetime_yr": {        # n
        "onshorewind": 25,
        "offshorewind": 25,
        "pv": 25,  # 若要 30 年，改这里
    }
}

# 技术列映射（与你的列名一致）
TECH_COLS = [
    ("onshorewind",  "Installed_onshorewind",  "Potential_onshorewind",  "cf_wind"),
    ("offshorewind", "Installed_offshorewind", "Potential_offshorewind", "cf_wind"),
    ("pv",           "Installed_pv",           "Potential_pv",           "cf_pv"),
]

def compute_lcoe_table(dp_df: pd.DataFrame,
                       cf_df: pd.DataFrame,
                       year: int) -> pd.DataFrame:
    """
    返回与 dp_df 同索引的 LCOE 表：['LCOE_onshorewind','LCOE_offshorewind','LCOE_pv'] ($/MWh)
    无潜力或 CF<=0 的位置填 np.inf（便于排序时自动跳过）
    """
    out = pd.DataFrame(index=dp_df.index.copy())
    for tech, col_inst, col_pot, cf_col in TECH_COLS:
        cf = np.asarray(cf_df[cf_col], dtype=float)
        pot = np.asarray(dp_df[col_pot], dtype=float)
        mask = (pot > 0) & np.isfinite(cf) & (cf > 0)

        r = FINANCE["discount_rate"]
        n = FINANCE["lifetime_yr"][tech]
        _crf = crf(r, n)
        capex_w   = _step_lookup(CAPEX_USD_PER_W, tech, year)         # $/W
        om_mwyr   = _step_lookup(OM_USD_PER_MW_YR, tech, year)        # $/MW·yr
        numer     = capex_w * 1_000_000 * _crf + om_mwyr              # $/MW·yr
        denom     = 8760.0 * np.maximum(cf, 1e-12)                    # MWh/(MW·yr)

        lcoe = np.full(len(dp_df), np.inf, dtype=float)
        lcoe[mask] = numer / denom[mask]                               # $/MWh
        out[f"LCOE_{tech}"] = lcoe
    return out

def save_lcoe_table(lcoe_df: pd.DataFrame, output_dir: Path, year: int, with_ids_from: pd.DataFrame | None = None):
    """保存 LCOE 年度表；如果 dp_df 里有 point_id/province 等字段，也一起带上方便追踪。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    to_save = lcoe_df.copy()
    if with_ids_from is not None:
        for col in ["point_id", "Provence_num", "city_id", "county_id"]:
            if col in with_ids_from.columns and col not in to_save.columns:
                to_save[col] = with_ids_from[col].values
    cols = [c for c in ["point_id","Provence_num","city_id","county_id"] if c in to_save.columns] + \
           [c for c in to_save.columns if c.startswith("LCOE_")]
    to_save[cols].to_excel(output_dir / f"LCOE_{year}.xlsx", index=False)
def calc_capacity_factors_mean(dp_df: pd.DataFrame,
                               *,
                               year: int = 2020,
                               months = range(1, 13),
                               n_runs: int = 500,
                               n_jobs: int = 1,
                               seed0: int = 20240101) -> pd.DataFrame:
    """
    对每个点重复计算 n_runs 次 CF，取均值。n_jobs>1 时并行（若 joblib 可用）。
    返回 DataFrame(index=dp_df.index, cols=['cf_wind','cf_pv'])
    """
    N = len(dp_df)
    wind_sum = np.zeros(N, dtype=float)
    pv_sum   = np.zeros(N, dtype=float)

    def _one(seed: int):
        # 给可能用到的随机库播种 —— 如果你内部用到了 np.random / random，这会生效
        np.random.seed(seed)
        random.seed(seed)
        cf = calc_capacity_factors(dp_df, year=year, months=months)
        w = np.asarray(cf["cf_wind"], dtype=float)
        p = np.asarray(cf["cf_pv"],   dtype=float)
        return w, p

    if n_jobs and n_jobs > 1:
        try:
            from joblib import Parallel, delayed
            chunks = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_one)(seed0 + i) for i in range(n_runs)
            )
            for w, p in chunks:
                wind_sum += w
                pv_sum   += p
        except Exception as e:
            print(f"[calc_capacity_factors_mean] 并行失败，改为串行：{e}")
            for i in range(n_runs):
                w, p = _one(seed0 + i)
                wind_sum += w
                pv_sum   += p
    else:
        for i in range(n_runs):
            w, p = _one(seed0 + i)
            wind_sum += w
            pv_sum   += p

    wind_mean = wind_sum / max(n_runs, 1)
    pv_mean   = pv_sum   / max(n_runs, 1)

    # 安全处理：理论上 CF∈[0,1]，并把“无潜力”的点强制为 0
    wind_mean = np.clip(wind_mean, 0.0, 1.0)
    pv_mean   = np.clip(pv_mean,   0.0, 1.0)

    # 若该点风电潜力都为 0（陆上+海上），则风 CF=0；若 PV 潜力为 0，则 PV CF=0
    pot_wind0 = (dp_df.get("Potential_onshorewind", 0) <= 0) & \
                (dp_df.get("Potential_offshorewind", 0) <= 0)
    pot_pv0   = (dp_df.get("Potential_pv", 0) <= 0)
    wind_mean = np.where(pot_wind0.values, 0.0, wind_mean)
    pv_mean   = np.where(pot_pv0.values,   0.0, pv_mean)

    out = pd.DataFrame({"cf_wind": wind_mean, "cf_pv": pv_mean}, index=dp_df.index)
    return out
def _step_lookup(table: dict, tech: str, year: int) -> float:
    """从字典中取 ≤year 的最近节点值"""
    nodes = sorted(table[tech].keys())
    y = max(n for n in nodes if n <= year)
    return table[tech][y]

def crf(r: float, n: int) -> float:
    """资本回收系数 Capital Recovery Factor"""
    if r == 0:
        return 1.0 / n
    return r * (1 + r) ** n / ((1 + r) ** n - 1)

def lcoe_usd_per_mwh(tech: str, cf: float, year: int) -> float:
    """按当年价格与给定 CF 计算 LCOE($/MWh)；CF<=0 返回 +inf"""
    if cf <= 0 or not math.isfinite(cf):
        return float("inf")
    r = FINANCE["discount_rate"]
    n = FINANCE["lifetime_yr"][tech]
    _crf = crf(r, n)
    capex_per_w = _step_lookup(CAPEX_USD_PER_W, tech, year)  # $/W
    om_per_mw_yr = _step_lookup(OM_USD_PER_MW_YR, tech, year)  # $/MW·yr
    annualized_capex_per_mw = capex_per_w * 1_000_000 * _crf  # $/MW·yr
    denom_mwh_per_mw_yr = 8760.0 * cf  # MWh/(MW·yr)
    return (annualized_capex_per_mw + om_per_mw_yr) / denom_mwh_per_mw_yr  # $/MWh

# ========= 基于 LCOE 的扩张（逐年重算 LCOE）=========
# def expand_by_LCOE_history(dp_df: pd.DataFrame,
#                            cf_df: pd.DataFrame,
#                            total_target_mw: float,
#                            start_year: int = 2020,
#                            end_year: int = 2050,
#                            step_years: int = 5,
#                            clip_tol: float = 1e-6) -> dict[int, pd.DataFrame]:
#     """
#     dp_df: 决策点表（含 Installed_* 与 Potential_*）
#     cf_df: 与 dp_df 同索引，含 cf_wind / cf_pv
#     返回：history[yr] = 该年扩张完成后的 dp 表
#     """
#     years = list(range(start_year, end_year + 1, step_years))
#     periods = len(years)
#     quota_per_period = total_target_mw / periods
#
#     dp = dp_df.copy()
#     history = {start_year: dp.copy()}
#
#     for yr in years:
#         # 1) 生成当期候选清单（计算 LCOE）
#         candidates = []
#         for idx in dp.index:
#             for tech, col_inst, col_pot, cf_col in TECH_COLS:
#                 avail = float(dp.at[idx, col_pot] - dp.at[idx, col_inst])
#                 if avail <= clip_tol:
#                     continue
#                 cf_val = float(cf_df.at[idx, cf_col])
#                 l = lcoe_usd_per_mwh(tech, cf_val, yr)
#                 if not math.isfinite(l):
#                     continue
#                 candidates.append((l, idx, tech, col_inst, col_pot, avail))
#
#         if not candidates:
#             print(f"[{yr}] 无可扩张候选。")
#             history[yr] = dp.copy()
#             continue
#
#         # 2) 按 LCOE 升序扩张
#         candidates.sort(key=lambda x: x[0])  # x[0]=LCOE
#         remaining = quota_per_period
#         picked_costs, picked_caps = [], []
#
#         for lcoe_val, idx, tech, col_inst, col_pot, avail in candidates:
#             if remaining <= 0:
#                 break
#             add = min(avail, remaining)
#             if add <= 0:
#                 continue
#             dp.at[idx, col_inst] += add
#             remaining -= add
#             picked_costs.append(lcoe_val)
#             picked_caps.append(add)
#
#         added = quota_per_period - remaining
#         # 选中容量加权的当期平均 LCOE（仅供反馈）
#         avg_lcoe = (sum(c*w for c, w in zip(picked_costs, picked_caps)) / max(sum(picked_caps), 1e-9)) if picked_caps else float("nan")
#         print(f"[{yr}] 基于 LCOE 扩建 {added:.1f} MW / 目标 {quota_per_period:.1f} MW；选中平均 LCOE ≈ {avg_lcoe:.2f} $/MWh")
#
#         history[yr] = dp.copy()
#
#     return history

def expand_by_LCOE_history(dp_df: pd.DataFrame,
                           cf_df: pd.DataFrame,
                           total_target_mw: float,
                           start_year: int = 2020,
                           end_year: int = 2050,
                           step_years: int = 5,
                           clip_tol: float = 1e-6,
                           debug_dir: Path | None = None) -> dict[int, pd.DataFrame]:

    years = list(range(start_year, end_year + 1, step_years))
    quota_per_period = total_target_mw / len(years)
    dp = dp_df.copy()
    history = {start_year: dp.copy()}

    for yr in years:
        # 👉 先算并（可选）保存当年的每点 LCOE
        lcoe_table = compute_lcoe_table(dp, cf_df, yr)
        if debug_dir is not None:
            save_lcoe_table(lcoe_table, debug_dir, yr, with_ids_from=dp)

        # 下面保持原来的“点×技术一起排队”的候选构建与排序
        candidates = []
        for idx in dp.index:
            for tech, col_inst, col_pot, cf_col in TECH_COLS:
                avail = float(dp.at[idx, col_pot] - dp.at[idx, col_inst])
                if avail <= clip_tol:
                    continue
                l = float(lcoe_table.at[idx, f"LCOE_{tech}"])
                if not np.isfinite(l):
                    continue
                candidates.append((l, idx, tech, col_inst, col_pot, avail))

        candidates.sort(key=lambda x: x[0])  # 按 LCOE 升序
        remaining = quota_per_period
        picked_costs, picked_caps = [], []
        for lcoe_val, idx, tech, col_inst, col_pot, avail in candidates:
            if remaining <= 0:
                break
            add = min(avail, remaining)
            if add <= 0:
                continue
            dp.at[idx, col_inst] += add
            remaining -= add
            picked_costs.append(lcoe_val)
            picked_caps.append(add)

        added = quota_per_period - remaining
        avg_lcoe = (np.dot(picked_costs, picked_caps) / max(sum(picked_caps), 1e-9)) if picked_caps else float("nan")
        print(f"[{yr}] 基于 LCOE 扩建 {added:.1f} MW；平均 LCOE ≈ {avg_lcoe:.2f} $/MWh")
        history[yr] = dp.copy()

    return history
def calc_capacity_factors(dp_df: pd.DataFrame,
                          year: int = 2020,
                          months: range = range(1, 13)) -> pd.DataFrame:
    N = len(dp_df)
    e_on  = np.zeros(N)          # 陆上风电年能量 (MWh)
    e_pv  = np.zeros(N)          # 光伏年能量   (MWh)

    for m in months:
        df_tmp = dp_df.copy()
        for i, p in [("Installed_onshorewind", "Potential_onshorewind"),
                     ("Installed_offshorewind", "Potential_offshorewind"),
                     ("Installed_pv",          "Potential_pv")]:
            df_tmp[i] = df_tmp[p]

        wdf, sdf = compute_wind_solar(year, m, df_tmp)
        e_on += np.array(wdf["wind_power_output"].apply(sum))   # 已确保与 dp_df 对齐
        e_pv += np.array(sdf["solar_power_output"].apply(sum))

    hours = 168 * len(months)          # 本轮统计对应小时数
    pot_on  = dp_df["Potential_onshorewind"].values
    pot_off = dp_df["Potential_offshorewind"].values
    pot_pv  = dp_df["Potential_pv"].values
    pot_wind = pot_on + pot_off

    # --- 安全除法：先全零，再写入有效位置 ---
    cf_wind = np.zeros(N, dtype=float)
    mask_w  = pot_wind > 0
    np.divide(e_on, pot_wind * hours, out=cf_wind, where=mask_w)

    cf_pv   = np.zeros(N, dtype=float)
    mask_p  = pot_pv > 0
    np.divide(e_pv, pot_pv  * hours, out=cf_pv,   where=mask_p)

    return pd.DataFrame({
        "cf_wind": cf_wind,
        "cf_pv"  : cf_pv,
    }, index=dp_df.index)


# ------------------------------------------------------------------------------
# 2. 贪婪扩张主流程
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# 2. 纯 CF Greedy 扩张，不区分风/光配比
# ------------------------------------------------------------------------------

def greedy_expand_pure_cf(dp_df: pd.DataFrame,
                          cf_df: pd.DataFrame,
                          total_target_mw: float = 500_000.0,
                          start_year: int = 2020,
                          end_year: int = 2050,
                          step_years: int = 5) -> pd.DataFrame:
    years            = list(range(start_year, end_year + 1, step_years))
    periods          = len(years)
    quota_per_period = total_target_mw / periods

    dp = dp_df.copy()

    # ① 构造所有“(点位, 技术, CF)”列表
    tech_list = []
    for idx in cf_df.index:
        # 陆风
        if dp.loc[idx, "Potential_onshorewind"]  > dp.loc[idx, "Installed_onshorewind"]:
            tech_list.append((idx, "Installed_onshorewind", cf_df.loc[idx, "cf_wind"]))
        # 海风
        if dp.loc[idx, "Potential_offshorewind"] > dp.loc[idx, "Installed_offshorewind"]:
            tech_list.append((idx, "Installed_offshorewind", cf_df.loc[idx, "cf_wind"]))
        # 光伏
        if dp.loc[idx, "Potential_pv"]           > dp.loc[idx, "Installed_pv"]:
            tech_list.append((idx, "Installed_pv", cf_df.loc[idx, "cf_pv"]))

    # 按 CF 降序
    tech_rank = sorted(tech_list, key=lambda x: x[2], reverse=True)

    # ② 每期分配
    for yr in years:
        remaining = quota_per_period
        for idx, col_inst, _ in tech_rank:
            if remaining <= 0:
                break
            # 对应的潜力列
            col_pot = "Potential" + col_inst[len("Installed"):]
            avail   = dp.loc[idx, col_pot] - dp.loc[idx, col_inst]
            add     = min(avail, remaining)
            dp.loc[idx, col_inst] += add
            remaining -= add

        print(f"[{yr}] 扩建了 {quota_per_period - remaining:.1f} MW (目标 {quota_per_period:.1f} MW)")

    return dp

# 2. 纯 CF Greedy 扩张 —— 返回每一步的快照
# ----------------------------------------------------------------------
def greedy_expand_pure_cf_history(dp_df: pd.DataFrame,
                                  cf_df: pd.DataFrame,
                                  total_target_mw: float ,
                                  start_year: int = 2020,
                                  end_year: int = 2050,
                                  step_years: int = 5) -> dict[int, pd.DataFrame]:
    """
    不改变原 dp_df，把每次扩建后的 dp 副本保存在 history 中：
        history[yr] = expansion 完成到 yr 这一年的 dp_df
    """
    years            = list(range(start_year, end_year + 1, step_years))
    periods          = len(years)
    quota_per_period = total_target_mw / periods

    dp = dp_df.copy()
    history = {start_year: dp.copy()}  # 2020 年初始状态

    # ① 构造所有“(点位, 技术, CF)”列表（和之前一样）
    tech_list = []
    for idx in cf_df.index:
        if dp.loc[idx, "Potential_onshorewind"]  > dp.loc[idx, "Installed_onshorewind"]:
            tech_list.append((idx, "Installed_onshorewind", cf_df.loc[idx, "cf_wind"]))
        if dp.loc[idx, "Potential_offshorewind"] > dp.loc[idx, "Installed_offshorewind"]:
            tech_list.append((idx, "Installed_offshorewind", cf_df.loc[idx, "cf_wind"]))
        if dp.loc[idx, "Potential_pv"]           > dp.loc[idx, "Installed_pv"]:
            tech_list.append((idx, "Installed_pv", cf_df.loc[idx, "cf_pv"]))

    tech_rank = sorted(tech_list, key=lambda x: x[2], reverse=True)

    # ② 每期分配，并保存快照
    for yr in years[0:]:
        remaining = quota_per_period
        for idx, col_inst, _ in tech_rank:
            if remaining <= 0:
                break
            col_pot = "Potential" + col_inst[len("Installed"):]
            avail   = dp.loc[idx, col_pot] - dp.loc[idx, col_inst]
            add     = min(avail, remaining)
            dp.loc[idx, col_inst] += add
            remaining -= add

        print(f"[{yr}] 扩建了 {quota_per_period - remaining:.1f} MW")
        history[yr] = dp.copy()  # 存下当年扩建完的 dp

    return history
# ------------------------------------------------------------------------------
# 3. 一键运行
# ------------------------------------------------------------------------------

def main(decision_points_path: Path,
         output_dir: Path,
         total_target_mw: float = 115000


):
    full_df = read_decision_points(decision_points_path)

    # 👉 若只做广东：PROVINCE_NUM=44；若做南方五省，可自行拼接 44/45/46/52/53
    PROVINCE_NUM = 53
    dp_df = full_df[full_df["Provence_num"] == PROVINCE_NUM].reset_index(drop=True)

    print("Calculating capacity factors …")
    #cf_df = calc_capacity_factors(dp_df, year=2020, months=range(1,13))
    cf_df = calc_capacity_factors_mean(
        dp_df,
        year=2020,
        months=range(1, 13),
        n_runs=500,
        n_jobs=20,  # 例如 16 核并行；没有 joblib 时会自动退化为串行
        seed0=20240101  # 固定随机种子以便可重复
    )
    # 可保存一下 CF 结果
    (output_dir / "debug").mkdir(parents=True, exist_ok=True)
    cf_df.to_excel(output_dir / "debug" / "CF.xlsx")
    dp_df.to_excel(output_dir / "debug" / "dp_base.xlsx")

    # === 用 LCOE 逐年扩张 ===
    history = expand_by_LCOE_history(dp_df, cf_df,
                                     total_target_mw=total_target_mw,
                                     start_year=2020, end_year=2050, step_years=5,
                                     debug_dir=output_dir / "debug" / "LCOE_tables"
                                     )

    output_dir.mkdir(parents=True, exist_ok=True)
    for yr, df_snap in history.items():
        path = output_dir / f"expansion_LCOE_{yr}.xlsx"
        df_snap.to_excel(path, index=False)
        print(f" 👉 保存 {yr} 年扩张结果到：{path}")




if __name__ == "__main__":
    # === 修改为您自己的路径 ===
    main(
        decision_points_path=input_path("decision_points"),
        output_dir        =input_path("output_dir") / "baselines/LCOE/YN"
    )