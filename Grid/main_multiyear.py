import os
import sys
# === 核心魔法：将项目根目录 REX-Grid 动态加入系统路径 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 退回上一级到 REX-Grid
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import copy
import numpy as np
import pandas as pd
import gurobipy as gp
from pathlib import Path

from utils_yearly import read_table, load_decision_points_for_year
from utils_intra import build_county_ts, plan_intra_transmission_v1, aggregate_to_city, load_intra_network_state, \
    dump_intra_network_state
from dispatch_model import initialize_annual_inputs, AnnualDispatchOptimizer, THERMAL_PARAMS, STORAGE_TECHS, \
    extract_results
# --- 跨文件夹导入 REX 模块 (将 env 改为 REX) ---
from REX.compute_power_gap import compute_gap
import dispatch_model as dm

SOUTH_IDS = [44, 45, 46, 52, 53]
THERMAL_TYPES = ["Coal_300to600", "Coal_lessthan300", "Coal_morethan600", "GAS"]

EPS_CAP = 1e-2  # 容量数值阈值


def run_case(expansion_mode: str, carbon_scenario: str):
    """
    运行多时间步调度模型，仅保留数据推进与 Excel 结果输出。
    """
    dm.CARBON_SCENARIO = carbon_scenario
    tag = f"{expansion_mode}_{carbon_scenario}".replace(".", "")
    out_dir = Path(r"D:\paper2\绘图\调度情景\0918补充\增速4") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    YEARS = list(range(2020, 2051, 5))
    DISCOUNT_RATE = 0.05

    # 路径配置
    thermal_csv_path = r"D:/paper2/data/Power_generate/火电/全国分级火电分布.csv"
    storage_excel_path = r"D:/paper2/data/Power_generate/energy_story_2020.xlsx"
    transmission_csv_path = r"D:\paper2\data\Power_generate\跨省传输线路\exist_trasmission.csv"
    nation_base_path = r"D:/paper2/data/resourcepoint/05已安装风光数据链接/RL_STATE_2020.xlsx"
    RETIRE_XLSX = r"D:\paper2\data\Power_generate\火电\retirement_capacity_bins.xlsx"
    city_center_path = r'D:\paper2\data\Power_generate\省内能源传输\city-center-0816.xls'

    nation_base = read_table(nation_base_path, id_col="point_id")

    # 读取强制退役清单
    retire_df = pd.read_excel(RETIRE_XLSX).rename(columns={"Provence_n": "prov_code", "Period": "year"})
    RETIRE_MAP = {
        yr: {u: grp.set_index("prov_code")[u].fillna(0.0).to_dict() for u in THERMAL_TYPES}
        for yr, grp in retire_df.groupby("year")
    }
    ret_credit = {u: {prov: 0.0 for prov in SOUTH_IDS} for u in THERMAL_TYPES}

    # 初始化 2020 基线
    data0 = initialize_annual_inputs(
        year=2020, thermal_csv=thermal_csv_path, storage_xlsx=storage_excel_path,
        line_csv=transmission_csv_path, decision_points=nation_base, compute_gap=compute_gap,
    )
    thermal_cap_vintage = copy.deepcopy(data0["thermal_cap_vintage"])
    storage_cap_p_vintage = copy.deepcopy(data0["storage_cap_p_vintage"])
    line_cap_state = data0["corridor_df"]["base_cap"].copy()
    ccs_cap_vintage = copy.deepcopy(data0["ccs_cap_vintage"])

    spur_dict, trunk_dict = load_intra_network_state(init_year=2020)

    npv_total = 0.0
    results_list = []
    annual_cost_series_list = []

    for k, yr in enumerate(YEARS):
        print(f"\n==========  {yr}  ==========")
        dec_points = load_decision_points_for_year(yr, nation_base, mode=expansion_mode)

        # 省内线路规划
        county_ts, _, ts_df, _ = build_county_ts(yr, dec_points, city_center_path=city_center_path)
        city_ts, city_meta = aggregate_to_city(county_ts, ts_df)
        spur_dict, trunk_dict, intra_cost, _, _ = plan_intra_transmission_v1(
            county_ts, city_ts, city_meta, ts_df, spur_prev=spur_dict, trunk_prev=trunk_dict,
            network_path="city_network-0816"
        )

        data_year = initialize_annual_inputs(
            year=yr, thermal_csv=thermal_csv_path, storage_xlsx=storage_excel_path,
            line_csv=transmission_csv_path, decision_points=dec_points, compute_gap=compute_gap,
            thermal_cap_vintage_override=copy.deepcopy(thermal_cap_vintage),
            ccs_cap_vintage_override=copy.deepcopy(ccs_cap_vintage),
            storage_cap_p_override=copy.deepcopy(storage_cap_p_vintage),
            line_cap_override=line_cap_state.copy(),
        )

        # 强制退役计算
        official_req = {u: {prov: 0.0 for prov in SOUTH_IDS} for u in THERMAL_TYPES}
        for u in THERMAL_TYPES:
            if yr in RETIRE_MAP and u in RETIRE_MAP[yr]:
                for prov, mw in RETIRE_MAP[yr][u].items():
                    official_req[u][prov] = mw

        mand_ret = {u: np.zeros(len(SOUTH_IDS)) for u in THERMAL_TYPES}
        for idx, prov in enumerate(SOUTH_IDS):
            for u in THERMAL_TYPES:
                mand_ret[u][idx] = max(official_req[u][prov] - ret_credit[u][prov], 0.0)

        # 调度求解
        opt = AnnualDispatchOptimizer(data_year, current_year=yr, threads=22, mandatory_retire=mand_ret)
        opt.add_intra_cost(intra_cost)
        status = opt.optimize()

        if status in (gp.GRB.OPTIMAL, gp.GRB.SUBOPTIMAL) and opt.m.SolCount > 0:
            best_obj = opt.m.ObjVal
            npv_total += (1 / (1 + DISCOUNT_RATE) ** k) * best_obj
            print(f"[{yr}] Optimal  Obj={best_obj:,.0f}  累计NPV={npv_total:,.0f} k RMB")
            cost_series = opt.get_cost_report()
        else:
            print(f"[{yr}] 模型无可行解 (status={status})，执行 IIS…")
            opt.m.computeIIS()
            opt.m.write(f"iisp_{yr}.ilp")
            zero_idx = ["capex_thermal", "fixom_thermal", "capex_ccs", "fixom_ccs", "capex_storage",
                        "capex_line_linear", "capex_substation", "intra_transmission", "fuel", "co2_chain",
                        "storage_var_om", "start", "investment_total", "opex_total", "total"]
            cost_series = pd.Series({k: 0.0 for k in zero_idx}, name=yr)

        cost_series.name = yr
        annual_cost_series_list.append(cost_series)

        year_res = extract_results(opt)
        year_res["year"] = yr
        results_list.append(year_res)

        # 更新基线
        Δvintage, Δ_ccs, Δ_sto_p, Δ_line = opt.get_capacity_deltas()
        for u in THERMAL_PARAMS:
            for pb in Δvintage[u]:
                if pb not in thermal_cap_vintage[u]:
                    thermal_cap_vintage[u][pb] = np.zeros_like(
                        thermal_cap_vintage[u][next(iter(thermal_cap_vintage[u]))], dtype=float)
                    ccs_cap_vintage[u][pb] = np.zeros_like(thermal_cap_vintage[u][next(iter(thermal_cap_vintage[u]))],
                                                           dtype=float)

                ccs_cap_vintage[u][pb] = np.minimum(ccs_cap_vintage[u][pb].astype(float),
                                                    thermal_cap_vintage[u][pb].astype(float))
                thermal_cap_vintage[u][pb] = np.maximum(thermal_cap_vintage[u][pb].astype(float) + Δvintage[u][pb], 0.0)
                ccs_cap_vintage[u][pb] = np.maximum(ccs_cap_vintage[u][pb].astype(float) + Δ_ccs[u][pb], 0.0)

                thermal_cap_vintage[u][pb][np.abs(thermal_cap_vintage[u][pb]) < 1e-5] = 0.0
                ccs_cap_vintage[u][pb][np.abs(ccs_cap_vintage[u][pb]) < 1e-5] = 0.0

        # 退役余额结转
        retire_now = {u: {prov: 0.0 for prov in SOUTH_IDS} for u in THERMAL_TYPES}
        for u in THERMAL_TYPES:
            for pb_arr in Δvintage[u].values():
                for idx, prov in enumerate(SOUTH_IDS):
                    if pb_arr[idx] < 0:
                        retire_now[u][prov] += -pb_arr[idx]

        for prov in SOUTH_IDS:
            for u in THERMAL_TYPES:
                ret_credit[u][prov] = max(ret_credit[u][prov] + retire_now[u][prov] - official_req[u][prov], 0.0)

        for tech in STORAGE_TECHS:
            if yr not in storage_cap_p_vintage[tech]:
                storage_cap_p_vintage[tech][yr] = np.zeros(opt.n_prov, dtype=float)
            storage_cap_p_vintage[tech][yr] = storage_cap_p_vintage[tech][yr].astype(float) + Δ_sto_p[tech]

        line_cap_state = line_cap_state.add(Δ_line, fill_value=0)
        dump_intra_network_state(spur_dict, trunk_dict, yr + 5)

    # ===== 保存详细逐时数据到 Excel =====
    with pd.ExcelWriter(out_dir / "multi_year_dispatch_results.xlsx") as xls:
        for res in results_list:
            y = res["year"]
            res["thermal"].to_excel(xls, sheet_name=f"Thermal_{y}")
            res["thermal_coal"].to_excel(xls, sheet_name=f"thermal_coal_{y}")
            res["thermal_gas"].to_excel(xls, sheet_name=f"thermal_gas{y}")
            for src, df in res["ren"].items():
                df.to_excel(xls, sheet_name=f"{src}_{y}")
            res["storage"].to_excel(xls, sheet_name=f"Storage_{y}")
            res["trans"].to_excel(xls, sheet_name=f"Trans_{y}")
            res["curtail"].to_excel(xls, sheet_name=f"Curtail_{y}")
            res["demand"].to_excel(xls, sheet_name=f"Load_{y}")
            res["emission"].to_frame().to_excel(xls, sheet_name=f"CO2_{y}")
            res["line_flow_fwd_send"].to_excel(xls, sheet_name=f"LineFwdSend_{y}")
            res["line_flow_rev_send"].to_excel(xls, sheet_name=f"LineRevSend_{y}")
            res["line_flow_fwd_arr"].to_excel(xls, sheet_name=f"LineFwdArr_{y}")
            res["line_flow_rev_arr"].to_excel(xls, sheet_name=f"LineRevArr_{y}")
            res["pair_flow_arrive"].to_excel(xls, sheet_name=f"PairArr_{y}")
            if y == results_list[0]["year"]:
                res["line_meta"].to_excel(xls, sheet_name="Corridors")

    print("✅ 已输出 multi_year_dispatch_results.xlsx")

    # ===== 提取宏观数据并生成年度报表 =====
    provs = SOUTH_IDS
    corr_ids = results_list[0]["line_meta"].index.sort_values()

    cap_th_coal = pd.DataFrame(0.0, index=YEARS, columns=provs)
    cap_th_gas = pd.DataFrame(0.0, index=YEARS, columns=provs)
    cap_th_coal_ccs = pd.DataFrame(0.0, index=YEARS, columns=provs)
    cap_th_gas_ccs = pd.DataFrame(0.0, index=YEARS, columns=provs)
    sto_caps = {tech: pd.DataFrame(0.0, index=YEARS, columns=provs) for tech in STORAGE_TECHS}
    line_caps = pd.DataFrame(0.0, index=YEARS, columns=corr_ids, dtype=float)
    co2_annual = pd.DataFrame(0.0, index=YEARS, columns=provs)

    for i, res in enumerate(results_list):
        y = res["year"]
        opt = res.get("optimizer")

        def sum_dict_arrays(d: dict) -> np.ndarray:
            out = np.zeros(len(provs), dtype=float)
            for _, a in d.items(): out += np.asarray(a, dtype=float)
            return out

        coal_total_eoy, coal_ccs_eoy = np.zeros(len(provs)), np.zeros(len(provs))
        for u in ["Coal_morethan600", "Coal_300to600", "Coal_lessthan300"]:
            t_eoy = sum_dict_arrays(opt.thermal_cap_vintage[u]) + sum_dict_arrays(opt.get_capacity_deltas()[0][u])
            c_eoy = np.minimum(
                sum_dict_arrays(opt.ccs_cap_vintage[u]) + sum_dict_arrays(opt.get_capacity_deltas()[1][u]), t_eoy)
            coal_total_eoy += t_eoy
            coal_ccs_eoy += c_eoy

        cap_th_coal.loc[y] = coal_total_eoy - coal_ccs_eoy
        cap_th_coal_ccs.loc[y] = coal_ccs_eoy

        g_eoy = sum_dict_arrays(opt.thermal_cap_vintage["GAS"]) + sum_dict_arrays(opt.get_capacity_deltas()[0]["GAS"])
        g_ccs_eoy = np.minimum(
            sum_dict_arrays(opt.ccs_cap_vintage["GAS"]) + sum_dict_arrays(opt.get_capacity_deltas()[1]["GAS"]), g_eoy)
        cap_th_gas.loc[y] = g_eoy - g_ccs_eoy
        cap_th_gas_ccs.loc[y] = g_ccs_eoy

        for tech in STORAGE_TECHS:
            sto_caps[tech].loc[y] = opt.storage_cap_p_base[tech] + np.array(
                [opt.add_sto_p[tech, idx].X for idx in range(opt.n_prov)])

        base = opt.corridor_df["base_cap"].astype(float)
        add = pd.Series([opt.add_corr[j].X for j in range(opt.n_corr)], index=opt.corridor_df.index, dtype=float)
        line_caps.loc[y, opt.corridor_df.index] = base + add
        co2_annual.loc[y] = res["emission"]

    with pd.ExcelWriter(out_dir / "summary_tables.xlsx") as xls:
        cap_th_coal.to_excel(xls, sheet_name="Coal_NoCC")
        cap_th_coal_ccs.to_excel(xls, sheet_name="Coal_CCS")
        cap_th_gas.to_excel(xls, sheet_name="Gas_NoCC")
        cap_th_gas_ccs.to_excel(xls, sheet_name="Gas_CCS")
        for tech, df in sto_caps.items():
            df.to_excel(xls, sheet_name=f"Storage_{tech}")
        line_caps.to_excel(xls, sheet_name="Lines_MW")
        co2_annual.to_excel(xls, sheet_name="CO2_t")
        pd.DataFrame(annual_cost_series_list).to_excel(xls, sheet_name="AnnualCosts")

    print(f"✅ 执行完毕，所有纯数据结果已保存到 {out_dir.resolve()}")


if __name__ == "__main__":
    for mode in ["RL", "Greedy", "LCOE"]:
        for scen in ["CN2050", "GM2.0", "NDC"]:
            print(f"\n\n===== Start Case: {mode} / {scen} =====")
            run_case(mode, scen)