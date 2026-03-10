import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from Grid.typical_week_selector import build_typical_gap_province_independent


SOUTH_IDS = [44, 45, 46, 52, 53]
prov_to_idx = {pid: k for k, pid in enumerate(SOUTH_IDS)}
idx_to_prov = {k: pid for k, pid in enumerate(SOUTH_IDS)}

OM = 4.387211256 * (8760 / 2016)
year_scale = 4.387211256
WINTER_MONTHS = {11, 12, 1, 2, 3}
HOURS_PER_MONTH = 168
T_YEAR = 12 * HOURS_PER_MONTH

INV_COST_THERMAL = {
    "Coal_morethan600": 573, "Coal_300to600": 574, "Coal_lessthan300": 605, "GAS": 480,
}

THERMAL_PARAMS = {
    "Coal_morethan600": {"pmax": 1.0, "pmin": 0.40, "pmin_w": 0.40, "rup": 0.25, "rdn": 0.45, "var_cost": 10.88,
                         "fuel_use": 281, "CO2_emission": 773},
    "Coal_300to600": {"pmax": 1.0, "pmin": 0.40, "pmin_w": 0.40, "rup": 0.3, "rdn": 0.5, "var_cost": 11.4,
                      "fuel_use": 302, "CO2_emission": 831},
    "Coal_lessthan300": {"pmax": 1.0, "pmin": 0.40, "pmin_w": 0.40, "rup": 0.35, "rdn": 0.55, "var_cost": 12.7,
                         "fuel_use": 345, "CO2_emission": 950},
    "GAS": {"pmax": 1.0, "pmin": 0.20, "pmin_w": 0.20, "rup": 0.50, "rdn": 0.50, "var_cost": 12, "fuel_use": 240,
            "CO2_emission": 413},
}

MIN_UP_HR = {"Coal_morethan600": 24, "Coal_300to600": 8, "Coal_lessthan300": 8, "GAS": 1}
MIN_DN_HR = {"Coal_morethan600": 48, "Coal_300to600": 8, "Coal_lessthan300": 4, "GAS": 1}

IDLE_FRAC = 0.20
EFF_MIN = {"Coal_morethan600": 1.1285, "Coal_300to600": 1.1579, "Coal_lessthan300": 1.1059, "GAS": 1.18}
ALPHA = {u: (THERMAL_PARAMS[u]["fuel_use"] * 29307 / 1_000_000) * (EFF_MIN[u] - 1) * IDLE_FRAC for u in THERMAL_PARAMS}

COAL_PRICE = {44: 5.93, 45: 6.85, 46: 4.80, 52: 4.95, 53: 4.23}
GAS_PRICE = {44: 10.58, 45: 9.95, 46: 8.62, 52: 8.88, 53: 8.88}

TRANS_COST_CO2 = 0.026
STORE_COST_CO2 = 5.0
CO2_DISTANCE_KM = 50
CAPTURE_RATE = 0.90

START_COST = {"Coal_morethan600": 142, "Coal_300to600": 142, "Coal_lessthan300": 142, "GAS": 85}

_NODE_YEARS = (2025, 2030, 2040, 2050, 2060)
_CAPEX_CC = {"Coal": [381.9, 305.5, 195.5, 158.4, 142.5], "GAS": [271.5, 217.2, 139.0, 112.6, 101.4]}
_FIXED_OM = {"Coal": [20.7, 16.6, 10.6, 8.6, 7.7], "GAS": [26.1, 20.9, 13.4, 10.8, 9.7]}
_PENALTY = {"Coal": [24.4, 22.2, 15.6, 13.3, 11.1], "GAS": [18.3, 16.7, 11.7, 10.0, 8.3]}


def _lookup(tech: str, year: int, table: dict) -> float:
    if year < _NODE_YEARS[0]: return table[tech][0]
    return table[tech][max(i for i, y in enumerate(_NODE_YEARS) if year >= y)]


def ccs_capex_kw(tech: str, yr: int) -> float: return _lookup(tech, yr, _CAPEX_CC)


def ccs_fix_om_kw(tech: str, yr: int) -> float: return _lookup(tech, yr, _FIXED_OM)


def ccs_energy_penalty(tech: str, yr: int) -> float: return _lookup(tech, yr, _PENALTY) / 100


CCS_RETROFIT = {u: {"capex": 1800, "eff_penalty": 0.08} for u in THERMAL_PARAMS}

STORAGE_TECHS = {
    "PHS": {"fix_om": 5460, "var_om": 0.2, "eff": 0.78, "dur": 8},
    "BAT": {"fix_om": 2520, "var_om": 3, "eff": 0.95, "dur": 4},
    "CAES": {"fix_om": 7000, "var_om": 0.5, "eff": 0.52, "dur": 20},
    "VRB": {"fix_om": 2520, "var_om": 3, "eff": 0.70, "dur": 10},
}
STO_LIFE = {"PHS": 40, "BAT": 15, "CAES": 30, "VRB": 15}
STO_SELF_DIS = {"PHS": 0.0, "BAT": 0.07, "CAES": 1.0, "VRB": 0.6}
STO_DURATION = {"PHS": 8, "BAT": 4, "CAES": 20, "VRB": 10}

CAPEX_USD_2020 = {"PHS": 77, "BAT": 350, "CAES": 50, "VRB": 100}
CAPEX_USD_2050 = {"PHS": 77, "BAT": 106, "CAES": 34, "VRB": 47}


def storage_capex_kwh_exp(tech: str, year: int) -> float:
    p0, p2050 = CAPEX_USD_2020[tech], CAPEX_USD_2050[tech]
    if year <= 2020: return p0
    if year >= 2050: return p2050
    return p0 * (p2050 / p0) ** ((year - 2020) / 30)


INV_COST_LINE = 4
SPIN_ALPHA = 0.05
SPIN_BETA = 0.3
CYCLE_LIMIT = 1000
RES_MARGIN = 0.15
R_EARTH = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    phi1, phi2 = np.radians([lat1, lat2])
    dphi, dlamb = phi2 - phi1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlamb / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


TECH_INFO = {
    "HVAC": dict(loss_per_km=0.07 / 1000, capex=450, substation=22.6),
    "HVDC": dict(loss_per_km=0.016 / 1000, capex=850, substation=84.2, conv_loss=0.014),
}
FOM_LINE_RATE, FOM_SUB_RATE = 0.035, 0.035

PRESET_TECH = {
    (44, 45): ("HVAC", 500), (44, 46): ("HVAC", 500), (44, 52): ("HVAC", 500),
    (44, 53): ("HVDC", 800), (45, 46): ("HVAC", 500), (45, 52): ("HVAC", 500),
    (45, 53): ("HVDC", 800), (52, 53): ("HVAC", 500),
}


def cref_hvac_500kv(distance_km: float) -> float:
    pts = [(0, 3000), (200, 2500), (400, 1800), (600, 1400), (800, 1200), (1000, 1050), (1200, 1000), (1600, 1000)]
    if distance_km <= pts[0][0]: return pts[0][1]
    if distance_km >= pts[-1][0]: return pts[-1][1]
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        if x1 <= distance_km <= x2:
            return y1 + (y2 - y1) * (distance_km - x1) / (x2 - x1)


def load_existing_lines(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["status"].isin(["Operating", "Construct"])]
    rec = []
    for _, r in df.iterrows():
        a, b = sorted((int(r["from"]), int(r["to"])))
        tech = "HVDC" if str(r["type"]).strip().upper() == "DC" else "HVAC"
        dist = float(r["distance(km)"]) if "distance(km)" in r and not pd.isna(r["distance(km)"]) else 1000.0
        rec.append(dict(
            prov_i=a, prov_j=b, tech=tech, capacity_GW=float(r["capacity_GW"]), distance_km=dist,
            v_kV=int(str(r["LEVEL"]).strip("±kVkV")) if isinstance(r["LEVEL"], str) else 0
        ))
    return pd.DataFrame(rec)


def build_corridors(line_csv: str) -> pd.DataFrame:
    exist_df = load_existing_lines(line_csv)
    agg = {}
    for _, r in exist_df.iterrows():
        key = (r.prov_i, r.prov_j)
        dlist = agg.setdefault(key, dict(prov_i=r.prov_i, prov_j=r.prov_j, tech=r.tech, v_kV=r.v_kV, base_cap=0.0,
                                         dist_list=[]))
        dlist["base_cap"] += r.capacity_GW
        dlist["dist_list"].append(r.distance_km)

    for key, (tech, v_kV) in PRESET_TECH.items():
        if key not in agg:
            agg[key] = dict(prov_i=key[0], prov_j=key[1], tech=tech, v_kV=v_kV, base_cap=0.0, dist_list=[])

    rows = []
    for rec in agg.values():
        dkm = sum(rec["dist_list"]) / len(rec["dist_list"]) if rec["dist_list"] else 500.0
        tinfo = TECH_INFO[rec["tech"]]
        loss_fac = (1 - tinfo["loss_per_km"]) ** dkm if rec["tech"] == "HVAC" else (1 - tinfo["conv_loss"]) * (
                    1 - tinfo["loss_per_km"]) ** dkm
        rows.append(dict(prov_i=rec["prov_i"], prov_j=rec["prov_j"], tech=rec["tech"], v_kV=rec["v_kV"],
                         base_cap=rec["base_cap"], dist_km=dkm, loss_factor=loss_fac))
    return pd.DataFrame(rows)


def split_into_vintage(cap_base: np.ndarray, start_year: int, end_year: int, step=5):
    vintages = {pb: np.zeros_like(cap_base) for pb in range(start_year, end_year + 1, step)}
    vintages[start_year] = cap_base.copy()
    return vintages


def _clip_nested(d, eps=1e-6):
    for k, v in d.items():
        if isinstance(v, dict):
            _clip_nested(v, eps)
        else:
            d[k] = np.maximum(v, 0.0)


CARBON_CAP_SCENARIOS = {
    "NDC": {
        2020: {44: 285.13, 45: 85.91, 46: 17.90, 52: 112.72, 53: 34.55},
        2025: {44: 288.22, 45: 86.84, 46: 18.09, 52: 113.94, 53: 34.92},
        2030: {44: 267.82, 45: 80.70, 46: 16.81, 52: 105.87, 53: 32.45},
        2035: {44: 233.23, 45: 70.28, 46: 14.64, 52: 92.20, 53: 28.26},
        2040: {44: 184.20, 45: 55.50, 46: 11.56, 52: 72.82, 53: 22.32},
        2045: {44: 142.50, 45: 42.94, 46: 8.95, 52: 56.33, 53: 17.27},
        2050: {44: 103.03, 45: 31.04, 46: 6.47, 52: 40.73, 53: 12.48},
    },
    "GM2.0": {
        2020: {44: 285.1313, 45: 85.914, 46: 17.89875, 52: 112.7205, 53: 34.54875},
        2025: {44: 275.6441, 45: 83.0554, 46: 17.30321, 52: 108.97, 53: 33.39922},
        2030: {44: 239.5943, 45: 72.19308543, 46: 15.04022613, 52: 94.71844736, 53: 29.03113416},
        2035: {44: 182.6572837, 45: 55.03717278, 46: 11.46607766, 52: 72.20962397, 53: 22.13219642},
        2040: {44: 126.2012, 45: 38.02616, 46: 7.922117, 52: 49.89091, 53: 15.29153},
        2045: {44: 75.06802, 45: 22.61903, 46: 4.712299, 52: 29.67652, 53: 9.095833},
        2050: {44: 30.11117, 45: 9.072912, 46: 1.89019, 52: 11.9038, 53: 3.648506},
    },
    "CN2050": {
        2020: {44: 285.1313, 45: 85.914, 46: 17.89875, 52: 112.7205, 53: 34.54875},
        2025: {44: 273.5269, 45: 82.41744, 46: 17.1703, 52: 108.133, 53: 33.14267},
        2030: {44: 218.6139684, 45: 65.87142056, 46: 13.72321262, 52: 86.42432504, 53: 26.4889918},
        2035: {44: 137.1395508, 45: 41.3220486, 46: 8.608760125, 52: 54.21516841, 53: 16.61690908},
        2040: {44: 47.77068, 45: 14.39397, 46: 2.998743, 52: 18.88511, 53: 5.788272},
        2045: {44: 10.62454, 45: 3.20132, 46: 0.666942, 52: 4.200182, 53: 1.287353},
        2050: {44: 0, 45: 0, 46: 0, 52: 0, 53: 0},
    },
}

CARBON_SCENARIO = "GM2.0"


def carbon_cap_t_sample(year: int, prov: int, sample_hours: int = 2016) -> float:
    caps = CARBON_CAP_SCENARIOS[CARBON_SCENARIO]
    key = max(y for y in caps if y <= year)
    return caps[key][prov] * 1e6 * (sample_hours / 8760)


def initialize_annual_inputs(
        year: int, thermal_csv: str, storage_xlsx: str, line_csv: str, decision_points: pd.DataFrame, compute_gap,
        *, thermal_cap_vintage_override=None, storage_cap_p_override=None, line_cap_override=None,
        ccs_cap_vintage_override=None
):
    thermal_df = pd.read_csv(thermal_csv).set_index("Provence_n").loc[SOUTH_IDS].reset_index()
    storage_df = pd.read_excel(storage_xlsx).set_index("Provence_num").reindex(SOUTH_IDS)
    corridor_df = build_corridors(line_csv=line_csv)

    if line_cap_override is not None:
        corridor_df["base_cap"] = pd.Series(line_cap_override, index=corridor_df.index).reindex(
            corridor_df.index).values

    unit_cols = ["Coal_morethan600", "Coal_300to600", "Coal_lessthan300", "GAS"]

    if thermal_cap_vintage_override:
        thermal_cap_vintage = thermal_cap_vintage_override
    else:
        thermal_df[unit_cols] = thermal_df[unit_cols].clip(lower=0).where(thermal_df[unit_cols].abs() > 1e-5, 0.0)
        thermal_cap_vintage = {u: split_into_vintage(thermal_df[u].to_numpy(), 2020, year, step=5) for u in unit_cols}

    if ccs_cap_vintage_override is not None:
        ccs_cap_vintage = ccs_cap_vintage_override
    else:
        ccs_cap_vintage = {u: {pb: np.zeros_like(arr) for pb, arr in thermal_cap_vintage[u].items()} for u in
                           THERMAL_PARAMS}

    for u in THERMAL_PARAMS:
        thermal_cap_vintage[u].setdefault(year, np.zeros_like(next(iter(thermal_cap_vintage[u].values()))))
        ccs_cap_vintage[u].setdefault(year, np.zeros_like(next(iter(ccs_cap_vintage[u].values()))))

        for pb, arr in ccs_cap_vintage[u].items():
            base = thermal_cap_vintage[u][pb].astype(float)
            ccs_cap_vintage[u][pb] = np.minimum(arr.astype(float), base)
            ccs_cap_vintage[u][pb][np.abs(ccs_cap_vintage[u][pb]) < 1e-5] = 0.0
            thermal_cap_vintage[u][pb][np.abs(thermal_cap_vintage[u][pb]) < 1e-5] = 0.0

    if storage_cap_p_override:
        storage_cap_p_vintage = storage_cap_p_override
    else:
        storage_cap_p_vintage = {
            tech: split_into_vintage(storage_df[f"{tech}(MW)"].to_numpy(), start_year=2020, end_year=year, step=5) for
            tech in STO_DURATION}

    storage_cap_p_base = {}
    for tech, vint in storage_cap_p_vintage.items():
        alive = {pb: arr for pb, arr in vint.items() if year - pb < STO_LIFE[tech]}
        storage_cap_p_vintage[tech] = alive
        storage_cap_p_base[tech] = sum(alive.values())

    gap_list, load_list, clean_dict = [], [], {}
    if line_cap_override is not None:
        for m in range(1, 13):
            gap_m, load_m, clean_m = build_typical_gap_province_independent(year, m, decision_points, n_runs=500)
            gap_list.append(gap_m)
            load_list.append(load_m)
            for k in clean_m: clean_dict.setdefault(k, []).append(clean_m[k])

        gap_df = pd.concat(gap_list, axis=0).reset_index(drop=True)[SOUTH_IDS]
        load_df = pd.concat(load_list, axis=0).reset_index(drop=True)[SOUTH_IDS]
        clean_df = {k: pd.concat(v, axis=0).reset_index(drop=True) for k, v in clean_dict.items()}
        clean_df = pd.concat(clean_df, axis=1).loc[:, pd.IndexSlice[:, SOUTH_IDS]]

        _clip_nested(thermal_cap_vintage)
        _clip_nested(ccs_cap_vintage)
        _clip_nested(storage_cap_p_vintage)
    else:
        gap_df, load_df, clean_df = gap_list, load_list, clean_dict

    return dict(
        gap_df=gap_df, load_df=load_df, clean_power_df=clean_df,
        thermal_cap_vintage=thermal_cap_vintage, storage_cap_p_base=storage_cap_p_base,
        storage_cap_p_vintage=storage_cap_p_vintage, ccs_cap_vintage=ccs_cap_vintage,
        corridor_df=corridor_df, provinces=SOUTH_IDS.copy(),
    )


class AnnualDispatchOptimizer:
    def __init__(self, data: dict, current_year: int, threads: int = 22, mandatory_retire=None):
        self.data = data
        self.current_year = current_year
        self.provinces = data["provinces"]
        self.n_prov = len(self.provinces)
        self.T = T_YEAR
        self.month_of_t = np.repeat(np.arange(1, 13), HOURS_PER_MONTH)
        self.is_winter = np.isin(self.month_of_t, list(WINTER_MONTHS))

        self.pbs = sorted(set(next(iter(data["thermal_cap_vintage"].values())).keys()) | {self.current_year})
        self.n_pb = len(self.pbs)

        self.gap_df = data["gap_df"].values
        self.load_df = data["load_df"]
        self.clean_power_df = data["clean_power_df"]
        self.thermal_cap_vintage = data["thermal_cap_vintage"]
        self.storage_cap_p_base = data["storage_cap_p_base"]
        self.storage_cap_p_vintage = data["storage_cap_p_vintage"]
        self.ccs_cap_vintage = data["ccs_cap_vintage"]
        self.corridor_df = data["corridor_df"]
        self.n_corr = len(self.corridor_df)
        self.loss_fac = self.corridor_df["loss_factor"].values
        self.mandatory_retire = mandatory_retire if mandatory_retire is not None else {u: np.zeros(self.n_prov) for u in
                                                                                       THERMAL_PARAMS}

        self.corridor_df["src_idx"] = self.corridor_df["prov_i"].map(prov_to_idx).astype(int)
        self.corridor_df["dst_idx"] = self.corridor_df["prov_j"].map(prov_to_idx).astype(int)
        self.corr_src = self.corridor_df["src_idx"].to_numpy()
        self.corr_dst = self.corridor_df["dst_idx"].to_numpy()

        self.incidence = [[] for _ in range(self.n_prov)]
        for j in range(self.n_corr):
            self.incidence[int(self.corr_src[j])].append((j, -1))
            self.incidence[int(self.corr_dst[j])].append((j, +1))

        self.m = gp.Model(f"dispatch_{current_year}")
        self.m.Params.OutputFlag = 0
        self.m.Params.Method = 2
        self.m.Params.Threads = threads
        self.m.Params.FeasibilityTol = 1e-2
        self.m.Params.NumericFocus = 1

        self.cost_terms = {
            "capex_thermal": gp.LinExpr(), "fixom_thermal": gp.LinExpr(),
            "capex_ccs": gp.LinExpr(), "fixom_ccs": gp.LinExpr(),
            "capex_storage": gp.LinExpr(), "capex_line_linear": gp.LinExpr(),
            "line_fix_OM": gp.LinExpr(), "capex_substation": gp.LinExpr(),
            "intra_transmission": 0.0, "fuel": gp.LinExpr(),
            "co2_chain": gp.LinExpr(), "storage_var_om": gp.LinExpr(),
            "storage_fix_om": gp.LinExpr(), "start": gp.LinExpr(), "Co2": gp.LinExpr()
        }

        self._build_capacity_layer()
        self._build_dispatch_layer()
        self._build_objective()

    def _build_capacity_layer(self):
        self.new_th = self.m.addVars(list(THERMAL_PARAMS), self.n_prov, lb=0, name="newTh")
        self.ret_th = self.m.addVars(list(THERMAL_PARAMS), self.n_pb, self.n_prov, lb=0, name="earlyRetire")
        self.add_ccs = self.m.addVars(list(CCS_RETROFIT), self.n_pb, self.n_prov, lb=0, name="newCCS")
        self.add_sto_p = self.m.addVars(list(STORAGE_TECHS), self.n_prov, lb=0, name="addStoP")

        self.inv_cost = gp.LinExpr()
        TOL, EPS = 1e-5, 1e-5

        for u in THERMAL_PARAMS:
            self.cost_terms["capex_thermal"] += INV_COST_THERMAL[u] * 1000 * gp.quicksum(
                self.new_th[u, i] for i in range(self.n_prov))
            tech = "Coal" if u.startswith("Coal") else "GAS"

            for i in range(self.n_prov):
                base = sum(float(self.thermal_cap_vintage[u][pb][i]) for pb in self.pbs)
                net = base + self.new_th[u, i] - gp.quicksum(self.ret_th[u, pbi, i] for pbi in range(self.n_pb))
                self.cost_terms["fixom_thermal"] += THERMAL_PARAMS[u]["var_cost"] * 1000 * net * year_scale

            capex_kw = ccs_capex_kw(tech, self.current_year)
            fix_om_kw = ccs_fix_om_kw(tech, self.current_year)

            self.cost_terms["capex_ccs"] += capex_kw * 1000 * gp.quicksum(
                self.add_ccs[u, pbi, i] for pbi in range(self.n_pb) for i in range(self.n_prov))

            ccs_hist = sum(float(np.sum(self.ccs_cap_vintage[u][pb])) for pb in self.pbs)
            add_new = gp.quicksum(self.add_ccs[u, pbi, i] for pbi in range(self.n_pb) for i in range(self.n_prov))
            self.cost_terms["fixom_ccs"] += fix_om_kw * 1000 * (ccs_hist + add_new) * year_scale

            headroom = np.zeros(self.n_prov, dtype=float)
            for pb in self.pbs: headroom += self.thermal_cap_vintage[u][pb].astype(float)
            headroom[np.abs(headroom) < EPS] = 0.0

            mand_req = np.minimum(self.mandatory_retire[u], headroom)
            mand_req[np.abs(mand_req) < EPS] = 0.0

            for i in range(self.n_prov):
                rhs = float(mand_req[i])
                if rhs > 0.0:
                    self.m.addConstr(gp.quicksum(self.ret_th[u, pbi, i] for pbi in range(self.n_pb)) >= rhs,
                                     name=f"mandRet_{u}_{i}")

            for pbi, pb in enumerate(self.pbs):
                base_vec = self.thermal_cap_vintage[u][pb].astype(float)
                for i in range(self.n_prov):
                    ub = max(base_vec[i], 0.0) if base_vec[i] >= EPS else 0.0
                    self.m.addConstr(self.ret_th[u, pbi, i] <= ub, name=f"noOverRet_{u}_{pb}_{i}")
                    self.m.addConstr(self.ret_th["GAS", pbi, i] == 0.0, name=f"fixRet_GAS_{pbi}_{i}")

                    base_minus_ccs = float(base_vec[i] - self.ccs_cap_vintage[u][pb][i])
                    base_minus_ccs_pos = base_minus_ccs if base_minus_ccs > EPS else 0.0
                    raw_avail_expr = base_minus_ccs_pos if pb < self.current_year else base_minus_ccs_pos + self.new_th[
                        u, i]
                    self.m.addConstr(self.add_ccs[u, pbi, i] <= raw_avail_expr, name=f"noOverCCS_{u}_{pb}_{i}")

        for tech in STORAGE_TECHS:
            cap_p = storage_capex_kwh_exp(tech, self.current_year) * 1000 * STO_DURATION[tech]
            self.cost_terms["capex_storage"] += cap_p * gp.quicksum(self.add_sto_p[tech, i] for i in range(self.n_prov))
            cap_p2 = gp.quicksum(self.storage_cap_p_base[tech][i] + self.add_sto_p[tech, i] for i in range(self.n_prov))
            self.cost_terms["storage_fix_om"] += STORAGE_TECHS[tech]["fix_om"] * cap_p2 * year_scale

        self.add_corr = self.m.addVars(self.n_corr, lb=0, name="addCorr")
        self.cap_corr = [float(self.corridor_df.at[j, "base_cap"]) + self.add_corr[j] for j in range(self.n_corr)]

        for j in range(self.n_corr):
            tech, dist = self.corridor_df.at[j, "tech"], self.corridor_df.at[j, "dist_km"]
            cap_add_MW = self.add_corr[j] * 1000
            cref = 8000.0 if tech == "HVDC" else cref_hvac_500kv(dist)

            line_cost = (cap_add_MW / cref) * TECH_INFO[tech]["capex"] * 1000 * dist
            sub_cost = cap_add_MW * TECH_INFO[tech]["substation"] * 1000
            self.cost_terms["capex_line_linear"] += line_cost
            self.cost_terms["capex_substation"] += sub_cost * 2

            cap_MW = self.cap_corr[j] * 1000
            line_capex_kUSD = TECH_INFO[tech]["capex"] * 1000 * dist * (cap_MW / cref)
            sub_capex_both_kUSD = TECH_INFO[tech]["substation"] * 1000 * cap_MW * 2
            self.cost_terms["line_fix_OM"] += (
                                                          FOM_LINE_RATE * line_capex_kUSD + FOM_SUB_RATE * sub_capex_both_kUSD) * year_scale

        self.inv_cost += sum([self.cost_terms[k] for k in
                              ["capex_thermal", "fixom_thermal", "capex_ccs", "fixom_ccs", "capex_storage",
                               "storage_fix_om", "capex_line_linear", "capex_substation", "line_fix_OM"]])

    def _build_dispatch_layer(self):
        self.on, self.start, self.stop, self.dp_noCC, self.dp_CC, self.comm_noCC, self.comm_CC = {}, {}, {}, {}, {}, {}, {}

        for u in THERMAL_PARAMS:
            self.on[u] = self.m.addVars(self.T, self.n_prov, lb=0, name=f"ON_{u}")
            self.start[u] = self.m.addVars(self.T, self.n_prov, lb=0, name=f"ST_{u}")
            self.stop[u] = self.m.addVars(self.T, self.n_prov, lb=0, name=f"SD_{u}")
            self.dp_noCC[u] = self.m.addVars(self.T, self.n_prov, lb=0, name=f"PnoCC_{u}")
            self.dp_CC[u] = self.m.addVars(self.T, self.n_prov, lb=0, name=f"PCC_{u}")
            self.comm_noCC[u] = self.m.addVars(self.T, self.n_prov, lb=0, name=f"comm_noCC_{u}")
            self.comm_CC[u] = self.m.addVars(self.T, self.n_prov, lb=0, name=f"comm_CC_{u}")

        for u in THERMAL_PARAMS:
            rup, rdn, Tu, Td = THERMAL_PARAMS[u]["rup"], THERMAL_PARAMS[u]["rdn"], MIN_UP_HR[u], MIN_DN_HR[u]
            for i in range(self.n_prov):
                base_cap = sum(
                    self.thermal_cap_vintage[u][pb][i] - self.ret_th[u, pbi, i] for pbi, pb in enumerate(self.pbs)) + \
                           self.new_th[u, i]
                Σ_ccs = sum(self.ccs_cap_vintage[u][pb][i] for pb in self.pbs) + gp.quicksum(
                    self.add_ccs[u, pbi, i] for pbi in range(self.n_pb))
                ret_ccs_tot = self.m.addVar(lb=0.0, name=f"retCC_tot_{u}_{i}")
                self.m.addConstr(ret_ccs_tot <= Σ_ccs)

                Sigma_ccs = Σ_ccs - ret_ccs_tot
                self.m.addConstr(Sigma_ccs <= base_cap)

                for t in range(self.T):
                    self.m.addConstr(self.comm_noCC[u][t, i] + self.comm_CC[u][t, i] == self.on[u][t, i])
                    self.m.addConstr(self.comm_CC[u][t, i] <= Sigma_ccs)
                    self.m.addConstr(self.comm_noCC[u][t, i] <= base_cap - Sigma_ccs)

                    if t == 0:
                        self.m.addConstr(self.on[u][t, i] == self.start[u][t, i] - self.stop[u][t, i])
                        self.m.addConstr(self.on[u][t, i] <= base_cap)
                    else:
                        self.m.addConstr(
                            self.on[u][t, i] == self.on[u][t - 1, i] + self.start[u][t, i] - self.stop[u][t, i])
                        self.m.addConstr(
                            self.dp_noCC[u][t, i] - self.dp_noCC[u][t - 1, i] <= rup * self.comm_noCC[u][t - 1, i])
                        self.m.addConstr(
                            self.dp_noCC[u][t - 1, i] - self.dp_noCC[u][t, i] <= rdn * self.comm_noCC[u][t - 1, i])
                        self.m.addConstr(
                            self.dp_CC[u][t, i] - self.dp_CC[u][t - 1, i] <= rup * self.comm_CC[u][t - 1, i])
                        self.m.addConstr(
                            self.dp_CC[u][t - 1, i] - self.dp_CC[u][t, i] <= rdn * self.comm_CC[u][t - 1, i])

                    pmin = THERMAL_PARAMS[u]["pmin_w"] if self.is_winter[t] else THERMAL_PARAMS[u]["pmin"]
                    pmax = THERMAL_PARAMS[u]["pmax"]
                    self.m.addConstr(self.dp_noCC[u][t, i] <= pmax * self.comm_noCC[u][t, i])
                    self.m.addConstr(self.dp_CC[u][t, i] <= pmax * self.comm_CC[u][t, i])
                    self.m.addConstr(self.dp_noCC[u][t, i] >= pmin * self.comm_noCC[u][t, i])
                    self.m.addConstr(self.dp_CC[u][t, i] >= pmin * self.comm_CC[u][t, i])

                    lhs_up = gp.quicksum(self.start[u][k, i] for k in range(max(0, t - Tu + 1), t + 1))
                    lhs_dn = gp.quicksum(self.stop[u][k, i] for k in range(max(0, t - Td + 1), t + 1))
                    self.m.addConstr(self.on[u][t, i] >= lhs_up)
                    self.m.addConstr(base_cap - self.on[u][t, i] >= lhs_dn)

        self.E_annual = self.m.addVars(self.n_prov, lb=0, name="E_CO2")
        self.co2_slack = self.m.addVars(self.n_prov, lb=0.0, name="co2_slack")
        for i, prov in enumerate(self.provinces):
            expr = gp.quicksum(THERMAL_PARAMS[u]["CO2_emission"] / 1000 * (
                        self.dp_noCC[u][t, i] + (1 - CAPTURE_RATE) * self.dp_CC[u][t, i]) for u in THERMAL_PARAMS for t
                               in range(self.T))
            self.m.addConstr(self.E_annual[i] == expr)
            self.m.addConstr(
                self.E_annual[i] <= carbon_cap_t_sample(self.current_year, prov, self.T) + self.co2_slack[i])

        self.stor, self.soc, self.stor_abs, self.P_chg, self.P_dis = {}, {}, {}, {}, {}
        for tech, p in STORAGE_TECHS.items():
            self.stor[tech] = self.m.addVars(self.T, self.n_prov, lb=-GRB.INFINITY, name=f"stor_{tech}")
            self.soc[tech] = self.m.addVars(self.T + 1, self.n_prov, lb=0, name=f"soc_{tech}")
            self.P_chg[tech] = self.m.addVars(self.T, self.n_prov, lb=0, name=f"Pchg_{tech}")
            self.P_dis[tech] = self.m.addVars(self.T, self.n_prov, lb=0, name=f"Pdis_{tech}")

            η_c = p["eff"] ** 0.5
            η_d = p["eff"] ** 0.5
            λ_hr = STO_SELF_DIS[tech] / 100 / 24

            for i in range(self.n_prov):
                cap_p = self.storage_cap_p_base[tech][i] + self.add_sto_p[tech, i]
                cap_e = STO_DURATION[tech] * cap_p

                for t in range(self.T):
                    self.m.addConstr(self.P_chg[tech][t, i] <= cap_p)
                    self.m.addConstr(self.P_dis[tech][t, i] <= cap_p)
                    self.m.addConstr(η_c * self.P_chg[tech][t, i] + self.P_dis[tech][t, i] / η_d <= cap_p)
                    self.m.addConstr(self.stor[tech][t, i] == self.P_dis[tech][t, i] - self.P_chg[tech][t, i])
                    self.m.addConstr(
                        self.soc[tech][t + 1, i] == (1 - λ_hr) * self.soc[tech][t, i] + η_c * self.P_chg[tech][t, i] -
                        self.P_dis[tech][t, i] / η_d)

                for t in range(1, self.T + 1):
                    self.m.addConstr(self.soc[tech][t, i] <= cap_e)
                self.m.addConstr(self.soc[tech][0, i] == self.soc[tech][self.T, i])
                self.m.addConstr(gp.quicksum(self.P_dis[tech][t, i] for t in range(self.T)) <= CYCLE_LIMIT * cap_p)

        self.F_fwd = self.m.addVars(self.T, self.n_corr, lb=0, name="Ffwd")
        self.F_rev = self.m.addVars(self.T, self.n_corr, lb=0, name="Frev")
        self.eta_line = self.corridor_df["loss_factor"].to_numpy()

        for j in range(self.n_corr):
            inv_eff = 1.0 / self.loss_fac[j]
            for t in range(self.T):
                self.m.addConstr(self.F_fwd[t, j] + self.F_rev[t, j] * inv_eff <= self.cap_corr[j])

        self.res_sto = self.m.addVars(self.T, self.n_prov, lb=0, name="Res_STO")
        self.res_th = self.m.addVars(self.T, self.n_prov, lb=0, name="Res_TH")
        self.res_tot = self.m.addVars(self.T, self.n_prov, lb=0, name="Res_TOT")
        self.res_sto_tech = {tech: self.m.addVars(self.T, self.n_prov, lb=0, name=f"Res_STO_{tech}") for tech in
                             STORAGE_TECHS}
        self.curtail = self.m.addVars(self.T, self.n_prov, lb=0, name="curtail")

        clean_arr = self.clean_power_df["Wind"].values + self.clean_power_df["Solar"].values + self.clean_power_df[
            "Hydro"].values
        for t in range(self.T):
            for i in range(self.n_prov):
                dp_eff_sum = gp.quicksum(self.dp_noCC[u][t, i] + (
                            1 - ccs_energy_penalty("Coal" if u.startswith("Coal") else "GAS", self.current_year)) *
                                         self.dp_CC[u][t, i] for u in THERMAL_PARAMS)
                stor_sum = gp.quicksum(self.stor[tech][t, i] for tech in STORAGE_TECHS)

                net_trans = gp.LinExpr()
                for j in range(self.n_corr):
                    η, r, s = self.eta_line[j], self.corr_src[j], self.corr_dst[j]
                    if i == r:
                        net_trans += 1000 * (self.F_rev[t, j] - self.F_fwd[t, j] / η)
                    elif i == s:
                        net_trans += 1000 * (self.F_fwd[t, j] - self.F_rev[t, j] / η)

                self.m.addConstr(dp_eff_sum + stor_sum + net_trans == self.gap_df[t, i] + self.curtail[t, i])
                self.m.addConstr(self.curtail[t, i] <= clean_arr[t, i])

                headroom_th = gp.quicksum(THERMAL_PARAMS[u]["pmax"] * self.on[u][t, i] - (self.dp_noCC[u][t, i] + (
                            1 - ccs_energy_penalty("Coal" if u.startswith("Coal") else "GAS", self.current_year)) *
                                                                                          self.dp_CC[u][t, i]) for u in
                                          THERMAL_PARAMS)
                self.m.addConstr(self.res_th[t, i] <= headroom_th)

                for tech in STORAGE_TECHS:
                    self.m.addConstr(
                        self.res_sto_tech[tech][t, i] <= self.storage_cap_p_base[tech][i] + self.add_sto_p[tech, i] -
                        self.P_dis[tech][t, i])
                    self.m.addConstr(self.res_sto_tech[tech][t, i] <= self.soc[tech][t, i])

                self.m.addConstr(
                    self.res_sto[t, i] == gp.quicksum(self.res_sto_tech[tech][t, i] for tech in STORAGE_TECHS))
                self.m.addConstr(self.res_tot[t, i] == self.res_th[t, i] + self.res_sto[t, i])

                req = SPIN_ALPHA * self.load_df.iloc[t, i] + SPIN_BETA * (
                            self.clean_power_df["Wind"].iloc[t, i] + self.clean_power_df["Solar"].iloc[t, i])
                self.m.addConstr(self.res_tot[t, i] >= req)

    def _build_objective(self):
        self.cost_terms["start"] += gp.quicksum(
            START_COST[u] * self.start[u][t, i] for u in self.start for t in range(self.T) for i in
            range(self.n_prov)) * OM

        for u in THERMAL_PARAMS:
            tech = "Coal" if u.startswith("Coal") else "GAS"
            coef, alpha = THERMAL_PARAMS[u]["fuel_use"] * 29307 / 1_000_000, ALPHA[u]
            for i, prov in enumerate(self.provinces):
                price = (COAL_PRICE if tech == "Coal" else GAS_PRICE)[prov]
                for t in range(self.T):
                    self.cost_terms["fuel"] += price * (
                                coef * (self.dp_noCC[u][t, i] + self.dp_CC[u][t, i]) + alpha * self.on[u][t, i])
        self.cost_terms["fuel"] *= OM

        unit_fee = TRANS_COST_CO2 * CO2_DISTANCE_KM + STORE_COST_CO2
        for u in self.dp_CC:
            ef_t = THERMAL_PARAMS[u]["CO2_emission"] / 1_000
            for i in range(self.n_prov):
                self.cost_terms["co2_chain"] += unit_fee * gp.quicksum(
                    self.dp_CC[u][t, i] for t in range(self.T)) * ef_t * CAPTURE_RATE
        self.cost_terms["co2_chain"] *= OM

        self.cost_terms["storage_var_om"] += gp.quicksum(
            STORAGE_TECHS[tech]["var_om"] * self.P_dis[tech][t, i] for tech in STORAGE_TECHS for t in range(self.T) for
            i in range(self.n_prov)) * OM
        self.cost_terms["Co2"] = 100 * gp.quicksum(self.co2_slack[i] for i in range(self.n_prov)) * OM

        total = sum(self.cost_terms[k] for k in ["fuel", "co2_chain", "storage_var_om", "start", "Co2"]) + self.inv_cost
        self.m.setObjective(total, GRB.MINIMIZE)
        self.m.Params.ObjScale = 1e-4

    def optimize(self):
        self.m.optimize()
        if self.m.Status == GRB.OPTIMAL: print(f"[{self.current_year}] optimal cost = {self.m.ObjVal:,.0f} k RMB")
        return self.m.Status

    def get_cost_breakdown(self) -> dict[str, float]:
        out = {k: float(expr.getValue() if hasattr(expr, "getValue") else expr) for k, expr in self.cost_terms.items()}
        out["total_cost"] = float(self.m.ObjVal)
        return out

    def get_capacity_deltas(self):
        Δ_vintage_th = {u: {pb: np.zeros(self.n_prov) for pb in self.pbs + [self.current_year]} for u in THERMAL_PARAMS}
        Δ_vintage_ccs = {u: {pb: np.zeros(self.n_prov) for pb in self.pbs + [self.current_year]} for u in
                         THERMAL_PARAMS}

        for u in THERMAL_PARAMS:
            Δ_vintage_th[u][self.current_year] = np.array([self.new_th[u, i].X for i in range(self.n_prov)])
            for pbi, pb in enumerate(self.pbs):
                Δ_vintage_th[u][pb] -= np.array([self.ret_th[u, pbi, i].X for i in range(self.n_prov)])
                Δ_vintage_ccs[u][pb] += np.array([self.add_ccs[u, pbi, i].X for i in range(self.n_prov)])

        Δ_sto_p = {tech: np.array([self.add_sto_p[tech, i].X for i in range(self.n_prov)]) for tech in STORAGE_TECHS}
        Δ_line = pd.Series([self.add_corr[j].X for j in range(self.n_corr)], index=self.corridor_df.index, name="ΔGW")
        return Δ_vintage_th, Δ_vintage_ccs, Δ_sto_p, Δ_line

    def add_intra_cost(self, value: float):
        self.cost_terms["intra_transmission"] += float(value)
        self.inv_cost += float(value)

    def get_cost_report(self) -> pd.Series:
        s = {k: float(v.getValue() if hasattr(v, "getValue") else v) for k, v in self.cost_terms.items()}
        s["investment_total"] = s["capex_thermal"] + s["capex_ccs"] + s["capex_storage"] + s["capex_line_linear"] + s[
            "capex_substation"] + s["intra_transmission"]
        s["opex_total"] = s["fuel"] + s["co2_chain"] + s["storage_var_om"] + s["start"] + s["fixom_thermal"] + s[
            "fixom_ccs"] + s["storage_fix_om"] + s["Co2"] + s["line_fix_OM"]
        s["total"] = s["investment_total"] + s["opex_total"]
        return pd.Series(s, name=self.current_year)


def extract_results(opt: AnnualDispatchOptimizer):
    T, P = opt.T, opt.n_prov
    hrs, provs = pd.RangeIndex(T, name="t"), opt.provinces

    ren_dict = {src: opt.clean_power_df[src][provs].copy() for src in ["Wind", "Solar", "Hydro", "Nuclear"] if
                src in opt.clean_power_df}

    coal_eff, gas_eff = np.zeros((T, P)), np.zeros((T, P))
    for u in THERMAL_PARAMS:
        tgt, tech = (coal_eff, "Coal") if u.startswith("Coal") else (gas_eff, "GAS")
        loss = ccs_energy_penalty(tech, opt.current_year)
        for t in range(T):
            tgt[t, :] += np.array([opt.dp_noCC[u][t, i].X for i in range(P)]) + (1.0 - loss) * np.array(
                [opt.dp_CC[u][t, i].X for i in range(P)])

    thermal_coal_df, thermal_gas_df = pd.DataFrame(coal_eff, index=hrs, columns=provs), pd.DataFrame(gas_eff, index=hrs,
                                                                                                     columns=provs)

    stor_net = np.zeros((T, P))
    for tech in STORAGE_TECHS:
        for t in range(T):
            stor_net[t, :] += np.array([opt.P_dis[tech][t, i].X - opt.P_chg[tech][t, i].X for i in range(P)])

    net_trans = np.zeros((T, P))
    fwd_send, rev_send, fwd_arr, rev_arr = np.zeros((T, opt.n_corr)), np.zeros((T, opt.n_corr)), np.zeros(
        (T, opt.n_corr)), np.zeros((T, opt.n_corr))

    for j in range(opt.n_corr):
        η, r, s = opt.eta_line[j], opt.corr_src[j], opt.corr_dst[j]
        for t in range(T):
            Ff, Fr = opt.F_fwd[t, j].X * 1000.0, opt.F_rev[t, j].X * 1000.0
            net_trans[t, r] += Fr - Ff / η
            net_trans[t, s] += Ff - Fr / η
            fwd_send[t, j], rev_send[t, j], fwd_arr[t, j], rev_arr[t, j] = Ff, Fr, η * Ff, η * Fr

    line_meta = opt.corridor_df[["prov_i", "prov_j", "tech", "v_kV", "loss_factor", "base_cap"]].copy()
    line_meta["default_dir"] = line_meta["prov_i"].astype(int).astype(str) + "→" + line_meta["prov_j"].astype(
        int).astype(str)

    lbl_fwd = line_meta.apply(lambda r: f"{int(r['prov_i'])}→{int(r['prov_j'])}", axis=1)
    lbl_rev = line_meta.apply(lambda r: f"{int(r['prov_j'])}→{int(r['prov_i'])}", axis=1)

    return dict(
        ren=ren_dict, thermal=thermal_coal_df + thermal_gas_df, thermal_coal=thermal_coal_df,
        thermal_gas=thermal_gas_df,
        emission=pd.Series([opt.E_annual[i].X for i in range(P)], index=provs, name="Annual_CO2_t"),
        storage=pd.DataFrame(stor_net, index=hrs, columns=provs),
        trans=pd.DataFrame(net_trans, index=hrs, columns=provs),
        demand=opt.load_df[provs].copy(),
        curtail=pd.DataFrame([[opt.curtail[t, i].X for i in range(P)] for t in range(T)], index=hrs, columns=provs),
        line_flow_fwd_send=pd.DataFrame(fwd_send, index=hrs, columns=opt.corridor_df.index),
        line_flow_rev_send=pd.DataFrame(rev_send, index=hrs, columns=opt.corridor_df.index),
        line_flow_fwd_arr=pd.DataFrame(fwd_arr, index=hrs, columns=opt.corridor_df.index),
        line_flow_rev_arr=pd.DataFrame(rev_arr, index=hrs, columns=opt.corridor_df.index),
        line_meta=line_meta,
        pair_flow_arrive=pd.DataFrame(fwd_arr, index=hrs, columns=opt.corridor_df.index).groupby(lbl_fwd,
                                                                                                 axis=1).sum().add(
            pd.DataFrame(rev_arr, index=hrs, columns=opt.corridor_df.index).groupby(lbl_rev, axis=1).sum(),
            fill_value=0.0)
    )