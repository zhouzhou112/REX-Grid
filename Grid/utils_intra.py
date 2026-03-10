import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import geopandas as gpd
from scipy.spatial import cKDTree
from math import radians, sin, cos, sqrt, atan2
from tqdm import trange
from collections import defaultdict

from REX.compute_wind_solar import compute_wind_solar

TRUNK_CAPEX_K = 1181.18  # $/ MW·km
SPUR_CAPEX_K = 1181.18  # 自定义
Sub_CAPEX = 38000  # $/ MW


def _haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def _nearest_city_by_prov(ts_df: pd.DataFrame, city_df: pd.DataFrame):
    N = len(ts_df)
    cid, c_lat, c_lon, c_pcent = np.empty(N, dtype=int), np.empty(N), np.empty(N), np.empty(N, dtype=int)

    for prov, group in ts_df.groupby('Provence_num'):
        rows = group.index.values
        sub = city_df[city_df['Provence_n'] == prov]
        if sub.empty: raise ValueError(f'省份 {prov} 在 city-center.xls 中找不到！')

        _, idx = cKDTree(sub[['y', 'x']].values).query(group[['lat', 'lon']].values, k=1)
        cid[rows] = sub.iloc[idx]['city_id'].values
        c_lat[rows] = sub.iloc[idx]['y'].values
        c_lon[rows] = sub.iloc[idx]['x'].values
        c_pcent[rows] = sub.iloc[idx]['is_powercenter'].values

    return cid, c_lat, c_lon, c_pcent


def build_county_ts(year: int, decision_points: pd.DataFrame, city_center_path: str):
    acc = {pid: [] for pid in decision_points['point_id']}
    for m in trange(1, 13, desc=f"WindSolar {year}"):
        wdf, sdf = compute_wind_solar(year, m, decision_points)
        for w, s in zip(wdf.itertuples(), sdf.itertuples()):
            acc[w.point_id].extend(np.asarray(w.wind_power_output) + np.asarray(s.solar_power_output))

    ts_df = decision_points.rename(columns={'x': 'lon', 'y': 'lat', 'name': 'location_type'}).copy()
    ts_df['power_output'] = [acc[pid] for pid in ts_df['point_id']]

    city_df = pd.read_excel(city_center_path)[['Provence_n', 'x', 'y', 'is_powercenter', 'city_id']]
    cid, c_lat, c_lon, c_pcent = _nearest_city_by_prov(ts_df, city_df)

    ts_df['city_id'], ts_df['city_lat'], ts_df['city_lon'], ts_df['is_powercenter'] = cid, c_lat, c_lon, c_pcent
    county_ts = np.array(ts_df['power_output'].tolist()).T

    uniq_city = np.unique(cid)
    city_ts = np.zeros((county_ts.shape[0], len(uniq_city)))
    idx_map = {c: i for i, c in enumerate(uniq_city)}
    for col, c in enumerate(cid):
        city_ts[:, idx_map[c]] += county_ts[:, col]

    return county_ts, city_ts, ts_df, np.column_stack([cid, c_lat, c_lon, c_pcent])


def aggregate_to_city(county_ts: np.ndarray, ts_df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    unique_city = list(dict.fromkeys(ts_df['city_id']))
    city_meta = \
    ts_df[['city_id', 'city_lat', 'city_lon', 'Provence_num', 'is_powercenter']].drop_duplicates('city_id').set_index(
        'city_id').loc[unique_city]

    city_ts = np.zeros((county_ts.shape[0], len(unique_city)))
    city_idx = {cid: i for i, cid in enumerate(unique_city)}
    for col, cid in enumerate(ts_df['city_id']):
        city_ts[:, city_idx[cid]] += county_ts[:, col]

    return city_ts, city_meta


def load_intra_network_state(init_year: int, initial_year: int = 2020, state_dir: str = ".") -> tuple[dict, dict]:
    if init_year == initial_year: return {}, {}
    path = os.path.join(state_dir, f"intra_network_state_{init_year}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data.get("spur_dict", {}), data.get("trunk_dict", {})
    return {}, {}


def dump_intra_network_state(spur_dict: dict, trunk_dict: dict, year: int, state_dir: str = ".") -> None:
    with open(os.path.join(state_dir, f"intra_network_state_{year}.pkl"), "wb") as f:
        pickle.dump({"spur_dict": spur_dict, "trunk_dict": trunk_dict}, f)


def plan_intra_transmission_v1(county_ts, city_ts, city_meta, ts_df, spur_prev, trunk_prev,
                               network_path="city_network-0816"):
    cost = 0.0
    T, N_county = county_ts.shape
    county_absorb, city_absorb = 0.10, 0.30

    p95 = np.quantile(np.clip(county_ts, 0, None) * (1.0 - county_absorb), 0.95, axis=0)
    if 'dist_to_city_km' not in ts_df:
        ts_df['dist_to_city_km'] = ts_df.apply(lambda r: _haversine(r.lat, r.lon, r.city_lat, r.city_lon), axis=1)

    eta = np.maximum(1.0 - (0.004 / 100.0) * ts_df['dist_to_city_km'].to_numpy(), 1e-3)
    cap_need = (p95 / eta) * 1.10

    spur_records = []
    for need, pid, dist, cidx in zip(cap_need, ts_df['point_id'], ts_df['dist_to_city_km'], ts_df['city_id']):
        base = spur_prev.get(pid, 0.0)
        if need > base:
            spur_prev[pid] = need
            cost += (need - base) * dist * SPUR_CAPEX_K + (need - base) * Sub_CAPEX

        row = ts_df.loc[ts_df['point_id'] == pid].iloc[0]
        spur_records.append(
            {'county_id': pid, 'city_id': cidx, 'x_county': row.lon, 'y_county': row.lat, 'x_city': row.city_lon,
             'y_city': row.city_lat, 'capacity': spur_prev.get(pid, 0.0)})

    with open(network_path, "rb") as f:
        net = pickle.load(f)

    G = nx.Graph()
    for (ci, cj), d in net['edge_dist'].items():
        if city_meta.at[ci, 'Provence_num'] == city_meta.at[cj, 'Provence_num']:
            G.add_edge(ci, cj, weight=d)

    centers_by_prov = {prov: city_meta.loc[sub][city_meta.loc[sub, 'is_powercenter'] == 1].index.tolist() for prov, sub
                       in city_meta.groupby('Provence_num').groups.items()}

    city_ids = list(city_meta.index)
    edge_to_cols = {}
    for col, cid in enumerate(city_ids):
        if city_meta.at[cid, 'is_powercenter'] == 1: continue

        best_center, best_dist, best_path = None, float('inf'), None
        for ctr in centers_by_prov.get(city_meta.at[cid, 'Provence_num'], []):
            try:
                path = nx.shortest_path(G, cid, ctr, weight='weight')
                d = sum(G[u][v]['weight'] for u, v in zip(path, path[1:]))
                if d < best_dist: best_dist, best_center, best_path = d, ctr, path
            except nx.NetworkXNoPath:
                continue

        if best_path:
            for i in range(len(best_path) - 1):
                edge_to_cols.setdefault(tuple(sorted((best_path[i], best_path[i + 1]))), []).append(col)

    city_net_export = np.clip(np.clip(city_ts, 0.0, None) * (1.0 - county_absorb) * (1.0 - city_absorb), 0.0, None)

    trunk_records = []
    for e, cols in edge_to_cols.items():
        idxs = [c for c in cols if c < city_net_export.shape[1]]
        peak = np.quantile(city_net_export[:, idxs].sum(axis=1), 0.95) * 1.10 if idxs else 0.0
        base = trunk_prev.get(e, 0.0)

        if peak > base:
            trunk_prev[e] = peak
            cost += (peak - base) * net['edge_dist'][e] * TRUNK_CAPEX_K + (peak - base) * Sub_CAPEX

        trunk_records.append({'city_i': e[0], 'city_j': e[1], 'x_i': city_meta.at[e[0], 'city_lon'],
                              'y_i': city_meta.at[e[0], 'city_lat'], 'x_j': city_meta.at[e[1], 'city_lon'],
                              'y_j': city_meta.at[e[1], 'city_lat'], 'capacity': max(base, peak)})

    return spur_prev, trunk_prev, cost, pd.DataFrame(spur_records), pd.DataFrame(trunk_records)