"""
typical_week_selector.py
-------------------------
* 完全独立省份：每省单独聚类 7 条日曲线(24h)，互不干扰。
* 并行安全：不在子进程里修改共享 dict，避免空样本；采样结果由主进程汇总。
* 极端日保障：强制保留最高 & 最低日。
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn_extra.cluster import KMedoids
from joblib import Parallel, delayed

from REX.compute_power_gap import compute_gap


def _sample_once(seed: int, year: int, month: int, dp) -> Dict[int, List[Tuple[np.ndarray, np.ndarray]]]:
    np.random.seed(seed)
    _, _, clean = compute_gap(year, month, dp)
    wind_df, solar_df = clean["Wind"], clean["Solar"]

    result = {}
    for p_idx, prov in enumerate(wind_df.columns):
        result[prov] = []
        for d in range(7):
            sl = slice(d * 24, (d + 1) * 24)
            result[prov].append((wind_df.iloc[sl, p_idx].to_numpy(float), solar_df.iloc[sl, p_idx].to_numpy(float)))
    return result


def _collect_samples(year: int, month: int, dp, n_runs: int = 60, n_jobs: int = 8) -> Dict[
    int, List[Tuple[np.ndarray, np.ndarray]]]:
    all_results = Parallel(n_jobs=n_jobs)(delayed(_sample_once)(seed, year, month, dp) for seed in range(n_runs))
    merged = {}
    for res in all_results:
        for prov, lst in res.items():
            merged.setdefault(prov, []).extend(lst)
    return merged


def _select_7_days(day_vecs: List[Tuple[np.ndarray, np.ndarray]], *, k: int = 7, inject_extremes: bool = False,
                   random_state: int = 42) -> List[int]:
    if len(day_vecs) < k: raise ValueError(f"样本不足：仅 {len(day_vecs)} 条日曲线，无法聚 {k} 类")

    X = np.array([np.concatenate(v) for v in day_vecs])
    km = KMedoids(n_clusters=k, init="k-medoids++", random_state=random_state).fit(X)
    idx = list(km.medoid_indices_)

    if inject_extremes:
        totals = X.sum(axis=1)
        idx.extend([int(np.argmin(totals)), int(np.argmax(totals))])

    idx = list(dict.fromkeys(idx))
    if len(idx) < k:
        selected_set = set(idx)
        dists = np.square(X[:, None] - X[idx]).sum(axis=2).min(axis=1)
        for i in np.argsort(-dists):
            if i not in selected_set:
                idx.append(int(i))
                if len(idx) == k: break

    idx.sort(key=lambda i: X[i].sum())
    return idx[:k]


def build_typical_gap_province_independent(year: int, month: int, decision_points, n_runs: int = 500, n_jobs: int = 22,
                                           k: int = 7):
    samples_by_prov = _collect_samples(year, month, decision_points, n_runs=n_runs, n_jobs=n_jobs)
    _, load_df, clean_tmp = compute_gap(year, month, decision_points)
    hydro_df, nuclear_df = clean_tmp["Hydro"], clean_tmp["Nuclear"]

    P, prov_codes = len(load_df.columns), list(load_df.columns)
    wind_mat, solar_mat = np.zeros((168, P)), np.zeros((168, P))

    for p_idx, prov in enumerate(prov_codes):
        day_list = samples_by_prov.get(prov, [])
        if not day_list: raise RuntimeError(f"省份 {prov} 未采到任何样本")

        chosen = _select_7_days(day_list, k=k, inject_extremes=True)
        w_rows, s_rows = zip(*[day_list[ix] for ix in chosen])

        wind_mat[:, p_idx] = np.concatenate(w_rows)
        solar_mat[:, p_idx] = np.concatenate(s_rows)

    clean_df = pd.concat({
        "Wind": pd.DataFrame(wind_mat, columns=prov_codes),
        "Solar": pd.DataFrame(solar_mat, columns=prov_codes),
        "Hydro": hydro_df.reset_index(drop=True),
        "Nuclear": nuclear_df.reset_index(drop=True)
    }, axis=1)

    gap_df = load_df.reset_index(drop=True) - clean_df.T.groupby(level=1).sum().T
    return gap_df, load_df.reset_index(drop=True), clean_df