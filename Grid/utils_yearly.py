from pathlib import Path
import pandas as pd

SOUTH_IDS = [44, 45, 46, 52, 53]

PROV_DIR_MAP_greedy = {p: rf"D:\codeenv\pycharmproject\RL_agent\RL_main\expansion\greedy\{p}\history" for p in
                       ["GD", "GX", "HN", "YN", "GZ"]}
PROV_DIR_MAP_rl = {p: rf"D:\codeenv\pycharmproject\RL_agent\RL_main\expansion\RL\{p}" for p in
                   ["GD", "GX", "HN", "YN", "GZ"]}
PROV_DIR_MAP_lcoe = {p: rf"D:\codeenv\pycharmproject\RL_agent\RL_main\expansion\LCOE\{p}" for p in
                     ["GD", "GX", "HN", "YN", "GZ"]}

UPDATE_COLS = ["Installed_onshorewind", "Installed_offshorewind", "Installed_pv", "Potential_onshorewind",
               "Potential_offshorewind", "Potential_pv"]


def read_table(path: str | Path, id_col: str = "point_id") -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path) if path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(path)
    df[id_col] = df[id_col].astype(str)
    return df


def update_decision_points(nation_df: pd.DataFrame, sub_df: pd.DataFrame, *, id_col: str = "point_id",
                           update_cols: list = None) -> pd.DataFrame:
    if update_cols is None: update_cols = [c for c in sub_df.columns if c != id_col]
    nat, sub = nation_df.set_index(id_col), sub_df.set_index(id_col)
    for col in update_cols:
        if col in nat.columns: nat.loc[sub.index, col] = sub[col]
    return nat.reset_index()[nation_df.columns]


def load_decision_points_for_year(year: int, nation_base: pd.DataFrame, mode: str = "RL") -> pd.DataFrame:
    mode_l = str(mode).lower()
    dir_map = PROV_DIR_MAP_rl if mode_l == "rl" else PROV_DIR_MAP_greedy if mode_l == "greedy" else PROV_DIR_MAP_lcoe
    fname_tmpl = f"expansion-RL_{year}.xlsx" if mode_l == "rl" else f"expansion_greedy_{year}.xlsx" if mode_l == "greedy" else f"expansion_LCOE_{year}.xlsx"

    nation_df = nation_base.copy()
    for prov, dir_path in dir_map.items():
        fp = Path(dir_path) / fname_tmpl
        if fp.exists():
            nation_df = update_decision_points(nation_df, read_table(fp), id_col="point_id", update_cols=UPDATE_COLS)

    return nation_df[nation_df["Provence_num"].isin(SOUTH_IDS)].reset_index(drop=True)