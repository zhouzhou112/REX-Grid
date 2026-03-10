import pandas as pd
import numpy as np


def calculate_nuclear_output(month: int, installed_capacity: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有省份在指定月份的核电出力。

    参数：
        month: 要计算的月份（1-12）。
        installed_capacity: 包含每个省份装机容量的数据。

    返回：
        pd.DataFrame: 所有省份在7*24小时(168小时)内的核电出力数据。
    """
    nuclear_cf = 0.95  # 核电固定容量系数

    # 转换为 MW (装机单位原为 GW)
    hourly_output = installed_capacity["Nuclear(GW)"] * nuclear_cf * 1000

    # 扩展为 168 小时序列
    hourly_output_df = pd.DataFrame(
        np.tile(hourly_output.values, (7 * 24, 1)),
        columns=installed_capacity["Province_num"]
    )

    return hourly_output_df


def calculate_hydro_output(month: int, installed_capacity: pd.DataFrame, hydro_cf_file: str) -> pd.DataFrame:
    """
    根据区域容量系数计算所有省份在指定月份的水电出力。

    参数：
        month: 要计算的月份（1-12）。
        installed_capacity: 包含每个省份装机容量的数据。
        hydro_cf_file: 包含区域水电容量系数的CSV文件路径。

    返回：
        pd.DataFrame: 所有省份在7*24小时(168小时)内的水电出力数据。
    """
    # 加载水电容量系数数据
    hydro_cf = pd.read_csv(hydro_cf_file)

    # 获取指定月份的容量系数
    month_cf = hydro_cf[["region", str(month)]]
    month_cf.columns = ["region", "Capacity_Factor"]

    # 合并装机容量与容量系数
    merged_cap = installed_capacity.merge(
        month_cf, left_on="region", right_on="region", how="left"
    )

    # 计算每小时水电出力并转换为 MW
    hourly_output = merged_cap["Hydropower(GW)"] * merged_cap["Capacity_Factor"] * 1000

    # 扩展为 168 小时序列
    hourly_output_df = pd.DataFrame(
        np.tile(hourly_output.values, (7 * 24, 1)),
        columns=merged_cap["Province_num"]
    )

    return hourly_output_df