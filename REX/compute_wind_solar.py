import pandas as pd

# 直接从重构好的中央引擎中引入全局单例
from REX.compute_power_gap import _global_calculator


def compute_wind_solar(year: int, month: int, decision_points: pd.DataFrame):
    """
    计算并返回指定年月、各决策点连续 7 天的风电与光伏出力。

    注意：底层已对接 REX.compute_power_gap 中的 _global_calculator，
    完美适配了新的参数结构，且不会产生任何额外的 NetCDF 文件读取开销。
    """
    # 直接调用单例内部的计算方法，它会自动处理 wind_data 和 solar_data 的传参
    wind_df, solar_df = _global_calculator.compute_county_wind_solar(year, month, decision_points)

    return wind_df, solar_df


# ==========================================
# 本地快速测试入口 (需要测试时取消注释即可)
# ==========================================
if __name__ == "__main__":
    pass
    # 测试文件路径
    # decision_points = pd.read_excel(test_excel)

    # print(f"正在计算测试数据的风光出力...")
    # w_df, s_df = compute_wind_solar(2025, 7, decision_points)

    # print("计算完成！风电数据预览：")
    # print(w_df[['point_id', 'Province_num', 'wind_power_output']].head())
