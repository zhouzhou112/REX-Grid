import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d

# ==========================================
# 风机功率曲线配置
# ==========================================
# V164-8.0 MW 海上风机功率曲线
wind_speeds_offshore = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
power_output_offshore = [0.0, 0.0, 0.0, 91.8, 526.7, 1123.1, 2043.9, 3134.6, 4486.4, 6393.2, 7363.8, 7834.4, 8026.4,
                         8077.2, 8077.2, 8077.2, 8077.2, 8077.2, 8077.2, 8077.2, 8077.2, 8077.2, 8077.2, 8077.2, 8077.2,
                         8077.2]

power_curve_offshore = interp1d(wind_speeds_offshore, power_output_offshore, kind='linear', fill_value="extrapolate")

# GE 2.5-100 陆地风机功率曲线
wind_speeds_onshore = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0,
                       9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0,
                       17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 24.0, 24.5, 25.0,
                       25.5, 26.0, 26.5, 27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 30.0, 30.5, 31.0, 31.5, 32.0, 32.5, 33.0,
                       33.5, 34.0, 34.5, 35.0]
power_output_onshore = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 63.0, 129.0, 194.0, 295.0, 395.0, 527.0, 658.0, 809.0,
                        959.0, 1152.0, 1345.0, 1604.0, 1862.0, 2060.0, 2248.0, 2340.0, 2426.0, 2475.0, 2495.0, 2500.0,
                        2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0,
                        2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 2500.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

power_curve_onshore = interp1d(wind_speeds_onshore, power_output_onshore, kind='linear', fill_value="extrapolate")

cut_in_offshore, cut_out_offshore = 4.0, 25.0
cut_in_onshore, cut_out_onshore = 3.0, 25.0


# ==========================================
# 核心计算函数
# ==========================================
def calculate_wind_power(wind_speed_data: xr.DataArray, decision_points: pd.DataFrame) -> pd.DataFrame:
    """
    向量化、无缓存的风电出力计算实现（0512 极速版）。
    """
    # 1) 提取底层数组和坐标
    ws_arr = wind_speed_data.values  # (T, n_lat, n_lon)
    lats = wind_speed_data['lat'].values  # (n_lat,)
    lons = wind_speed_data['lon'].values  # (n_lon,)

    # 2) 最近邻索引
    pt_lat = decision_points['y'].values  # (N,)
    pt_lon = decision_points['x'].values  # (N,)
    lat_idx = np.abs(lats[:, None] - pt_lat[None, :]).argmin(axis=0)  # (N,)
    lon_idx = np.abs(lons[:, None] - pt_lon[None, :]).argmin(axis=0)  # (N,)

    # 3) 切片得到 (T, N) 的风速
    ws = ws_arr[:, lat_idx, lon_idx]  # (T, N)

    # 4) 批量计算功率
    cap_on = decision_points['Installed_onshorewind'].values  # (N,)
    cap_off = decision_points['Installed_offshorewind'].values  # (N,)

    single_rated_power_on = 2500.0  # 单台风机额定功率kW
    single_rated_power_off = 8000.0  # 单台风机额定功率kW

    # 计算容量因子 (0~1)
    cf_on = power_curve_onshore(ws) / single_rated_power_on
    cf_off = power_curve_offshore(ws) / single_rated_power_off

    pow_on = cf_on * cap_on[None, :]
    pow_off = cf_off * cap_off[None, :]

    # 切入/切出掩码
    mask_on = (ws >= cut_in_onshore) & (ws <= cut_out_onshore)
    mask_off = (ws >= cut_in_offshore) & (ws <= cut_out_offshore)
    pow_on *= mask_on
    pow_off *= mask_off

    power_output = pow_on + pow_off  # (T, N)

    # 5) 构造结果 DataFrame
    return pd.DataFrame({
        'point_id': decision_points['point_id'].values,
        'lat': decision_points['y'].values,
        'lon': decision_points['x'].values,
        'location_type': decision_points['name'].values,
        'Province_num': decision_points['Provence_num'].values,
        'region_num': decision_points['region_num'].values,
        'wind_power_output': list(power_output.T)
    })


def aggregate_wind_output_by_province_with_all_codes(wind_result: pd.DataFrame,
                                                     province_codes: pd.DataFrame) -> pd.DataFrame:
    """将风力发电量结果按省份统计，确保所有省份均有数据（缺失填0）"""
    province_wind_output = {code: np.zeros(168) for code in province_codes['Province_num']}
    grouped = wind_result.groupby('Province_num')

    for province, group in grouped:
        province_wind_output[province] = np.sum(np.vstack(group['wind_power_output'].values), axis=0)

    province_wind_df = pd.DataFrame(province_wind_output)[province_codes['Province_num']]
    province_wind_df.columns = province_codes['Province_num']
    return province_wind_df