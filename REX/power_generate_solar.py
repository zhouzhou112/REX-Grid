import numpy as np
import pandas as pd
import xarray as xr

# —— 常量 & 经验系数 ——
rho_s, T_cell, sigma_T = 0.2, 44, -0.0041
b_0, b_1, c_1 = 0.00692, -0.400, 0.0703
b_2, c_2 = -0.006768, 0.000907
b_3, c_3 = -0.00270, 0.00148
a_0, a_1, a_2, a_3 = 1.3793, 1.2011, -0.0144, 8.051e-5
gamma_loss = 0.1944

def calculate_solar_power(solar_rad_data: xr.Dataset, decision_points: pd.DataFrame) -> pd.DataFrame:
    """
    向量化、无缓存的光伏出力计算实现（0512 极速版）：
      - 每次调用都重新计算索引与时间参数
      - 全程 NumPy 操作，无 Python 循环
      - 考虑了详细的经纬度差异与物理光照衰减
    """
    # 1) 网格经纬度与点位经纬度
    lats   = solar_rad_data["lat"].values    # (L,)
    lons   = solar_rad_data["lon"].values    # (M,)
    pt_lat = decision_points["y"].values     # (N,)
    pt_lon = decision_points["x"].values     # (N,)

    # 2) 最近邻索引
    lat_idx = np.abs(lats[:, None] - pt_lat[None, :]).argmin(axis=0)  # (N,)
    lon_idx = np.abs(lons[:, None] - pt_lon[None, :]).argmin(axis=0)  # (N,)

    # 3) 时间向量
    times       = pd.DatetimeIndex(solar_rad_data["time"].values)
    day_of_year = times.day_of_year.values   # (T,)
    hour_vec    = times.hour.values          # (T,)
    minute_vec  = times.minute.values        # (T,)

    # 4) 太阳偏角 δ
    d_A   = 2 * np.pi * (day_of_year - 1) / 365.0           # (T,)
    delta = (
        b_0 + b_1*np.cos(d_A) + c_1*np.sin(d_A)
      + b_2*np.cos(2*d_A) + c_2*np.sin(2*d_A)
      + b_3*np.cos(3*d_A) + c_3*np.sin(3*d_A)
    )[:, None]                                              # (T,1)

    # 5) 本地太阳时修正
    offset     = np.round(pt_lon / 15).astype(int) - 8      # (N,)
    local_hour = (hour_vec[:, None] + offset[None, :] + 24) % 24  # (T,N)

    # 6) φ, ω, 装机容量
    theta    = np.deg2rad(pt_lat)[None, :]                  # (1,N)，地面纬度
    phi      = np.deg2rad(pt_lat)[None, :]                  # (1,N)
    omega    = (np.pi/180) * (a_0 + a_1 * pt_lat + a_2 * pt_lat**2 + a_3 * pt_lat**3)[None, :] # (1,N)
    cap_pv   = decision_points["Installed_pv"].values       # (N,)

    # 7) 提取辐射与温度矩阵
    SWGDN = solar_rad_data["SWGDN"].values[:, lat_idx, lon_idx]  # 总辐射
    SWGNT = solar_rad_data["SWGNT"].values[:, lat_idx, lon_idx]  # 净辐射（吸收）
    T2M   = solar_rad_data["T2M"].values[:, lat_idx, lon_idx]    # (T,N)

    # 地表反射辐射量与散射辐射估计
    R_up = SWGDN - SWGNT
    albedo = np.divide(R_up, SWGDN, out=np.zeros_like(R_up), where=SWGDN != 0)
    diffuse_fraction = np.clip(1 - 0.75 * albedo, 0, 1)
    R_diff = SWGDN * diffuse_fraction
    R_direct = SWGDN - R_diff

    # 8) 太阳高度角 β
    hour_angle = np.deg2rad(15 * (local_hour + minute_vec[:, None]/60 - 12))  # (T,N)
    beta = np.arcsin(
        np.sin(theta) * np.sin(delta) + np.cos(theta) * np.cos(delta) * np.cos(hour_angle)
    )
    beta = np.clip(beta, np.deg2rad(2), np.pi / 2) # 防止beta数值爆炸

    cos_phi = (np.sin(beta) * np.sin(theta) - np.sin(delta)) / (np.cos(beta) * np.cos(theta))
    cot_beta = 1 / np.tan(beta)
    sin_omega = np.sin(omega)
    cos_omega = np.cos(omega)

    # 9) 面板入射辐射 I_panel
    term_direct = (cot_beta * cos_phi * sin_omega + cos_omega) * R_direct
    term_diff = (1 + cos_omega) / 2 * R_diff
    term_ref = (1 - cos_omega) / 2 * rho_s * (R_direct + R_diff)
    I_panel = term_direct + term_diff + term_ref

    # 10) 温度系数 & 输出功率 (MW)
    gamma_temp = 1 + sigma_T * (T2M - 273.15 + ((T_cell - 20)/0.8) - 25)
    gamma_shade = np.clip(1 - np.abs(np.sin(omega * cot_beta * cos_phi)) / 12, 0, 1)

    power_output = cap_pv[None, :] * (I_panel/1000) * gamma_shade * gamma_temp * (1 - gamma_loss)

    # 11) 构造结果 DataFrame
    return pd.DataFrame({
        "point_id":           decision_points["point_id"].values,
        "lat":                decision_points["y"].values,
        "lon":                decision_points["x"].values,
        "Province_num":       decision_points["Provence_num"].values,
        "region_num":         decision_points["region_num"].values,
        "solar_power_output": list(power_output.T)
    })

def aggregate_solar_output_by_province_with_all_codes(solar_result: pd.DataFrame, province_codes: pd.DataFrame) -> pd.DataFrame:
    """将光伏发电量结果按省份统计，确保所有省份均有数据（缺失填0）"""
    province_solar_output = {code: np.zeros(168) for code in province_codes['Province_num']}
    grouped = solar_result.groupby('Province_num')

    for province, group in grouped:
        province_solar_output[province] = np.sum(np.vstack(group['solar_power_output'].values), axis=0)

    province_solar_df = pd.DataFrame(province_solar_output)[province_codes['Province_num']]
    province_solar_df.columns = province_codes['Province_num']
    return province_solar_df