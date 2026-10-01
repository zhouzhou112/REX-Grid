import pandas as pd
import time
from pathlib import Path
import xarray as xr
from config.paths import input_path
# --- 假设下列函数均已定义或从其他模块导入 ---
from REX.load_curve import ProvinceLoadManager
from REX.wind_solar_generate import generate_seven_days_wind_solar
from REX.power_generate_wind import calculate_wind_power, aggregate_wind_output_by_province_with_all_codes
from REX.power_generate_solar import calculate_solar_power, aggregate_solar_output_by_province_with_all_codes
from REX.power_generate_othersource import calculate_hydro_output, calculate_nuclear_output


# ==========================================
# 1. 集中配置区域 (建议后续开源时独立为 config.py)
# ==========================================
class Config:
    # Input paths are configured relative to the repository root.
    LOAD_FOLDER = input_path("load_folder")
    WIND_NC_PATH = input_path("wind_nc")
    SOLAR_NC_PATH = input_path("solar_nc")
    HYDRO_CF_CSV = input_path("hydro_cf")
    INSTALLED_CAP_CSV = input_path("installed_cap")
    HYDRO_COEFF_XLSX = input_path("hydro_coeff")
    NUCLEAR_COEFF_XLSX = input_path("nuclear_coeff")

    # 省份定义
    PROVINCE_CODES = pd.DataFrame({
        'Province': ["北京市", "天津市", "河北省", "山西省", "内蒙古自治区", "辽宁省", "吉林省", "黑龙江省", "上海市",
                     "江苏省", "浙江省", "安徽省", "福建省", "江西省", "山东省", "河南省", "湖北省", "湖南省", "广东省",
                     "广西壮族自治区", "海南省", "重庆市", "四川省", "贵州省", "云南省", "西藏自治区", "陕西省",
                     "甘肃省",
                     "青海省", "宁夏回族自治区", "新疆维吾尔自治区"],
        'Province_num': [11, 12, 13, 14, 15, 21, 22, 23, 31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 50, 51, 52,
                         53, 54, 61, 62, 63, 64, 65]
    })


# ==========================================
# 2. 辅助数据加载函数
# ==========================================
def load_expansion_coeff(kind: str) -> pd.DataFrame:
    """读取水电或核电的扩张系数表"""
    path = Config.HYDRO_COEFF_XLSX if kind == "hydro" else Config.NUCLEAR_COEFF_XLSX
    return pd.read_excel(path).set_index("year").rename_axis(index=None)


def get_year_coeff(df: pd.DataFrame, year: int, target_codes: list) -> pd.Series:
    """获取特定年份的扩张系数，若缺失则回退使用最近的历史年份"""
    if year in df.index:
        row = df.loc[year]
    else:
        past = df.index[df.index <= year]
        row = df.loc[past.max()] if len(past) > 0 else pd.Series(1.0, df.columns)
    return row.reindex(target_codes, fill_value=1.0)


def read_decision_points(path: Path, id_col: str = 'point_id') -> pd.DataFrame:
    """读取决策点(安装点)数据"""
    df = pd.read_excel(path) if path.suffix.lower() in ('.xlsx', '.xls') else pd.read_csv(path)
    df[id_col] = df[id_col].astype(str)
    return df


# ==========================================
# 3. 核心计算流程 (面向对象封装)
# ==========================================
class PowerGapCalculator:
    def __init__(self):
        # 实例化负荷管理器 (全局只建一次，避免重复读取)
        self.province_manager = ProvinceLoadManager(Config.LOAD_FOLDER, Config.PROVINCE_CODES)

        # 预加载基础装机容量与系数表
        self.installed_cap_df = pd.read_csv(Config.INSTALLED_CAP_CSV)
        self.hydro_coeff_df = load_expansion_coeff("hydro")
        self.nuclear_coeff_df = load_expansion_coeff("nuclear")

        # === 在初始化时预加载庞大的气象数据集（全局只读一次） ===
        print("正在预加载气象 NetCDF 数据，请稍候...")
        self.wind_data = xr.open_dataset(Config.WIND_NC_PATH)['Wind_Speed_100M']
        self.solar_data = xr.open_dataset(Config.SOLAR_NC_PATH)
        print("气象数据预加载完成！")

    def compute_gap(self, year: int, month: int, decision_points: pd.DataFrame):
        """主计算函数：串联所有模块，计算目标省份的电力供需缺口"""
        target_codes = sorted(decision_points["Provence_num"].unique())
        subset_codes_df = Config.PROVINCE_CODES[Config.PROVINCE_CODES["Province_num"].isin(target_codes)]

        # 1. 计算负载 (Load)
        load_curves_dict = self.province_manager.update_load_curves_for_all_provinces(year, month)
        load_df = self.province_manager.get_combined_load_curves(load_curves_dict)
        load_df = load_df[target_codes].reset_index(drop=True)

        # 2. 获取气象数据 (Wind & Solar 168小时切片)
        wind_speed_data, solar_rad_data = generate_seven_days_wind_solar(
            year, month, self.wind_data, self.solar_data
        )

        # 3. 计算风电出力
        wind_result = calculate_wind_power(wind_speed_data, decision_points)
        wind_df = aggregate_wind_output_by_province_with_all_codes(wind_result, subset_codes_df)[target_codes]

        # 4. 计算光伏出力
        solar_result = calculate_solar_power(solar_rad_data, decision_points)
        solar_df = aggregate_solar_output_by_province_with_all_codes(solar_result, subset_codes_df)[target_codes]

        # 5. 计算水电 & 核电出力 (包含扩张系数修正)
        hydro_base = calculate_hydro_output(month, self.installed_cap_df, Config.HYDRO_CF_CSV)[target_codes]
        nuclear_base = calculate_nuclear_output(month, self.installed_cap_df)[target_codes]

        coef_hydro = get_year_coeff(self.hydro_coeff_df, year, target_codes)
        coef_nuclear = get_year_coeff(self.nuclear_coeff_df, year, target_codes)

        hydro_df = hydro_base.mul(coef_hydro, axis=1)
        nuclear_df = nuclear_base.mul(coef_nuclear, axis=1)

        # 6. 拼合数据与计算缺口
        clean_power_df = pd.concat(
            {'Wind': wind_df, 'Solar': solar_df, 'Hydro': hydro_df, 'Nuclear': nuclear_df},
            axis=1
        )

        # Gap = Load - (Wind + Solar + Hydro + Nuclear)
        total_clean = clean_power_df.T.groupby(level=1).sum().T
        gap_df = load_df - total_clean

        return gap_df, load_df, clean_power_df

    def compute_county_wind_solar(self, year: int, month: int, decision_points: pd.DataFrame):
        """
        专门用于省内电网规划：返回未按省份聚合的县级(点级)风光出力
        """
        wind_speed_data, solar_rad_data = generate_seven_days_wind_solar(
            year, month, self.wind_data, self.solar_data
        )
        wind_result = calculate_wind_power(wind_speed_data, decision_points)
        solar_result = calculate_solar_power(solar_rad_data, decision_points)

        return wind_result, solar_result


# ==========================================
# ★ 全局单例与接口暴露 ★
# ==========================================
# Load external weather on first use; importing the package needs no data files.
class _LazyCalculator:
    def __init__(self):
        self._instance = None

    def __getattr__(self, name):
        if self._instance is None:
            self._instance = PowerGapCalculator()
        return getattr(self._instance, name)


_global_calculator = _LazyCalculator()


# 2. 暴露一个与旧版同名的独立函数，供外部脚本直接 import 和调用
def compute_gap(year: int, month: int, decision_points: pd.DataFrame):
    """
    这是一个包装函数，外部调用它时，实际上是在调用全局唯一的 _global_calculator
    """
    return _global_calculator.compute_gap(year, month, decision_points)


# ==========================================
# 4. 执行入口
# ==========================================
if __name__ == "__main__":
    pass
    # 执行测试
    # NATION_PATH = Config.WORK_BASE_DIR / "HKV2.xlsx"
    # OUTPUT_PATH = Config.WORK_BASE_DIR / "HK9.xlsx"
    # ... (原有测试代码)