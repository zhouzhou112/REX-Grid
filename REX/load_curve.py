import os
import pandas as pd
import numpy as np


class LoadCurveModule:
    def __init__(self, workday_file: str, holiday_file: str, growth_rate: float = 0.04):
        """
        初始化负载数据加载器
        :param workday_file: 工作日负载数据的文件路径
        :param holiday_file: 节假日负载数据的文件路径
        :param growth_rate: 年度负载增长率，默认为4%
        """
        self.workday_load = pd.read_excel(workday_file)
        self.holiday_load = pd.read_excel(holiday_file)
        self.growth_rate = growth_rate
        self.process_data()

    def process_data(self):
        """预处理负载数据，对Hour列取整，并按小时求均值。"""
        self.workday_load['Hour'] = self.workday_load['Hour'].apply(np.floor).astype(int)
        self.holiday_load['Hour'] = self.holiday_load['Hour'].apply(np.floor).astype(int)

        self.workday_load = self.workday_load.sort_values(by='Hour').reset_index(drop=True)
        self.holiday_load = self.holiday_load.sort_values(by='Hour').reset_index(drop=True)

        self.workday_load = self.workday_load.groupby('Hour', as_index=False)['Load'].mean()
        self.holiday_load = self.holiday_load.groupby('Hour', as_index=False)['Load'].mean()

    def generate_monthly_load_curve(self, year: int, month: int) -> pd.DataFrame:
        """生成指定年份和月份的负载曲线，考虑年度增长和季节性波动"""
        workday_load_adjusted = self.workday_load.copy()
        holiday_load_adjusted = self.holiday_load.copy()

        # 负载随年份分段增长
        if year <= 2030:
            growth_factor = (1 + self.growth_rate) ** (year - 2020)
        else:
            # 2030年后增长率放缓
            growth_factor = ((1 + self.growth_rate - 0.02) ** (year - 2030)) * ((1 + self.growth_rate) ** 10)

        workday_load_adjusted['Load'] *= growth_factor
        holiday_load_adjusted['Load'] *= growth_factor

        # 模拟月度季节性波动
        seasonal_factor = [1.0, 0.95, 0.9, 1.05, 1.1, 1.15, 1.2, 1.15, 1.1, 1.05, 1.0, 0.95][month - 1]
        workday_load_adjusted['Load'] *= seasonal_factor
        holiday_load_adjusted['Load'] *= seasonal_factor

        # 拼接连续7天：5个工作日 + 2个节假日
        typical_month_load = pd.DataFrame()
        for _ in range(5):
            typical_month_load = pd.concat([typical_month_load, workday_load_adjusted], ignore_index=True)
        for _ in range(2):
            typical_month_load = pd.concat([typical_month_load, holiday_load_adjusted], ignore_index=True)

        return typical_month_load

    def get_load_curve_for_year_and_month(self, year: int, month: int) -> pd.DataFrame:
        return self.generate_monthly_load_curve(year, month)


class ProvinceLoadManager:
    def __init__(self, base_folder: str, province_codes: pd.DataFrame):
        """初始化省份负载管理器"""
        self.provinces = {}
        self.province_codes = province_codes
        self.load_all_provinces(base_folder)

    def load_all_provinces(self, base_folder: str):
        """加载所有省份的负载数据"""
        for province in os.listdir(base_folder):
            province_folder = os.path.join(base_folder, province)
            if os.path.isdir(province_folder) and province in self.province_codes['Province'].values:
                weekday_file = os.path.join(province_folder, '工作日.xlsx')
                holiday_file = os.path.join(province_folder, '节假日.xlsx')
                self.provinces[province] = LoadCurveModule(weekday_file, holiday_file)

    def update_load_curves_for_all_provinces(self, year: int, month: int) -> dict:
        """更新所有省份的负载曲线"""
        return {province: module.get_load_curve_for_year_and_month(year, month)
                for province, module in self.provinces.items()}

    def get_combined_load_curves(self, load_curves: dict) -> pd.DataFrame:
        """
        返回一个 DataFrame，每列代表一个省份的负载曲线
        确保所有省份数据长度均为 168，以对齐强化学习环境步长
        """
        combined_data = {}

        for province, load_curve in load_curves.items():
            province_num = self.province_codes[self.province_codes['Province'] == province]['Province_num'].values[0]
            load_values = load_curve['Load'].values

            # 强制对齐 168 小时
            if len(load_values) < 168:
                load_values = np.pad(load_values, (0, 168 - len(load_values)), 'edge')  # 用最后一个值填充
            elif len(load_values) > 168:
                load_values = load_values[:168]  # 截断超出部分

            combined_data[province_num] = load_values

        # 转换为 DataFrame，索引为 1-168
        combined_df = pd.DataFrame(combined_data, index=np.arange(1, 169))
        return combined_df[sorted(combined_df.columns)]