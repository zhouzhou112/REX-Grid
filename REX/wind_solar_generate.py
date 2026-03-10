import random
import numpy as np
import xarray as xr


def generate_seven_days_wind_solar(year: int, month: int, wind_data: xr.DataArray, solar_data: xr.Dataset):
    """
    向量化、一致地获取连续7天的风速和太阳辐射数据。
    - 按北京时间0点对齐，每天从0点开始随机选一段7天（168小时）
    - 保证风速和辐射使用同一时间切片

    参数:
        year: 目标年份
        month: 目标月份
        wind_data: 预先加载的风速 DataArray dims=('time','lat','lon')
        solar_data: 预先加载的太阳辐射 Dataset dims=('time','lat','lon')

    返回:
        wind7  : xarray.DataArray (time=168, lat, lon)
        solar7 : xarray.Dataset   (time=168, lat, lon)
    """
    # 1) 统一把风速的 UTC 时间转为北京时间（Dataset 和 DataArray 都有相同 time 轴）
    times_utc = wind_data['time'].values  # datetime64[ns]
    times_bj = times_utc + np.timedelta64(8, 'h')  # 北京时间

    # 2) 筛选出该月所有的时间下标
    month_nums = (
                         (times_bj.astype('datetime64[M]') - np.datetime64('1970-01'))
                         // np.timedelta64(1, 'M')
                 ) % 12 + 1  # 得到 1–12
    month_idxs = np.nonzero(month_nums == month)[0]

    if month_idxs.size < 168:
        raise ValueError(f"第 {month} 月数据少于168小时，无法抽取连续7天")

    # 3) 找到每天北京时 0 点对应的首个索引
    days = times_bj[month_idxs].astype('datetime64[D]')
    unique_days, first = np.unique(days, return_index=True)

    # 4) 过滤保证 168 小时不越界的起点
    valid_bases = [
        month_idxs[first[i]]
        for i in range(len(unique_days))
        if month_idxs[first[i]] + 168 <= len(times_bj)
    ]
    if not valid_bases:
        raise ValueError("无法找到不越界的连续7天区段")

    # 5) 随机选一个起点，生成连续168小时的索引
    base = random.choice(valid_bases)
    span = np.arange(base, base + 168)

    # 6) 同时对风速和辐射做一次 isel 切片
    wind7 = wind_data.isel(time=span).transpose('time', 'lat', 'lon')
    solar7 = solar_data.isel(time=span).transpose('time', 'lat', 'lon')

    # 7) 赋回北京时间坐标
    wind7 = wind7.assign_coords(time=times_bj[span])
    solar7 = solar7.assign_coords(time=times_bj[span])

    return wind7, solar7