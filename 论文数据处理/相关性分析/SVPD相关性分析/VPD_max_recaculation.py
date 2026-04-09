import os
import xarray as xr
import pandas as pd
import numpy as np
#该文件为将每日VPDmax重采样为30天滑动平均 -> 8天采样
# --- 1. 路径与参数设置 ---
input_base_dir = r'E:\data\meteorologicaldata\VPD'
# 结果保存路径
output_dir = r'E:\data\meteorologicaldata\VPD_8Day_result'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

YEAR_START = 2000
YEAR_END = 2022


def run_vpd_8day_yearly_save():
    print(f"🚀 开始处理 VPDmax 并保存至: {output_dir}")
    print(f"模式：30天滑动平均 -> 8天采样 (按年独立保存)")
    print("-" * 60)

    buffer_prev_days = None

    for year in range(YEAR_START, YEAR_END + 1):
        # 匹配输入路径
        file_path = os.path.join(input_base_dir, str(year), f'ChinaMet_010deg_VPDmax_{year}_Annual.nc')

        if not os.path.exists(file_path):
            print(f"⚠️ {year} 年原始文件未找到，跳过。")
            continue

        # 1. 加载年度文件
        ds_year = xr.open_dataset(file_path).load()

        # 2. 坐标维度标准化
        if 'day' in ds_year.dims:
            ds_year = ds_year.rename({'day': 'time'})

        # 3. 修复时间索引 (确保 resample 可用)
        if not np.issubdtype(ds_year.time.dtype, np.datetime64):
            time_dates = pd.date_range(start=f"{year}-01-01", periods=len(ds_year.time), freq='D')
            ds_year = ds_year.assign_coords(time=time_dates)

        # 获取变量名
        var_name = 'vpd_max' if 'vpd_max' in ds_year.data_vars else list(ds_year.data_vars)[0]
        da_year = ds_year[var_name]

        # 4. 跨年拼接逻辑
        if buffer_prev_days is not None:
            da_to_calc = xr.concat([buffer_prev_days, da_year], dim='time')
        else:
            da_to_calc = da_year

        # 5. 计算 30 天滑动平均
        da_rolling = da_to_calc.rolling(time=30, center=False, min_periods=30).mean()

        # 6. 精准截取当年数据 (使用 .values 避免之前的 TypeError)
        actual_start = da_year.time.min().values
        actual_end = da_year.time.max().values
        da_year_only = da_rolling.sel(time=slice(actual_start, actual_end))

        # 7. 8天采样
        da_8d = da_year_only.resample(time='8D').last()

        # 8. 保存当年结果
        ds_output = da_8d.to_dataset(name='VPDmax_30d_avg')
        save_path = os.path.join(output_dir, f'VPDmax_30d_mean_8day_{year}.nc')

        encoding = {'VPDmax_30d_avg': {'zlib': True, 'complevel': 4}}
        ds_output.to_netcdf(save_path, encoding=encoding)

        # 9. 更新缓冲区 (提取今年最后29天原始数据)
        buffer_prev_days = da_year.isel(time=slice(-29, None))

        print(f"✅ 已完成 {year} 年处理并保存至: {os.path.basename(save_path)}")

    print("-" * 60)
    print(f"🎉 任务全部完成！共有 {len(range(YEAR_START, YEAR_END + 1))} 个年度文件存放在 {output_dir}")


if __name__ == "__main__":
    run_vpd_8day_yearly_save()