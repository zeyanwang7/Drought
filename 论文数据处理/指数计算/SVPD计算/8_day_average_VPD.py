import os
import xarray as xr
import pandas as pd
import numpy as np

# --- 1. 路径与参数设置 ---
# 输入路径：包含每日 VPDmax 的 NC 文件（按年存放）
input_base_dir = r'E:\data\meteorologicaldata\VPD'
# 输出路径：存储 8天平均 后的结果
output_dir = r'E:\data\meteorologicaldata\VPD_8Day_Mean_Result'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

YEAR_START = 2000
YEAR_END = 2022


def run_vpd_8day_mean_sync():
    print(f"🚀 开始处理 VPDmax 8天同步平均...")
    print(f"模式：每日数据 -> 8天区间直接平均 (无滑动窗口，物理时间对齐)")
    print("-" * 60)

    # 存储所有年份处理后的 DataArray
    all_years_8d = []

    for year in range(YEAR_START, YEAR_END + 1):
        file_path = os.path.join(input_base_dir, str(year), f'ChinaMet_010deg_VPDmax_{year}_Annual.nc')

        if not os.path.exists(file_path):
            print(f"⚠️ {year} 年原始文件未找到，跳过。")
            continue

        # 1. 加载年度文件 (使用 dask 优化大数据处理)
        ds_year = xr.open_dataset(file_path)

        # 2. 坐标维度统一
        if 'day' in ds_year.dims:
            ds_year = ds_year.rename({'day': 'time'})

        # 3. 修复/建立标准时间索引 (确保重采样起始日为 1月1日)
        time_dates = pd.date_range(start=f"{year}-01-01", periods=len(ds_year.time), freq='D')
        ds_year = ds_year.assign_coords(time=time_dates)

        # 获取变量名
        var_name = 'vpd_max' if 'vpd_max' in ds_year.data_vars else list(ds_year.data_vars)[0]
        da_daily = ds_year[var_name]

        # 4. 执行 8天平均聚合 (Resample Mean)
        # label='left' 确保 1月1日的标签代表 1月1日-1月8日的平均
        # closed='left' 确保区间是 [1-1, 1-9)
        da_8d = da_daily.resample(time='8D', label='left', closed='left').mean()

        # 5. 保存年度结果
        ds_output = da_8d.to_dataset(name='VPDmax_8d_mean')
        save_path = os.path.join(output_dir, f'VPDmax_8d_mean_{year}.nc')

        encoding = {'VPDmax_8d_mean': {'zlib': True, 'complevel': 4}}
        ds_output.to_netcdf(save_path, encoding=encoding)

        print(f"✅ {year} 年处理完成：得到 {len(da_8d.time)} 个 8天周期")

    print("-" * 60)
    print(f"🎉 聚合完成！请检查目录: {output_dir}")


if __name__ == "__main__":
    run_vpd_8day_mean_sync()