import xarray as xr
import rioxarray
import os
import glob
import pandas as pd
import numpy as np

# --- 1. 路径与参数设置 ---
# NDVI 原始数据所在目录
ndvi_dir = r'E:\data\NDVI\NDVI_China'
# 之前计算的 SPEI 结果作为空间模版（0.1度）
spei_path = r'E:\data\meteorologicaldata\D_value_8day\D_value_8day_2000.nc'
# 输出结果路径
output_path = r'E:\data\NDVI\NDVI_8D\NDVI_8d_01deg_China_Aligned.nc'

# 加载 SPEI 模版以获取空间网格
ds_spei_template = xr.open_dataset(spei_path)
# 只要空间坐标，不要时间轴
spei_grid = ds_spei_template.isel(time=0).drop_vars('time')


def process_ndvi_workflow():
    all_years_data = []

    # 循环处理 2000-2022 年
    for year in range(2000, 2023):
        file_name = f'Daily_Gap-filled_NDVI_{year}.nc4'
        file_path = os.path.join(ndvi_dir, file_name)

        if not os.path.exists(file_path):
            print(f"⚠️ 跳过文件 (未找到): {file_name}")
            continue

        print(f"🚀 正在处理 {year} 年数据...")

        # 1. 读取单年日尺度数据
        ds_daily = xr.open_dataset(file_path)

        # 2. 8天最大值合成 (MVC)
        # 注意：你的文件维度是 'Time' (首字母大写)
        # .compute() 确保计算立即执行，避免惰性加载导致的内存积压
        ndvi_8d_005 = ds_daily['NDVI'].resample(Time='8D').max().compute()

        # 3. 统一坐标名称：将 'Time' 改为小写 'time'，以便与后续及 SPEI 对齐
        ndvi_8d_005 = ndvi_8d_005.rename({'Time': 'time'})

        # 4. 空间重采样：将 0.05° 对齐到 0.1° SPEI 网格
        # 使用 linear (双线性插值) 保证植被指数过渡自然
        ndvi_8d_01 = ndvi_8d_005.interp_like(spei_grid, method='linear')

        all_years_data.append(ndvi_8d_01)

    # --- 2. 合并全序列与时间对齐 ---
    print("-" * 50)
    print("📦 正在合并全序列并执行强制时间裁剪...")
    full_ndvi = xr.concat(all_years_data, dim='time')

    # 5. 核心：强制从 2000-01-25 开始裁剪
    # 这将自动剔除 2000 年前三期 (1.01, 1.09, 1.17)
    ndvi_final = full_ndvi.sel(time=slice('2000-01-25', '2022-12-31'))

    # 6. 增加压缩编码以减小文件体积
    encoding = {'NDVI': {'zlib': True, 'complevel': 4}}

    # 7. 保存为 NetCDF
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    ndvi_final.to_dataset(name='NDVI').to_netcdf(output_path, encoding=encoding)

    print("-" * 50)
    print(f"🎉 任务成功完成！")
    print(f"起始日期: {ndvi_final.time.values[0]}")
    print(f"截止日期: {ndvi_final.time.values[-1]}")
    print(f"总计期数: {len(ndvi_final.time)}")
    print(f"保存路径: {output_path}")


if __name__ == "__main__":
    process_ndvi_workflow()