import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import norm
from osgeo import gdal
import os

# --- 1. 路径设置 ---
input_resampled_path = r'E:\data\meteorologicaldata\SPEI\D_30d_accumulated_8day_step.nc'
mask_path = r'E:\data\全国1Km作物种植数据\水稻\Rice_Distribution_010deg_Average.tif'
# 建议在文件名中标注 2001_2022 以便区分
output_spei_path = r'E:\data\meteorologicaldata\SPEI_30d_8dStep_Rice_2001_2022.nc'


def calculate_spei_gringorten():
    print("🚀 开始 SPEI 标准化计算 (基准期: 2001-2022, Gringorten 方法)...")

    # --- 2. 加载数据并截取时间轴 ---
    ds = xr.open_dataset(input_resampled_path)

    # 【关键修改】：将计算基准期限定为 2001-2022
    # 这一步会自动过滤掉 2000 年那些受 1999 年缺失影响的、不稳定的初期数据
    ds_filtered = ds.sel(time=slice('2001-01-01', '2022-12-31'))
    d_values = ds_filtered['D_30d_acc']

    # --- 3. 加载并对齐小麦掩膜 ---
    print("读取作物种植区掩膜...")
    mask_ds = gdal.Open(mask_path)
    mask_array = mask_ds.ReadAsArray()
    mask_array_fixed = np.flipud(mask_array)

    # 创建与截取后的数据坐标一致的掩膜
    wheat_mask = xr.DataArray(
        mask_array_fixed > 0,
        coords=[d_values.lat, d_values.lon],
        dims=['lat', 'lon']
    )

    # --- 4. 定义核心计算函数 ---
    def gringorten_norm(x):
        """对一个时间序列应用 Gringorten 概率转换"""
        valid_idx = ~np.isnan(x)
        valid_x = x[valid_idx]
        n = len(valid_x)

        # 此时 n 最大为 22 (2001-2022)，样本量足够稳定
        if n < 10:
            return np.full(x.shape, np.nan)

        # 计算秩 (rank)
        ranks = pd.Series(valid_x).rank(method='average').values

        # Gringorten 公式
        p = (ranks - 0.44) / (n + 0.12)
        p = np.clip(p, 0.0001, 0.9999)

        # 转换为正态分布 Z-score
        spei_subset = norm.ppf(p)

        result = np.full(x.shape, np.nan)
        result[valid_idx] = spei_subset
        return result

    # --- 5. 执行分组计算 ---
    print(f"正在执行标准化计算 (样本年份: {len(np.unique(d_values.time.dt.year))} 年)...")

    spei_da = xr.apply_ufunc(
        gringorten_norm,
        d_values.groupby('time.dayofyear'),
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='forbidden',
        output_dtypes=[np.float32]
    )

    # --- 6. 掩膜过滤与导出 ---
    print("应用种植区过滤并导出结果...")
    spei_wheat = spei_da.where(wheat_mask)
    spei_wheat = spei_wheat.transpose('time', 'lat', 'lon')

    if 'dayofyear' in spei_wheat.coords:
        spei_wheat = spei_wheat.drop_vars('dayofyear')

    # 保存文件并添加说明
    spei_ds = spei_wheat.to_dataset(name='SPEI_30d')
    spei_ds.attrs['description'] = "SPEI-1 month calculated with 2001-2022 base period"

    encoding = {'SPEI_30d': {'zlib': True, 'complevel': 4}}

    if os.path.exists(output_spei_path):
        os.remove(output_spei_path)

    spei_ds.to_netcdf(output_spei_path, encoding=encoding)

    print("-" * 50)
    print(f"🎉 SPEI 计算完成！")
    print(f"最终结果: {output_spei_path}")
    print(f"有效时间起止: {spei_wheat.time.values[0]} 至 {spei_wheat.time.values[-1]}")


if __name__ == "__main__":
    calculate_spei_gringorten()