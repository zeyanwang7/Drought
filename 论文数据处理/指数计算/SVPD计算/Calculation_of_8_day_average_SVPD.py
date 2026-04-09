import os
import xarray as xr
import numpy as np
from scipy.stats import norm
import pandas as pd
import rioxarray

# --- 1. 路径设置 ---
input_dir = r'E:\data\meteorologicaldata\VPD_8Day_Mean_Result'
mask_path = r'E:\data\全国1Km作物种植数据\玉米\Maize_Distribution_010deg_Average.tif'
output_folder = r'E:\data\meteorologicaldata\IndexCalculationResults\SVPD'
# 明确标注 2001-2022
output_path = os.path.join(output_folder, 'SVPD_8day_Maize_2001_2022.nc')


def calculate_svpd_2001_start():
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # --- 2. 加载 VPD 数据并截断至 2001 年 ---
    print("正在加载并筛选 2001-2022 年 VPD 8天均值数据...")
    # 使用 open_mfdataset 合并所有文件
    ds_all = xr.open_mfdataset(os.path.join(input_dir, "*.nc"), combine='by_coords', chunks='auto')

    # 核心步骤：强制筛选 2001 及以后的数据，排除 2000 年的残缺干扰
    ds = ds_all.sel(time=slice('2001-01-01', '2022-12-31'))

    var_name = 'VPDmax_8d_mean' if 'VPDmax_8d_mean' in ds.data_vars else list(ds.data_vars)[0]

    # --- 3. 加载并对齐掩膜 ---
    print("正在对齐小麦种植区掩膜...")
    mask_ds = xr.open_dataset(mask_path, engine="rasterio")
    mask_da = mask_ds.band_data.isel(band=0)
    if 'x' in mask_da.coords:
        mask_da = mask_da.rename({'x': 'lon', 'y': 'lat'})

    mask_resampled = mask_da.interp(lat=ds.lat, lon=ds.lon, method='nearest')
    crop_mask = mask_resampled > 0

    # --- 4. 标准化计算核心函数 (Gringorten) ---
    def compute_svpd_core(x, is_crop):
        if not is_crop or np.all(np.isnan(x)):
            return np.full(x.shape, np.nan)

        valid_indices = np.where(~np.isnan(x))[0]
        valid_data = x[valid_indices]
        n = len(valid_data)

        # 22 年样本量足够稳定
        if n < 10:
            return np.full(x.shape, np.nan)

        # 经验概率计算
        ranks = pd.Series(valid_data).rank(method='average').values
        p = (ranks - 0.44) / (n + 0.12)
        p = np.clip(p, 0.0001, 0.9999)

        # 转换为标准正态分布，乘以 -1 使负值代表干旱
        result = np.full(x.shape, np.nan)
        result[valid_indices] = -1 * norm.ppf(p)
        return result

    # --- 5. 分组标准化 (按 8 天周期) ---
    print(f"开始执行标准化计算 (样本量 n = {len(np.unique(ds.time.dt.year))} 年)...")
    crop_mask_da = xr.DataArray(crop_mask, coords=[ds.lat, ds.lon], dims=['lat', 'lon'])

    svpd_final = xr.apply_ufunc(
        compute_svpd_core,
        ds[var_name].groupby('time.dayofyear'),
        crop_mask_da,
        input_core_dims=[['time'], []],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float32]
    )

    # --- 6. 结果整理与输出 ---
    svpd_final = svpd_final.transpose('time', 'lat', 'lon')

    if isinstance(svpd_final, xr.DataArray):
        svpd_ds = svpd_final.to_dataset(name='SVPD')
    else:
        svpd_ds = svpd_final.rename({list(svpd_final.data_vars)[0]: 'SVPD'})

    svpd_ds['SVPD'].attrs = {
        'units': 'dimensionless',
        'long_name': 'Standardized Vapor Pressure Deficit',
        'base_period': '2001-2022',
        'note': 'Negative values indicate drought'
    }

    print(f"写入文件至: {output_path}")
    encoding = {'SVPD': {'zlib': True, 'complevel': 4}}
    svpd_ds.to_netcdf(output_path, encoding=encoding)
    print("\n✅ 2001-2022 年 SVPD 序列生成完毕！")


if __name__ == "__main__":
    calculate_svpd_2001_start()