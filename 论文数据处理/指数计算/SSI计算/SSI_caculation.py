import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import rioxarray
from scipy.stats import norm

# --- 1. 路径设置 ---
sm_dir = r'E:\data\Soil_Humidity\SM2000-2022'
mask_path = r'E:\data\全国1Km作物种植数据\玉米\Maize_Distribution_010deg_Average.tif'
ref_nc_path = r'E:\data\meteorologicaldata\SPEI\D_30d_accumulated_8day_step.nc'
output_dir = r'E:\data\meteorologicaldata\IndexCalculationResults\SSI'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def gringorten_ssi(data):
    valid_mask = ~np.isnan(data)
    valid_data = data[valid_mask]
    n = len(valid_data)
    if n < 15: return np.full(data.shape, np.nan)

    m = pd.Series(valid_data).rank(method='average').values
    p = (m - 0.44) / (n + 0.12)
    ssi_values = norm.ppf(p)

    result = np.full(data.shape, np.nan)
    result[valid_mask] = ssi_values
    return result


def calculate_rice_ssi_main():
    print("🚀 启动水稻 SSI 计算 (变量名: SMCI, 公式: Gringorten)...")

    # 1. 加载 8 天步长的参考时间轴
    ds_ref = xr.open_dataset(ref_nc_path)
    target_times = ds_ref.time.values

    # 2. 读取 2000-2022 所有土壤湿度文件
    print("正在合并 2000-2022 每日数据 (SMCI)...")
    sm_list = []
    for year in range(2000, 2023):
        f_path = os.path.join(sm_dir, f"SMCI_9km_{year}_100cm.nc")
        if os.path.exists(f_path):
            ds_tmp = xr.open_dataset(f_path)['SMCI']
            sm_list.append(ds_tmp)

    sm_daily = xr.concat(sm_list, dim='time')

    # 3. 计算 30 天滑动平均
    print("计算 30 天滑动平均...")
    sm_30d = sm_daily.rolling(time=30, center=False).mean()

    # 4. 重采样
    print("重采样至 8 天步长...")
    sm_8d = sm_30d.sel(time=target_times, method='nearest')

    # 5. 空间掩膜处理 (修复后的部分)
    print("应用水稻种植区掩膜...")
    rice_mask = rioxarray.open_rasterio(mask_path).isel(band=0)
    if 'x' in rice_mask.dims:
        rice_mask = rice_mask.rename({'x': 'lon', 'y': 'lat'})

    rice_mask_aligned = rice_mask.reindex(lat=sm_8d.lat, lon=sm_8d.lon, method="nearest")
    sm_8d_masked = sm_8d.where(rice_mask_aligned > 0)

    # 6. 核心标准化计算
    print("执行 Gringorten 标准化拟合 (按 dayofyear 分组)...")
    ssi = xr.apply_ufunc(
        gringorten_ssi,
        sm_8d_masked.groupby('time.dayofyear'),
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized'
    )

    ssi = ssi.transpose('time', 'lat', 'lon')
    ssi.name = "SSI"

    # 7. 保存
    output_path = os.path.join(output_dir, 'SSI_30d_8day_Maize_Gringorten.nc')
    encoding = {'SSI': {'zlib': True, 'complevel': 5}}
    ssi.to_netcdf(output_path, encoding=encoding)
    print(f"🎉 任务完成！保存至: {output_path}")


if __name__ == "__main__":
    calculate_rice_ssi_main()