import xarray as xr
import rioxarray
import pandas as pd
import numpy as np
from pathlib import Path
from rasterio.enums import Resampling

# ==================== 1. 路径设置 ====================
precip_8day_dir = Path(r"E:\data\meteorologicaldata\Precip_8Day_Standard")
corn_dir = Path(r"E:\data\全国1Km作物种植数据\玉米")
output_dir = Path(r"E:\data\TCI_Corn_Results_01deg")
output_dir.mkdir(parents=True, exist_ok=True)

# ==================== 2. 计算历史极值基准 ====================
print("第一步：正在读取 2000-2019 数据并计算历史极值...")

all_precip_files = sorted(list(precip_8day_dir.glob("*.nc")))
ds_list = [xr.open_dataset(f, engine="netcdf4") for f in all_precip_files]
ds_all = xr.concat(ds_list, dim="time")

# 标记 1-46 周期
day_indices = ((ds_all.time.dt.dayofyear - 1) // 8) + 1
ds_all.coords['period'] = ("time", day_indices.values)

print("正在生成历史极值背景矩阵 (P_max & P_min)...")
p_max = ds_all.groupby("period").max(dim="time").compute()
p_min = ds_all.groupby("period").min(dim="time").compute()

# ==================== 3. 逐年计算 TCI 并应用掩膜 ====================
print("\n第二步：开始逐年计算 0.1° TCI (玉米种植比例 > 0)...")

for year in range(2000, 2020):
    print(f"--- 正在处理 {year} 年 ---")

    # 提取当年数据并对齐极值
    ds_year = ds_all.sel(time=str(year))
    year_periods = ds_year.period.values
    mx = p_max.sel(period=year_periods).prec
    mn = p_min.sel(period=year_periods).prec

    # 计算 TCI 并降维（解决黑色长条问题）
    tci_raw = 100 * (ds_year.prec - mn) / (mx - mn + 1e-6)

    if 'period' in tci_raw.dims:
        tci_year = tci_raw.isel(period=0).squeeze()
    else:
        tci_year = tci_raw

    # 规范化数值范围在 0-100 之间
    tci_year = tci_year.clip(0, 100)

    # 规范化坐标
    if 'latitude' in tci_year.coords:
        tci_year = tci_year.rename({'latitude': 'lat', 'longitude': 'lon'})
    tci_year.rio.write_crs("EPSG:4326", inplace=True)
    tci_year = tci_year.transpose("time", "lat", "lon")

    # 处理玉米掩膜 (1km -> 0.1deg)
    corn_tif = corn_dir / f"CHN_Maize_{year}.tif"
    if corn_tif.exists():
        corn_1km = rioxarray.open_rasterio(corn_tif)
        if corn_1km.rio.crs is None:
            corn_1km.rio.write_crs("EPSG:4326", inplace=True)

        # 使用平均值法重采样，得到 0.1度像元内的玉米种植比例
        corn_purity = corn_1km.rio.reproject_match(
            tci_year,
            resampling=Resampling.average
        )

        if 'band' in corn_purity.dims:
            corn_purity = corn_purity.squeeze('band').drop_vars('band')

        # 强制空间坐标对齐，防止 where 运算失败
        corn_purity = corn_purity.rename({'x': 'lon', 'y': 'lat'})
        corn_purity = corn_purity.assign_coords(lat=tci_year.lat, lon=tci_year.lon)

        # 【核心改动】：阈值改为 > 0
        tci_final = tci_year.where(corn_purity > 0)

        # 清理冗余变量
        if 'spatial_ref' in tci_final.coords:
            tci_final = tci_final.drop_vars('spatial_ref')
        if 'period' in tci_final.coords:
            tci_final = tci_final.drop_vars('period')

        # 保存文件
        out_path = output_dir / f"China_Corn_TCI_8Day_0.1deg_{year}.nc"
        tci_final.to_netcdf(out_path, engine="netcdf4")

        print(f"成功导出 {year} 年。有效像元数: {int(tci_final.isel(time=0).notnull().sum())}")

        # 释放内存
        del tci_final, corn_purity, corn_1km, tci_year
    else:
        print(f"警告：未找到 {year} 年玉米图层，跳过掩膜步骤处理。")

print("\n✅ 处理完成！请在 QGIS 中检查 2019 年结果是否已恢复正常。")