import xarray as xr
import geopandas as gpd
import rioxarray
import pandas as pd
import numpy as np
import os

# --- 1. 路径设置 ---
nc_path = r'E:\data\meteorologicaldata\SPEI\SPEI_30d_8dStep_Rice.nc'
shp_path = r'E:\数据集\中国_省\中国_省2.shp'
output_csv = r'E:\data\计算结果\Provincial_Rice_SPEI_Steps.csv'


def final_extraction_fix():
    print("🚀 开始省级提取 (多步长扫描版)...")

    # 加载 NC 并修正坐标
    ds = xr.open_dataset(nc_path)
    ds = ds.sortby(['lon', 'lat'])
    ds.coords['lon'] = np.round(ds.coords['lon'], 4)
    ds.coords['lat'] = np.round(ds.coords['lat'], 4)
    ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    ds.rio.write_crs("EPSG:4326", inplace=True)
    spei_da = ds['SPEI_30d']

    # 加载 SHP (UTF-8)
    gdf = gpd.read_file(shp_path, encoding='utf-8')
    gdf = gdf.to_crs("EPSG:4326")
    name_col = next((c for c in ['name', '省', '省名', 'NAME'] if c in gdf.columns), gdf.columns[0])

    # --- 关键：生成地理掩膜 ---
    # 之前失败是因为 time=0 是空的。现在我们压缩时间维度，
    # 只要一个格点在整个序列中出现过有效值，我们就认为它是小麦格点。
    print("🧠 正在生成全局地理有效性掩膜...")
    valid_mask_2d = spei_da.notnull().any(dim='time')

    results = {}

    print("--- 正在逐省计算 ---")
    for _, row in gdf.iterrows():
        p_name = str(row[name_col]).strip()
        geom = [row['geometry']]

        try:
            # 1. 裁剪掩膜，检查该省到底有没有小麦格点
            mask_clipped = valid_mask_2d.rio.clip(geom, gdf.crs, drop=True, all_touched=True)
            spatial_valid_count = int(mask_clipped.sum().values)

            if spatial_valid_count > 5:  # 阈值设为 5
                # 2. 只有确认有数，才裁剪完整的数据集进行均值计算
                data_clipped = spei_da.rio.clip(geom, gdf.crs, drop=True, all_touched=True)

                # 空间平均：保留时间轴
                p_series = data_clipped.mean(dim=['lat', 'lon'], skipna=True)
                results[p_name] = p_series.to_series()
                print(f"✅ {p_name:<10} | 地理格点数: {spatial_valid_count}")
            else:
                if spatial_valid_count > 0:
                    print(f"⏭️ 跳过 {p_name}: 种植区过小 ({spatial_valid_count})")
        except Exception:
            continue

    # --- 保存结果 ---
    if results:
        final_df = pd.concat(results, axis=1)
        final_df.index = pd.to_datetime(final_df.index)
        final_df = final_df.reindex(sorted(final_df.columns), axis=1)

        # 结果导出
        final_df.to_csv(output_csv, encoding='utf-8-sig')
        print(f"\n🎉 提取完成！CSV 已保存至: {output_csv}")
        print(f"📊 包含省份: {list(final_df.columns)}")
    else:
        print("❌ 提取失败：请检查空间重合度。")


if __name__ == "__main__":
    final_extraction_fix()