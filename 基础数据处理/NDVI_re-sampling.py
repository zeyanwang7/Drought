import xarray as xr
import matplotlib.pyplot as plt
import os

# =========================
# 输入路径
# =========================
ndvi_dir = r"E:\data\NDVI\NDVI_China"
meteo_sample = r"E:\data\meteorologicaldata\precipitation\2000\ChinaMet_010deg_prec_2000_01_01.nc"
output_dir = r"E:\data\NDVI\NDVI_China\NDVI_010deg"

os.makedirs(output_dir, exist_ok=True)

# =========================
# 1. 读取目标网格（只执行一次）
# =========================
ds_meteo = xr.open_dataset(meteo_sample)

target_lat = ds_meteo['lat']
target_lon = ds_meteo['lon']

print("✅ 目标气象网格加载完成")

# =========================
# 2. 年份循环
# =========================
for year in range(2000, 2021):

    print(f"\n🚀 正在处理年份: {year}")

    try:
        # ---------------------
        # 2.1 构建NDVI路径
        # ---------------------
        ndvi_path = os.path.join(
            ndvi_dir,
            f"Daily_Gap-filled_NDVI_{year}.nc4"
        )

        if not os.path.exists(ndvi_path):
            print(f"⚠️ 文件不存在，跳过: {ndvi_path}")
            continue

        # ---------------------
        # 2.2 读取NDVI
        # ---------------------
        ds_ndvi = xr.open_dataset(ndvi_path)

        # 坐标标准化
        ds_ndvi = ds_ndvi.rename({
            'Lat': 'lat',
            'Lon': 'lon',
            'Time': 'time'
        })

        # 纬度方向修正
        ds_ndvi = ds_ndvi.sortby('lat')

        ndvi = ds_ndvi['NDVI']

        # ---------------------
        # 2.3 coarsen（降噪）
        # ---------------------
        ndvi_coarse = ndvi.coarsen(
            lat=2,
            lon=2,
            boundary='trim'
        ).mean()

        # ---------------------
        # 2.4 interp（对齐气象网格）
        # ---------------------
        ndvi_final = ndvi_coarse.interp(
            lat=target_lat,
            lon=target_lon,
            method='linear'
        )

        # ---------------------
        # 2.5 数据类型优化（关键）
        # ---------------------
        ndvi_final = ndvi_final.astype('float32')

        # ---------------------
        # 2.6 保存（压缩）
        # ---------------------
        output_path = os.path.join(
            output_dir,
            f"NDVI_010deg_{year}.nc"
        )

        encoding = {
            'NDVI': {
                'zlib': True,
                'complevel': 4
            }
        }

        ndvi_final.to_netcdf(output_path, encoding=encoding)

        print(f"✅ 完成: {year}")

        # ---------------------
        # 2.7 可视化检查（每5年）
        # ---------------------
        if year % 5 == 0:
            plt.figure(figsize=(6,4))
            ndvi_final.isel(time=0).plot(
                cmap='YlGn',
                vmin=0,
                vmax=1
            )
            plt.title(f"NDVI {year}")
            plt.show()

    except Exception as e:
        print(f"❌ 处理失败 {year}: {e}")

print("\n🎉 全部年份处理完成！")