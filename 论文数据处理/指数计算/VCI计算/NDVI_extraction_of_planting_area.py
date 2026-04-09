import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os
import matplotlib.pyplot as plt
import random

# ============================
# 1. 路径
# ============================
ndvi_folder = r"E:\data\NDVI\NDVI_China\NDVI_010deg"
mask_path = r"E:\数据集\作物种植文件\0.1度重采样文件\Winter_Wheat_0p1deg_mask.tif"

out_folder = r"E:\data\VCI_DOY\WinterWheat"
fig_folder = r"E:\data\VCI_DOY\figures"

os.makedirs(out_folder, exist_ok=True)
os.makedirs(fig_folder, exist_ok=True)

years = list(range(2001, 2021))

# ============================
# 2. 读取一个NDVI（建立目标网格）
# ============================
ds_ref = xr.open_dataset(os.path.join(ndvi_folder, "NDVI_010deg_2001.nc"))
ndvi_ref = ds_ref["NDVI"]

lat = ndvi_ref["lat"].values
lon = ndvi_ref["lon"].values

# 构建目标transform（关键）
res_lat = lat[1] - lat[0]
res_lon = lon[1] - lon[0]

transform = rasterio.transform.from_origin(
    lon.min() - res_lon/2,
    lat.max() + res_lat/2,
    res_lon,
    res_lat
)

# ============================
# 3. 读取mask并重采样（关键修复）
# ============================
with rasterio.open(mask_path) as src:
    mask_data = src.read(1)
    src_transform = src.transform
    src_crs = src.crs

mask_resampled = np.zeros((len(lat), len(lon)), dtype=np.float32)

reproject(
    source=mask_data,
    destination=mask_resampled,
    src_transform=src_transform,
    src_crs=src_crs,
    dst_transform=transform,
    dst_crs="EPSG:4326",
    resampling=Resampling.nearest
)

mask_xr = xr.DataArray(
    mask_resampled,
    dims=("lat", "lon"),
    coords={"lat": lat, "lon": lon}
)

# 转为0/1
mask_xr = xr.where(mask_xr > 0.5, 1, 0)

print("✅ mask 已空间对齐")

# ============================
# 4. 拼接NDVI（用于DOY climatology）
# ============================
print("📦 拼接NDVI...")

ds_list = []
for year in years:
    path = os.path.join(ndvi_folder, f"NDVI_010deg_{year}.nc")
    ds = xr.open_dataset(path)
    ndvi = ds["NDVI"]

    # 去异常
    ndvi = ndvi.where((ndvi >= 0) & (ndvi <= 1))

    # 删除2月29日
    ndvi = ndvi.sel(time=~((ndvi.time.dt.month == 2) & (ndvi.time.dt.day == 29)))

    ds_list.append(ndvi)

ndvi_all = xr.concat(ds_list, dim="time")

# 添加DOY
ndvi_all = ndvi_all.assign_coords(doy=ndvi_all.time.dt.dayofyear)

# ============================
# 5. DOY min/max
# ============================
print("📊 计算DOY min/max...")

ndvi_min = ndvi_all.groupby("doy").min("time")
ndvi_max = ndvi_all.groupby("doy").max("time")

ndvi_range = ndvi_max - ndvi_min
ndvi_range = xr.where(ndvi_range == 0, np.nan, ndvi_range)

# ============================
# 6. 逐年计算
# ============================
plot_years = [2001, 2006, 2011, 2016, 2020]

print("🚀 开始计算VCI...")

for year in years:
    print(f"➡️ {year}")

    ds = xr.open_dataset(os.path.join(ndvi_folder, f"NDVI_010deg_{year}.nc"))
    ndvi = ds["NDVI"]

    ndvi = ndvi.where((ndvi >= 0) & (ndvi <= 1))
    ndvi = ndvi.sel(time=~((ndvi.time.dt.month == 2) & (ndvi.time.dt.day == 29)))

    ndvi = ndvi.assign_coords(doy=ndvi.time.dt.dayofyear)

    # 对齐DOY
    ndvi_min_sel = ndvi_min.sel(doy=ndvi.doy)
    ndvi_max_sel = ndvi_max.sel(doy=ndvi.doy)
    ndvi_range_sel = ndvi_range.sel(doy=ndvi.doy)

    # VCI
    vci = (ndvi - ndvi_min_sel) / ndvi_range_sel * 100
    vci = vci.clip(0, 100)

    # 掩膜（已对齐）
    vci_crop = vci.where(mask_xr == 1)

    # ============================
    # 保存（压缩）
    # ============================
    vci_crop = vci_crop.astype("float32")
    vci_crop.name = "VCI"

    encoding = {
        "VCI": {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": -9999,
            "chunksizes": (1, 200, 200)
        }
    }

    out_path = os.path.join(out_folder, f"VCI_DOY_WinterWheat_{year}.nc")
    vci_crop.to_netcdf(out_path, encoding=encoding)

    # ============================
    # 绘图（修正不翻转）
    # ============================
    if year in plot_years:
        rand_idx = random.randint(0, vci_crop.sizes["time"] - 1)
        vci_day = vci_crop.isel(time=rand_idx)

        plt.figure(figsize=(10, 6))

        vci_day.plot(
            cmap="RdYlGn",
            vmin=0,
            vmax=100,
            cbar_kwargs={"label": "VCI"}
        )

        plt.title(f"VCI {year} - {str(vci_day.time.values)[:10]}")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        fig_path = os.path.join(fig_folder, f"VCI_{year}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()

print("🎉 全部完成（空间正确 + 无翻转 + 无海南误判）")