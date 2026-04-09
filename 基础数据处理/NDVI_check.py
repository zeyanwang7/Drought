import xarray as xr
import rasterio
import numpy as np
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling

# ============================
# 📌 文件路径（你的）
# ============================
crop_file = r"E:\数据集\作物种植文件\0.1度重采样文件\Winter_Wheat_0p1deg_mask.tif"
ndvi_file = r"E:\data\NDVI\NDVI_China\NDVI_010deg\NDVI_010deg_2001.nc"

# ============================
# 1️⃣ 读取 Crop（作为标准网格）
# ============================
with rasterio.open(crop_file) as src:
    crop_data = src.read(1)
    crop_transform = src.transform
    crop_crs = src.crs
    crop_shape = (src.height, src.width)

    left, bottom, right, top = src.bounds

print("✅ Crop 网格读取完成")

# ============================
# 2️⃣ 读取 NDVI
# ============================
ds = xr.open_dataset(ndvi_file)
ndvi = ds["NDVI"]

lat = ds["lat"].values
lon = ds["lon"].values

print("✅ NDVI 读取完成")

# ============================
# 3️⃣ 修正纬度方向（关键）
# ============================
if lat[0] < lat[-1]:
    print("🔧 NDVI 纬度为升序 → 执行翻转")
    ndvi = ndvi.sel(lat=lat[::-1])
    lat = lat[::-1]
else:
    print("✅ NDVI 纬度方向正确")

# ============================
# 4️⃣ 构建 NDVI transform
# ============================
ndvi_transform = from_bounds(
    lon.min(), lat.min(),
    lon.max(), lat.max(),
    len(lon), len(lat)
)

# ============================
# 5️⃣ 重投影对齐（核心步骤）
# ============================
print("🚀 开始重投影对齐...")

ndvi_aligned = np.zeros((ndvi.shape[0], crop_shape[0], crop_shape[1]), dtype=np.float32)

for t in range(ndvi.shape[0]):
    if t % 20 == 0:
        print(f"处理时间步: {t}/{ndvi.shape[0]}")

    src_array = ndvi.isel(time=t).values

    dst_array = np.zeros(crop_shape, dtype=np.float32)

    reproject(
        source=src_array,
        destination=dst_array,
        src_transform=ndvi_transform,
        src_crs="EPSG:4326",
        dst_transform=crop_transform,
        dst_crs=crop_crs,
        resampling=Resampling.bilinear
    )

    ndvi_aligned[t] = dst_array

print("✅ 重投影完成")

# ============================
# 6️⃣ 保存结果
# ============================
out_nc = r"E:\data\NDVI\NDVI_China\NDVI_010deg\NDVI_010deg_2001_aligned.nc"

xr.Dataset(
    {
        "NDVI": (["time", "y", "x"], ndvi_aligned)
    },
    coords={
        "time": ds["time"].values,
        "y": np.arange(crop_shape[0]),
        "x": np.arange(crop_shape[1])
    }
).to_netcdf(out_nc)

print(f"🎉 完成！输出文件：{out_nc}")