import os
import xarray as xr
import rioxarray as rxr
import numpy as np

# ==============================
# 1. 路径设置
# ==============================
ndvi_dir = r"E:\data\NDVI\NDVI_China\NDVI_010deg"
mask_path = r"E:\数据集\作物种植文件\0.1度重采样文件\Summer_Maize_Stable_0.1deg.tif"
out_dir = r"E:\data\VCI\Summer_Maize_2001_2020"

os.makedirs(out_dir, exist_ok=True)

years = [str(y) for y in range(2001, 2021)]

# ==============================
# 2. 读取 crop mask
# ==============================
print("🚀 读取作物掩膜...")

mask = rxr.open_rasterio(mask_path).squeeze()

# ❗必须转 uint8（否则报错）
mask = (mask > 0).astype("uint8")

print("✅ mask唯一值:", np.unique(mask.values))

# ==============================
# 3. 读取并拼接 NDVI（2001–2020）
# ==============================
print("🚀 合并 NDVI 数据...")

datasets = []

for year in years:
    file = f"NDVI_010deg_{year}.nc"
    path = os.path.join(ndvi_dir, file)

    print(f"加载: {file}")

    ds = xr.open_dataset(path)
    var_name = list(ds.data_vars)[0]
    ndvi = ds[var_name]

    # 坐标统一
    if "lon" in ndvi.coords:
        ndvi = ndvi.rename({"lon": "x"})
    if "lat" in ndvi.coords:
        ndvi = ndvi.rename({"lat": "y"})

    # CRS写入
    ndvi.rio.write_crs("EPSG:4326", inplace=True)

    datasets.append(ndvi)

# 拼接时间维度
ndvi_all = xr.concat(datasets, dim="time")

# chunk（防止爆内存）
ndvi_all = ndvi_all.chunk({"time": 50})

# ==============================
# 4. mask 对齐 + 应用
# ==============================
print("🚀 对齐 mask...")

mask_match = mask.rio.reproject_match(ndvi_all)

ndvi_all = ndvi_all.where(mask_match == 1)

# ==============================
# 5. 计算多年 min / max（核心）
# ==============================
print("🚀 计算气候态 NDVI_min / NDVI_max...")

ndvi_min = ndvi_all.min(dim="time")
ndvi_max = ndvi_all.max(dim="time")

# ==============================
# 6. 逐年计算 VCI + 压缩保存
# ==============================
print("🚀 开始计算 VCI...")

for year in years:
    print(f"\n👉 处理 {year}")

    file = f"NDVI_010deg_{year}.nc"
    path = os.path.join(ndvi_dir, file)

    ds = xr.open_dataset(path)
    var_name = list(ds.data_vars)[0]
    ndvi = ds[var_name]

    # 坐标统一
    if "lon" in ndvi.coords:
        ndvi = ndvi.rename({"lon": "x"})
    if "lat" in ndvi.coords:
        ndvi = ndvi.rename({"lat": "y"})

    ndvi.rio.write_crs("EPSG:4326", inplace=True)

    # mask对齐
    mask_match = mask.rio.reproject_match(ndvi)

    # 应用mask
    ndvi = ndvi.where(mask_match == 1)

    # ==============================
    # 计算VCI
    # ==============================
    vci = (ndvi - ndvi_min) / (ndvi_max - ndvi_min) * 100

    # 防止除0
    vci = vci.where((ndvi_max - ndvi_min) != 0)

    # 限制范围（保险）
    vci = vci.clip(0, 100)

    # 保留2位小数（减少体积）
    vci = vci.round(2)

    # 命名（必须）
    vci.name = "VCI"

    # ==============================
    # 压缩保存（核心）
    # ==============================
    out_path = os.path.join(out_dir, f"VCI_{year}.nc")

    encoding = {
        "VCI": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 5
        }
    }

    vci.to_netcdf(out_path, encoding=encoding)

    print(f"✅ 已保存（压缩）: {out_path}")

print("\n🎉 2001–2020 VCI 计算完成（压缩版）！")