import os
import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import random
import gc
import calendar   # ⭐ 新增

# ===============================
# 1. 路径与参数设置
# ===============================
precip_dir = r"E:\data\meteorologicaldata\precipitation"
mask_path = r"E:\数据集\作物种植文件\0.1度重采样文件\Spring_Wheat_01deg.tif"
output_dir = r"E:\data\SPI\SPI_1month_daily"
os.makedirs(output_dir, exist_ok=True)

SCALE = 30  # 30天滑动累计


# ===============================
# 2. SPI 计算函数 (Gringorten)
# ===============================
def gringorten_spi(values):
    valid_mask = ~np.isnan(values)
    v = values[valid_mask]
    if len(v) < 15:
        return np.full(len(values), np.nan)

    ranks = pd.Series(v).rank(method='average')
    p = (ranks - 0.44) / (len(v) + 0.12)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    spi_values = norm.ppf(p)

    result = np.full(len(values), np.nan)
    result[valid_mask] = spi_values
    return result


# ===============================
# 3. 加载并对齐 Mask
# ===============================
print("🚀 正在加载并对齐掩膜文件...")

first_year_dir = os.path.join(precip_dir, "2000")
first_file = [f for f in os.listdir(first_year_dir) if f.endswith(".nc")][0]

with xr.open_dataset(os.path.join(first_year_dir, first_file)) as ref_ds:
    ref_ds = ref_ds.sortby(['lat', 'lon'], ascending=True)
    lat_coords = ref_ds.lat.values
    lon_coords = ref_ds.lon.values

da_mask = xr.open_dataarray(mask_path, engine="rasterio").squeeze()
if 'y' in da_mask.dims:
    da_mask = da_mask.rename({'y': 'lat', 'x': 'lon'})

da_mask = da_mask.assign_coords(lat=lat_coords, lon=lon_coords)
mask_np = da_mask.values

mask_flat = mask_np.flatten()
valid_idx = np.where(mask_flat > 0)[0]

print(f"✅ 识别到作物像元数量: {len(valid_idx)}")


# ===============================
# 4. 构建滑动累加降水（已修复闰年问题）
# ===============================
print("🚀 Step 1: 计算 30 天累积降水...")

window = deque(maxlen=SCALE)

# ⭐ 改成 365 天
doy_data = {i: [] for i in range(1, 366)}
date_list = []

for year in range(2000, 2022):
    year_path = os.path.join(precip_dir, str(year))
    if not os.path.exists(year_path):
        continue

    files = sorted([f for f in os.listdir(year_path) if f.endswith(".nc")])

    for f in files:
        with xr.open_dataset(os.path.join(year_path, f)) as ds:
            ds = ds.sortby(['lat', 'lon'], ascending=True)
            prec = ds['prec'].values

        window.append(prec)
        if len(window) < SCALE:
            continue

        p30 = np.sum(window, axis=0)

        # ===============================
        # ⭐ 关键修改开始
        # ===============================
        date_str = "_".join(f.split("_")[-3:]).replace(".nc", "")
        dt = datetime.strptime(date_str, "%Y_%m_%d")

        # ✔ 删除 2 月 29 日
        if dt.month == 2 and dt.day == 29:
            continue

        doy = dt.timetuple().tm_yday

        # ✔ 修正闰年 DOY 错位
        if calendar.isleap(dt.year) and dt.month > 2:
            doy -= 1
        # ===============================
        # ⭐ 修改结束
        # ===============================

        doy_data[doy].append(p30)

        if dt.year >= 2001:
            date_list.append((dt, doy))


# ===============================
# 5. 计算 SPI
# ===============================
print("🚀 Step 2: 计算 SPI (仅限种植区)...")

spi_results_by_doy = {}
ny, nx = mask_np.shape

for doy in range(1, 366):
    if not doy_data[doy]:
        continue

    stack = np.stack(doy_data[doy], axis=0)
    t = stack.shape[0]

    reshaped = stack.reshape(t, -1)
    spi_matrix = np.full((t, ny * nx), np.nan, dtype=np.float32)

    crop_pixels = reshaped[:, valid_idx]
    for i in range(len(valid_idx)):
        spi_matrix[:, valid_idx[i]] = gringorten_spi(crop_pixels[:, i])

    spi_results_by_doy[doy] = spi_matrix.reshape(t, ny, nx)

    doy_data[doy] = None
    gc.collect()


# ===============================
# 6. 重组序列
# ===============================
print("🚀 Step 3: 重组序列...")

final_time = [d[0] for d in date_list]
final_spi = np.zeros((len(final_time), ny, nx), dtype=np.float32)

doy_counter = {i: 0 for i in range(1, 366)}

for i, (dt, doy) in enumerate(date_list):
    final_spi[i] = spi_results_by_doy[doy][doy_counter[doy]]
    doy_counter[doy] += 1


# ===============================
# 7. Sanity Check
# ===============================
rand_idx = random.randint(0, len(final_time) - 1)
check_date = final_time[rand_idx]
check_data = final_spi[rand_idx]

plt.figure(figsize=(10, 6))
plt.imshow(check_data, origin='lower', cmap='RdBu', vmin=-2.5, vmax=2.5,
           extent=[lon_coords.min(), lon_coords.max(), lat_coords.min(), lat_coords.max()])
plt.colorbar(label="SPI Value")
plt.title(f"Spring Wheat SPI Check\nDate: {check_date.date()}")
plt.show()


# ===============================
# 8. 保存
# ===============================
print("🚀 Step 4: 保存结果...")

ds_out = xr.Dataset(
    {"SPI": (["time", "lat", "lon"], final_spi)},
    coords={"time": final_time, "lat": lat_coords, "lon": lon_coords}
)

encoding = {
    "SPI": {
        "zlib": True,
        "complevel": 5,
        "dtype": "float32",
        "chunksizes": (100, ny, nx),
        "_FillValue": -9999.0
    }
}

output_file = os.path.join(output_dir, "SPI_1month_daily_Spring_Wheat_noLeap.nc")
ds_out.to_netcdf(output_file, encoding=encoding)

print(f"🎉 完成！文件输出于: {output_file}")