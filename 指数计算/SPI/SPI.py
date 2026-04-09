import os
import xarray as xr
import numpy as np
import rasterio
import pandas as pd
from scipy.stats import norm
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import random
import calendar
import gc

# ===============================
# 1. 路径设置
# ===============================
precip_dir = r"E:\data\meteorologicaldata\precipitation"
mask_path = r"E:\数据集\作物种植文件\0.1度重采样文件\Summer_Maize_01deg.tif"
output_dir = r"E:\data\SPI\SPI_3month_daily"

os.makedirs(output_dir, exist_ok=True)

SCALE = 90  # 1个月尺度 = 30天滑动累计

# ===============================
# 2. 工具函数
# ===============================
def get_no_leap_doy(date_obj):
    """
    将日期转换为“无闰日DOY”
    规则：
    1) 2月29日直接跳过，不参与计算
    2) 闰年中 3月1日及之后的日期，DOY减1
       这样所有年份都统一成 365 天
    """
    if date_obj.month == 2 and date_obj.day == 29:
        return None

    doy = date_obj.timetuple().tm_yday

    if calendar.isleap(date_obj.year) and (date_obj.month > 2):
        doy -= 1

    return doy


def gringorten_spi(values):
    """
    对单个像元的一条时间序列做 Gringorten 经验概率拟合，
    再转为标准正态分位数（SPI）
    """
    valid_mask = ~np.isnan(values)
    v = values[valid_mask]

    # 样本太少，不计算
    if len(v) < 15:
        return np.full(len(values), np.nan, dtype=np.float32)

    # 平均秩，兼容重复值
    ranks = pd.Series(v).rank(method="average").values

    # Gringorten plotting position
    p = (ranks - 0.44) / (len(v) + 0.12)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    spi_values = norm.ppf(p)

    out = np.full(len(values), np.nan, dtype=np.float32)
    out[valid_mask] = spi_values.astype(np.float32)

    return out


# ===============================
# 3. 读取 mask（修正方向）
# ===============================
print("🚀 Step 0: 读取作物掩膜...")

with rasterio.open(mask_path) as src:
    mask = src.read(1).astype(np.float32)

# 若你的 tif 与 nc 在南北方向相反，这里保留 flipud
mask = np.flipud(mask)

# 转为 NaN 掩膜
mask[mask == 0] = np.nan

mask_flat = mask.flatten()
valid_idx = np.where(~np.isnan(mask_flat))[0]

print(f"✅ 作物像元数量: {len(valid_idx)}")
print(f"✅ Mask shape: {mask.shape}")


# ===============================
# 4. 构建 30 天滑动累计降水
#    已删除 2 月 29 日，并修正 DOY
# ===============================
window = deque(maxlen=SCALE)

# 只保留 365 天
doy_data = {i: [] for i in range(1, 366)}
date_list = []

lat = None
lon = None

print(f"🚀 Step 1: 构建 {SCALE} 天累计降水（删除 2 月 29 日）...")

for year in range(2000, 2021):
    year_path = os.path.join(precip_dir, str(year))
    if not os.path.exists(year_path):
        print(f"⚠️ 年份目录不存在，跳过: {year_path}")
        continue

    files = sorted(os.listdir(year_path))
    print(f"  正在处理 {year} 年，共 {len(files)} 个文件...")

    for f in files:
        if not f.endswith(".nc"):
            continue

        file_path = os.path.join(year_path, f)

        with xr.open_dataset(file_path) as ds:
            prec = ds["prec"].values.astype(np.float32)

            if lat is None:
                lat = ds["lat"].values
                lon = ds["lon"].values

        # 从文件名解析日期
        date_str = "_".join(f.split("_")[-3:]).replace(".nc", "")
        date = datetime.strptime(date_str, "%Y_%m_%d")

        # 删除 2 月 29 日
        if date.month == 2 and date.day == 29:
            continue

        # 放入滑动窗口
        window.append(prec)

        if len(window) < SCALE:
            continue

        # 30 天累计降水
        p_acc = np.sum(window, axis=0, dtype=np.float32)

        # 计算无闰日 DOY
        doy = get_no_leap_doy(date)
        if doy is None:
            continue

        doy_data[doy].append(p_acc)

        # 保持和你原代码一致：输出从 2001 年开始
        if date.year >= 2001:
            date_list.append((date, doy))

print("✅ Step 1 完成")


# ===============================
# 5. 计算 SPI（仅作物像元）
# ===============================
print("🚀 Step 2: 计算 SPI（仅作物区）...")

spi_result = {}

for doy in range(1, 366):
    if len(doy_data[doy]) == 0:
        continue

    print(f"  正在处理 DOY = {doy}")

    stack = np.stack(doy_data[doy], axis=0).astype(np.float32)
    t, lat_size, lon_size = stack.shape

    reshaped = stack.reshape(t, -1)
    spi = np.full((t, reshaped.shape[1]), np.nan, dtype=np.float32)

    # 只对作物区计算
    crop_pixels = reshaped[:, valid_idx]  # [t, n_valid]

    for j in range(len(valid_idx)):
        spi[:, valid_idx[j]] = gringorten_spi(crop_pixels[:, j])

    spi = spi.reshape(t, lat_size, lon_size)
    spi_result[doy] = spi

    # 释放内存
    doy_data[doy] = None
    del stack, reshaped, spi, crop_pixels
    gc.collect()

print("✅ Step 2 完成")


# ===============================
# 6. 重建 SPI 时间序列
# ===============================
print("🚀 Step 3: 重建 SPI 序列...")

spi_daily = []
counter = {doy: 0 for doy in range(1, 366)}

for date, doy in date_list:
    spi = spi_result[doy][counter[doy]]
    counter[doy] += 1
    spi_daily.append(spi)

spi_daily = np.stack(spi_daily, axis=0).astype(np.float32)
spi_daily[np.isinf(spi_daily)] = np.nan

time_coord = [d[0] for d in date_list]

print("✅ Step 3 完成")
print(f"✅ 输出时间长度: {len(time_coord)}")


# ===============================
# 7. Sanity Check
# ===============================
print("🚀 Step 3.5: Sanity Check...")

rand_idx = random.randint(0, spi_daily.shape[0] - 1)
rand_date = time_coord[rand_idx]
spi_sample = spi_daily[rand_idx]

plt.figure(figsize=(10, 6))

plt.imshow(
    spi_sample,
    origin="lower",
    cmap="RdBu",
    vmin=-2,
    vmax=2,
    extent=[lon.min(), lon.max(), lat.min(), lat.max()],
)

plt.colorbar(label="SPI")
plt.title(f"SPI-3month (Summer_Maize)\nDate: {rand_date.strftime('%Y-%m-%d')}")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.show()

print(f"✅ 已绘制日期: {rand_date.strftime('%Y-%m-%d')}")
print("SPI min:", np.nanmin(spi_sample))
print("SPI max:", np.nanmax(spi_sample))


# ===============================
# 8. 保存 NetCDF
# ===============================
print("🚀 Step 4: 保存文件...")

# 保存前把 NaN 替换成 _FillValue，避免不同软件读取时不一致
spi_to_save = np.where(np.isnan(spi_daily), -9999.0, spi_daily).astype(np.float32)

ds_out = xr.Dataset(
    {
        "SPI": (["time", "lat", "lon"], spi_to_save)
    },
    coords={
        "time": time_coord,
        "lat": lat,
        "lon": lon
    }
)

encoding = {
    "SPI": {
        "zlib": True,
        "complevel": 5,
        "dtype": "float32",
        "_FillValue": np.float32(-9999.0),
        "chunksizes": (100, len(lat), len(lon))
    }
}

output_path = os.path.join(output_dir, "SPI_3month_daily_Summer_Maize_noLeap.nc")
ds_out.to_netcdf(output_path, encoding=encoding)

print(f"🎉 完成！输出文件：{output_path}")