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
# 1. 路径配置
# ===============================
p_minus_pet_root = r"E:\data\meteorologicaldata\P_minus_PET"
output_root = r"E:\data\SPEI\Maize"
os.makedirs(output_root, exist_ok=True)

crop_configs = {
    "Summer_Maize": r"E:\数据集\作物种植文件\0.1度重采样文件\Summer_Maize_01deg.tif",
    "Spring_Maize": r"E:\数据集\作物种植文件\0.1度重采样文件\Spring_Maize_01deg.tif"
}

scales = {
    "1month": 30,
    "3month": 90,
    "6month": 180,
    "9month": 270,
    "12month": 365

}

# 🔥 包含历史年份（滑动窗口）
read_years = range(2001, 2023)

# 🔥 输出年份
output_start_year = 2002


# ===============================
# 2. SPEI函数
# ===============================
def gringorten_standardize(values):
    valid_mask = ~np.isnan(values)
    v = values[valid_mask]

    if len(v) < 15:
        return np.full(len(values), np.nan, dtype=np.float32)

    ranks = pd.Series(v).rank(method="average").values
    p = (ranks - 0.44) / (len(v) + 0.12)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    z = norm.ppf(p)

    out = np.full(len(values), np.nan, dtype=np.float32)
    out[valid_mask] = z.astype(np.float32)
    return out


def get_no_leap_doy(date):
    if date.month == 2 and date.day == 29:
        return None

    doy = date.timetuple().tm_yday
    if calendar.isleap(date.year) and date.month > 2:
        doy -= 1
    return doy


def infer_data_var(ds):
    coord_like = {"lat", "latitude", "lon", "longitude", "time"}
    candidates = [v for v in ds.data_vars if v.lower() not in coord_like]

    if len(candidates) == 1:
        return candidates[0]

    for v in candidates:
        if ds[v].ndim in [2, 3]:
            return v

    raise ValueError("无法识别变量")


# ===============================
# 3. 主程序
# ===============================
for crop_name, mask_path in crop_configs.items():

    print(f"\n================ {crop_name} ================")

    # ===============================
    # 读取mask
    # ===============================
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.float32)

    mask = np.flipud(mask)
    mask[mask == 0] = np.nan

    rows = np.any(~np.isnan(mask), axis=1)
    cols = np.any(~np.isnan(mask), axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    mask_crop = mask[rmin:rmax + 1, cmin:cmax + 1]
    valid_idx = np.where(~np.isnan(mask_crop.flatten()))[0]

    print(f"✅ 作物像元数: {len(valid_idx)}")

    # ===============================
    # 循环尺度
    # ===============================
    for scale_name, SCALE in scales.items():

        print(f"\n🚀 {crop_name} - {scale_name}")

        window = deque(maxlen=SCALE)
        doy_data = {i: [] for i in range(1, 366)}
        date_list = []

        lat, lon = None, None
        used_count = 0

        # ===============================
        # Step1 构建滑动窗口
        # ===============================
        for year in read_years:

            year_dir = os.path.join(p_minus_pet_root, str(year))
            if not os.path.exists(year_dir):
                print(f"⚠️ 缺失年份: {year}")
                continue

            print(f"📅 {year}")

            for month in range(1, 13):
                n_days = calendar.monthrange(year, month)[1]

                for day in range(1, n_days + 1):

                    date = datetime(year, month, day)

                    if date.month == 2 and date.day == 29:
                        continue

                    file_name = f"ChinaMet_010deg_D_{year:04d}_{month:02d}_{day:02d}.nc"
                    file_path = os.path.join(year_dir, file_name)

                    if not os.path.exists(file_path):
                        continue

                    with xr.open_dataset(file_path) as ds:

                        # 🔥 统一方向
                        ds = ds.sortby(['lat', 'lon'], ascending=True)

                        data_var = infer_data_var(ds)
                        arr = ds[data_var].values.astype(np.float32)

                        if arr.ndim == 3:
                            arr = arr[0]

                        if lat is None:
                            lat = ds["lat"].values[rmin:rmax + 1]
                            lon = ds["lon"].values[cmin:cmax + 1]

                    arr = arr[rmin:rmax + 1, cmin:cmax + 1]
                    arr = np.where(np.isnan(mask_crop), np.nan, arr)

                    window.append(arr)

                    if len(window) < SCALE:
                        continue

                    acc = np.sum(window, axis=0, dtype=np.float32)

                    doy = get_no_leap_doy(date)
                    if doy is None:
                        continue

                    doy_data[doy].append(acc)

                    if date.year >= output_start_year:
                        date_list.append((date, doy))
                        used_count += 1

        print(f"✅ 有效时次: {used_count}")

        if used_count == 0:
            continue

        # ===============================
        # Step2 SPEI
        # ===============================
        spei_result = {}

        for doy in range(1, 366):
            if len(doy_data[doy]) == 0:
                continue

            stack = np.stack(doy_data[doy])
            t, h, w = stack.shape

            reshaped = stack.reshape(t, -1)
            spei = np.full_like(reshaped, np.nan, dtype=np.float32)

            for j in valid_idx:
                spei[:, j] = gringorten_standardize(reshaped[:, j])

            spei_result[doy] = spei.reshape(t, h, w)
            doy_data[doy] = None
            gc.collect()

        print("✅ SPEI完成")

        # ===============================
        # Step3 重建
        # ===============================
        spei_daily = []
        counter = {doy: 0 for doy in range(1, 366)}

        for date, doy in date_list:
            spei_daily.append(spei_result[doy][counter[doy]])
            counter[doy] += 1

        spei_daily = np.stack(spei_daily).astype(np.float32)

        print(f"✅ 输出长度: {len(spei_daily)}")

        # ===============================
        # Step4 绘图
        # ===============================
        idx = random.randint(0, len(spei_daily) - 1)

        plt.imshow(
            spei_daily[idx],
            origin='lower',
            cmap='RdBu_r',
            vmin=-2,
            vmax=2,
            extent=[lon.min(), lon.max(), lat.min(), lat.max()]
        )
        plt.colorbar()
        plt.title(date_list[idx][0])
        plt.show()

        # ===============================
        # Step5 保存（🔥最终压缩优化）
        # ===============================
        ds_out = xr.Dataset(
            {"SPEI": (["time", "lat", "lon"], spei_daily)},
            coords={
                "time": [d[0] for d in date_list],
                "lat": lat,
                "lon": lon
            }
        )

        out_path = os.path.join(output_root, f"SPEI_{scale_name}_{crop_name}.nc")

        ds_out.to_netcdf(
            out_path,
            encoding={
                "SPEI": {
                    "zlib": True,
                    "complevel": 6,
                    "dtype": "float32",
                    "chunksizes": (1, len(lat), len(lon))
                }
            }
        )

        print(f"🎉 保存完成: {out_path}")

        del spei_daily, spei_result
        gc.collect()

print("\n✅ 全部完成")