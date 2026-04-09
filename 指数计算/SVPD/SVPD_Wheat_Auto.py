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
vpd_dir = r"E:\data\meteorologicaldata\VPD"
output_root = r"E:\data\SVPD\Wheat"
os.makedirs(output_root, exist_ok=True)

crop_configs = {
    "Winter_Wheat": r"E:\数据集\作物种植文件\0.1度重采样文件_修正版\Winter_Wheat_01deg_filtered.tif",
    "Spring_Wheat": r"E:\数据集\作物种植文件\0.1度重采样文件_修正版\Spring_Wheat_01deg_filtered.tif"
}

scales = {
    "1month": 30,
    "3month": 90,
    "6month": 180,
    "9month": 270,
    "12month": 365
}

# ===============================
# 2. SVPD函数
# ===============================
def gringorten_svpd(values):
    valid_mask = ~np.isnan(values)
    v = values[valid_mask]

    if len(v) < 15:
        return np.full(len(values), np.nan, dtype=np.float32)

    ranks = pd.Series(v).rank(method="average").values
    p = (ranks - 0.44) / (len(v) + 0.12)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    svpd = -norm.ppf(p)

    out = np.full(len(values), np.nan, dtype=np.float32)
    out[valid_mask] = svpd.astype(np.float32)
    return out


def get_no_leap_doy(date):
    if date.month == 2 and date.day == 29:
        return None
    doy = date.timetuple().tm_yday
    if calendar.isleap(date.year) and date.month > 2:
        doy -= 1
    return doy


def parse_date(filename):
    date_str = "_".join(filename.split("_")[-3:]).replace(".nc", "")
    return datetime.strptime(date_str, "%Y_%m_%d")


# ===============================
# 3. 主程序
# ===============================
for crop_name, mask_path in crop_configs.items():

    print(f"\n================ {crop_name} ================")

    # ===============================
    # Step0: 读取mask
    # ===============================
    with rasterio.open(mask_path) as src:
        mask_raw = src.read(1).astype(np.float32)
        transform = src.transform

    rows, cols = mask_raw.shape

    lon_full = transform.c + (np.arange(cols) + 0.5) * transform.a
    lat_full = transform.f + (np.arange(rows) + 0.5) * transform.e

    mask = mask_raw.copy()
    mask[mask <= 0] = np.nan
    mask[mask > 0] = 1

    # ===============================
    # 自动裁剪
    # ===============================
    valid_rows = np.any(~np.isnan(mask), axis=1)
    valid_cols = np.any(~np.isnan(mask), axis=0)

    rmin, rmax = np.where(valid_rows)[0][[0, -1]]
    cmin, cmax = np.where(valid_cols)[0][[0, -1]]

    print(f"✅ 裁剪范围: 行 {rmin}-{rmax}, 列 {cmin}-{cmax}")

    mask = mask[rmin:rmax+1, cmin:cmax+1]
    lat = lat_full[rmin:rmax+1]
    lon = lon_full[cmin:cmax+1]

    valid_idx = np.where(~np.isnan(mask.flatten()))[0]
    print(f"✅ 作物像元数: {len(valid_idx)}")

    # ===============================
    # 循环尺度
    # ===============================
    for scale_name, SCALE in scales.items():

        print(f"\n🚀 开始计算 {crop_name} - {scale_name}")

        window = deque(maxlen=SCALE)
        doy_data = {i: [] for i in range(1, 366)}
        date_list = []

        # ===============================
        # Step2: VPD处理
        # ===============================
        for year in range(2001, 2023):   # 🔥 2001作为预热年

            year_path = os.path.join(vpd_dir, str(year))
            if not os.path.exists(year_path):
                continue

            files = sorted([f for f in os.listdir(year_path) if f.endswith(".nc")])
            print(f"  📅 {year}: {len(files)}")

            for f in files:
                date = parse_date(f)

                if date.month == 2 and date.day == 29:
                    continue

                file_path = os.path.join(year_path, f)

                with xr.open_dataset(file_path) as ds:
                    vpd_var = [v for v in ds.data_vars if "vpd" in v.lower()][0]
                    da = ds[vpd_var]

                    if da.dims != ("lat", "lon"):
                        da = da.transpose("lat", "lon")

                    da_interp = da.interp(lat=lat, lon=lon, method="linear")
                    vpd = da_interp.values.astype(np.float32)

                vpd[np.isnan(mask)] = np.nan

                window.append(vpd)

                if len(window) < SCALE:
                    continue

                vpd_mean = np.mean(window, axis=0, dtype=np.float32)

                doy = get_no_leap_doy(date)
                if doy is None:
                    continue

                doy_data[doy].append(vpd_mean)

                # 🔥 只输出2002-2022
                if date.year >= 2002:
                    date_list.append((date, doy))

        print("✅ VPD处理完成")

        # ===============================
        # Step3: SVPD计算
        # ===============================
        print("🚀 开始 SVPD...")

        svpd_result = {}

        for doy in range(1, 366):

            if len(doy_data[doy]) == 0:
                continue

            stack = np.stack(doy_data[doy]).astype(np.float32)
            t, h, w = stack.shape

            reshaped = stack.reshape(t, -1)
            svpd = np.full_like(reshaped, np.nan, dtype=np.float32)

            crop_pixels = reshaped[:, valid_idx]

            for j in range(len(valid_idx)):
                svpd[:, valid_idx[j]] = gringorten_svpd(crop_pixels[:, j])

            svpd = svpd.reshape(t, h, w)
            svpd_result[doy] = svpd

            if doy % 30 == 0 or doy == 365:
                print(f"  ... DOY {doy}/365")

        print("✅ SVPD完成")

        # ===============================
        # Step4: 重建时间序列
        # ===============================
        svpd_daily = []
        counter = {doy: 0 for doy in range(1, 366)}

        for date, doy in date_list:
            svpd = svpd_result[doy][counter[doy]]
            counter[doy] += 1
            svpd_daily.append(svpd)

        svpd_daily = np.stack(svpd_daily).astype(np.float32)
        svpd_daily[np.isinf(svpd_daily)] = np.nan

        print(f"✅ 时间长度: {len(svpd_daily)}")

        # ===============================
        # Step5: 保存
        # ===============================
        print("💾 保存中...")

        ds_out = xr.Dataset(
            {"SVPD": (["time", "lat", "lon"], svpd_daily)},
            coords={
                "time": [d[0] for d in date_list],
                "lat": lat,
                "lon": lon
            }
        )

        encoding = {
            "SVPD": {
                "zlib": True,
                "complevel": 6,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": None,
                "chunksizes": (5, 100, 100)
            }
        }

        out_path = os.path.join(
            output_root,
            f"SVPD_{scale_name}_{crop_name}.nc"
        )

        ds_out.to_netcdf(out_path, encoding=encoding)

        print(f"🎉 完成: {out_path}")

        del svpd_daily
        gc.collect()