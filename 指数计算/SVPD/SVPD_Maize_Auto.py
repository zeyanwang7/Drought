import os
import xarray as xr
import numpy as np
import rasterio
import pandas as pd
from scipy.stats import norm
from collections import deque
from datetime import datetime
import calendar
import matplotlib.pyplot as plt
import random
import gc

# ===============================
# 路径
# ===============================
vpd_dir = r"E:\data\meteorologicaldata\VPD"
output_root = r"E:\data\SVPD\Maize_fast"
os.makedirs(output_root, exist_ok=True)

crop_configs = {
    "Spring_Maize": r"E:\数据集\作物种植文件\0.1度重采样文件\Spring_Maize_01deg.tif",
    "Summer_Maize": r"E:\数据集\作物种植文件\0.1度重采样文件\Summer_Maize_01deg.tif"
}

scales = {
    # "1month": 30,
    # "3month": 90,
    # "6month": 180,
    "9month": 270
    # "12month": 365

}

# ===============================
# SVPD函数
# ===============================
def gringorten(v):
    valid = ~np.isnan(v)
    x = v[valid]

    if len(x) < 15:
        return np.full(len(v), np.nan, dtype=np.float32)

    r = pd.Series(x).rank().values
    p = (r - 0.44) / (len(x) + 0.12)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    z = -norm.ppf(p)

    out = np.full(len(v), np.nan, dtype=np.float32)
    out[valid] = z.astype(np.float32)
    return out


def parse_date(f):
    s = "_".join(f.split("_")[-3:]).replace(".nc", "")
    return datetime.strptime(s, "%Y_%m_%d")


def get_doy(d):
    if d.month == 2 and d.day == 29:
        return None
    doy = d.timetuple().tm_yday
    if calendar.isleap(d.year) and d.month > 2:
        doy -= 1
    return doy


# ===============================
# 主循环
# ===============================
for crop_name, mask_path in crop_configs.items():

    print(f"\n================ {crop_name} ================")

    # ===============================
    # 读取mask
    # ===============================
    with rasterio.open(mask_path) as src:
        mask_raw = src.read(1).astype(np.float32)
        transform = src.transform

    rows, cols = mask_raw.shape

    lat_full = transform.f + (np.arange(rows) + 0.5) * transform.e
    lon_full = transform.c + (np.arange(cols) + 0.5) * transform.a

    if transform.e < 0:
        mask_raw = np.flipud(mask_raw)
        lat_full = lat_full[::-1]

    mask = mask_raw.copy()
    mask[mask <= 0] = np.nan
    mask[mask > 0] = 1

    # 裁剪
    valid_rows = np.any(~np.isnan(mask), axis=1)
    valid_cols = np.any(~np.isnan(mask), axis=0)

    rmin, rmax = np.where(valid_rows)[0][[0, -1]]
    cmin, cmax = np.where(valid_cols)[0][[0, -1]]

    mask = mask[rmin:rmax+1, cmin:cmax+1]
    lat = lat_full[rmin:rmax+1]
    lon = lon_full[cmin:cmax+1]

    h, w = mask.shape

    # 🔥 作物像元索引（核心）
    valid_idx = np.where(~np.isnan(mask.flatten()))[0]
    print("作物像元数:", len(valid_idx))

    # ===============================
    # 多尺度
    # ===============================
    for scale_name, SCALE in scales.items():

        print(f"\n🚀 {crop_name} - {scale_name}")

        window = deque(maxlen=SCALE)
        doy_data = {i: [] for i in range(1, 366)}
        date_list = []

        # ===============================
        # 读取VPD
        # ===============================
        for year in range(2001, 2023):

            year_path = os.path.join(vpd_dir, str(year))
            if not os.path.exists(year_path):
                continue

            files = sorted([f for f in os.listdir(year_path) if f.endswith(".nc")])
            print(f"  📅 {year}: {len(files)}")

            for f in files:
                date = parse_date(f)

                if date.month == 2 and date.day == 29:
                    continue

                with xr.open_dataset(os.path.join(year_path, f)) as ds:
                    var = [v for v in ds.data_vars if "vpd" in v.lower()][0]
                    da = ds[var]

                    if da.dims != ("lat", "lon"):
                        da = da.transpose("lat", "lon")

                    da_interp = da.interp(lat=lat_full, lon=lon_full)
                    vpd_full = da_interp.values.astype(np.float32)

                # 🔥 只提取作物像元
                vpd = vpd_full[rmin:rmax+1, cmin:cmax+1].reshape(-1)
                vpd_crop = vpd[valid_idx]

                window.append(vpd_crop)

                if len(window) < SCALE:
                    continue

                mean_vpd = np.mean(window, axis=0)

                doy = get_doy(date)
                if doy is None:
                    continue

                doy_data[doy].append(mean_vpd)

                if date.year >= 2002:
                    date_list.append((date, doy))

        print("✅ VPD完成")

        # ===============================
        # SVPD计算（1D）
        # ===============================
        svpd_res = {}

        for doy in range(1, 366):

            if len(doy_data[doy]) == 0:
                continue

            stack = np.stack(doy_data[doy])  # (t, pixels)
            t, n = stack.shape

            svpd = np.zeros_like(stack)

            for j in range(n):
                svpd[:, j] = gringorten(stack[:, j])

            svpd_res[doy] = svpd

        print("✅ SVPD完成")

        # ===============================
        # 重建二维
        # ===============================
        out_list = []
        counter = {d: 0 for d in range(1, 366)}

        for date, doy in date_list:
            row = svpd_res[doy][counter[doy]]
            counter[doy] += 1

            full = np.full(h*w, np.nan, dtype=np.float32)
            full[valid_idx] = row

            out_list.append(full.reshape(h, w))

        svpd = np.stack(out_list)

        print("时间长度:", len(svpd))
        print("NaN比例:", np.isnan(svpd).mean())

        # ===============================
        # 保存（压缩）
        # ===============================
        ds_out = xr.Dataset(
            {"SVPD": (["time", "lat", "lon"], svpd)},
            coords={
                "time": [d[0] for d in date_list],
                "lat": lat,
                "lon": lon
            }
        )

        out_path = os.path.join(output_root, f"SVPD_{scale_name}_{crop_name}.nc")

        encoding = {
            "SVPD": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "shuffle": True
            }
        }

        ds_out.to_netcdf(out_path, encoding=encoding)
        print("💾 保存:", out_path)

        # ===============================
        # 随机绘图
        # ===============================
        idx = random.randint(0, len(svpd)-1)
        img = svpd[idx]
        t = str(date_list[idx][0])[:10]

        plt.figure(figsize=(8,5))
        plt.imshow(img, origin="lower", cmap="RdBu_r",
                   vmin=-2, vmax=2,
                   extent=[lon.min(), lon.max(), lat.min(), lat.max()])
        plt.colorbar(label="SVPD")
        plt.title(f"{crop_name}-{scale_name}\n{t}")
        plt.tight_layout()
        plt.show()

        print("🎯 绘图:", t)

        del svpd, ds_out
        gc.collect()