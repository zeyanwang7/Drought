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
import calendar
import time

# ===============================
# 1. 路径与参数
# ===============================
precip_dir = r"E:\data\meteorologicaldata\precipitation"
output_root = r"E:\data\SPI\Wheat"
os.makedirs(output_root, exist_ok=True)

crop_dict = {
    "Spring_Wheat": r"E:\数据集\作物种植文件\0.1度重采样文件_修正版\Spring_Wheat_01deg_filtered.tif",
    "Winter_Wheat": r"E:\数据集\作物种植文件\0.1度重采样文件_修正版\Winter_Wheat_01deg_filtered.tif",
}

scale_dict = {
    # "1month": 30,
    # "3month": 90,
    # "6month": 180,
    # "12month": 365
"9month": 270
}

calc_years = range(2001, 2023)
output_start_year = 2002


# ===============================
# 2. SPI函数
# ===============================
def gringorten_spi(values):
    valid_mask = ~np.isnan(values)
    v = values[valid_mask]

    if len(v) < 15:
        return np.full(len(values), np.nan, dtype=np.float32)

    ranks = pd.Series(v).rank(method='average')
    p = (ranks - 0.44) / (len(v) + 0.12)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    spi_values = norm.ppf(p)

    result = np.full(len(values), np.nan, dtype=np.float32)
    result[valid_mask] = spi_values.astype(np.float32)

    return result


# ===============================
# 3. 辅助函数
# ===============================
def print_divider(char="=", n=60):
    print(char * n)


def get_progress_step(total, target_steps=12):
    if total <= 0:
        return 1
    return max(1, total // target_steps)


# ===============================
# 4. 主程序
# ===============================
all_start = time.time()

for crop_name, mask_path in crop_dict.items():

    crop_start = time.time()
    print("\n")
    print_divider("=")
    print(f"🌾 开始处理作物: {crop_name}")
    print(f"📂 掩膜文件: {mask_path}")
    print_divider("=")

    # ===============================
    # Step 0
    # ===============================
    print("📌 Step 0/5: 读取参考降水网格并对齐掩膜 ...")

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

    rows = np.any(mask_np > 0, axis=1)
    cols = np.any(mask_np > 0, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    mask_np = mask_np[rmin:rmax + 1, cmin:cmax + 1]
    lat_coords = lat_coords[rmin:rmax + 1]
    lon_coords = lon_coords[cmin:cmax + 1]

    ny, nx = mask_np.shape
    mask_flat = mask_np.flatten()
    valid_idx = np.where(mask_flat > 0)[0]

    print(f"✅ 作物像元数: {len(valid_idx)}")
    print(f"✅ 裁剪后网格尺寸: {ny} × {nx}")

    # ===============================
    # 循环尺度
    # ===============================
    for scale_name, SCALE in scale_dict.items():

        scale_start = time.time()
        print("\n")
        print_divider("-")
        print(f"🚀 开始尺度计算: {crop_name} - {scale_name} (窗口={SCALE}天)")
        print_divider("-")

        window = deque(maxlen=SCALE)
        doy_data = {i: [] for i in range(1, 366)}
        date_list = []

        # ===============================
        # Step1
        # ===============================
        print("📌 Step 1/5: 构建滑动累计降水序列 ...")

        for year in calc_years:
            year_path = os.path.join(precip_dir, str(year))
            files = sorted([f for f in os.listdir(year_path) if f.endswith(".nc")])

            print(f"  📅 正在处理年份: {year}，文件数: {len(files)}")

            for f in files:

                file_path = os.path.join(year_path, f)

                try:
                    with xr.open_dataset(file_path) as ds:
                        ds = ds.sortby(['lat', 'lon'], ascending=True)
                        prec = ds['prec'].values[rmin:rmax + 1, cmin:cmax + 1]
                except:
                    continue

                window.append(prec.astype(np.float32))

                if len(window) < SCALE:
                    continue

                p_acc = np.sum(window, axis=0, dtype=np.float32)

                try:
                    date_str = "_".join(f.split("_")[-3:]).replace(".nc", "")
                    dt = datetime.strptime(date_str, "%Y_%m_%d")
                except:
                    continue

                if dt.month == 2 and dt.day == 29:
                    continue

                doy = dt.timetuple().tm_yday

                if calendar.isleap(dt.year) and dt.month > 2:
                    doy -= 1

                if dt.year >= output_start_year:
                    doy_data[doy].append(p_acc)
                    date_list.append((dt, doy))

        print("✅ Step 1 完成")

        # ===============================
        # Step2
        # ===============================
        print("\n📌 Step 2/5: 按 DOY 分组计算 SPI ...")

        spi_results_by_doy = {}
        valid_doys = [d for d in range(1, 366) if doy_data[d]]

        print(f"✅ 有样本的 DOY 组数: {len(valid_doys)} / 365")

        doy_progress_step = get_progress_step(len(valid_doys), target_steps=12)

        for doy_i, doy in enumerate(valid_doys, 1):

            stack = np.stack(doy_data[doy], axis=0).astype(np.float32)
            t = stack.shape[0]

            if doy_i == 1 or doy_i % doy_progress_step == 0 or doy_i == len(valid_doys):
                print(f"  ... DOY组进度: {doy_i}/{len(valid_doys)} (当前 DOY={doy:03d}, 样本数={t})")

            reshaped = stack.reshape(t, -1)
            spi_matrix = np.full((t, ny * nx), np.nan, dtype=np.float32)

            crop_pixels = reshaped[:, valid_idx]

            pixel_step = get_progress_step(len(valid_idx), 6)

            for i in range(len(valid_idx)):
                spi_matrix[:, valid_idx[i]] = gringorten_spi(crop_pixels[:, i])

                if (i + 1) % pixel_step == 0 or (i + 1) == len(valid_idx):
                    if doy_i == 1 or doy_i % doy_progress_step == 0:
                        print(f"    DOY {doy:03d} | 像元进度: {i + 1}/{len(valid_idx)}")

            spi_results_by_doy[doy] = spi_matrix.reshape(t, ny, nx)

            gc.collect()

        print("✅ Step 2 完成")

        # ===============================
        # Step3
        # ===============================
        print("\n📌 Step 3/5: 重组最终时间序列 ...")

        final_time = [d[0] for d in date_list]
        final_spi = np.zeros((len(final_time), ny, nx), dtype=np.float32)

        doy_counter = {i: 0 for i in range(1, 366)}

        for i, (dt, doy) in enumerate(date_list):
            final_spi[i] = spi_results_by_doy[doy][doy_counter[doy]]
            doy_counter[doy] += 1

        print(f"✅ 最终数组形状: {final_spi.shape}")

        # ===============================
        # Step4
        # ===============================
        print("\n📌 Step 4/5: 随机绘图检查 ...")

        idx = random.randint(0, len(final_time) - 1)

        plt.imshow(final_spi[idx], cmap='RdBu_r', vmin=-2.5, vmax=2.5, origin='lower')
        plt.colorbar(label="SPI")
        plt.title(f"{crop_name}-{scale_name}\n{final_time[idx]}")
        plt.show()

        # ===============================
        # Step5（🔥已修改）
        # ===============================
        print("\n📌 Step 5/5: 保存文件 (压缩 + float32) ...")

        ds_out = xr.Dataset(
            {"SPI": (["time", "lat", "lon"], final_spi)},
            coords={"time": final_time, "lat": lat_coords, "lon": lon_coords}
        )

        encoding = {
            "SPI": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "_FillValue": np.nan
            }
        }

        out_path = os.path.join(output_root, f"SPI_{scale_name}_{crop_name}.nc")
        ds_out.to_netcdf(out_path, encoding=encoding)

        print(f"✅ 文件保存完成: {out_path}")

        gc.collect()

print("\n🎉 全部计算完成")