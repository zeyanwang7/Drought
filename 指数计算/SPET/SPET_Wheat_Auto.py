# -*- coding: utf-8 -*-
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
# 这里改成你前面已经生成好的 P-ET 数据目录
data_dir = r"E:\data\P_minus_ET_Daily"

# 输出目录建议单独建一个
output_root = r"E:\data\SPET\Wheat"
os.makedirs(output_root, exist_ok=True)

crop_dict = {
    "Spring_Wheat": r"E:\数据集\作物种植文件\0.1度重采样文件_修正版\Spring_Wheat_01deg_filtered.tif",
    "Winter_Wheat": r"E:\数据集\作物种植文件\0.1度重采样文件_修正版\Winter_Wheat_01deg_filtered.tif",
}

scale_dict = {
    "1month": 30,
    "3month": 90,
    "6month": 180,
    "9month": 270,
    "12month": 365
}

# 2001 为预热年，输出从 2002 开始
calc_years = range(2001, 2023)
output_start_year = 2002

# 绘图显示范围，可按需要调整
plot_vmin = -2.5
plot_vmax = 2.5

# 固定随机种子，便于复现
random.seed(42)
np.random.seed(42)


# ===============================
# 2. SPET 标准化函数
#    （沿用你原 SPEI 代码中的 Gringorten 思路）
# ===============================
def gringorten_spet(values):
    """
    对单像元时间序列做标准化
    values: 某一DOY下，不同年份同一像元的累计 P-ET 序列
    """
    valid_mask = ~np.isnan(values)
    v = values[valid_mask]

    # 样本太少则不计算
    if len(v) < 15:
        return np.full(len(values), np.nan, dtype=np.float32)

    ranks = pd.Series(v).rank(method='average')
    p = (ranks - 0.44) / (len(v) + 0.12)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    spet_values = norm.ppf(p)

    result = np.full(len(values), np.nan, dtype=np.float32)
    result[valid_mask] = spet_values.astype(np.float32)

    return result


# ===============================
# 3. 辅助函数
# ===============================
def print_divider(char="=", n=70):
    print(char * n)


def get_progress_step(total, target_steps=12):
    if total <= 0:
        return 1
    return max(1, total // target_steps)


def guess_var_name(ds):
    """
    自动识别变量名
    优先识别 P_minus_ET
    """
    data_vars = list(ds.data_vars)

    if len(data_vars) == 0:
        raise ValueError("数据集中未找到数据变量")

    preferred = ["P_minus_ET", "p_minus_et", "PET", "pet", "wb", "water_balance"]
    for v in preferred:
        if v in data_vars:
            return v

    return data_vars[0]


def parse_date_from_filename(filename):
    """
    解析文件名中的日期
    适配:
    P_ET_2001-01-01.nc
    """
    base = os.path.basename(filename).replace(".nc", "")
    # 预期格式: P_ET_2001-01-01
    parts = base.split("_")
    date_str = parts[-1]
    return datetime.strptime(date_str, "%Y-%m-%d")


def read_reference_grid():
    """
    从第一个有效文件读取参考网格
    """
    for year in calc_years:
        year_dir = os.path.join(data_dir, str(year))
        if not os.path.isdir(year_dir):
            continue

        files = sorted([f for f in os.listdir(year_dir) if f.endswith(".nc")])
        if len(files) == 0:
            continue

        first_file = os.path.join(year_dir, files[0])

        with xr.open_dataset(first_file) as ref_ds:
            ref_ds = ref_ds.sortby(['lat', 'lon'], ascending=True)
            lat_coords = ref_ds.lat.values
            lon_coords = ref_ds.lon.values

        return lat_coords, lon_coords

    raise FileNotFoundError("未在输入目录中找到任何可用的 P-ET nc 文件。")


def load_crop_mask(mask_path, lat_coords, lon_coords):
    """
    读取并对齐作物掩膜
    """
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
    lat_crop = lat_coords[rmin:rmax + 1]
    lon_crop = lon_coords[cmin:cmax + 1]

    ny, nx = mask_np.shape
    mask_flat = mask_np.flatten()
    valid_idx = np.where(mask_flat > 0)[0]

    return mask_np, lat_crop, lon_crop, rmin, rmax, cmin, cmax, ny, nx, valid_idx


# ===============================
# 4. 主程序
# ===============================
all_start = time.time()

print("\n")
print_divider("=")
print("🚀 开始计算标准化 SPET（基于 P-ET）")
print(f"📂 输入目录: {data_dir}")
print(f"📂 输出目录: {output_root}")
print_divider("=")

# 参考网格
lat_coords_full, lon_coords_full = read_reference_grid()

for crop_name, mask_path in crop_dict.items():

    crop_start = time.time()
    print("\n")
    print_divider("=")
    print(f"🌾 开始处理作物: {crop_name}")
    print(f"📂 掩膜文件: {mask_path}")
    print_divider("=")

    # ===============================
    # Step 0: 网格 & 掩膜
    # ===============================
    print("📌 Step 0/5: 读取参考网格并对齐掩膜 ...")

    (
        mask_np, lat_coords, lon_coords,
        rmin, rmax, cmin, cmax,
        ny, nx, valid_idx
    ) = load_crop_mask(mask_path, lat_coords_full, lon_coords_full)

    print(f"✅ 作物像元数: {len(valid_idx)}")
    print(f"✅ 裁剪后网格尺寸: {ny} × {nx}")

    # ===============================
    # 循环尺度
    # ===============================
    for scale_name, SCALE in scale_dict.items():

        scale_start = time.time()

        print("\n")
        print_divider("-")
        print(f"🚀 SPET计算: {crop_name} - {scale_name} (窗口={SCALE}天)")
        print_divider("-")

        window = deque(maxlen=SCALE)
        doy_data = {i: [] for i in range(1, 366)}
        date_list = []

        # ===============================
        # Step 1: 滑动累计
        # ===============================
        print("📌 Step 1/5: 构建滑动累计序列 ...")

        for year in calc_years:
            year_path = os.path.join(data_dir, str(year))

            if not os.path.isdir(year_path):
                print(f"  ⚠️ 年份目录不存在，跳过: {year_path}")
                continue

            files = sorted([f for f in os.listdir(year_path) if f.endswith(".nc")])
            print(f"  📅 正在处理年份: {year}，文件数: {len(files)}")

            for f in files:
                file_path = os.path.join(year_path, f)

                try:
                    with xr.open_dataset(file_path) as ds:
                        ds = ds.sortby(['lat', 'lon'], ascending=True)

                        var_name = guess_var_name(ds)

                        data = ds[var_name].values[
                            rmin:rmax + 1, cmin:cmax + 1
                        ].astype(np.float32)

                        # 若有缺失值编码，转成 nan
                        fill_values = [-999, -9999, 9999, 32767, -32768, 1e20, -1e20]
                        for fv in fill_values:
                            data[data == fv] = np.nan

                except Exception as e:
                    print(f"    ⚠️ 读取失败，跳过: {f} | {e}")
                    continue

                window.append(data)

                if len(window) < SCALE:
                    continue

                wb_acc = np.sum(window, axis=0, dtype=np.float32)

                try:
                    dt = parse_date_from_filename(f)
                except Exception as e:
                    print(f"    ⚠️ 日期解析失败，跳过: {f} | {e}")
                    continue

                # 删除 2 月 29 日
                if dt.month == 2 and dt.day == 29:
                    continue

                doy = dt.timetuple().tm_yday
                if calendar.isleap(dt.year) and dt.month > 2:
                    doy -= 1

                if dt.year >= output_start_year:
                    doy_data[doy].append(wb_acc)
                    date_list.append((dt, doy))

        print("✅ Step 1 完成")

        # ===============================
        # Step 2: 按 DOY 分组计算 SPET
        # ===============================
        print("\n📌 Step 2/5: 按 DOY 分组计算 SPET ...")

        spet_results_by_doy = {}
        valid_doys = [d for d in range(1, 366) if len(doy_data[d]) > 0]
        doy_progress_step = get_progress_step(len(valid_doys))

        for doy_i, doy in enumerate(valid_doys, 1):
            stack = np.stack(doy_data[doy], axis=0).astype(np.float32)
            t = stack.shape[0]

            if doy_i == 1 or doy_i % doy_progress_step == 0:
                print(f"  ... DOY={doy:03d} 样本数={t}")

            reshaped = stack.reshape(t, -1)
            spet_matrix = np.full((t, ny * nx), np.nan, dtype=np.float32)

            crop_pixels = reshaped[:, valid_idx]

            for i in range(len(valid_idx)):
                spet_matrix[:, valid_idx[i]] = gringorten_spet(crop_pixels[:, i])

            spet_results_by_doy[doy] = spet_matrix.reshape(t, ny, nx)

            del stack, reshaped, spet_matrix, crop_pixels
            gc.collect()

        print("✅ Step 2 完成")

        # ===============================
        # Step 3: 重组时间序列
        # ===============================
        print("\n📌 Step 3/5: 重组时间序列 ...")

        final_time = [d[0] for d in date_list]
        final_spet = np.full((len(final_time), ny, nx), np.nan, dtype=np.float32)

        doy_counter = {i: 0 for i in range(1, 366)}

        for i, (dt, doy) in enumerate(date_list):
            final_spet[i] = spet_results_by_doy[doy][doy_counter[doy]]
            doy_counter[doy] += 1

        print(f"✅ 最终数组形状: {final_spet.shape}")

        # ===============================
        # Step 4: 随机绘图检查
        # ===============================
        print("\n📌 Step 4/5: 随机绘图检查 ...")

        if len(final_time) > 0:
            idx = random.randint(0, len(final_time) - 1)

            plt.figure(figsize=(10, 6))
            plt.imshow(
                final_spet[idx],
                cmap='RdBu_r',
                vmin=plot_vmin,
                vmax=plot_vmax,
                origin='lower'
            )
            plt.colorbar(label="SPET")
            plt.title(f"{crop_name}-{scale_name}\n{final_time[idx]}")
            plt.tight_layout()
            plt.show()

        # ===============================
        # Step 5: 保存
        # ===============================
        print("\n📌 Step 5/5: 保存文件 ...")

        ds_out = xr.Dataset(
            {"SPET": (["time", "lat", "lon"], final_spet)},
            coords={"time": final_time, "lat": lat_coords, "lon": lon_coords}
        )

        encoding = {
            "SPET": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "_FillValue": np.nan
            }
        }

        out_path = os.path.join(
            output_root,
            f"SPET_{scale_name}_{crop_name}.nc"
        )

        ds_out.to_netcdf(out_path, encoding=encoding)

        print(f"✅ 文件保存完成: {out_path}")
        print(f"⏱️ 本尺度耗时: {(time.time() - scale_start) / 60:.2f} 分钟")

        del ds_out, final_spet, spet_results_by_doy, doy_data, date_list, window
        gc.collect()

    print(f"\n🌾 作物 {crop_name} 全部完成，用时: {(time.time() - crop_start) / 60:.2f} 分钟")

print("\n🎉 全部 SPET 计算完成！")
print(f"⏱️ 总耗时: {(time.time() - all_start) / 3600:.2f} 小时")