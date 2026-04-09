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
etpet_root = r"E:\data\ET_minus_PET_Daily"     # 例如 E:\data\ET_minus_PET_Daily\2001\Diff_2001-01-01.nc
output_root = r"E:\data\SEDI\Wheat"
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

# 2001 作为预热年，只输出 2002-2022
calc_years = range(2001, 2023)
output_start_year = 2002

# ET-PET 文件中变量名
var_name = "Diff"


# ===============================
# 2. SEDI函数（非参数经验分布标准化）
#    这里延续你SPI代码中的Gringorten思想
# ===============================
def gringorten_sedi(values):
    """
    对某一个像元的同一DOY跨年样本序列进行非参数标准化
    values: shape = (n_time,)
    return : shape = (n_time,)
    """
    valid_mask = ~np.isnan(values)
    v = values[valid_mask]

    # 样本太少则不计算
    if len(v) < 15:
        return np.full(len(values), np.nan, dtype=np.float32)

    # Gringorten plotting position
    ranks = pd.Series(v).rank(method='average')
    p = (ranks - 0.44) / (len(v) + 0.12)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    sedi_values = norm.ppf(p)

    result = np.full(len(values), np.nan, dtype=np.float32)
    result[valid_mask] = sedi_values.astype(np.float32)

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


def get_first_nc_file(year_dir):
    files = sorted([f for f in os.listdir(year_dir) if f.endswith(".nc")])
    if len(files) == 0:
        raise FileNotFoundError(f"文件夹下没有nc文件: {year_dir}")
    return os.path.join(year_dir, files[0])


def parse_date_from_filename(filename):
    """
    文件名格式: Diff_2001-01-01.nc
    """
    base = os.path.basename(filename).replace(".nc", "")
    date_str = base.split("_")[-1]
    return datetime.strptime(date_str, "%Y-%m-%d")


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
    # Step 0: 读取参考ET-PET网格并对齐掩膜
    # ===============================
    print("📌 Step 0/5: 读取参考 ET-PET 网格并对齐掩膜 ...")

    first_year_dir = os.path.join(etpet_root, "2001")
    first_file = get_first_nc_file(first_year_dir)

    with xr.open_dataset(first_file) as ref_ds:
        ref_ds = ref_ds.sortby(['lat', 'lon'], ascending=True)
        lat_coords = ref_ds.lat.values
        lon_coords = ref_ds.lon.values

    # 读取作物掩膜
    da_mask = xr.open_dataarray(mask_path, engine="rasterio").squeeze()

    # 重命名维度
    rename_dict = {}
    if 'y' in da_mask.dims:
        rename_dict['y'] = 'lat'
    if 'x' in da_mask.dims:
        rename_dict['x'] = 'lon'
    if rename_dict:
        da_mask = da_mask.rename(rename_dict)

    mask_np = da_mask.values

    # 若掩膜纬度是降序，则翻转
    if da_mask.lat.values[0] > da_mask.lat.values[-1]:
        mask_np = np.flipud(mask_np)

    # 直接赋参考网格坐标（前提是你的ET-PET已与目标网格对齐）
    da_mask = xr.DataArray(
        mask_np,
        coords={"lat": lat_coords, "lon": lon_coords},
        dims=("lat", "lon")
    )

    mask_np = da_mask.values

    # 裁剪到最小包围框，减少输出体积
    rows = np.any(mask_np > 0, axis=1)
    cols = np.any(mask_np > 0, axis=0)

    if not np.any(rows) or not np.any(cols):
        raise ValueError(f"{crop_name} 掩膜中没有有效种植像元")

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    mask_np = mask_np[rmin:rmax + 1, cmin:cmax + 1]
    lat_coords_sub = lat_coords[rmin:rmax + 1]
    lon_coords_sub = lon_coords[cmin:cmax + 1]

    ny, nx = mask_np.shape
    mask_flat = mask_np.flatten()
    valid_idx = np.where(mask_flat > 0)[0]

    print(f"✅ 作物像元数: {len(valid_idx)}")
    print(f"✅ 裁剪后网格尺寸: {ny} × {nx}")
    print(f"✅ 纬度范围: {lat_coords_sub[0]} ~ {lat_coords_sub[-1]}")
    print(f"✅ 经度范围: {lon_coords_sub[0]} ~ {lon_coords_sub[-1]}")

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
        # Step 1: 构建滑动累计 ET-PET 序列
        # ===============================
        print("📌 Step 1/5: 构建滑动累计 ET-PET 序列 ...")

        total_valid_days = 0

        for year in calc_years:
            year_path = os.path.join(etpet_root, str(year))
            if not os.path.exists(year_path):
                print(f"⚠️ 跳过不存在年份目录: {year_path}")
                continue

            files = sorted([f for f in os.listdir(year_path) if f.endswith(".nc")])
            print(f"  📅 正在处理年份: {year}，文件数: {len(files)}")

            for f in files:
                file_path = os.path.join(year_path, f)

                try:
                    dt = parse_date_from_filename(f)
                except:
                    print(f"⚠️ 文件名日期解析失败，跳过: {f}")
                    continue

                # 删除闰年2月29日
                if dt.month == 2 and dt.day == 29:
                    continue

                try:
                    with xr.open_dataset(file_path) as ds:
                        ds = ds.sortby(['lat', 'lon'], ascending=True)

                        if var_name not in ds.data_vars:
                            print(f"⚠️ 文件中未找到变量 {var_name}，跳过: {f}")
                            continue

                        data = ds[var_name].values

                        # 若存在 time 维，则压掉
                        if data.ndim == 3:
                            data = data[0]

                        data = data[rmin:rmax + 1, cmin:cmax + 1].astype(np.float32)

                        # 将非有限值转为NaN
                        data[~np.isfinite(data)] = np.nan

                except Exception as e:
                    print(f"⚠️ 读取失败，跳过: {f} | {e}")
                    continue

                window.append(data)

                if len(window) < SCALE:
                    continue

                # 滑动累计
                acc = np.sum(window, axis=0, dtype=np.float32)

                doy = dt.timetuple().tm_yday
                if calendar.isleap(dt.year) and dt.month > 2:
                    doy -= 1

                if dt.year >= output_start_year:
                    doy_data[doy].append(acc)
                    date_list.append((dt, doy))
                    total_valid_days += 1

            gc.collect()

        print(f"✅ Step 1 完成")
        print(f"✅ 输出时间步数: {total_valid_days}")

        if total_valid_days == 0:
            print(f"⚠️ {crop_name}-{scale_name} 无有效输出，跳过")
            continue

        # ===============================
        # Step 2: 按 DOY 分组计算 SEDI
        # ===============================
        print("\n📌 Step 2/5: 按 DOY 分组计算 SEDI ...")

        sedi_results_by_doy = {}
        valid_doys = [d for d in range(1, 366) if len(doy_data[d]) > 0]

        print(f"✅ 有样本的 DOY 组数: {len(valid_doys)} / 365")

        doy_progress_step = max(1, 30)  # 每隔30个DOY打印一次

        for doy_i, doy in enumerate(valid_doys, 1):

            stack = np.stack(doy_data[doy], axis=0).astype(np.float32)
            t = stack.shape[0]

            if doy_i == 1 or doy_i % doy_progress_step == 0 or doy_i == len(valid_doys):
                print(f"  ... DOY组进度: {doy_i}/{len(valid_doys)} (当前 DOY={doy:03d}, 样本数={t})")

            reshaped = stack.reshape(t, -1)
            sedi_matrix = np.full((t, ny * nx), np.nan, dtype=np.float32)

            crop_pixels = reshaped[:, valid_idx]

            pixel_step = get_progress_step(len(valid_idx), 6)

            for i in range(len(valid_idx)):
                sedi_matrix[:, valid_idx[i]] = gringorten_sedi(crop_pixels[:, i])

                if (i + 1) % pixel_step == 0 or (i + 1) == len(valid_idx):
                    if doy_i == 1 or doy_i % doy_progress_step == 0 or doy_i == len(valid_doys):
                        print(f"    DOY {doy:03d} | 像元进度: {i + 1}/{len(valid_idx)}")

            sedi_results_by_doy[doy] = sedi_matrix.reshape(t, ny, nx)

            del stack, reshaped, sedi_matrix, crop_pixels
            gc.collect()

        print("✅ Step 2 完成")

        # ===============================
        # Step 3: 重组最终时间序列
        # ===============================
        print("\n📌 Step 3/5: 重组最终时间序列 ...")

        final_time = [d[0] for d in date_list]
        final_sedi = np.full((len(final_time), ny, nx), np.nan, dtype=np.float32)

        doy_counter = {i: 0 for i in range(1, 366)}

        for i, (dt, doy) in enumerate(date_list):
            final_sedi[i] = sedi_results_by_doy[doy][doy_counter[doy]]
            doy_counter[doy] += 1

        print(f"✅ 最终数组形状: {final_sedi.shape}")

        # 非作物区域强制设为 NaN
        crop_mask_2d = mask_np > 0
        final_sedi[:, ~crop_mask_2d] = np.nan

        # ===============================
        # Step 4: 随机绘图检查
        # ===============================
        print("\n📌 Step 4/5: 随机绘图检查 ...")

        idx = random.randint(0, len(final_time) - 1)
        rand_date = final_time[idx]

        arr = final_sedi[idx]
        valid_pixels = np.isfinite(arr)

        print("-" * 60)
        print(f"🖼️ 随机时间索引: {idx}")
        print(f"随机日期    : {rand_date.strftime('%Y-%m-%d')}")
        print(f"有效像元数  : {valid_pixels.sum()}")
        if valid_pixels.sum() > 0:
            print(f"最小值      : {np.nanmin(arr):.4f}")
            print(f"最大值      : {np.nanmax(arr):.4f}")
            print(f"均值        : {np.nanmean(arr):.4f}")
        print("-" * 60)

        plt.figure(figsize=(8, 5))
        plt.imshow(arr, cmap='RdBu_r', vmin=-2.5, vmax=2.5, origin='lower')
        plt.colorbar(label="SEDI")
        plt.title(f"{crop_name}-{scale_name}\n{rand_date.strftime('%Y-%m-%d')}")
        plt.tight_layout()
        plt.show()

        # ===============================
        # Step 5: 保存文件
        # ===============================
        print("\n📌 Step 5/5: 保存文件 (压缩 + float32) ...")

        ds_out = xr.Dataset(
            {"SEDI": (["time", "lat", "lon"], final_sedi)},
            coords={"time": final_time, "lat": lat_coords_sub, "lon": lon_coords_sub}
        )

        encoding = {
            "SEDI": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 4,
                "shuffle": True,
                "_FillValue": np.nan
            }
        }

        out_path = os.path.join(output_root, f"SEDI_{scale_name}_{crop_name}.nc")
        ds_out.to_netcdf(out_path, encoding=encoding)

        print(f"✅ 文件保存完成: {out_path}")

        del final_sedi, ds_out, sedi_results_by_doy, doy_data, date_list
        gc.collect()

        print(f"⏱️ 当前尺度耗时: {(time.time() - scale_start)/60:.2f} 分钟")

    print(f"\n✅ {crop_name} 全部完成，用时: {(time.time() - crop_start)/60:.2f} 分钟")

print("\n🎉 全部计算完成")
print(f"总耗时: {(time.time() - all_start)/3600:.2f} 小时")