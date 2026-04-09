# -*- coding: utf-8 -*-
"""
计算春玉米 / 夏玉米种植区每日 SPET 指数
基于每日 P-ET 数据，采用 Gringorten plotting position 标准化
输出 1/3/6/9/12 month 五个尺度，2001 为预热年，输出 2002-2022

输入:
  E:\data\P_minus_ET_Daily\YYYY\P_ET_YYYY-MM-DD.nc

输出:
  E:\data\SPET\Maize\SPET_1month_Summer_Maize.nc
  E:\data\SPET\Maize\SPET_1month_Spring_Maize.nc
  ...

说明:
1. 参考 SPI 代码框架，将输入由 precip 替换为 P_minus_ET
2. 删除 2 月 29 日，保证各年 DOY 一致
3. 只对作物掩膜范围内像元做标准化
4. 保存为压缩 NetCDF
"""

import os
import gc
import random
import calendar
from datetime import datetime
from collections import deque

import xarray as xr
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from scipy.stats import norm

# ===============================
# 1. 路径配置
# ===============================
pet_diff_dir = r"E:\data\P_minus_ET_Daily"   # 这里是你第二个代码生成的 P-ET 数据目录
output_root = r"E:\data\SPET\Maize"
os.makedirs(output_root, exist_ok=True)

crop_configs = {
    "Summer_Maize": r"E:\数据集\作物种植文件\0.1度重采样文件\Summer_Maize_01deg.tif",
    "Spring_Maize": r"E:\数据集\作物种植文件\0.1度重采样文件\Spring_Maize_01deg.tif"
}

# 时间尺度（按日滑动）
scales = {
    "1month": 30,
    "3month": 90,
    "6month": 180,
    "9month": 270,
    "12month": 365
}

START_YEAR = 2001   # 预热年
END_YEAR   = 2022   # 输出到 2022
PLOT_CHECK = True   # 是否随机绘图检查


# ===============================
# 2. 工具函数
# ===============================
def gringorten_standardize(values):
    """
    对一个像元的时间序列做 Gringorten plotting position 标准化
    values: 1D array
    """
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
    """
    去掉闰年2月29日后，返回对应的 DOY（1~365）
    """
    if date.month == 2 and date.day == 29:
        return None

    doy = date.timetuple().tm_yday
    if calendar.isleap(date.year) and date.month > 2:
        doy -= 1
    return doy


def replace_invalid_with_nan(arr):
    """
    将常见无效值替换为 NaN
    """
    arr = arr.astype(np.float32, copy=True)
    invalid_values = [-999, -9999, 9999, 32767, -32768, 1e20, -1e20]
    for v in invalid_values:
        arr[np.isclose(arr, v)] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr


def guess_main_var(ds):
    """
    自动识别数据变量名
    优先找 P_minus_ET
    """
    preferred = ["P_minus_ET", "p_minus_et", "P_ET", "pet", "diff"]
    data_vars = list(ds.data_vars)

    for name in preferred:
        if name in data_vars:
            return name

    if len(data_vars) == 0:
        raise ValueError("未找到数据变量。")

    return data_vars[0]


# ===============================
# 3. 主程序
# ===============================
for crop_name, mask_path in crop_configs.items():

    print("\n" + "=" * 80)
    print(f"开始处理作物: {crop_name}")
    print("=" * 80)

    # ===============================
    # 3.1 读取作物掩膜，并裁剪到有效区域
    # ===============================
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.float32)

    # 与你原 SPI 代码保持一致
    mask = np.flipud(mask)
    mask[mask == 0] = np.nan

    rows = np.any(~np.isnan(mask), axis=1)
    cols = np.any(~np.isnan(mask), axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    mask = mask[rmin:rmax + 1, cmin:cmax + 1]
    valid_idx = np.where(~np.isnan(mask.flatten()))[0]

    print(f"✅ 作物有效像元数: {len(valid_idx)}")
    print(f"✅ 裁剪后网格尺寸: {mask.shape}")

    # ===============================
    # 3.2 按尺度计算
    # ===============================
    for scale_name, SCALE in scales.items():

        print("\n" + "-" * 80)
        print(f"🚀 开始计算 {crop_name} - {scale_name} SPET")
        print("-" * 80)

        window = deque(maxlen=SCALE)
        doy_data = {i: [] for i in range(1, 366)}
        date_list = []

        lat = None
        lon = None

        # ===============================
        # Step 1: 构建累计 P-ET
        # ===============================
        for year in range(START_YEAR, END_YEAR + 1):

            year_dir = os.path.join(pet_diff_dir, str(year))
            if not os.path.exists(year_dir):
                print(f"⚠️ 跳过年份 {year}，目录不存在: {year_dir}")
                continue

            files = sorted([f for f in os.listdir(year_dir) if f.endswith(".nc")])
            print(f"  📅 {year} 年文件数: {len(files)}")

            for f in files:
                file_path = os.path.join(year_dir, f)

                with xr.open_dataset(file_path) as ds:
                    var_name = guess_main_var(ds)
                    data = ds[var_name].values.astype(np.float32)

                    if lat is None:
                        lat = ds["lat"].values[rmin:rmax + 1]
                        lon = ds["lon"].values[cmin:cmax + 1]

                # 无效值处理
                data = replace_invalid_with_nan(data)

                # 裁剪到作物区域 bounding box
                data = data[rmin:rmax + 1, cmin:cmax + 1]

                # 从文件名解析日期
                # P_ET_2001-01-01.nc
                date_str = f.replace("P_ET_", "").replace(".nc", "")
                date = datetime.strptime(date_str, "%Y-%m-%d")

                # 删除 2 月 29 日
                if date.month == 2 and date.day == 29:
                    continue

                window.append(data)

                if len(window) < SCALE:
                    continue

                acc = np.sum(window, axis=0, dtype=np.float32)

                doy = get_no_leap_doy(date)
                if doy is None:
                    continue

                doy_data[doy].append(acc)

                # 只输出 2002-2022
                if date.year >= 2002:
                    date_list.append((date, doy))

            gc.collect()

        print("✅ 累计 P-ET 完成")

        # ===============================
        # Step 2: 按 DOY 做标准化，得到 SPET
        # ===============================
        print("🚀 开始 SPET 标准化...")

        spet_result = {}

        for doy in range(1, 366):
            if len(doy_data[doy]) == 0:
                continue

            stack = np.stack(doy_data[doy], axis=0).astype(np.float32)  # [t, h, w]
            t, h, w = stack.shape

            reshaped = stack.reshape(t, -1)
            spet = np.full_like(reshaped, np.nan, dtype=np.float32)

            crop_pixels = reshaped[:, valid_idx]

            for j in range(len(valid_idx)):
                spet[:, valid_idx[j]] = gringorten_standardize(crop_pixels[:, j])

            spet = spet.reshape(t, h, w)
            spet_result[doy] = spet

            # 及时释放
            doy_data[doy] = None
            gc.collect()

        print("✅ SPET 标准化完成")

        # ===============================
        # Step 3: 重建 2002-2022 日序列
        # ===============================
        print("🚀 开始重建时间序列...")

        spet_daily = []
        counter = {doy: 0 for doy in range(1, 366)}

        for date, doy in date_list:
            spet = spet_result[doy][counter[doy]]
            counter[doy] += 1
            spet_daily.append(spet)

        spet_daily = np.stack(spet_daily).astype(np.float32)
        spet_daily[np.isinf(spet_daily)] = np.nan

        print(f"✅ 时间序列长度: {len(spet_daily)}")

        # ===============================
        # Step 4: 随机绘图检查
        # ===============================
        if PLOT_CHECK and len(spet_daily) > 0:
            idx = random.randint(0, len(spet_daily) - 1)

            plt.figure(figsize=(10, 6))
            plt.imshow(
                spet_daily[idx],
                origin="lower",
                cmap="RdBu",
                vmin=-2,
                vmax=2,
                extent=[lon.min(), lon.max(), lat.min(), lat.max()]
            )
            plt.colorbar(label="SPET")
            plt.title(f"{crop_name} - {scale_name}\n{date_list[idx][0].strftime('%Y-%m-%d')}")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.tight_layout()
            plt.show()

            print("✅ 随机绘图检查完成")

        # ===============================
        # Step 5: 保存 NetCDF
        # ===============================
        print("💾 正在保存 NetCDF...")

        spet_to_save = np.where(np.isnan(spet_daily), -9999.0, spet_daily)

        ds_out = xr.Dataset(
            {
                "SPET": (["time", "lat", "lon"], spet_to_save)
            },
            coords={
                "time": [d[0] for d in date_list],
                "lat": lat,
                "lon": lon
            }
        )

        ds_out["SPET"].attrs = {
            "long_name": "Standardized Precipitation minus Evapotranspiration Index",
            "description": f"SPET computed from accumulated daily P-ET at {scale_name} scale using Gringorten plotting position",
            "units": "standardized z-score"
        }

        ds_out.attrs = {
            "title": f"{crop_name} {scale_name} SPET",
            "crop": crop_name,
            "scale": scale_name,
            "time_range": "2002-2022",
            "source_data": "Daily P-ET"
        }

        encoding = {
            "SPET": {
                "zlib": True,
                "complevel": 6,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": np.float32(-9999.0),
                "chunksizes": (50, 100, 100)
            }
        }

        out_path = os.path.join(output_root, f"SPET_{scale_name}_{crop_name}.nc")
        ds_out.to_netcdf(out_path, encoding=encoding)

        print(f"🎉 保存完成: {out_path}")

        # 清理内存
        del spet_daily, ds_out, spet_result
        gc.collect()

print("\n" + "=" * 80)
print("✅ 全部作物、全部尺度 SPET 计算完成")
print("=" * 80)