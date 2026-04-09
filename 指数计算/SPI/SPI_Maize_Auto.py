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
precip_dir = r"E:\data\meteorologicaldata\precipitation"
output_root = r"E:\data\SPI\Maize"
os.makedirs(output_root, exist_ok=True)

# ⚠️ 注意：这里你确认路径是否正确
crop_configs = {
    "Summer_Maize": r"E:\数据集\作物种植文件\0.1度重采样文件\Summer_Maize_01deg.tif",
    "Spring_Maize": r"E:\数据集\作物种植文件\0.1度重采样文件\Spring_Maize_01deg.tif"
}

# 时间尺度
scales = {
    "1month": 30,
    "3month": 90,
    "6month": 180,
    "9month": 270,
    "12month": 365

}

# ===============================
# 2. 工具函数
# ===============================
def gringorten_spi(values):
    valid_mask = ~np.isnan(values)
    v = values[valid_mask]

    if len(v) < 15:
        return np.full(len(values), np.nan, dtype=np.float32)

    ranks = pd.Series(v).rank(method="average").values
    p = (ranks - 0.44) / (len(v) + 0.12)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    spi = norm.ppf(p)

    out = np.full(len(values), np.nan, dtype=np.float32)
    out[valid_mask] = spi.astype(np.float32)
    return out


def get_no_leap_doy(date):
    if date.month == 2 and date.day == 29:
        return None
    doy = date.timetuple().tm_yday
    if calendar.isleap(date.year) and date.month > 2:
        doy -= 1
    return doy


# ===============================
# 3. 主程序
# ===============================
for crop_name, mask_path in crop_configs.items():

    print(f"\n================ {crop_name} ================")

    # ===============================
    # 读取mask + 裁剪空白区域
    # ===============================
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.float32)

    mask = np.flipud(mask)
    mask[mask == 0] = np.nan

    # 👉 裁剪 bounding box（关键优化）
    rows = np.any(~np.isnan(mask), axis=1)
    cols = np.any(~np.isnan(mask), axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    mask = mask[rmin:rmax+1, cmin:cmax+1]

    valid_idx = np.where(~np.isnan(mask.flatten()))[0]

    print(f"✅ 作物像元数: {len(valid_idx)}")
    print(f"✅ 裁剪后尺寸: {mask.shape}")

    # ===============================
    # 循环尺度
    # ===============================
    for scale_name, SCALE in scales.items():

        print(f"\n🚀 开始计算 {crop_name} - {scale_name}")

        window = deque(maxlen=SCALE)
        doy_data = {i: [] for i in range(1, 366)}
        date_list = []

        lat = None
        lon = None

        # ===============================
        # Step1: 构建累计降水
        # ===============================
        for year in range(2001, 2023):

            year_path = os.path.join(precip_dir, str(year))
            if not os.path.exists(year_path):
                continue

            files = sorted(os.listdir(year_path))
            print(f"  📅 {year} 年: {len(files)} files")

            for f in files:
                if not f.endswith(".nc"):
                    continue

                file_path = os.path.join(year_path, f)

                with xr.open_dataset(file_path) as ds:
                    prec = ds["prec"].values.astype(np.float32)

                    if lat is None:
                        lat = ds["lat"].values[rmin:rmax+1]
                        lon = ds["lon"].values[cmin:cmax+1]

                # 裁剪
                prec = prec[rmin:rmax+1, cmin:cmax+1]

                # 时间解析
                date_str = "_".join(f.split("_")[-3:]).replace(".nc", "")
                date = datetime.strptime(date_str, "%Y_%m_%d")

                # 删除2月29日
                if date.month == 2 and date.day == 29:
                    continue

                window.append(prec)

                if len(window) < SCALE:
                    continue

                p_acc = np.sum(window, axis=0, dtype=np.float32)

                doy = get_no_leap_doy(date)
                if doy is None:
                    continue

                doy_data[doy].append(p_acc)

                if date.year >= 2002:
                    date_list.append((date, doy))

        print("✅ 累计降水完成")

        # ===============================
        # Step2: SPI计算
        # ===============================
        print("🚀 开始 SPI 计算...")

        spi_result = {}

        for doy in range(1, 366):

            if len(doy_data[doy]) == 0:
                continue

            stack = np.stack(doy_data[doy], axis=0).astype(np.float32)
            t, h, w = stack.shape

            reshaped = stack.reshape(t, -1)
            spi = np.full_like(reshaped, np.nan, dtype=np.float32)

            crop_pixels = reshaped[:, valid_idx]

            for j in range(len(valid_idx)):
                spi[:, valid_idx[j]] = gringorten_spi(crop_pixels[:, j])

            spi = spi.reshape(t, h, w)
            spi_result[doy] = spi

            doy_data[doy] = None
            gc.collect()

        print("✅ SPI计算完成")

        # ===============================
        # Step3: 重建时间序列
        # ===============================
        spi_daily = []
        counter = {doy: 0 for doy in range(1, 366)}

        for date, doy in date_list:
            spi = spi_result[doy][counter[doy]]
            counter[doy] += 1
            spi_daily.append(spi)

        spi_daily = np.stack(spi_daily).astype(np.float32)
        spi_daily[np.isinf(spi_daily)] = np.nan

        print(f"✅ 时间长度: {len(spi_daily)}")

        # ===============================
        # Step4: 随机绘图（sanity check）
        # ===============================
        idx = random.randint(0, len(spi_daily) - 1)
        plt.figure(figsize=(10, 6))

        plt.imshow(
            spi_daily[idx],
            origin="lower",
            cmap="RdBu",
            vmin=-2,
            vmax=2,
            extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        )

        plt.colorbar(label="SPI")
        plt.title(f"{crop_name}-{scale_name}\n{date_list[idx][0]}")
        plt.tight_layout()
        plt.show()

        print("✅ 随机绘图完成")

        # ===============================
        # Step5: 保存（float32压缩优化版）
        # ===============================
        print("💾 正在保存...")

        spi_to_save = np.where(np.isnan(spi_daily), -9999.0, spi_daily)

        ds_out = xr.Dataset(
            {"SPI": (["time", "lat", "lon"], spi_to_save)},
            coords={
                "time": [d[0] for d in date_list],
                "lat": lat,
                "lon": lon
            }
        )

        encoding = {
            "SPI": {
                "zlib": True,
                "complevel": 6,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": np.float32(-9999.0),
                "chunksizes": (50, 100, 100)
            }
        }

        out_path = os.path.join(
            output_root,
            f"SPI_{scale_name}_{crop_name}.nc"
        )

        ds_out.to_netcdf(out_path, encoding=encoding)

        print(f"🎉 完成: {out_path}")

        # 清理内存
        del spi_daily
        gc.collect()