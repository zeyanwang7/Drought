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
diff_root = r"E:\data\ET_minus_PET_Daily"
output_root = r"E:\data\SEDI\Maize"
os.makedirs(output_root, exist_ok=True)

crop_configs = {
    "Summer_Maize": r"E:\数据集\作物种植文件\0.1度重采样文件\Summer_Maize_01deg.tif",
    "Spring_Maize": r"E:\数据集\作物种植文件\0.1度重采样文件\Spring_Maize_01deg.tif"
}

# 时间尺度
scales = {
    "1month": 30,
    "3month": 90,
    "6month": 180,
    # "9month": 270,
    "12month": 365

}

# 预热年 + 输出年份
start_year = 2001   # 预热年
output_start_year = 2002
end_year = 2022

sample_file = os.path.join(diff_root, "2001", "Diff_2001-01-01.nc")


# ===============================
# 2. 工具函数
# ===============================
def gringorten_sedi(values):
    """
    非参数标准化，与SPI相同思路：
    对每个DOY组的时间序列做经验分布 → 正态分位映射
    """
    valid_mask = ~np.isnan(values)
    v = values[valid_mask]

    if len(v) < 15:
        return np.full(len(values), np.nan, dtype=np.float32)

    ranks = pd.Series(v).rank(method="average").values
    p = (ranks - 0.44) / (len(v) + 0.12)
    p = np.clip(p, 1e-6, 1 - 1e-6)

    sedi = norm.ppf(p)

    out = np.full(len(values), np.nan, dtype=np.float32)
    out[valid_mask] = sedi.astype(np.float32)
    return out


def get_no_leap_doy(date):
    """
    删除2月29日后，对应的365天DOY
    """
    if date.month == 2 and date.day == 29:
        return None
    doy = date.timetuple().tm_yday
    if calendar.isleap(date.year) and date.month > 2:
        doy -= 1
    return doy


def parse_date_from_filename(filename):
    """
    Diff_2001-01-01.nc -> datetime
    """
    date_str = filename.replace("Diff_", "").replace(".nc", "")
    return datetime.strptime(date_str, "%Y-%m-%d")


def read_mask_and_build_target_grid(mask_path):
    """
    读取mask，统一翻转为纬度递增；
    然后裁剪到作物区域bounding box；
    返回：
        mask_crop, target_lat_crop, target_lon_crop, valid_idx
    """
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.float32)
        transform = src.transform
        height, width = mask.shape

        lon = np.array([transform.c + (i + 0.5) * transform.a for i in range(width)], dtype=np.float64)
        lat = np.array([transform.f + (j + 0.5) * transform.e for j in range(height)], dtype=np.float64)

    print(f"标准shape: {mask.shape}")

    # 如果mask纬度递减，则翻转到递增
    if lat[1] < lat[0]:
        mask = np.flipud(mask)
        lat = lat[::-1]
        print("⚠️ mask降序 → flip")

    mask[mask == 0] = np.nan

    # 裁剪作物有效区 bounding box
    rows = np.any(~np.isnan(mask), axis=1)
    cols = np.any(~np.isnan(mask), axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    mask_crop = mask[rmin:rmax + 1, cmin:cmax + 1]
    lat_crop = lat[rmin:rmax + 1]
    lon_crop = lon[cmin:cmax + 1]

    valid_idx = np.where(~np.isnan(mask_crop.flatten()))[0]

    print(f"裁剪后shape: {mask_crop.shape}")
    print(f"lat范围: {lat_crop.min()} ~ {lat_crop.max()}")
    print(f"lon范围: {lon_crop.min()} ~ {lon_crop.max()}")
    print(f"有效作物像元数: {len(valid_idx)}")

    return mask_crop, lat_crop, lon_crop, valid_idx


def read_diff_aligned(file_path, target_lat, target_lon):
    """
    读取单日 Diff，并插值到目标 mask 网格
    强制返回二维数组: (lat, lon)
    """
    with xr.open_dataset(file_path) as ds:
        if "Diff" not in ds.data_vars:
            raise ValueError(f"文件中未找到变量 'Diff': {file_path}")

        da = ds["Diff"]

        # 如果存在长度为1的time维，去掉
        da = da.squeeze(drop=True)

        # 如果还有别的多余维度，也尽量压缩
        da = da.squeeze()

        # 保证纬度升序，避免interp时方向不一致
        if "lat" in da.dims and da.lat.size > 1:
            if da.lat.values[1] < da.lat.values[0]:
                da = da.sortby("lat")

        # 插值到目标网格
        da = da.interp(
            lat=target_lat,
            lon=target_lon,
            method="nearest"
        )

        # 再次压缩，确保只剩(lat, lon)
        da = da.squeeze(drop=True)

        # 若维度顺序不是(lat, lon)，强制调整
        expected_dims = [d for d in ["lat", "lon"] if d in da.dims]
        if tuple(da.dims) != tuple(expected_dims):
            da = da.transpose("lat", "lon")

        arr = da.values.astype(np.float32)

        # 最终强制检查
        if arr.ndim != 2:
            raise ValueError(
                f"读取后的 Diff 不是二维数组，而是 {arr.ndim} 维，shape={arr.shape}，文件={file_path}"
            )

    return arr


def random_plot_one_day(data_3d, lat, lon, date_list, crop_name, scale_name):
    """
    随机抽取一天绘图检查
    """
    if len(data_3d) == 0:
        print("⚠️ 无可绘图数据")
        return

    idx = random.randint(0, len(data_3d) - 1)

    plt.figure(figsize=(10, 6))
    plt.imshow(
        data_3d[idx],
        origin="lower",
        cmap="RdBu",
        vmin=-2,
        vmax=2,
        extent=[lon.min(), lon.max(), lat.min(), lat.max()]
    )
    plt.colorbar(label="SEDI")
    plt.title(f"{crop_name}-{scale_name}\n{date_list[idx][0].strftime('%Y-%m-%d')}")
    plt.tight_layout()
    plt.show()

    print("✅ 随机绘图完成")


# ===============================
# 3. 主程序
# ===============================
for crop_name, mask_path in crop_configs.items():

    print(f"\n================ {crop_name} ================")

    # 读取作物mask并生成目标网格
    mask, target_lat, target_lon, valid_idx = read_mask_and_build_target_grid(mask_path)

    # 循环尺度
    for scale_name, SCALE in scales.items():

        print(f"\n🚀 开始计算 {crop_name} - {scale_name}")

        window = deque(maxlen=SCALE)
        doy_data = {i: [] for i in range(1, 366)}
        date_list = []

        file_count = 0
        valid_day_count = 0

        # ===============================
        # Step1: 构建累计 ET-PET
        # ===============================
        for year in range(start_year, end_year + 1):

            year_path = os.path.join(diff_root, str(year))
            if not os.path.exists(year_path):
                print(f"⚠️ 跳过不存在年份文件夹: {year_path}")
                continue

            files = sorted([f for f in os.listdir(year_path) if f.endswith(".nc")])
            print(f"  📅 {year} 年: {len(files)} files")

            for f in files:
                file_count += 1
                file_path = os.path.join(year_path, f)

                # 解析时间
                date = parse_date_from_filename(f)

                # 删除2月29日
                if date.month == 2 and date.day == 29:
                    continue

                # 读取并对齐到mask网格
                diff = read_diff_aligned(file_path, target_lat, target_lon)

                window.append(diff)

                if len(window) < SCALE:
                    continue

                # 滑动累计 ET-PET
                d_acc = np.sum(window, axis=0, dtype=np.float32)

                doy = get_no_leap_doy(date)
                if doy is None:
                    continue

                doy_data[doy].append(d_acc)

                # 只输出2002-2022
                if date.year >= output_start_year:
                    date_list.append((date, doy))
                    valid_day_count += 1

            print(f"    ✅ {year} 年处理完成")

        print("✅ 累计 ET-PET 完成")
        print(f"总读取文件数: {file_count}")
        print(f"有效输出日数: {valid_day_count}")

        # ===============================
        # Step2: SEDI计算
        # ===============================
        print("🚀 开始 SEDI 计算...")

        sedi_result = {}

        for doy in range(1, 366):

            if len(doy_data[doy]) == 0:
                continue

            stack = np.stack(doy_data[doy], axis=0).astype(np.float32)
            t, h, w = stack.shape

            reshaped = stack.reshape(t, -1)
            sedi = np.full_like(reshaped, np.nan, dtype=np.float32)

            crop_pixels = reshaped[:, valid_idx]

            # 只有每隔50个DOY，才显示该DOY内部的像元进度
            show_pixel_progress = (doy % 50 == 0 or doy == 365)

            for j in range(len(valid_idx)):
                sedi[:, valid_idx[j]] = gringorten_sedi(crop_pixels[:, j])

                if show_pixel_progress and ((j + 1) % 5000 == 0 or (j + 1) == len(valid_idx)):
                    print(f"    DOY {doy:03d} | 像元进度: {j + 1}/{len(valid_idx)}")

            sedi = sedi.reshape(t, h, w)
            sedi_result[doy] = sedi

            doy_data[doy] = None
            gc.collect()

            if doy % 50 == 0 or doy == 365:
                print(f"  ... DOY组进度: {doy}/365")

        print("✅ SEDI计算完成")

        # ===============================
        # Step3: 重建时间序列
        # ===============================
        print("🚀 正在重建时间序列...")

        sedi_daily = []
        counter = {doy: 0 for doy in range(1, 366)}

        for date, doy in date_list:
            sedi = sedi_result[doy][counter[doy]]
            counter[doy] += 1
            sedi_daily.append(sedi)

        sedi_daily = np.stack(sedi_daily).astype(np.float32)
        sedi_daily[np.isinf(sedi_daily)] = np.nan

        print(f"✅ 时间长度: {len(sedi_daily)}")

        # ===============================
        # Step4: 随机绘图（sanity check）
        # ===============================
        random_plot_one_day(
            sedi_daily,
            target_lat,
            target_lon,
            date_list,
            crop_name,
            scale_name
        )

        # ===============================
        # Step5: 保存（float32压缩优化版）
        # ===============================
        print("💾 正在保存...")

        sedi_to_save = np.where(np.isnan(sedi_daily), -9999.0, sedi_daily).astype(np.float32)

        ds_out = xr.Dataset(
            {"SEDI": (["time", "lat", "lon"], sedi_to_save)},
            coords={
                "time": [d[0] for d in date_list],
                "lat": target_lat,
                "lon": target_lon
            }
        )

        encoding = {
            "SEDI": {
                "zlib": True,
                "complevel": 6,
                "shuffle": True,
                "dtype": "float32",
                "_FillValue": np.float32(-9999.0),
                "chunksizes": (50, min(100, len(target_lat)), min(100, len(target_lon)))
            }
        }

        out_path = os.path.join(
            output_root,
            f"SEDI_{scale_name}_{crop_name}.nc"
        )

        ds_out.to_netcdf(out_path, encoding=encoding)

        print(f"🎉 完成: {out_path}")

        # ===============================
        # Step6: 清理内存
        # ===============================
        del window
        del doy_data
        del sedi_result
        del sedi_daily
        del ds_out
        gc.collect()

print("\n✅ 全部计算完成")