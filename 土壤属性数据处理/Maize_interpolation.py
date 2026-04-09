# -*- coding: utf-8 -*-
"""
对玉米作物种植像元中的缺失值进行邻域均值插补
规则：
1. 只在原始作物种植区(mask==1)内处理
2. 仅对种植区中的 NaN 像元做插补
3. 用其周围邻近有效像元的平均值填补
4. 支持 depth 分层逐层处理
5. 支持多轮迭代，直到无法继续填补或达到最大迭代次数
"""

import os
import time
import numpy as np
import xarray as xr


# =============================================================================
# 1. 路径设置
# =============================================================================
NC_DIR = r"E:\data\SoilData\Maize_0p1deg_Masked"

CROP_MASKS = {
    "Spring_Maize": r"E:\data\CropMask\Spring_Maize_01deg.tif",
    "Summer_Maize": r"E:\data\CropMask\Summer_Maize_01deg.tif",
}

OUT_DIR = r"E:\data\SoilData\Maize_0p1deg_Masked_Filled"
os.makedirs(OUT_DIR, exist_ok=True)

# 若只想处理部分文件，可手动指定；否则默认处理目录下全部 nc
TARGET_FILES = None
# TARGET_FILES = [
#     r"E:\data\SoilData\Maize_0p1deg_Masked\TH1500_Spring_Maize_0p1deg_masked.nc",
#     r"E:\data\SoilData\Maize_0p1deg_Masked\TH33_Summer_Maize_0p1deg_masked.nc",
# ]

# 邻域半径
# radius=1 表示 3×3 邻域
# radius=2 表示 5×5 邻域
NEIGHBOR_RADIUS = 1

# 最大迭代次数
MAX_ITER = 20


# =============================================================================
# 2. 工具函数
# =============================================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def standardize_lat_lon_da(da):
    """统一坐标名为 lat/lon，并保证坐标升序。"""
    rename_dict = {}
    for c in da.coords:
        lc = c.lower()
        if lc in ["latitude", "lat", "y"]:
            rename_dict[c] = "lat"
        elif lc in ["longitude", "lon", "x"]:
            rename_dict[c] = "lon"

    if rename_dict:
        da = da.rename(rename_dict)

    if "lat" not in da.coords or "lon" not in da.coords:
        raise ValueError(f"未找到 lat/lon 坐标，当前坐标为: {list(da.coords)}")

    if da["lat"].values[0] > da["lat"].values[-1]:
        da = da.sortby("lat")
    if da["lon"].values[0] > da["lon"].values[-1]:
        da = da.sortby("lon")

    return da


def find_main_var(ds):
    """自动识别主变量。"""
    var_names = list(ds.data_vars)

    for preferred in ["TH1500", "TH33"]:
        if preferred in var_names:
            return preferred

    exclude_keywords = ["cv", "std", "err", "uncertainty", "quality", "flag"]
    main_vars = [v for v in var_names if not any(k in v.lower() for k in exclude_keywords)]

    if len(main_vars) == 1:
        return main_vars[0]
    if len(var_names) == 1:
        return var_names[0]

    raise ValueError(f"无法唯一识别主变量，请检查变量名：{var_names}")


def parse_crop_and_soil_from_filename(nc_path):
    """从文件名中解析 SoilType / CropType。"""
    name = os.path.splitext(os.path.basename(nc_path))[0]

    soil_type = "Unknown"
    crop_type = "Unknown"

    if "TH1500" in name:
        soil_type = "TH1500"
    elif "TH33" in name:
        soil_type = "TH33"

    if "Spring_Maize" in name:
        crop_type = "Spring_Maize"
    elif "Summer_Maize" in name:
        crop_type = "Summer_Maize"

    return soil_type, crop_type


def load_crop_mask(mask_path):
    """读取原始作物掩膜 tif，输出 0/1 的 DataArray(lat, lon)。"""
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"作物掩膜不存在: {mask_path}")

    da = xr.open_dataarray(mask_path, engine="rasterio").squeeze(drop=True)

    rename_dict = {}
    for c in da.coords:
        lc = c.lower()
        if lc == "x":
            rename_dict[c] = "lon"
        elif lc == "y":
            rename_dict[c] = "lat"
    if rename_dict:
        da = da.rename(rename_dict)

    if "band" in da.dims:
        da = da.squeeze("band", drop=True)

    da = standardize_lat_lon_da(da)
    mask = xr.where(np.isfinite(da) & (da > 0), 1, 0)
    mask.name = "crop_mask"
    return mask


def align_mask_to_data(mask_da, data_da):
    """把原始作物掩膜对齐到 nc 数据网格。"""
    mask_da = standardize_lat_lon_da(mask_da)
    data_da = standardize_lat_lon_da(data_da)

    mask_aligned = mask_da.interp(
        lat=data_da.lat,
        lon=data_da.lon,
        method="nearest"
    )

    mask_aligned = xr.where(np.isfinite(mask_aligned) & (mask_aligned > 0), 1, 0)
    mask_aligned.name = "crop_mask_aligned"
    return mask_aligned


def get_output_path(nc_path, out_dir):
    """构建输出路径。"""
    base = os.path.splitext(os.path.basename(nc_path))[0]
    return os.path.join(out_dir, f"{base}_filled.nc")


# =============================================================================
# 3. 核心插补函数
# =============================================================================
def fill_missing_by_neighbor_mean(arr2d, crop_mask2d, radius=1, max_iter=20):
    """
    对二维数组在作物种植区内的缺失值进行邻域均值插补。

    参数
    ----
    arr2d : 2D numpy array
        某一层数据
    crop_mask2d : 2D numpy array (0/1)
        作物种植掩膜
    radius : int
        邻域半径；1=3x3, 2=5x5
    max_iter : int
        最大迭代次数

    返回
    ----
    filled : 2D numpy array
        插补后的数组
    """
    filled = arr2d.copy()
    crop_bool = (crop_mask2d == 1)

    # 只统计种植区内缺失
    initial_missing = int((crop_bool & ~np.isfinite(filled)).sum())
    log(f"初始种植区缺失像元数: {initial_missing}")

    if initial_missing == 0:
        return filled

    nrows, ncols = filled.shape

    for it in range(1, max_iter + 1):
        prev_missing = int((crop_bool & ~np.isfinite(filled)).sum())
        if prev_missing == 0:
            log(f"第 {it-1} 轮后已无缺失")
            break

        new_arr = filled.copy()
        fill_count_this_round = 0

        missing_positions = np.argwhere(crop_bool & ~np.isfinite(filled))

        for r, c in missing_positions:
            r0 = max(0, r - radius)
            r1 = min(nrows, r + radius + 1)
            c0 = max(0, c - radius)
            c1 = min(ncols, c + radius + 1)

            window = filled[r0:r1, c0:c1]
            window_crop = crop_bool[r0:r1, c0:c1]

            # 只用作物区内的有效邻居做平均
            valid_neighbors = window[window_crop & np.isfinite(window)]

            if valid_neighbors.size > 0:
                new_arr[r, c] = valid_neighbors.mean()
                fill_count_this_round += 1

        filled = new_arr
        curr_missing = int((crop_bool & ~np.isfinite(filled)).sum())

        log(
            f"第 {it} 轮插补: "
            f"本轮填补 {fill_count_this_round} 个, "
            f"剩余缺失 {curr_missing} 个"
        )

        # 如果这一轮一个都没填上，说明无法继续
        if fill_count_this_round == 0:
            log("后续无法继续插补，提前停止")
            break

    final_missing = int((crop_bool & ~np.isfinite(filled)).sum())
    log(f"最终种植区缺失像元数: {final_missing}")

    return filled


def fill_one_file(nc_path, out_dir, radius=1, max_iter=20):
    """
    对单个 nc 文件进行缺失值插补，并输出新文件。
    """
    log("=" * 100)
    log(f"开始处理: {nc_path}")

    ds = xr.open_dataset(nc_path)
    var_name = find_main_var(ds)
    da = standardize_lat_lon_da(ds[var_name])

    soil_type, crop_type = parse_crop_and_soil_from_filename(nc_path)
    if crop_type not in CROP_MASKS:
        raise ValueError(f"无法根据文件名识别作物类型: {nc_path}")

    mask_raw = load_crop_mask(CROP_MASKS[crop_type])
    crop_mask = align_mask_to_data(mask_raw, da)

    crop_pixels = int((crop_mask.values == 1).sum())
    log(f"识别变量: {var_name}")
    log(f"作物类型: {crop_type}")
    log(f"种植像元数: {crop_pixels}")

    filled_da = da.copy(deep=True)

    if "depth" in da.dims:
        depth_values = da["depth"].values

        for i, depth_val in enumerate(depth_values):
            log("-" * 80)
            log(f"处理第 {i+1}/{len(depth_values)} 层, depth={depth_val}")

            arr2d = da.isel(depth=i).values
            crop_mask2d = crop_mask.values

            arr2d_filled = fill_missing_by_neighbor_mean(
                arr2d=arr2d,
                crop_mask2d=crop_mask2d,
                radius=radius,
                max_iter=max_iter
            )

            filled_da.values[i, :, :] = arr2d_filled
    else:
        log("处理单层数据")
        arr2d = da.values
        crop_mask2d = crop_mask.values

        arr2d_filled = fill_missing_by_neighbor_mean(
            arr2d=arr2d,
            crop_mask2d=crop_mask2d,
            radius=radius,
            max_iter=max_iter
        )
        filled_da.values[:, :] = arr2d_filled

    # 保存输出
    filled_da = filled_da.astype(np.float32)
    filled_da.name = var_name

    out_path = get_output_path(nc_path, out_dir)
    encoding = {
        var_name: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan)
        }
    }

    filled_da.to_dataset(name=var_name).to_netcdf(out_path, encoding=encoding)
    log(f"已输出: {out_path}")


# =============================================================================
# 4. 主程序
# =============================================================================
def main():
    log("开始对玉米作物种植像元中的缺失值进行邻域均值插补")

    if TARGET_FILES is not None:
        nc_files = TARGET_FILES
    else:
        nc_files = [
            os.path.join(NC_DIR, f)
            for f in os.listdir(NC_DIR)
            if f.lower().endswith(".nc")
        ]

    if len(nc_files) == 0:
        log(f"未找到 nc 文件: {NC_DIR}")
        return

    log(f"共找到 {len(nc_files)} 个 nc 文件")

    for nc_path in nc_files:
        try:
            fill_one_file(
                nc_path=nc_path,
                out_dir=OUT_DIR,
                radius=NEIGHBOR_RADIUS,
                max_iter=MAX_ITER
            )
        except Exception as e:
            log(f"处理失败: {nc_path}")
            log(f"错误信息: {e}")

    print("\n" + "#" * 100)
    log("全部插补完成")
    print("#" * 100)


if __name__ == "__main__":
    main()