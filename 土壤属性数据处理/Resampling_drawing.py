# -*- coding: utf-8 -*-
"""
在 VSCode 中绘制重采样后的小麦/玉米土壤文件
功能：
1. 自动读取小麦和玉米重采样后的 nc 文件
2. 每个文件随机抽取一个深度层
3. 绘制二维空间分布图
"""

import os
import random
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# =============================================================================
# 1. 路径设置
# =============================================================================
WHEAT_DIR = r"E:\data\SoilData\Wheat_0p1deg_Masked_Filled_Resampled"
MAIZE_DIR = r"E:\data\SoilData\Maize_0p1deg_Masked_Filled_Resampled"

# 也可以只画某几个文件，默认画目录下全部 *_resampled_4depths.nc
TARGET_FILES = None
# TARGET_FILES = [
#     r"E:\data\SoilData\Wheat_0p1deg_Masked_Filled_Resampled\TH1500_Winter_Wheat_0p1deg_masked_filled_resampled_4depths.nc",
#     r"E:\data\SoilData\Maize_0p1deg_Masked_Filled_Resampled\TH33_Spring_Maize_0p1deg_masked_filled_resampled_4depths.nc",
# ]


# =============================================================================
# 2. 绘图参数
# =============================================================================
RANDOM_SEED = 42   # 固定随机种子，保证每次抽到的层一致；不想固定可设为 None
FIGSIZE = (8, 6)
CMAP = "viridis"


# =============================================================================
# 3. 工具函数
# =============================================================================
def log(msg):
    print(msg, flush=True)


def set_matplotlib_font():
    """
    尽量解决 VSCode/Windows 下中文显示问题。
    若你不需要中文，可删掉。
    """
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def find_main_var(ds):
    """
    自动识别主变量
    """
    var_names = list(ds.data_vars)

    for preferred in ["TH1500", "TH33"]:
        if preferred in var_names:
            return preferred

    if len(var_names) == 1:
        return var_names[0]

    raise ValueError(f"无法唯一识别主变量，请检查变量名: {var_names}")


def standardize_lat_lon_da(da):
    """
    统一坐标名为 lat/lon，并保证升序
    """
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


def collect_nc_files():
    """
    收集小麦和玉米重采样后的 nc 文件
    """
    if TARGET_FILES is not None:
        return TARGET_FILES

    nc_files = []

    for folder in [WHEAT_DIR, MAIZE_DIR]:
        if not os.path.exists(folder):
            log(f"目录不存在，跳过: {folder}")
            continue

        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith("_resampled_4depths.nc")
        ]
        nc_files.extend(files)

    return sorted(nc_files)


def plot_random_depth_from_file(nc_path):
    """
    从单个 nc 文件中随机抽取一个深度层并绘图
    """
    log("=" * 100)
    log(f"处理文件: {nc_path}")

    ds = xr.open_dataset(nc_path)
    var_name = find_main_var(ds)
    da = standardize_lat_lon_da(ds[var_name])

    if "depth" not in da.dims:
        raise ValueError(f"文件不包含 depth 维度: {nc_path}")

    depth_values = list(da["depth"].values)
    chosen_depth = random.choice(depth_values)

    log(f"变量名: {var_name}")
    log(f"可选深度层: {depth_values}")
    log(f"随机抽中深度层: {chosen_depth}")

    da2d = da.sel(depth=chosen_depth)

    arr = da2d.values
    valid_count = int(np.isfinite(arr).sum())
    nan_count = int(np.isnan(arr).sum())

    log(f"二维图 shape: {arr.shape}")
    log(f"有效像元数: {valid_count}")
    log(f"缺失像元数: {nan_count}")

    plt.figure(figsize=FIGSIZE)

    im = plt.imshow(
        arr,
        origin="lower",
        cmap=CMAP
    )

    plt.colorbar(im, label=var_name)

    title = (
        f"{os.path.basename(nc_path)}\n"
        f"Variable: {var_name} | Random depth: {chosen_depth}"
    )
    plt.title(title, fontsize=10)
    plt.xlabel("Column index")
    plt.ylabel("Row index")
    plt.tight_layout()
    plt.show()


# =============================================================================
# 4. 主程序
# =============================================================================
def main():
    set_matplotlib_font()

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    nc_files = collect_nc_files()

    if len(nc_files) == 0:
        log("未找到任何重采样后的 nc 文件，请检查目录。")
        return

    log(f"共找到 {len(nc_files)} 个重采样文件。")

    for nc_path in nc_files:
        try:
            plot_random_depth_from_file(nc_path)
        except Exception as e:
            log(f"绘图失败: {nc_path}")
            log(f"错误信息: {e}")

    log("全部绘图完成。")


if __name__ == "__main__":
    main()