# -*- coding: utf-8 -*-
"""
在 VSCode / PyCharm 中绘制冬春小麦、春夏玉米四个深度层的 SMCI 空间分布
—— 自动扫描实际存在的 nc 文件，并随机抽取一天绘图

绘图布局：
- 行：作物（Winter_Wheat, Spring_Wheat, Spring_Maize, Summer_Maize）
- 列：深度层（0_10cm, 10_40cm, 40_80cm, 0_80cm）

功能：
1. 自动扫描各作物、各深度层目录下实际存在的年度 nc 文件
2. 从第一个可用文件中随机抽取一天
3. 用该随机日期绘制 4 类作物 × 4 深度层 的 SMCI 空间图
4. 若某文件不存在或该日无数据，则子图显示 No Data
"""

import os
import glob
import random
import re
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# =============================================================================
# 1. 参数设置
# =============================================================================
# 是否固定随机种子；设为 None 则每次运行随机不同
RANDOM_SEED = 42

# 四类作物目录
crop_configs = {
    "Winter_Wheat": {
        "base_dir": r"E:\data\SMCI\Winter_Wheat"
    },
    "Spring_Wheat": {
        "base_dir": r"E:\data\SMCI\Spring_Wheat"
    },
    "Spring_Maize": {
        "base_dir": r"E:\data\SMCI\Maize\Spring_Maize"
    },
    "Summer_Maize": {
        "base_dir": r"E:\data\SMCI\Maize\Summer_Maize"
    }
}

# 深度层顺序
soil_layers = ["0_10cm", "10_40cm", "40_80cm", "0_80cm"]

# 统一色标范围
VMIN = 0.0
VMAX = 1.0

# 是否保存图片
SAVE_FIG = False
SAVE_PATH = r"E:\data\SMCI\SMCI_4crop_4layer_random_day_map.png"

# 字体设置（防止中文乱码；如不需要可删除）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False


# =============================================================================
# 2. 工具函数
# =============================================================================
def find_main_var(ds):
    """
    优先找 SMCI 变量，否则自动取第一个数据变量
    """
    if "SMCI" in ds.data_vars:
        return "SMCI"
    return list(ds.data_vars.keys())[0]


def standardize_lat_lon(da):
    """
    保证坐标名为 lat/lon，且纬度升序
    """
    rename_dict = {}

    if "latitude" in da.coords:
        rename_dict["latitude"] = "lat"
    if "longitude" in da.coords:
        rename_dict["longitude"] = "lon"
    if "y" in da.coords:
        rename_dict["y"] = "lat"
    if "x" in da.coords:
        rename_dict["x"] = "lon"

    if rename_dict:
        da = da.rename(rename_dict)

    if "lat" not in da.coords or "lon" not in da.coords:
        raise ValueError("数据中未找到 lat/lon 坐标。")

    lat_vals = da["lat"].values
    if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
        da = da.sortby("lat")

    return da


def scan_year_files(base_dir, crop_name, layer_name):
    """
    扫描某作物某深度层目录下的年度文件
    例如：
    E:\\data\\SMCI\\Maize\\Spring_Maize\\0_10cm\\SMCI_Index_Spring_Maize_0_10cm_2002.nc
    """
    layer_dir = os.path.join(base_dir, layer_name)
    pattern = os.path.join(layer_dir, f"SMCI_Index_{crop_name}_{layer_name}_*.nc")
    files = sorted(glob.glob(pattern))

    year_file_map = {}
    for fp in files:
        fname = os.path.basename(fp)
        m = re.search(r"_(\d{4})\.nc$", fname)
        if m:
            year = int(m.group(1))
            year_file_map[year] = fp

    return year_file_map


def get_available_times(file_path):
    """
    读取某个 nc 文件中的全部时间
    """
    if not os.path.exists(file_path):
        return None, f"文件不存在:\n{file_path}"

    ds = xr.open_dataset(file_path)
    var_name = find_main_var(ds)
    da = ds[var_name]

    if "time" not in da.dims:
        ds.close()
        return None, "缺少 time 维"

    time_values = pd.to_datetime(da["time"].values)
    ds.close()

    return time_values, None


def choose_random_date():
    """
    从所有作物、所有深度中扫描第一个可用文件，随机抽取一天
    """
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    for crop_name, cfg in crop_configs.items():
        for layer_name in soil_layers:
            year_file_map = scan_year_files(cfg["base_dir"], crop_name, layer_name)

            if len(year_file_map) == 0:
                print(f"⚠️ 未找到文件: {crop_name} - {layer_name}")
                continue

            # 取第一个可用年份文件
            first_year = sorted(year_file_map.keys())[0]
            ref_file = year_file_map[first_year]

            print(f"✅ 随机日期参考文件: {ref_file}")

            time_values, err = get_available_times(ref_file)
            if time_values is None or len(time_values) == 0:
                print(f"⚠️ 参考文件没有可用时间: {err}")
                continue

            random_time = random.choice(time_values)
            return pd.Timestamp(random_time).strftime("%Y-%m-%d")

    raise FileNotFoundError("所有作物和深度层目录中都未找到可用 nc 文件，请检查路径。")


def read_smci_map(file_path, target_date):
    """
    读取某个 nc 文件中指定日期的 SMCI 空间图
    """
    if not os.path.exists(file_path):
        return None, f"文件不存在:\n{file_path}"

    ds = xr.open_dataset(file_path)
    var_name = find_main_var(ds)
    da = ds[var_name]
    da = standardize_lat_lon(da)

    if "time" not in da.dims:
        ds.close()
        return None, "缺少 time 维"

    time_values = pd.to_datetime(da["time"].values)
    target_ts = pd.Timestamp(target_date)

    if target_ts not in time_values:
        ds.close()
        return None, f"日期不存在:\n{target_date}"

    da_day = da.sel(time=target_ts).load()
    ds.close()

    return da_day, None


def get_file_for_year(base_dir, crop_name, layer_name, year):
    """
    根据实际扫描结果返回某一年对应文件
    """
    year_file_map = scan_year_files(base_dir, crop_name, layer_name)
    return year_file_map.get(year, None)


# =============================================================================
# 3. 主绘图函数
# =============================================================================
def plot_smci_maps():
    target_date = choose_random_date()
    target_year = pd.Timestamp(target_date).year

    print("=" * 90)
    print(f"随机抽取的日期: {target_date}")
    print(f"对应年份: {target_year}")
    print("=" * 90)

    nrows = len(crop_configs)
    ncols = len(soil_layers)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.6 * ncols, 4.2 * nrows),
        constrained_layout=True
    )

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)
    if ncols == 1:
        axes = np.expand_dims(axes, axis=1)

    last_im = None

    for i, (crop_name, cfg) in enumerate(crop_configs.items()):
        base_dir = cfg["base_dir"]

        for j, layer_name in enumerate(soil_layers):
            ax = axes[i, j]

            file_path = get_file_for_year(base_dir, crop_name, layer_name, target_year)

            if file_path is None:
                ax.text(
                    0.5, 0.5,
                    f"No Data\n{crop_name}\n{layer_name}\n未找到 {target_year} 年文件",
                    ha="center", va="center", fontsize=10, transform=ax.transAxes
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{crop_name}\n{layer_name}", fontsize=11)
                continue

            da_day, err = read_smci_map(file_path, target_date)

            if da_day is None:
                ax.text(
                    0.5, 0.5,
                    f"No Data\n{crop_name}\n{layer_name}\n{err}",
                    ha="center", va="center", fontsize=10, transform=ax.transAxes
                )
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{crop_name}\n{layer_name}", fontsize=11)
                continue

            im = ax.imshow(
                da_day.values,
                origin="lower",
                cmap="RdYlBu",
                vmin=VMIN,
                vmax=VMAX
            )
            last_im = im

            ax.set_title(f"{crop_name}\n{layer_name}", fontsize=11)
            ax.set_xticks([])
            ax.set_yticks([])

    if last_im is not None:
        cbar = fig.colorbar(
            last_im,
            ax=axes,
            shrink=0.95,
            pad=0.02
        )
        cbar.set_label("SMCI", fontsize=12)

    fig.suptitle(f"SMCI Spatial Distribution on Random Day: {target_date}", fontsize=16)

    if SAVE_FIG:
        plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
        print(f"✅ 图片已保存: {SAVE_PATH}")

    plt.show()


# =============================================================================
# 4. 程序入口
# =============================================================================
if __name__ == "__main__":
    plot_smci_maps()