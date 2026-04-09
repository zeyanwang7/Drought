# -*- coding: utf-8 -*-
"""
基于已裁剪到春夏玉米种植区的土壤湿度文件计算 SMCI（彻底内存优化版）

适配目录结构：
E:\data\Soil_Humidity_CropClip_ToCropGrid\Spring_Maize\0_10cm\*.nc
E:\data\Soil_Humidity_CropClip_ToCropGrid\Spring_Maize\10_40cm\*.nc
E:\data\Soil_Humidity_CropClip_ToCropGrid\Spring_Maize\40_80cm\*.nc
E:\data\Soil_Humidity_CropClip_ToCropGrid\Spring_Maize\0_80cm\*.nc

E:\data\Soil_Humidity_CropClip_ToCropGrid\Summer_Maize\0_10cm\*.nc
E:\data\Soil_Humidity_CropClip_ToCropGrid\Summer_Maize\10_40cm\*.nc
E:\data\Soil_Humidity_CropClip_ToCropGrid\Summer_Maize\40_80cm\*.nc
E:\data\Soil_Humidity_CropClip_ToCropGrid\Summer_Maize\0_80cm\*.nc

SMCI公式：
    SMCI = (SM - SM_min) / (SM_max - SM_min)

说明：
1. 按“像元 × 历日(dayofyear)”统计多年 min/max
2. 删除 2月29日，统一为365天
3. 2000-2022 用于统计 min/max
4. 正式输出 2002-2022 年
5. 按年分块计算，并按年直接保存，不在内存中拼接多年结果
"""

import os
import glob
import gc
import random
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# 1. 路径设置
# =============================================================================
# soil_root = r"E:\data\Soil_Humidity_CropClip_ToCropGrid"
# crop_names = ["Spring_Maize", "Summer_Maize"]
# soil_layer_dirs = ["0_10cm", "10_40cm", "40_80cm", "0_80cm"]

# output_root = r"E:\data\SMCI\Maize"

soil_root = r"E:\data\Soil_Humidity_CropClip_ToCropGrid"
crop_names = ["Winter_Wheat"]
soil_layer_dirs = ["0_10cm", "10_40cm", "40_80cm", "0_80cm"]

output_root = r"E:\data\SMCI\Wheat"
os.makedirs(output_root, exist_ok=True)

# 是否绘图检查（建议第一次先关掉）
CHECK_PLOT = False

# 是否输出年度进度
VERBOSE_YEAR_PROGRESS = True

# 是否保存年度文件列表
SAVE_FILE_LIST = True

ENCODING = {
    "zlib": True,
    "complevel": 4,
    "dtype": "float32",
    "_FillValue": np.float32(np.nan)
}

# =============================================================================
# 2. 工具函数
# =============================================================================
def detect_main_var(ds):
    exclude_names = {"time", "lat", "lon", "latitude", "longitude", "x", "y"}
    data_vars = [v for v in ds.data_vars if v.lower() not in exclude_names]
    if len(data_vars) == 0:
        raise ValueError("未识别到有效数据变量，请检查 nc 文件。")
    return data_vars[0]


def get_lat_lon_names(da):
    lat_candidates = ["lat", "latitude", "y"]
    lon_candidates = ["lon", "longitude", "x"]

    lat_name, lon_name = None, None

    for n in lat_candidates:
        if n in da.coords:
            lat_name = n
            break
    for n in lon_candidates:
        if n in da.coords:
            lon_name = n
            break

    if lat_name is None or lon_name is None:
        raise ValueError("未找到 lat/lon 坐标名。")

    return lat_name, lon_name


def ensure_lat_ascending(da, lat_name):
    lat_vals = da[lat_name].values
    if len(lat_vals) > 1 and lat_vals[1] < lat_vals[0]:
        print("  ⚠️ 检测到纬度为降序，自动翻转为升序")
        da = da.sortby(lat_name)
    return da


def remove_leap_day(ds_or_da):
    if "time" not in ds_or_da.coords:
        raise ValueError("数据中没有 time 坐标。")
    time_index = pd.DatetimeIndex(ds_or_da["time"].values)
    mask = ~((time_index.month == 2) & (time_index.day == 29))
    return ds_or_da.sel(time=mask)


def build_history_minmax(da_hist):
    """
    基于历史数据构建像元 × 历日(dayofyear) 的 min/max
    """
    time_index = pd.DatetimeIndex(da_hist["time"].values)
    doy = xr.DataArray(
        time_index.dayofyear,
        dims=["time"],
        coords={"time": da_hist["time"]}
    )
    da_hist = da_hist.assign_coords(dayofyear=doy)

    sm_min = da_hist.groupby("dayofyear").min(dim="time", skipna=True)
    sm_max = da_hist.groupby("dayofyear").max(dim="time", skipna=True)

    return sm_min.astype(np.float32), sm_max.astype(np.float32)


def compute_smci_one_year(da_year, sm_min, sm_max):
    """
    单年计算 SMCI，不保留多年结果
    """
    time_index = pd.DatetimeIndex(da_year["time"].values)
    doy = xr.DataArray(
        time_index.dayofyear,
        dims=["time"],
        coords={"time": da_year["time"]}
    )
    da_year = da_year.assign_coords(dayofyear=doy)

    sm_min_year = sm_min.sel(dayofyear=da_year["dayofyear"])
    sm_max_year = sm_max.sel(dayofyear=da_year["dayofyear"])

    denom = sm_max_year - sm_min_year

    smci_year = xr.where(
        denom > 0,
        (da_year - sm_min_year) / denom,
        np.nan
    )

    smci_year = xr.where(smci_year < 0, 0, smci_year)
    smci_year = xr.where(smci_year > 1, 1, smci_year)

    if "dayofyear" in smci_year.coords:
        smci_year = smci_year.drop_vars("dayofyear")

    smci_year = smci_year.astype(np.float32)
    smci_year.name = "SMCI"
    smci_year.attrs["long_name"] = "Soil Moisture Condition Index"
    smci_year.attrs["units"] = "1"
    smci_year.attrs["formula"] = "(SM - SM_min) / (SM_max - SM_min)"
    smci_year.attrs["description"] = "Pixel-wise daily SMCI computed year-by-year"

    return smci_year


def quick_check_plot(da, save_dir, crop_name, layer_name):
    if da.sizes["time"] == 0:
        return

    idx = random.randint(0, da.sizes["time"] - 1)
    da_plot = da.isel(time=idx)
    date_str = pd.to_datetime(str(da_plot["time"].values)).strftime("%Y-%m-%d")

    plt.figure(figsize=(8, 6))
    im = plt.imshow(
        da_plot.values,
        origin="lower",
        cmap="RdYlBu",
        vmin=0,
        vmax=1
    )
    plt.colorbar(im, label="SMCI")
    plt.title(f"{crop_name} {layer_name} SMCI | {date_str}")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"SMCI_Check_{crop_name}_{layer_name}_{date_str}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"  ✅ 检查图已保存: {save_path}")


def concat_layer_files(layer_dir):
    """
    读取某一深度层下全部 nc 文件，并拼接为一个 DataArray
    """
    nc_files = sorted(glob.glob(os.path.join(layer_dir, "*.nc")))
    if len(nc_files) == 0:
        raise FileNotFoundError(f"未找到文件: {layer_dir}")

    print(f"  找到文件数: {len(nc_files)}")

    da_list = []
    var_name_ref = None

    for i, fp in enumerate(nc_files, 1):
        print(f"    [{i}/{len(nc_files)}] 读取: {os.path.basename(fp)}")

        ds = xr.open_dataset(fp)
        ds = remove_leap_day(ds)

        var_name = detect_main_var(ds)
        da = ds[var_name]

        lat_name, lon_name = get_lat_lon_names(da)
        da = ensure_lat_ascending(da, lat_name)

        rename_dict = {}
        if lat_name != "lat":
            rename_dict[lat_name] = "lat"
        if lon_name != "lon":
            rename_dict[lon_name] = "lon"
        if rename_dict:
            da = da.rename(rename_dict)

        if "time" not in da.dims:
            raise ValueError(f"文件缺少 time 维，请先检查该文件: {fp}")

        if var_name_ref is None:
            var_name_ref = var_name

        da = da.astype(np.float32)
        da_list.append(da)

        ds.close()

    da_all = xr.concat(da_list, dim="time").sortby("time")

    # 去重时间
    _, unique_idx = np.unique(da_all["time"].values, return_index=True)
    da_all = da_all.isel(time=np.sort(unique_idx))

    return da_all, var_name_ref


def save_yearly_smci_files(da_out, sm_min, sm_max, save_dir, crop_name, layer_name, sm_var_name):
    """
    逐年计算并逐年保存
    """
    years = np.unique(pd.DatetimeIndex(da_out["time"].values).year)
    saved_files = []

    for yr in years:
        if VERBOSE_YEAR_PROGRESS:
            print(f"    -> 正在计算并保存 {yr} 年 SMCI")

        da_year = da_out.sel(time=slice(f"{yr}-01-01", f"{yr}-12-31"))
        if da_year.sizes["time"] == 0:
            continue

        smci_year = compute_smci_one_year(da_year, sm_min, sm_max)

        out_ds = smci_year.to_dataset(name="SMCI")
        out_ds.attrs["crop_mask"] = crop_name
        out_ds.attrs["soil_layer"] = layer_name
        out_ds.attrs["source_variable"] = sm_var_name
        out_ds.attrs["year"] = str(yr)
        out_ds.attrs["time_range"] = f"{yr}-01-01 to {yr}-12-31"
        out_ds.attrs["history_range_for_minmax"] = "2000-2022"
        out_ds.attrs["note"] = "SMCI computed from crop-clipped soil humidity files (year-by-year saved version)"

        save_path = os.path.join(
            save_dir,
            f"SMCI_Index_{crop_name}_{layer_name}_{yr}.nc"
        )

        out_ds.to_netcdf(
            save_path,
            encoding={"SMCI": ENCODING}
        )

        print(f"       ✅ 已保存: {save_path}")
        print(f"       年内最小值: {float(np.nanmin(smci_year.values)):.4f}")
        print(f"       年内最大值: {float(np.nanmax(smci_year.values)):.4f}")

        if CHECK_PLOT and yr == years[0]:
            quick_check_plot(smci_year, save_dir, crop_name, layer_name)

        saved_files.append(save_path)

        del da_year, smci_year, out_ds
        gc.collect()

    return saved_files


# =============================================================================
# 3. 主程序
# =============================================================================
def main():
    print("=" * 100)
    print("开始计算 Spring_Maize / Summer_Maize 各深度层逐日 SMCI（逐年保存版）")
    print("=" * 100)

    for crop_name in crop_names:
        print("\n" + "#" * 100)
        print(f"处理作物: {crop_name}")
        print("#" * 100)

        for layer_name in soil_layer_dirs:
            print("\n" + "=" * 100)
            print(f"处理深度层: {layer_name}")
            print("=" * 100)

            layer_dir = os.path.join(soil_root, crop_name, layer_name)
            if not os.path.exists(layer_dir):
                print(f"  ⚠️ 目录不存在，跳过: {layer_dir}")
                continue

            try:
                da_all, sm_var_name = concat_layer_files(layer_dir)

                # 限定时间范围
                da_all = da_all.sel(time=slice("2000-01-01", "2022-12-31"))

                print("  拼接后数据概况:")
                print(f"  dims: {da_all.dims}")
                print(f"  shape: {da_all.shape}")
                print(f"  time range: {str(da_all.time.values[0])[:10]} ~ {str(da_all.time.values[-1])[:10]}")

                da_hist = da_all.sel(time=slice("2000-01-01", "2022-12-31"))
                da_out = da_all.sel(time=slice("2002-01-01", "2022-12-31"))

                if da_hist.sizes["time"] == 0:
                    raise ValueError("历史样本为空，无法计算 min/max。")
                if da_out.sizes["time"] == 0:
                    raise ValueError("输出时间段为空，请检查时间范围。")

                print(f"  历史样本长度: {da_hist.sizes['time']} 天")
                print(f"  输出样本长度: {da_out.sizes['time']} 天")

                print("  -> 计算像元×历日 min/max")
                sm_min, sm_max = build_history_minmax(da_hist)

                save_dir = os.path.join(output_root, crop_name, layer_name)
                os.makedirs(save_dir, exist_ok=True)

                print("  -> 逐年计算并保存 SMCI")
                saved_files = save_yearly_smci_files(
                    da_out=da_out,
                    sm_min=sm_min,
                    sm_max=sm_max,
                    save_dir=save_dir,
                    crop_name=crop_name,
                    layer_name=layer_name,
                    sm_var_name=sm_var_name
                )

                if SAVE_FILE_LIST:
                    list_path = os.path.join(save_dir, f"SMCI_Index_{crop_name}_{layer_name}_filelist.txt")
                    with open(list_path, "w", encoding="utf-8") as f:
                        for fp in saved_files:
                            f.write(fp + "\n")
                    print(f"  ✅ 年度文件清单已保存: {list_path}")

                del da_all, da_hist, da_out, sm_min, sm_max, saved_files
                gc.collect()

            except Exception as e:
                print(f"  ❌ 处理失败: {crop_name} - {layer_name}")
                print(f"  错误信息: {e}")

    print("\n全部处理完成。")


if __name__ == "__main__":
    main()