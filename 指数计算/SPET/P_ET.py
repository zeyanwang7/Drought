# -*- coding: utf-8 -*-
"""
批量计算 2001-2022 年每日尺度 P - ET，并随机抽取一天绘图检查（修正版）

输入:
  降水: E:\data\meteorologicaldata\precipitation\YYYY\ChinaMet_010deg_prec_YYYY_MM_DD.nc
  ET  : E:\data\ET_China_Clean_Aligned_Daily\YYYY\ET_YYYY-MM-DD.nc

输出:
  E:\data\P_minus_ET_Daily\YYYY\P_ET_YYYY-MM-DD.nc

修正说明:
1. replace_invalid_with_nan 改为 numpy 一次性掩膜，避免反复 xarray.where 太慢
2. 提高整体稳定性
3. 每年结束后随机绘图检查
"""

import os
import random
import warnings
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =========================================================
# 1. 路径与参数设置
# =========================================================
PREC_BASE = r"E:\data\meteorologicaldata\precipitation"
ET_BASE   = r"E:\data\ET_China_Clean_Aligned_Daily"
OUT_BASE  = r"E:\data\P_minus_ET_Daily"

START_YEAR = 2001
END_YEAR   = 2022

# 是否覆盖已存在输出
OVERWRITE = False

# 每个年份结束后是否随机绘图检查
ENABLE_RANDOM_PLOT = True

# 是否固定随机种子（便于复现）
USE_RANDOM_SEED = True
RANDOM_SEED = 42

# 常见无效值
INVALID_VALUES = [-999, -9999, 9999, 32767, -32768, 1e20, -1e20]


# =========================================================
# 2. 基础工具函数
# =========================================================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def generate_dates_of_year(year):
    """生成某年的全部日期"""
    start_date = datetime(year, 1, 1)
    end_date = datetime(year + 1, 1, 1)
    dates = []
    current = start_date
    while current < end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


def unify_lat_lon_names(ds):
    """统一经纬度维度/坐标名为 lat lon"""
    rename_dict = {}

    for dim in ds.dims:
        dim_l = dim.lower()
        if dim_l in ["latitude", "y"]:
            rename_dict[dim] = "lat"
        elif dim_l in ["longitude", "x"]:
            rename_dict[dim] = "lon"

    for coord in ds.coords:
        coord_l = coord.lower()
        if coord_l in ["latitude", "y"] and coord not in rename_dict:
            rename_dict[coord] = "lat"
        elif coord_l in ["longitude", "x"] and coord not in rename_dict:
            rename_dict[coord] = "lon"

    if rename_dict:
        ds = ds.rename(rename_dict)

    return ds


def guess_data_var(ds, preferred_names=None):
    """自动猜测主数据变量名"""
    data_vars = list(ds.data_vars)
    if len(data_vars) == 0:
        raise ValueError("未找到数据变量 data_vars。")

    if preferred_names:
        for name in preferred_names:
            if name in data_vars:
                return name

    for v in data_vars:
        vl = v.lower()
        for key in ["prec", "pre", "precip", "rain", "tp", "et", "aet"]:
            if key in vl:
                return v

    return data_vars[0]


def replace_invalid_with_nan_fast(da):
    """
    快速将常见无效值替换为 NaN
    核心修复：不再循环多次 da.where()，改成 numpy 一次性处理
    """
    arr = da.values.astype("float32", copy=True)

    invalids = list(INVALID_VALUES)

    # 读取属性中的 fill value
    for att in ["_FillValue", "missing_value", "fill_value"]:
        if att in da.attrs:
            val = da.attrs[att]
            if np.isscalar(val):
                invalids.append(val)
            else:
                try:
                    invalids.extend(list(val))
                except Exception:
                    pass

    # 去重
    clean_invalids = []
    for v in invalids:
        try:
            if v is not None and np.isfinite(float(v)):
                clean_invalids.append(float(v))
        except Exception:
            pass

    clean_invalids = list(set(clean_invalids))

    # 一次性构建掩膜
    mask = ~np.isfinite(arr)
    for v in clean_invalids:
        mask |= np.isclose(arr, v, equal_nan=True)

    arr[mask] = np.nan

    da_out = xr.DataArray(
        arr,
        coords=da.coords,
        dims=da.dims,
        attrs=da.attrs,
        name=da.name
    )
    return da_out


def sort_lat_lon_if_needed(da):
    """保证 lat/lon 为递增"""
    if "lat" in da.coords:
        lat_vals = da["lat"].values
        if len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
            da = da.sortby("lat")

    if "lon" in da.coords:
        lon_vals = da["lon"].values
        if len(lon_vals) > 1 and lon_vals[0] > lon_vals[-1]:
            da = da.sortby("lon")

    return da


def squeeze_extra_dims(da, var_type="变量"):
    """
    将除 lat/lon 外长度为1的维度去掉
    如果有长度>1的额外维度，则报错
    """
    extra_dims = [d for d in da.dims if d not in ("lat", "lon")]
    for d in extra_dims:
        if da.sizes[d] == 1:
            da = da.isel({d: 0})
        else:
            raise ValueError(f"{var_type}存在非单长度额外维度: {d}, shape={da.shape}")
    return da


def grids_equal(da1, da2, tol=1e-6):
    """判断两个网格是否一致"""
    try:
        if da1.sizes["lat"] != da2.sizes["lat"]:
            return False
        if da1.sizes["lon"] != da2.sizes["lon"]:
            return False
        if not np.allclose(da1["lat"].values, da2["lat"].values, atol=tol, equal_nan=True):
            return False
        if not np.allclose(da1["lon"].values, da2["lon"].values, atol=tol, equal_nan=True):
            return False
        return True
    except Exception:
        return False


# =========================================================
# 3. 文件读取函数
# =========================================================
def open_prec_file(file_path):
    """读取降水文件"""
    ds = xr.open_dataset(file_path)
    ds = unify_lat_lon_names(ds)

    var_name = guess_data_var(ds, preferred_names=["prec", "pre", "precipitation"])
    da = ds[var_name]

    da = replace_invalid_with_nan_fast(da)
    da = squeeze_extra_dims(da, var_type="降水变量")
    da = sort_lat_lon_if_needed(da)
    da.name = "prec"

    return da


def open_et_file(file_path):
    """读取ET文件"""
    ds = xr.open_dataset(file_path)
    ds = unify_lat_lon_names(ds)

    var_name = guess_data_var(ds, preferred_names=["ET", "et", "AET", "aet"])
    da = ds[var_name]

    da = replace_invalid_with_nan_fast(da)
    da = squeeze_extra_dims(da, var_type="ET变量")
    da = sort_lat_lon_if_needed(da)
    da.name = "ET"

    return da


# =========================================================
# 4. 输出构建
# =========================================================
def build_output_dataset(diff_da, date_str):
    ds_out = xr.Dataset(
        {
            "P_minus_ET": diff_da.astype("float32")
        }
    )

    ds_out["P_minus_ET"].attrs = {
        "long_name": "Daily precipitation minus actual evapotranspiration",
        "description": "P - ET",
        "units": "mm/day"
    }

    ds_out.attrs = {
        "title": "Daily precipitation minus actual evapotranspiration",
        "source": "ChinaMet precipitation and daily ET",
        "date": date_str
    }

    return ds_out


def print_stats(da, name="Data"):
    vals = da.values
    total_count = vals.size
    valid_mask = np.isfinite(vals)
    valid_count = int(np.sum(valid_mask))

    print(f"{name} 信息:")
    print(f"  shape     : {da.shape}")
    print(f"  总像元数  : {total_count}")
    print(f"  有效像元数: {valid_count}")

    if valid_count > 0:
        print(f"  最小值    : {np.nanmin(vals):.4f}")
        print(f"  最大值    : {np.nanmax(vals):.4f}")
        print(f"  均值      : {np.nanmean(vals):.4f}")
    else:
        print("  该图层全为空值")


# =========================================================
# 5. 绘图函数
# =========================================================
def plot_random_file_of_year(year_dir, year):
    """
    从某一年输出目录中随机抽取一个 nc 文件绘图
    """
    nc_files = [
        os.path.join(year_dir, f)
        for f in os.listdir(year_dir)
        if f.lower().endswith(".nc")
    ]

    if len(nc_files) == 0:
        print(f"⚠️ {year} 年无输出文件，无法绘图检查。")
        return

    sample_file = random.choice(nc_files)

    print("\n" + "=" * 80)
    print(f"🖼️ {year} 年随机抽样绘图检查")
    print("-" * 80)
    print(f"随机文件: {sample_file}")
    print("=" * 80)

    ds = xr.open_dataset(sample_file)

    data_vars = list(ds.data_vars)
    if len(data_vars) == 0:
        print("⚠️ 文件中未找到数据变量。")
        return

    var_name = "P_minus_ET" if "P_minus_ET" in data_vars else data_vars[0]
    da = ds[var_name]

    da = squeeze_extra_dims(da, var_type="绘图变量")
    da = replace_invalid_with_nan_fast(da)
    da = sort_lat_lon_if_needed(da)

    print_stats(da, name=var_name)

    plt.figure(figsize=(10, 6))
    da.plot(cmap="RdBu_r")
    plt.title(f"Random Check of P - ET ({year})\n{os.path.basename(sample_file)}", fontsize=13)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()


# =========================================================
# 6. 单年处理函数
# =========================================================
def calc_p_minus_et_for_year(year):
    print("\n" + "=" * 100)
    print(f"开始处理年份: {year}")
    print("=" * 100)

    prec_year_dir = os.path.join(PREC_BASE, str(year))
    et_year_dir = os.path.join(ET_BASE, str(year))
    out_year_dir = os.path.join(OUT_BASE, str(year))

    ensure_dir(out_year_dir)

    if not os.path.isdir(prec_year_dir):
        print(f"❌ 降水目录不存在: {prec_year_dir}")
        return

    if not os.path.isdir(et_year_dir):
        print(f"❌ ET目录不存在: {et_year_dir}")
        return

    dates = generate_dates_of_year(year)

    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, dt in enumerate(dates, start=1):
        date_prec = dt.strftime("%Y_%m_%d")
        date_std = dt.strftime("%Y-%m-%d")

        prec_file = os.path.join(
            prec_year_dir,
            f"ChinaMet_010deg_prec_{date_prec}.nc"
        )
        et_file = os.path.join(
            et_year_dir,
            f"ET_{date_std}.nc"
        )
        out_file = os.path.join(
            out_year_dir,
            f"P_ET_{date_std}.nc"
        )

        if (not OVERWRITE) and os.path.exists(out_file):
            skip_count += 1
            if i % 50 == 0 or i == len(dates):
                print(f"[{year}] 进度 {i}/{len(dates)} | 成功:{success_count} 跳过:{skip_count} 失败:{fail_count}")
            continue

        if not os.path.exists(prec_file):
            print(f"⚠️ 缺少降水文件: {prec_file}")
            fail_count += 1
            continue

        if not os.path.exists(et_file):
            print(f"⚠️ 缺少ET文件: {et_file}")
            fail_count += 1
            continue

        try:
            prec = open_prec_file(prec_file)
            et = open_et_file(et_file)

            # 若网格不一致，则将 ET 插值到降水网格
            if not grids_equal(prec, et):
                et = et.interp(
                    lat=prec["lat"],
                    lon=prec["lon"],
                    method="linear"
                )

            prec = sort_lat_lon_if_needed(prec)
            et = sort_lat_lon_if_needed(et)

            diff = prec - et
            diff.name = "P_minus_ET"

            ds_out = build_output_dataset(diff, date_std)

            encoding = {
                "P_minus_ET": {
                    "dtype": "float32",
                    "zlib": True,
                    "complevel": 4,
                    "_FillValue": -9999.0
                }
            }

            ds_out.to_netcdf(out_file, encoding=encoding)

            # 显式关闭，减少文件句柄占用
            try:
                prec.close()
            except Exception:
                pass
            try:
                et.close()
            except Exception:
                pass
            try:
                ds_out.close()
            except Exception:
                pass

            success_count += 1

            if i % 50 == 0 or i == len(dates):
                print(f"[{year}] 进度 {i}/{len(dates)} | 成功:{success_count} 跳过:{skip_count} 失败:{fail_count}")

        except Exception as e:
            print(f"❌ 处理失败: {date_std} | {e}")
            fail_count += 1

    print("\n" + "-" * 100)
    print(f"{year} 年处理完成")
    print(f"成功: {success_count}")
    print(f"跳过: {skip_count}")
    print(f"失败: {fail_count}")
    print("-" * 100)

    if ENABLE_RANDOM_PLOT:
        plot_random_file_of_year(out_year_dir, year)


# =========================================================
# 7. 主函数
# =========================================================
def main():
    if USE_RANDOM_SEED:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

    print("=" * 100)
    print("批量计算 2001-2022 年每日 P - ET，并随机抽样绘图检查（修正版）")
    print("=" * 100)
    print(f"降水目录: {PREC_BASE}")
    print(f"ET目录  : {ET_BASE}")
    print(f"输出目录: {OUT_BASE}")
    print(f"处理年份: {START_YEAR} - {END_YEAR}")
    print(f"覆盖输出: {OVERWRITE}")
    print(f"随机绘图: {ENABLE_RANDOM_PLOT}")
    print("=" * 100)

    ensure_dir(OUT_BASE)

    for year in range(START_YEAR, END_YEAR + 1):
        calc_p_minus_et_for_year(year)

    print("\n" + "=" * 100)
    print("全部年份处理完成")
    print("=" * 100)


if __name__ == "__main__":
    main()