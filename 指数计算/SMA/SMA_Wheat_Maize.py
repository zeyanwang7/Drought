# -*- coding: utf-8 -*-
"""
基于已经裁剪到中国作物种植区网格的多层土壤湿度年度文件，
计算中国大陆地区冬春小麦、春夏玉米的日尺度标准化土壤湿度距平 Standardized SMA

【修正版：两遍法 + month-day 对齐 + 防内存爆炸】

标准化 SMA 定义：
    SMA_std = (SM(d) - mean(month_day)) / std(month_day)

其中：
    SM(d)            : 某日土壤湿度
    mean(month_day)  : 该月-日对应的多年平均土壤湿度
    std(month_day)   : 该月-日对应的多年标准差

注意：
1. 不再使用 dayofyear，对闰年和非标准自然年文件更稳；
2. 使用 month-day（如 03-01）作为统一对齐键；
3. 适合你的时间范围：例如 2021-01-02 ~ 2022-01-01。
"""

import os
import glob
import re
import gc
import numpy as np
import pandas as pd
import xarray as xr

# =========================================================
# 1. 路径设置
# =========================================================
input_root = r"E:\data\Soil_Humidity_CropClip_ToCropGrid"
output_root = r"E:\data\SMAstd_CropClip_ToCropGrid"

crop_types = [
    "Winter_Wheat",
    "Spring_Wheat",
    "Spring_Maize",
    "Summer_Maize"
]

soil_layer_dirs = [
    "0_10cm",
    "10_40cm",
    "40_80cm",
    "0_80cm"
]

os.makedirs(output_root, exist_ok=True)

# =========================================================
# 2. 参数设置
# =========================================================
REMOVE_LEAP_DAY = True

HISTORY_START_YEAR = 2001
HISTORY_END_YEAR   = 2022

OUTPUT_START_YEAR = 2002
OUTPUT_END_YEAR   = 2022

SAVE_CLIMATOLOGY = True
VERBOSE = True

STD_EPS = 1e-6
STD_DDOF = 0

# 固定的 365 个 month-day
MONTH_DAY_365 = pd.date_range("2001-01-01", "2001-12-31", freq="D").strftime("%m-%d").tolist()

# =========================================================
# 3. 工具函数
# =========================================================
def log(msg):
    if VERBOSE:
        print(msg)


def detect_main_var(ds):
    exclude_names = {"time", "lat", "lon", "latitude", "longitude", "x", "y"}
    data_vars = [v for v in ds.data_vars if v.lower() not in exclude_names]
    if len(data_vars) == 0:
        raise ValueError("未识别到有效数据变量，请检查 nc 文件。")
    return data_vars[0]


def drop_leap_day(ds):
    if "time" not in ds.coords:
        raise ValueError("数据中不存在 time 坐标，无法计算指标。")
    time_index = pd.DatetimeIndex(ds["time"].values)
    mask = ~((time_index.month == 2) & (time_index.day == 29))
    return ds.isel(time=mask)


def ensure_float32(ds, var_name):
    ds[var_name] = ds[var_name].astype(np.float32)
    return ds


def build_encoding(var_name):
    return {
        var_name: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }
    }


def build_encoding_multi(var_names):
    encoding = {}
    for var_name in var_names:
        encoding[var_name] = {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }
    return encoding


def sanitize_filename(name):
    name = re.sub(r'[\\/:*?"<>|]+', '_', name)
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'__+', '_', name)
    return name.strip('_ ')


def make_unique_path(path):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    version = 2
    while True:
        new_path = f"{base}_v{version}{ext}"
        if not os.path.exists(new_path):
            return new_path
        version += 1


def extract_year_from_filename(fp):
    fname = os.path.basename(fp)
    m = re.search(r'(?<!\d)(20\d{2})(?!\d)', fname)
    if m:
        return int(m.group(1))
    return None


def build_smastd_output_filename(src_fp, suffix="SMAstd"):
    base = os.path.splitext(os.path.basename(src_fp))[0]
    out_name = f"{base}_{suffix}.nc"
    return sanitize_filename(out_name)


def build_clim_output_filename(crop_name, layer_name, suffix="Climatology_MeanStd"):
    out_name = f"{crop_name}_{layer_name}_{suffix}_{HISTORY_START_YEAR}_{HISTORY_END_YEAR}.nc"
    return sanitize_filename(out_name)


def open_preprocess_file(fp):
    ds = xr.open_dataset(fp)
    var_name = detect_main_var(ds)

    if "time" not in ds[var_name].dims:
        ds.close()
        raise ValueError(f"文件不存在 time 维度，无法计算指标: {fp}")

    ds = ensure_float32(ds, var_name)

    if REMOVE_LEAP_DAY:
        ds = drop_leap_day(ds)

    return ds, var_name


def get_file_map_by_year(in_dir):
    nc_files = sorted(glob.glob(os.path.join(in_dir, "*.nc")))
    year_map = {}

    for fp in nc_files:
        year = extract_year_from_filename(fp)
        if year is None:
            log(f"⚠️ 文件名未识别出年份，跳过: {fp}")
            continue
        year_map[year] = fp

    return year_map


def add_month_day_coord(da):
    """
    给 DataArray 增加 month_day 辅助坐标，格式如 01-01, 03-15
    """
    md = pd.DatetimeIndex(da["time"].values).strftime("%m-%d")
    return da.assign_coords(month_day=("time", md))


# =========================================================
# 4. 第一遍：计算 mean/std climatology（按 month-day）
# =========================================================
def first_pass_build_mean_std(in_dir, crop_name, layer_name):
    year_map = get_file_map_by_year(in_dir)
    hist_years = [y for y in sorted(year_map.keys()) if HISTORY_START_YEAR <= y <= HISTORY_END_YEAR]
    log(f"历史样本年份: {hist_years}")

    if len(hist_years) == 0:
        raise ValueError("历史样本期内没有可用文件。")

    sum_by_md = None
    sumsq_by_md = None
    count_by_md = None
    template_attrs = None
    template_ds_attrs = None
    var_name_ref = None

    for i, year in enumerate(hist_years, start=1):
        fp = year_map[year]
        log("\n" + "-" * 100)
        log(f"第一遍 [{i}/{len(hist_years)}] 处理历史样本年份: {year}")
        log(f"文件: {fp}")

        ds = None
        try:
            ds, var_name = open_preprocess_file(fp)

            if var_name_ref is None:
                var_name_ref = var_name
            elif var_name != var_name_ref:
                raise ValueError(f"变量名不一致：{var_name_ref} vs {var_name}")

            da = ds[var_name]
            da = add_month_day_coord(da)

            if template_attrs is None:
                template_attrs = da.attrs.copy()
            if template_ds_attrs is None:
                template_ds_attrs = ds.attrs.copy()

            log(f"  变量名: {var_name}")
            log(f"  shape: {da.shape}")
            log(f"  time range: {str(ds.time.values[0])[:10]} ~ {str(ds.time.values[-1])[:10]}")

            yearly_sum = da.groupby("month_day").sum(dim="time", skipna=True).astype(np.float32)
            yearly_sumsq = (da * da).groupby("month_day").sum(dim="time", skipna=True).astype(np.float32)
            yearly_count = da.groupby("month_day").count(dim="time").astype(np.int16)

            # 强制重排为固定 365 天顺序
            yearly_sum = yearly_sum.reindex(month_day=MONTH_DAY_365)
            yearly_sumsq = yearly_sumsq.reindex(month_day=MONTH_DAY_365)
            yearly_count = yearly_count.reindex(month_day=MONTH_DAY_365, fill_value=0)

            if sum_by_md is None:
                sum_by_md = yearly_sum
                sumsq_by_md = yearly_sumsq
                count_by_md = yearly_count
            else:
                sum_by_md = sum_by_md + yearly_sum
                sumsq_by_md = sumsq_by_md + yearly_sumsq
                count_by_md = count_by_md + yearly_count

            log("  ✅ 本年累计完成")

            del da, yearly_sum, yearly_sumsq, yearly_count
            ds.close()
            del ds
            gc.collect()

        except Exception as e:
            log(f"  ❌ 第一遍处理失败: {fp}")
            log(f"  错误信息: {e}")
            if ds is not None:
                ds.close()
            gc.collect()

    count_f = count_by_md.astype(np.float32)
    mean_da = xr.where(count_by_md > 0, sum_by_md / count_f, np.nan).astype(np.float32)

    var_pop = xr.where(
        count_by_md > 0,
        sumsq_by_md / count_f - mean_da * mean_da,
        np.nan
    )
    var_pop = xr.where(var_pop < 0, 0, var_pop)

    if STD_DDOF == 0:
        var_da = var_pop
    elif STD_DDOF == 1:
        var_da = xr.where(count_by_md > 1, var_pop * count_f / (count_f - 1.0), np.nan)
    else:
        raise ValueError("STD_DDOF 只能取 0 或 1")

    std_da = np.sqrt(var_da).astype(np.float32)

    mean_da.name = "mean_md"
    std_da.name = "std_md"

    mean_da.attrs = template_attrs.copy() if template_attrs else {}
    mean_da.attrs["long_name"] = "Multi-year daily mean soil moisture by month-day"

    std_da.attrs = template_attrs.copy() if template_attrs else {}
    std_da.attrs["long_name"] = "Multi-year daily std soil moisture by month-day"

    del sum_by_md, sumsq_by_md, count_by_md, count_f, var_pop, var_da
    gc.collect()

    return mean_da, std_da, template_attrs, template_ds_attrs, var_name_ref


# =========================================================
# 5. 第二遍：逐年输出标准化 SMA（按 month-day 索引）
# =========================================================
def second_pass_output_smastd(in_dir, out_dir, crop_name, layer_name,
                              mean_da, std_da, template_attrs, template_ds_attrs, var_name_ref):
    year_map = get_file_map_by_year(in_dir)
    out_years = [y for y in sorted(year_map.keys()) if OUTPUT_START_YEAR <= y <= OUTPUT_END_YEAR]
    log(f"输出样本年份: {out_years}")

    if len(out_years) == 0:
        log("⚠️ 输出样本期内没有可用文件。")
        return

    for i, year in enumerate(out_years, start=1):
        fp = year_map[year]
        log("\n" + "=" * 100)
        log(f"第二遍 [{i}/{len(out_years)}] 输出年份: {year}")
        log(f"文件: {fp}")
        log("=" * 100)

        ds = None
        out_ds = None

        try:
            ds, var_name = open_preprocess_file(fp)

            if var_name != var_name_ref:
                raise ValueError(f"变量名不一致：{var_name_ref} vs {var_name}")

            da = ds[var_name]

            # 构造每个 time 对应的 month_day 索引
            md_index = xr.DataArray(
                pd.DatetimeIndex(ds.time.values).strftime("%m-%d"),
                coords={"time": ds.time},
                dims="time",
                name="month_day"
            )

            mean_for_time = mean_da.sel(month_day=md_index)
            std_for_time = std_da.sel(month_day=md_index)
            safe_std = xr.where(std_for_time > STD_EPS, std_for_time, np.nan)

            smastd_da = ((da - mean_for_time) / safe_std).astype(np.float32)

            smastd_da.name = "SMA_std"
            smastd_da.attrs = template_attrs.copy() if template_attrs else {}
            smastd_da.attrs["long_name"] = "Standardized Soil Moisture Anomaly"
            smastd_da.attrs["description"] = (
                "Standardized daily soil moisture anomaly relative to multi-year "
                "mean and std of the same month-day"
            )

            out_ds = smastd_da.to_dataset(name="SMA_std")
            out_ds.attrs = template_ds_attrs.copy() if template_ds_attrs else {}
            out_ds.attrs["title"] = "Standardized Soil Moisture Anomaly (SMA_std)"
            out_ds.attrs["crop_type"] = crop_name
            out_ds.attrs["soil_layer"] = layer_name
            out_ds.attrs["history_period"] = f"{HISTORY_START_YEAR}-01-01 to {HISTORY_END_YEAR}-12-31"
            out_ds.attrs["output_period"] = f"{year}"
            out_ds.attrs["remove_leap_day"] = str(REMOVE_LEAP_DAY)
            out_ds.attrs["method"] = "SMA_std = (SM - mean(month_day)) / std(month_day)"
            out_ds.attrs["std_ddof"] = str(STD_DDOF)
            out_ds.attrs["std_epsilon"] = str(STD_EPS)
            out_ds.attrs["note"] = "Computed by two-pass memory-safe method using month-day alignment"

            out_name = build_smastd_output_filename(fp, suffix="SMAstd")
            save_path = os.path.join(out_dir, out_name)
            save_path = make_unique_path(save_path)

            out_ds.to_netcdf(save_path, encoding=build_encoding("SMA_std"))
            log(f"  ✅ 输出完成: {save_path}")

            del da, md_index, mean_for_time, std_for_time, safe_std, smastd_da
            out_ds.close()
            ds.close()
            del out_ds, ds
            gc.collect()

        except Exception as e:
            log(f"  ❌ 第二遍输出失败: {fp}")
            log(f"  错误信息: {e}")
            if out_ds is not None:
                try:
                    out_ds.close()
                except:
                    pass
            if ds is not None:
                try:
                    ds.close()
                except:
                    pass
            gc.collect()


# =========================================================
# 6. 保存 mean/std climatology
# =========================================================
def save_mean_std_climatology(out_dir, crop_name, layer_name, mean_da, std_da, template_ds_attrs):
    if not SAVE_CLIMATOLOGY:
        return

    clim_ds = xr.Dataset({
        "mean_md": mean_da,
        "std_md": std_da
    })

    clim_ds.attrs = template_ds_attrs.copy() if template_ds_attrs else {}
    clim_ds.attrs["title"] = "Multi-year daily mean and std of soil moisture by month-day"
    clim_ds.attrs["crop_type"] = crop_name
    clim_ds.attrs["soil_layer"] = layer_name
    clim_ds.attrs["history_period"] = f"{HISTORY_START_YEAR}-01-01 to {HISTORY_END_YEAR}-12-31"
    clim_ds.attrs["remove_leap_day"] = str(REMOVE_LEAP_DAY)
    clim_ds.attrs["std_ddof"] = str(STD_DDOF)
    clim_ds.attrs["note"] = "Computed by two-pass memory-safe method using month-day alignment"

    clim_name = build_clim_output_filename(crop_name, layer_name, suffix="Climatology_MeanStd")
    clim_path = os.path.join(out_dir, clim_name)
    clim_path = make_unique_path(clim_path)

    clim_ds.to_netcdf(clim_path, encoding=build_encoding_multi(["mean_md", "std_md"]))
    log(f"✅ mean/std climatology 已保存: {clim_path}")

    clim_ds.close()
    del clim_ds
    gc.collect()


# =========================================================
# 7. 主流程
# =========================================================
def main():
    print("=" * 100)
    print("开始计算中国大陆地区冬春小麦和春夏玉米标准化土壤湿度距平 SMA_std（修正版）")
    print("=" * 100)
    print(f"输入目录: {input_root}")
    print(f"输出目录: {output_root}")
    print(f"历史样本年份: {HISTORY_START_YEAR} ~ {HISTORY_END_YEAR}")
    print(f"输出样本年份: {OUTPUT_START_YEAR} ~ {OUTPUT_END_YEAR}")
    print(f"是否删除 2月29日: {REMOVE_LEAP_DAY}")
    print(f"是否保存 climatology: {SAVE_CLIMATOLOGY}")
    print(f"标准差 ddof: {STD_DDOF}")
    print(f"标准差最小阈值: {STD_EPS}")

    for crop_name in crop_types:
        for layer_name in soil_layer_dirs:
            in_dir = os.path.join(input_root, crop_name, layer_name)
            out_dir = os.path.join(output_root, crop_name, layer_name)
            os.makedirs(out_dir, exist_ok=True)

            print("\n" + "=" * 100)
            print(f"作物: {crop_name} | 深度层: {layer_name}")
            print(f"输入目录: {in_dir}")
            print(f"输出目录: {out_dir}")
            print("=" * 100)

            if not os.path.exists(in_dir):
                print(f"⚠️ 输入目录不存在，跳过: {in_dir}")
                continue

            try:
                mean_da, std_da, template_attrs, template_ds_attrs, var_name_ref = \
                    first_pass_build_mean_std(in_dir, crop_name, layer_name)

                save_mean_std_climatology(out_dir, crop_name, layer_name, mean_da, std_da, template_ds_attrs)

                second_pass_output_smastd(
                    in_dir=in_dir,
                    out_dir=out_dir,
                    crop_name=crop_name,
                    layer_name=layer_name,
                    mean_da=mean_da,
                    std_da=std_da,
                    template_attrs=template_attrs,
                    template_ds_attrs=template_ds_attrs,
                    var_name_ref=var_name_ref
                )

                del mean_da, std_da, template_attrs, template_ds_attrs, var_name_ref
                gc.collect()

            except Exception as e:
                print(f"❌ 该目录处理失败: {in_dir}")
                print(f"错误信息: {e}")
                gc.collect()

    print("\n全部标准化 SMA 计算完成。")


if __name__ == "__main__":
    main()