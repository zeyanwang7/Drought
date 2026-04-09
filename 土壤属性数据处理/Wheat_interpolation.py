# -*- coding: utf-8 -*-
"""
插补前后对比检查：
只针对原始作物种植像元(mask==1)统计

功能：
1. 对比原始 nc 与 filled nc
2. 统计每层插补前缺失数、插补后缺失数、补了多少
3. 输出全深度汇总
4. 输出 Excel
"""

import os
import time
import numpy as np
import pandas as pd
import xarray as xr

# =============================================================================
# 1. 路径设置
# =============================================================================
ORIG_DIR = r"E:\data\SoilData\Wheat_0p1deg_Masked"
FILLED_DIR = r"E:\data\SoilData\Wheat_0p1deg_Masked_Filled"

CROP_MASKS = {
    "Winter_Wheat": r"E:\data\CropMask\Winter_Wheat_01deg_filtered.tif",
    "Spring_Wheat": r"E:\data\CropMask\Spring_Wheat_01deg_filtered.tif",
}

OUT_EXCEL = r"E:\data\SoilData\Wheat_0p1deg_Masked_Filled\插补前后对比检查.xlsx"


# =============================================================================
# 2. 工具函数
# =============================================================================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def standardize_lat_lon_da(da):
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


def parse_crop_and_soil_from_filename(file_name):
    soil_type = "Unknown"
    crop_type = "Unknown"

    if "TH1500" in file_name:
        soil_type = "TH1500"
    elif "TH33" in file_name:
        soil_type = "TH33"

    if "Winter_Wheat" in file_name:
        crop_type = "Winter_Wheat"
    elif "Spring_Wheat" in file_name:
        crop_type = "Spring_Wheat"

    return soil_type, crop_type


def load_crop_mask(mask_path):
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


def build_filled_path(orig_path):
    base_name = os.path.splitext(os.path.basename(orig_path))[0]
    filled_name = f"{base_name}_filled.nc"
    return os.path.join(FILLED_DIR, filled_name)


# =============================================================================
# 3. 核心对比函数
# =============================================================================
def compare_one_pair(orig_path, filled_path):
    log(f"开始对比: {orig_path}")

    file_name = os.path.basename(orig_path)
    soil_type, crop_type = parse_crop_and_soil_from_filename(file_name)

    if crop_type not in CROP_MASKS:
        raise ValueError(f"无法识别作物类型: {file_name}")

    if not os.path.exists(filled_path):
        raise FileNotFoundError(f"未找到对应插补后文件: {filled_path}")

    ds_orig = xr.open_dataset(orig_path)
    ds_filled = xr.open_dataset(filled_path)

    var_orig = find_main_var(ds_orig)
    var_filled = find_main_var(ds_filled)

    da_orig = standardize_lat_lon_da(ds_orig[var_orig])
    da_filled = standardize_lat_lon_da(ds_filled[var_filled])

    mask_raw = load_crop_mask(CROP_MASKS[crop_type])
    crop_mask = align_mask_to_data(mask_raw, da_orig)
    crop_bool = (crop_mask.values == 1)

    crop_pixels = int(crop_bool.sum())

    log(f"作物类型: {crop_type}")
    log(f"土壤类型: {soil_type}")
    log(f"原始种植像元数: {crop_pixels}")
    log(f"原始数据 shape: {da_orig.shape}")
    log(f"插补数据 shape: {da_filled.shape}")

    # 检查维度一致
    if da_orig.dims != da_filled.dims:
        raise ValueError(f"原始与插补后维度不一致: {da_orig.dims} vs {da_filled.dims}")
    if da_orig.shape != da_filled.shape:
        raise ValueError(f"原始与插补后 shape 不一致: {da_orig.shape} vs {da_filled.shape}")

    layer_records = []
    summary_records = []

    if "depth" in da_orig.dims:
        depth_values = da_orig["depth"].values

        orig_3d = da_orig.values
        filled_3d = da_filled.values

        for i, depth_val in enumerate(depth_values):
            orig_vals = orig_3d[i][crop_bool]
            filled_vals = filled_3d[i][crop_bool]

            before_missing = int(np.isnan(orig_vals).sum())
            after_missing = int(np.isnan(filled_vals).sum())
            filled_count = int(before_missing - after_missing)

            before_valid = int(np.isfinite(orig_vals).sum())
            after_valid = int(np.isfinite(filled_vals).sum())

            before_missing_ratio = before_missing / crop_pixels * 100
            after_missing_ratio = after_missing / crop_pixels * 100

            layer_records.append({
                "SoilType": soil_type,
                "CropType": crop_type,
                "VarName": var_orig,
                "DepthIndex": i + 1,
                "DepthValue": float(depth_val) if np.isscalar(depth_val) else depth_val,
                "CropPixels": crop_pixels,
                "BeforeValid": before_valid,
                "BeforeMissing": before_missing,
                "BeforeMissingRatio(%)": round(before_missing_ratio, 4),
                "AfterValid": after_valid,
                "AfterMissing": after_missing,
                "AfterMissingRatio(%)": round(after_missing_ratio, 4),
                "FilledCount": filled_count,
                "AllMissingFilled": bool(after_missing == 0),
            })

        # 全深度统计
        orig_valid_3d = np.isfinite(orig_3d[:, crop_bool])
        filled_valid_3d = np.isfinite(filled_3d[:, crop_bool])

        orig_all_depth_valid = orig_valid_3d.all(axis=0)
        filled_all_depth_valid = filled_valid_3d.all(axis=0)

        before_all_depth_valid_pixels = int(orig_all_depth_valid.sum())
        after_all_depth_valid_pixels = int(filled_all_depth_valid.sum())

        before_missing_any_depth = crop_pixels - before_all_depth_valid_pixels
        after_missing_any_depth = crop_pixels - after_all_depth_valid_pixels

        summary_records.append({
            "SoilType": soil_type,
            "CropType": crop_type,
            "VarName": var_orig,
            "DepthCount": len(depth_values),
            "CropPixels": crop_pixels,
            "BeforePixelsValidAtAllDepths": before_all_depth_valid_pixels,
            "BeforePixelsMissingAtLeastOneDepth": before_missing_any_depth,
            "BeforeCompleteRatio(%)": round(before_all_depth_valid_pixels / crop_pixels * 100, 4),
            "AfterPixelsValidAtAllDepths": after_all_depth_valid_pixels,
            "AfterPixelsMissingAtLeastOneDepth": after_missing_any_depth,
            "AfterCompleteRatio(%)": round(after_all_depth_valid_pixels / crop_pixels * 100, 4),
            "ImprovedPixels": after_all_depth_valid_pixels - before_all_depth_valid_pixels,
            "AllPixelsCompleteAfterFill": bool(after_missing_any_depth == 0),
        })

    else:
        orig_vals = da_orig.values[crop_bool]
        filled_vals = da_filled.values[crop_bool]

        before_missing = int(np.isnan(orig_vals).sum())
        after_missing = int(np.isnan(filled_vals).sum())
        filled_count = int(before_missing - after_missing)

        before_valid = int(np.isfinite(orig_vals).sum())
        after_valid = int(np.isfinite(filled_vals).sum())

        before_missing_ratio = before_missing / crop_pixels * 100
        after_missing_ratio = after_missing / crop_pixels * 100

        layer_records.append({
            "SoilType": soil_type,
            "CropType": crop_type,
            "VarName": var_orig,
            "DepthIndex": 1,
            "DepthValue": np.nan,
            "CropPixels": crop_pixels,
            "BeforeValid": before_valid,
            "BeforeMissing": before_missing,
            "BeforeMissingRatio(%)": round(before_missing_ratio, 4),
            "AfterValid": after_valid,
            "AfterMissing": after_missing,
            "AfterMissingRatio(%)": round(after_missing_ratio, 4),
            "FilledCount": filled_count,
            "AllMissingFilled": bool(after_missing == 0),
        })

        summary_records.append({
            "SoilType": soil_type,
            "CropType": crop_type,
            "VarName": var_orig,
            "DepthCount": 1,
            "CropPixels": crop_pixels,
            "BeforePixelsValidAtAllDepths": before_valid,
            "BeforePixelsMissingAtLeastOneDepth": before_missing,
            "BeforeCompleteRatio(%)": round(before_valid / crop_pixels * 100, 4),
            "AfterPixelsValidAtAllDepths": after_valid,
            "AfterPixelsMissingAtLeastOneDepth": after_missing,
            "AfterCompleteRatio(%)": round(after_valid / crop_pixels * 100, 4),
            "ImprovedPixels": after_valid - before_valid,
            "AllPixelsCompleteAfterFill": bool(after_missing == 0),
        })

    return pd.DataFrame(layer_records), pd.DataFrame(summary_records)


# =============================================================================
# 4. 主程序
# =============================================================================
def main():
    log("开始进行插补前后对比检查")

    orig_files = [
        os.path.join(ORIG_DIR, f)
        for f in os.listdir(ORIG_DIR)
        if f.lower().endswith(".nc") and not f.lower().endswith("_filled.nc")
    ]

    if len(orig_files) == 0:
        log(f"未找到原始 nc 文件: {ORIG_DIR}")
        return

    log(f"共找到 {len(orig_files)} 个原始 nc 文件")

    all_layer_stats = []
    all_summary_stats = []

    for orig_path in orig_files:
        try:
            filled_path = build_filled_path(orig_path)
            df_layer, df_summary = compare_one_pair(orig_path, filled_path)
            all_layer_stats.append(df_layer)
            all_summary_stats.append(df_summary)
        except Exception as e:
            log(f"❌ 对比失败: {orig_path}")
            log(f"错误信息: {e}")

    if len(all_layer_stats) == 0:
        log("没有成功生成任何对比结果")
        return

    df_layer_all = pd.concat(all_layer_stats, ignore_index=True)
    df_summary_all = pd.concat(all_summary_stats, ignore_index=True)

    print("\n" + "=" * 100)
    print("一、每层插补前后对比统计")
    print("=" * 100)
    print(df_layer_all.to_string(index=False))

    print("\n" + "=" * 100)
    print("二、全深度完整性汇总")
    print("=" * 100)
    print(df_summary_all.to_string(index=False))

    with pd.ExcelWriter(OUT_EXCEL, engine="openpyxl") as writer:
        df_layer_all.to_excel(writer, sheet_name="每层前后对比", index=False)
        df_summary_all.to_excel(writer, sheet_name="全深度汇总", index=False)

    log(f"统计结果已保存: {OUT_EXCEL}")

    print("\n" + "#" * 100)
    log("插补前后对比检查完成")
    print("#" * 100)


if __name__ == "__main__":
    main()