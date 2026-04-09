import os
import numpy as np
import pandas as pd
import xarray as xr

# =============================================================================
# 1. 路径设置
# =============================================================================
INPUT_DIR = r"E:\data\SoilData\Wheat_0p1deg_Masked_Filled"
OUTPUT_DIR = r"E:\data\SoilData\Wheat_0p1deg_Masked_Filled_Resampled"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECK_EXCEL = os.path.join(OUTPUT_DIR, "Soil_Resampled_CheckTable.xlsx")

# 只处理插补后的文件
TARGET_FILES = None
# TARGET_FILES = [
#     r"E:\data\SoilData\Wheat_0p1deg_Masked_Filled\TH1500_Winter_Wheat_0p1deg_masked_filled.nc",
#     r"E:\data\SoilData\Wheat_0p1deg_Masked_Filled\TH33_Winter_Wheat_0p1deg_masked_filled.nc"
# ]


# =============================================================================
# 2. 原始7层边界（cm）
# =============================================================================
src_bounds_cm = np.array([
    [0.0, 4.5],
    [4.5, 9.1],
    [9.1, 16.6],
    [16.6, 28.9],
    [28.9, 49.3],
    [49.3, 82.9],
    [82.9, 138.3]
], dtype=float)

# =============================================================================
# 3. 目标层（只到80cm）
# =============================================================================
target_layers = {
    "0_10cm":  (0.0, 10.0),
    "10_40cm": (10.0, 40.0),
    "40_80cm": (40.0, 80.0),
    "0_80cm":  (0.0, 80.0),
}


# =============================================================================
# 4. 工具函数
# =============================================================================
def log(msg):
    print(msg, flush=True)


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

    if len(var_names) == 1:
        return var_names[0]

    raise ValueError(f"无法唯一识别主变量，请检查变量名: {var_names}")


def calc_overlap_thickness(src_top, src_bottom, tgt_top, tgt_bottom):
    return max(0.0, min(src_bottom, tgt_bottom) - max(src_top, tgt_top))


def resample_soil_profile(da, src_bounds_cm, target_layers):
    if "depth" not in da.dims:
        raise ValueError("输入数据必须包含 depth 维度")

    if da.sizes["depth"] != len(src_bounds_cm):
        raise ValueError(
            f"depth维长度({da.sizes['depth']}) 与原始层数({len(src_bounds_cm)})不一致"
        )

    out_layers = []

    for layer_name, (tgt_top, tgt_bottom) in target_layers.items():
        weighted_sum = None
        total_w = 0.0

        for i, (src_top, src_bottom) in enumerate(src_bounds_cm):
            overlap = calc_overlap_thickness(src_top, src_bottom, tgt_top, tgt_bottom)

            if overlap > 0:
                layer_da = da.isel(depth=i)
                tmp = layer_da * overlap

                if weighted_sum is None:
                    weighted_sum = tmp
                else:
                    weighted_sum = weighted_sum + tmp

                total_w += overlap

        if total_w == 0:
            raise ValueError(f"目标层 {layer_name} 没有与任何原始层重叠")

        new_layer = (weighted_sum / total_w).expand_dims(dim={"depth": [layer_name]})
        out_layers.append(new_layer)

    return xr.concat(out_layers, dim="depth")


def build_check_table(da_resampled, file_name, var_name):
    records = []

    for i in range(da_resampled.sizes["depth"]):
        depth_label = str(da_resampled["depth"].values[i])
        arr = da_resampled.isel(depth=i).values

        total_count = arr.size
        valid_count = int(np.isfinite(arr).sum())
        missing_count = int(np.isnan(arr).sum())
        missing_ratio = missing_count / total_count * 100.0

        if valid_count > 0:
            min_val = float(np.nanmin(arr))
            max_val = float(np.nanmax(arr))
            mean_val = float(np.nanmean(arr))
        else:
            min_val = np.nan
            max_val = np.nan
            mean_val = np.nan

        records.append({
            "File": file_name,
            "Variable": var_name,
            "Depth": depth_label,
            "Total_Pixels": total_count,
            "Valid_Pixels": valid_count,
            "Missing_Pixels": missing_count,
            "Missing_Ratio_%": round(missing_ratio, 4),
            "Min": min_val,
            "Max": max_val,
            "Mean": mean_val
        })

    return pd.DataFrame(records)


def get_output_nc_path(input_path):
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(OUTPUT_DIR, f"{base}_resampled_4depths.nc")


# =============================================================================
# 5. 主程序
# =============================================================================
def main():
    log("=" * 100)
    log("开始对插补后的土壤文件进行垂向重采样")
    log("=" * 100)

    if TARGET_FILES is not None:
        nc_files = TARGET_FILES
    else:
        nc_files = [
            os.path.join(INPUT_DIR, f)
            for f in os.listdir(INPUT_DIR)
            if f.lower().endswith(".nc") and f.lower().endswith("_filled.nc")
        ]

    if len(nc_files) == 0:
        log(f"未找到插补后的 nc 文件: {INPUT_DIR}")
        return

    log(f"共找到 {len(nc_files)} 个插补后文件")

    all_check_tables = []

    for nc_path in nc_files:
        log("\n" + "-" * 100)
        log(f"处理文件: {nc_path}")

        try:
            ds = xr.open_dataset(nc_path)
            var_name = find_main_var(ds)
            da = standardize_lat_lon_da(ds[var_name])

            log(f"变量名: {var_name}")
            log(f"dims: {da.dims}")
            log(f"shape: {da.shape}")

            da_resampled = resample_soil_profile(da, src_bounds_cm, target_layers)

            new_ds = da_resampled.to_dataset(name=var_name)
            new_ds[var_name].attrs = da.attrs.copy()
            new_ds[var_name].attrs["description"] = (
                "Thickness-weighted vertical resampling from 7 original layers "
                "to 0-10 cm, 10-40 cm, 40-80 cm, and 0-80 cm"
            )
            new_ds[var_name].attrs["note"] = (
                "Resampling was performed on filled crop-masked soil files. "
                "Only the portion above 80 cm was used."
            )

            output_nc = get_output_nc_path(nc_path)
            encoding = {
                var_name: {
                    "zlib": True,
                    "complevel": 4,
                    "dtype": "float32",
                    "_FillValue": np.float32(np.nan)
                }
            }
            new_ds.to_netcdf(output_nc, encoding=encoding)

            log(f"输出完成: {output_nc}")

            check_df = build_check_table(
                da_resampled=da_resampled,
                file_name=os.path.basename(nc_path),
                var_name=var_name
            )
            all_check_tables.append(check_df)

        except Exception as e:
            log(f"处理失败: {nc_path}")
            log(f"错误信息: {e}")

    if len(all_check_tables) > 0:
        final_check_df = pd.concat(all_check_tables, ignore_index=True)

        with pd.ExcelWriter(CHECK_EXCEL, engine="openpyxl") as writer:
            final_check_df.to_excel(writer, sheet_name="Check_Table", index=False)

        log("\n" + "=" * 100)
        log("全部重采样完成")
        log(f"检查表已输出: {CHECK_EXCEL}")
        log("=" * 100)
        print(final_check_df)
    else:
        log("没有成功处理任何文件。")


if __name__ == "__main__":
    main()