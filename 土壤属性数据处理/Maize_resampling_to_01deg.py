# -*- coding: utf-8 -*-
"""
将 TH1500 / TH33 重采样到春夏玉米 0.1° 种植区
功能：
1. 自动识别主变量（TH1500 / TH33）
2. 将土壤数据插值到玉米作物掩膜网格
3. 仅保留玉米种植区
4. 裁剪到有效种植范围，减少空白区
5. 输出压缩 NetCDF
"""

import os
import warnings
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")


# =============================================================================
# 1. 路径设置
# =============================================================================
SOIL_FILES = [
    r"E:\data\SoilData\TH1500.nc",
    r"E:\data\SoilData\TH33.nc"
]

CROP_MASKS = {
    "Spring_Maize": r"E:\data\CropMask\Spring_Maize_01deg.tif",
    "Summer_Maize": r"E:\data\CropMask\Summer_Maize_01deg.tif",
}

OUTPUT_DIR = r"E:\data\SoilData\Maize_0p1deg_Masked"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# 2. 通用函数
# =============================================================================
def open_nc_safely(nc_path):
    """安全打开 nc 文件，优先默认方式，失败后尝试其他 engine。"""
    print(f"尝试打开文件: {nc_path}")
    print(f"os.path.exists = {os.path.exists(nc_path)}")

    if not os.path.exists(nc_path):
        raise FileNotFoundError(f"文件不存在: {nc_path}")

    engines_to_try = [None, "h5netcdf", "netcdf4", "scipy"]
    last_error = None

    for eng in engines_to_try:
        try:
            if eng is None:
                ds = xr.open_dataset(nc_path)
                print("✅ 默认引擎打开成功")
            else:
                ds = xr.open_dataset(nc_path, engine=eng)
                print(f"✅ 使用 engine={eng} 打开成功")
            return ds
        except Exception as e:
            last_error = e
            print(f"⚠️ 打开失败 engine={eng}: {e}")

    raise RuntimeError(f"所有方式均无法打开文件: {nc_path}\n最后错误: {last_error}")


def find_data_var(ds):
    """
    自动识别主变量。
    优先跳过 cv / std / err / flag 等辅助变量。
    """
    var_names = list(ds.data_vars)

    preferred_order = ["TH1500", "TH33"]
    for name in preferred_order:
        if name in var_names:
            return name

    exclude_keywords = ["cv", "std", "err", "uncertainty", "quality", "flag"]
    main_vars = [
        v for v in var_names
        if not any(k in v.lower() for k in exclude_keywords)
    ]

    if len(main_vars) == 1:
        return main_vars[0]

    if len(var_names) == 1:
        return var_names[0]

    raise ValueError(f"无法唯一识别主变量，请检查变量名：{var_names}")


def standardize_lat_lon(da):
    """统一坐标名为 lat/lon，并保证 lat/lon 升序。"""
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
        raise ValueError("数据中未找到 lat/lon 坐标，请检查原始数据。")

    if da["lat"].values[0] > da["lat"].values[-1]:
        da = da.sortby("lat")

    if da["lon"].values[0] > da["lon"].values[-1]:
        da = da.sortby("lon")

    return da


def load_crop_mask(mask_path):
    """
    读取作物掩膜 tif
    输出 DataArray: dims = (lat, lon)
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"作物掩膜不存在: {mask_path}")

    da = xr.open_dataarray(mask_path, engine="rasterio").squeeze(drop=True)

    rename_dict = {}
    for c in da.coords:
        if c.lower() == "x":
            rename_dict[c] = "lon"
        elif c.lower() == "y":
            rename_dict[c] = "lat"
    if rename_dict:
        da = da.rename(rename_dict)

    if "band" in da.dims:
        da = da.squeeze("band", drop=True)

    da = standardize_lat_lon(da)

    mask = xr.where(np.isfinite(da) & (da > 0), 1, 0)
    mask.name = "crop_mask"

    return mask


def prepare_soil_dataarray(soil_path):
    """读取土壤主变量，并统一坐标。"""
    print(f"\n读取土壤文件: {soil_path}")
    ds = open_nc_safely(soil_path)

    var_name = find_data_var(ds)
    print(f"识别到的数据变量: {var_name}")

    da = ds[var_name]
    da = standardize_lat_lon(da)

    return da, var_name


def interpolate_to_crop_grid(soil_da, crop_mask):
    """
    将土壤数据插值到作物掩膜网格
    """
    target_lat = crop_mask["lat"]
    target_lon = crop_mask["lon"]

    soil_interp = soil_da.interp(
        lat=target_lat,
        lon=target_lon,
        method="nearest"
    )

    return soil_interp


def apply_crop_mask(soil_interp, crop_mask):
    """仅保留作物种植区。"""
    masked = soil_interp.where(crop_mask == 1)
    return masked


def crop_to_valid_extent(masked_da, crop_mask):
    """
    裁剪到作物有效范围，减少空白区。
    """
    valid_rows = np.where(crop_mask.sum(dim="lon").values > 0)[0]
    valid_cols = np.where(crop_mask.sum(dim="lat").values > 0)[0]

    if len(valid_rows) == 0 or len(valid_cols) == 0:
        raise ValueError("作物掩膜没有有效种植区。")

    lat_min_idx, lat_max_idx = valid_rows.min(), valid_rows.max()
    lon_min_idx, lon_max_idx = valid_cols.min(), valid_cols.max()

    cropped = masked_da.isel(
        lat=slice(lat_min_idx, lat_max_idx + 1),
        lon=slice(lon_min_idx, lon_max_idx + 1)
    )

    return cropped


def build_output_path(soil_path, crop_name, output_dir):
    soil_base = os.path.splitext(os.path.basename(soil_path))[0]
    out_name = f"{soil_base}_{crop_name}_0p1deg_masked.nc"
    return os.path.join(output_dir, out_name)


def save_to_netcdf(da, out_path, var_name):
    """保存为压缩 NetCDF。"""
    da = da.astype(np.float32)
    da.name = var_name

    encoding = {
        var_name: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan)
        }
    }

    ds_out = da.to_dataset(name=var_name)
    ds_out.to_netcdf(out_path, encoding=encoding)
    print(f"✅ 已输出: {out_path}")


# =============================================================================
# 3. 主处理流程
# =============================================================================
def process_one_soil_for_one_crop(soil_path, crop_name, crop_mask_path, output_dir):
    print("=" * 100)
    print(f"开始处理: soil={os.path.basename(soil_path)} | crop={crop_name}")
    print("=" * 100)

    # 1) 读取作物掩膜
    crop_mask = load_crop_mask(crop_mask_path)
    print(f"作物掩膜 shape: {crop_mask.shape}")
    print(f"作物有效像元数: {int((crop_mask == 1).sum().values)}")
    print(f"lat范围: {float(crop_mask.lat.min().values):.6f} ~ {float(crop_mask.lat.max().values):.6f}")
    print(f"lon范围: {float(crop_mask.lon.min().values):.6f} ~ {float(crop_mask.lon.max().values):.6f}")

    # 2) 读取土壤数据
    soil_da, src_var_name = prepare_soil_dataarray(soil_path)
    print(f"原始土壤数据 dims: {soil_da.dims}")
    print(f"原始土壤数据 shape: {soil_da.shape}")

    # 3) 插值到作物网格
    soil_interp = interpolate_to_crop_grid(soil_da, crop_mask)
    print(f"插值后 shape: {soil_interp.shape}")

    # 4) 掩膜
    masked_da = apply_crop_mask(soil_interp, crop_mask)

    # 5) 裁剪到有效种植区
    cropped_da = crop_to_valid_extent(masked_da, crop_mask)
    print(f"裁剪后 shape: {cropped_da.shape}")

    # 6) 输出
    out_path = build_output_path(soil_path, crop_name, output_dir)
    save_to_netcdf(cropped_da, out_path, src_var_name)


def main():
    print("=" * 100)
    print("开始重采样 TH1500 / TH33 到春夏玉米 0.1° 种植区")
    print("=" * 100)

    for soil_path in SOIL_FILES:
        for crop_name, crop_mask_path in CROP_MASKS.items():
            process_one_soil_for_one_crop(
                soil_path=soil_path,
                crop_name=crop_name,
                crop_mask_path=crop_mask_path,
                output_dir=OUTPUT_DIR
            )

    print("\n" + "=" * 100)
    print("全部处理完成")
    print("=" * 100)


if __name__ == "__main__":
    main()