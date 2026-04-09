# -*- coding: utf-8 -*-
"""
将四个新深度层土壤湿度文件：
先重采样到作物掩膜网格，再裁剪为作物种植区域文件

输出对象：
- Winter_Wheat
- Spring_Wheat
- Spring_Maize
- Summer_Maize
"""

import os
import glob
import numpy as np
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling

# =========================================================
# 1. 路径设置
# =========================================================
soil_root = r"E:\data\Soil_Humidity_ReSample"
output_root = r"E:\data\Soil_Humidity_CropClip_ToCropGrid"

soil_layer_dirs = ["0_10cm", "10_40cm", "40_80cm", "0_80cm"]

crop_masks = {
    "Winter_Wheat": r"E:\data\CropMask\Winter_Wheat_01deg_filtered.tif",
    "Spring_Wheat": r"E:\data\CropMask\Spring_Wheat_01deg_filtered.tif",
    "Spring_Maize": r"E:\data\CropMask\Spring_Maize_01deg.tif",
    "Summer_Maize": r"E:\data\CropMask\Summer_Maize_01deg.tif",
}

os.makedirs(output_root, exist_ok=True)

# =========================================================
# 2. 工具函数
# =========================================================
def detect_main_var(ds):
    """自动识别主变量名"""
    exclude_names = {"time", "lat", "lon", "latitude", "longitude", "x", "y"}
    data_vars = [v for v in ds.data_vars if v.lower() not in exclude_names]
    if len(data_vars) == 0:
        raise ValueError("未识别到有效数据变量，请检查 nc 文件。")
    return data_vars[0]


def get_lat_lon_names(da):
    """自动识别 lat/lon 坐标名"""
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
    """若纬度降序，则转成升序"""
    lat_vals = da[lat_name].values
    if len(lat_vals) > 1 and lat_vals[1] < lat_vals[0]:
        print("  ⚠️ 检测到土壤湿度纬度为降序，自动翻转为升序")
        da = da.sortby(lat_name)
    return da


def prepare_soil_da_for_rio(da, lat_name, lon_name):
    """
    将土壤 DataArray 准备成 rioxarray 可重投影对象
    坐标重命名为 x/y，并写入 CRS
    """
    rename_dict = {}
    if lon_name != "x":
        rename_dict[lon_name] = "x"
    if lat_name != "y":
        rename_dict[lat_name] = "y"

    da2 = da.rename(rename_dict)

    if not da2.rio.crs:
        da2 = da2.rio.write_crs("EPSG:4326")

    da2 = da2.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    return da2


def prepare_mask(mask_path):
    """
    读取作物掩膜，作为目标网格模板
    返回：
    - mask_da: 原始掩膜
    - mask_bin: 二值化掩膜（0/1）
    """
    mask_da = rxr.open_rasterio(mask_path, masked=True).squeeze(drop=True)

    if not mask_da.rio.crs:
        mask_da = mask_da.rio.write_crs("EPSG:4326")

    mask_da = mask_da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

    mask_bin = xr.where(np.isfinite(mask_da) & (mask_da > 0), 1, 0).astype(np.uint8)
    mask_bin = mask_bin.rio.write_crs(mask_da.rio.crs)

    return mask_da, mask_bin


def restore_coord_names(da, lat_name, lon_name):
    """把 x/y 改回原始经纬度名"""
    rename_back = {}
    if "x" != lon_name:
        rename_back["x"] = lon_name
    if "y" != lat_name:
        rename_back["y"] = lat_name
    return da.rename(rename_back)


def build_encoding(var_name):
    return {
        var_name: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
            "_FillValue": np.float32(np.nan),
        }
    }


# =========================================================
# 3. 预读取四类作物掩膜
# =========================================================
print("=" * 100)
print("读取作物掩膜（作为目标标准网格）")
print("=" * 100)

prepared_crop_info = {}
for crop_name, mask_path in crop_masks.items():
    print(f"\n读取掩膜: {crop_name}")
    mask_da, mask_bin = prepare_mask(mask_path)
    prepared_crop_info[crop_name] = {
        "mask_da": mask_da,
        "mask_bin": mask_bin
    }
    print(f"  shape: {mask_da.shape}")
    print(f"  x range: {float(mask_da.x.min()):.6f} ~ {float(mask_da.x.max()):.6f}")
    print(f"  y range: {float(mask_da.y.min()):.6f} ~ {float(mask_da.y.max()):.6f}")
    print(f"  CRS: {mask_da.rio.crs}")

# =========================================================
# 4. 逐深度层、逐文件处理
# =========================================================
for layer_name in soil_layer_dirs:
    in_dir = os.path.join(soil_root, layer_name)
    nc_files = sorted(glob.glob(os.path.join(in_dir, "*.nc")))

    if len(nc_files) == 0:
        print(f"\n⚠️ 未找到文件: {in_dir}")
        continue

    print("\n" + "=" * 100)
    print(f"开始处理深度层: {layer_name} | 文件数: {len(nc_files)}")
    print("=" * 100)

    for fp in nc_files:
        print("\n" + "-" * 100)
        print(f"处理文件: {fp}")
        print("-" * 100)

        ds = None
        try:
            ds = xr.open_dataset(fp)
            var_name = detect_main_var(ds)
            da = ds[var_name]

            lat_name, lon_name = get_lat_lon_names(da)
            da = ensure_lat_ascending(da, lat_name)

            print(f"  变量名: {var_name}")
            print(f"  dims: {da.dims}")
            print(f"  shape: {da.shape}")

            # 先准备 soil DataArray
            soil_da = prepare_soil_da_for_rio(da, lat_name, lon_name)

            # 对四种作物分别处理
            for crop_name, crop_info in prepared_crop_info.items():
                print(f"    -> 重采样并裁剪到: {crop_name}")

                target_mask_da = crop_info["mask_da"]
                target_mask_bin = crop_info["mask_bin"]

                # 1) 将土壤湿度重采样到作物掩膜网格
                # 土壤湿度是连续变量，所以建议 bilinear
                soil_on_crop_grid = soil_da.rio.reproject_match(
                    target_mask_da,
                    resampling=Resampling.bilinear
                )

                # 2) 用作物掩膜裁剪
                clipped = soil_on_crop_grid.where(target_mask_bin == 1)

                # 3) 将坐标名改回 lat/lon 风格（与掩膜一致）
                # 掩膜一般是 x/y，这里统一改成 lat/lon 方便后续使用
                clipped = clipped.rename({"x": "lon", "y": "lat"})

                # 若后续希望纬度升序，可在这里再排序一次
                if clipped.lat.size > 1 and clipped.lat.values[1] < clipped.lat.values[0]:
                    clipped = clipped.sortby("lat")

                # 4) 转成 Dataset
                out_ds = clipped.to_dataset(name=var_name)

                # 5) 添加属性
                out_ds.attrs = ds.attrs.copy()
                out_ds.attrs["crop_mask"] = crop_name
                out_ds.attrs["source_file"] = os.path.basename(fp)
                out_ds.attrs["grid_standard"] = "crop mask grid"
                out_ds.attrs["note"] = "Soil moisture first resampled to crop-mask grid, then clipped by crop mask"

                out_ds[var_name].attrs = da.attrs.copy()
                out_ds[var_name].attrs["crop_mask"] = crop_name
                out_ds[var_name].attrs["resampling_method"] = "bilinear"
                out_ds[var_name].attrs["clip_method"] = "mask == 1"

                # 6) 输出
                save_dir = os.path.join(output_root, crop_name, layer_name)
                os.makedirs(save_dir, exist_ok=True)

                out_name = os.path.basename(fp).replace(".nc", f"_{crop_name}.nc")
                save_path = os.path.join(save_dir, out_name)

                out_ds.to_netcdf(save_path, encoding=build_encoding(var_name))
                print(f"       ✅ 输出完成: {save_path}")

        except Exception as e:
            print(f"  ❌ 处理失败: {fp}")
            print(f"  错误信息: {e}")

        finally:
            if ds is not None:
                ds.close()

print("\n全部处理完成。")