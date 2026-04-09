import os
import random
import numpy as np
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

# =========================================================
# 1. 路径设置
# =========================================================
input_dir = r"E:\data\ET_China_Clean"   # 清理负值后的 ET 年文件
pet_template_path = r"E:\data\meteorologicaldata\pet\2009\ChinaMet_010deg_petPM_2009_01_01.nc"
output_root = r"E:\data\ET_China_Clean_Aligned_Daily"

os.makedirs(output_root, exist_ok=True)

start_year = 2014
end_year = 2022

# 是否随机绘图检查
PLOT_RANDOM_DAY = True

# =========================================================
# 2. 坐标识别函数
# =========================================================
def detect_lat_lon(da):
    lat_candidates = ["lat", "latitude", "Latitude", "LAT", "y"]
    lon_candidates = ["lon", "longitude", "Longitude", "LON", "x"]

    lat_name = None
    lon_name = None

    for c in lat_candidates:
        if c in da.coords:
            lat_name = c
            break

    for c in lon_candidates:
        if c in da.coords:
            lon_name = c
            break

    if lat_name is None or lon_name is None:
        raise ValueError(f"无法识别经纬度坐标，当前 coords: {list(da.coords)}")

    return lat_name, lon_name


# =========================================================
# 3. 数据变量识别函数
# =========================================================
def detect_main_var(ds):
    exclude_vars = {"time_bnds", "lat_bnds", "lon_bnds", "crs", "spatial_ref"}
    vars_ = [v for v in ds.data_vars if v not in exclude_vars]
    if len(vars_) == 0:
        raise ValueError("未识别到主变量")
    return vars_[0]


# =========================================================
# 4. 打开 PET 模板，作为目标网格
# =========================================================
if not os.path.exists(pet_template_path):
    raise FileNotFoundError(f"未找到 PET 模板文件: {pet_template_path}")

ds_pet = xr.open_dataset(pet_template_path)
pet_var = detect_main_var(ds_pet)
da_pet = ds_pet[pet_var]

pet_lat, pet_lon = detect_lat_lon(da_pet)

# 保证 PET 模板坐标递增
if np.any(np.diff(da_pet[pet_lat].values) < 0):
    da_pet = da_pet.sortby(pet_lat)
if np.any(np.diff(da_pet[pet_lon].values) < 0):
    da_pet = da_pet.sortby(pet_lon)

da_pet = da_pet.rio.set_spatial_dims(x_dim=pet_lon, y_dim=pet_lat, inplace=False)
da_pet = da_pet.rio.write_crs("EPSG:4326", inplace=False)

print("=" * 100)
print("✅ PET 模板读取成功")
print("=" * 100)
print(f"PET 主变量名: {pet_var}")
print(f"PET 坐标名  : lat={pet_lat}, lon={pet_lon}")
print(f"PET shape   : ({da_pet.sizes[pet_lat]}, {da_pet.sizes[pet_lon]})")
print(f"PET lat范围 : {float(da_pet[pet_lat].min())} ~ {float(da_pet[pet_lat].max())}")
print(f"PET lon范围 : {float(da_pet[pet_lon].min())} ~ {float(da_pet[pet_lon].max())}")

saved_daily_files = []
failed_years = []

# =========================================================
# 5. 批量处理 2001-2022
# =========================================================
for year in range(start_year, end_year + 1):
    print("\n" + "=" * 100)
    print(f"🚀 开始处理 {year}")
    print("=" * 100)

    in_file = os.path.join(input_dir, f"ET.SiTHv2.A{year}_ChinaClip_clean.nc")
    out_year_dir = os.path.join(output_root, str(year))
    os.makedirs(out_year_dir, exist_ok=True)

    if not os.path.exists(in_file):
        print(f"❌ 文件不存在，跳过: {in_file}")
        failed_years.append(year)
        continue

    try:
        ds_et = xr.open_dataset(in_file)
        et_var = detect_main_var(ds_et)
        da_et = ds_et[et_var]

        et_lat, et_lon = detect_lat_lon(da_et)

        print(f"ET 主变量名: {et_var}")
        print(f"ET 坐标名  : lat={et_lat}, lon={et_lon}")
        print(f"ET 原始维度: {da_et.dims}")
        print(f"ET 原始形状: {da_et.shape}")

        # 统一维度顺序
        if "time" in da_et.dims:
            da_et = da_et.transpose("time", et_lat, et_lon)
        else:
            da_et = da_et.transpose(et_lat, et_lon)

        # 坐标递增
        if np.any(np.diff(da_et[et_lat].values) < 0):
            da_et = da_et.sortby(et_lat)
        if np.any(np.diff(da_et[et_lon].values) < 0):
            da_et = da_et.sortby(et_lon)

        da_et = da_et.rio.set_spatial_dims(x_dim=et_lon, y_dim=et_lat, inplace=False)
        da_et = da_et.rio.write_crs("EPSG:4326", inplace=False)

        # 如果没有 time 维，则无法按天拆分
        if "time" not in da_et.dims:
            raise ValueError(f"{year} 年文件没有 time 维，无法按天保存。")

        nt = da_et.sizes["time"]
        print(f"该年份时间长度: {nt}")

        for i in range(nt):
            da_day = da_et.isel(time=i)
            date_val = da_et["time"].values[i]
            date_str = str(np.datetime_as_string(date_val, unit="D"))

            # 重采样并匹配到 PET 模板网格
            da_resampled = da_day.rio.reproject_match(
                da_pet,
                resampling=Resampling.bilinear
            )

            # 可能返回 x/y，统一改名
            rename_map = {}
            if "y" in da_resampled.dims:
                rename_map["y"] = pet_lat
            if "x" in da_resampled.dims:
                rename_map["x"] = pet_lon
            if len(rename_map) > 0:
                da_resampled = da_resampled.rename(rename_map)

            # 强制覆盖坐标，确保和 PET 完全一致
            da_resampled = da_resampled.assign_coords({
                pet_lat: da_pet[pet_lat].values,
                pet_lon: da_pet[pet_lon].values
            })

            da_resampled = da_resampled.transpose(pet_lat, pet_lon)

            # 恢复 time 维，保存为单日文件
            da_resampled = da_resampled.expand_dims(time=[date_val]).astype("float32")

            ds_out = da_resampled.to_dataset(name=et_var)
            ds_out[et_var].attrs.update(da_et.attrs)
            ds_out[et_var].attrs["note"] = "Resampled and aligned to PET template grid; saved as daily file"
            ds_out[et_var].attrs["target_grid_file"] = pet_template_path
            ds_out[et_var].attrs["dtype_saved"] = "float32"

            out_file = os.path.join(out_year_dir, f"ET_{date_str}.nc")

            encoding = {
                et_var: {
                    "zlib": True,
                    "complevel": 4,
                    "dtype": "float32",
                    "_FillValue": np.float32(np.nan)
                }
            }
            for c in ds_out.coords:
                if c not in encoding:
                    encoding[c] = {}

            ds_out.to_netcdf(out_file, encoding=encoding)

            if year == start_year and i < 3:
                saved_daily_files.append(out_file)
            elif random.random() < 0.003:
                saved_daily_files.append(out_file)

            if (i + 1) % 30 == 0 or (i + 1) == 1 or (i + 1) == nt:
                print(f"  ✅ {year} 已完成 {i + 1}/{nt} | {date_str}")

            ds_out.close()

        ds_et.close()
        print(f"✅ {year} 年处理完成")

    except Exception as e:
        print(f"❌ {year} 年处理失败: {e}")
        failed_years.append(year)

# =========================================================
# 6. 随机抽取一个生成文件绘图检查
# =========================================================
if PLOT_RANDOM_DAY and len(saved_daily_files) > 0:
    print("\n" + "=" * 100)
    print("🖼️ 随机抽取一个单日文件绘图检查")
    print("=" * 100)

    rand_file = random.choice(saved_daily_files)
    print(f"随机抽中的文件: {rand_file}")

    ds_check = xr.open_dataset(rand_file)
    check_var = detect_main_var(ds_check)
    da_check = ds_check[check_var]

    check_lat, check_lon = detect_lat_lon(da_check)

    if "time" in da_check.dims and da_check.sizes["time"] > 0:
        rand_time = str(np.datetime_as_string(da_check["time"].values[0], unit="D"))
        da_plot = da_check.isel(time=0)
    else:
        rand_time = "NoTime"
        da_plot = da_check

    da_plot = da_plot.transpose(check_lat, check_lon)

    arr = da_plot.values
    valid = np.isfinite(arr)

    print(f"随机日期: {rand_time}")
    print(f"有效像元数: {valid.sum()}")
    if valid.sum() > 0:
        print(f"最小值: {np.nanmin(arr):.4f}")
        print(f"最大值: {np.nanmax(arr):.4f}")
        print(f"均值  : {np.nanmean(arr):.4f}")

    plt.figure(figsize=(10, 8))
    im = plt.pcolormesh(
        da_plot[check_lon].values,
        da_plot[check_lat].values,
        arr,
        shading="auto",
        cmap="viridis"
    )
    plt.colorbar(im, label=check_var)
    plt.title(f"Random Check: {os.path.basename(rand_file)} | {rand_time}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()

    ds_check.close()

# =========================================================
# 7. 汇总
# =========================================================
print("\n" + "=" * 100)
print("🎯 全部处理完成")
print("=" * 100)

if len(failed_years) == 0:
    print("✅ 2001-2022 全部年份处理成功")
else:
    print("⚠️ 以下年份处理失败或缺文件：")
    print(failed_years)