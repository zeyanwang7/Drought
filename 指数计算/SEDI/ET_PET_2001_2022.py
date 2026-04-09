import os
import random
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# =========================================================
# 1. 路径设置
# =========================================================
et_root = r"E:\data\ET_China_Clean_Aligned_Daily"
pet_root = r"E:\data\meteorologicaldata\pet"
output_root = r"E:\data\ET_minus_PET_Daily"

start_year = 2001
end_year = 2022

PLOT_RANDOM_DAY = True

os.makedirs(output_root, exist_ok=True)

# =========================================================
# 2. 工具函数
# =========================================================
def detect_main_var(ds):
    exclude_vars = {"time_bnds", "lat_bnds", "lon_bnds", "crs", "spatial_ref"}
    vars_ = [v for v in ds.data_vars if v not in exclude_vars]
    if len(vars_) == 0:
        raise ValueError("未识别到主变量")
    return vars_[0]

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
        raise ValueError(f"无法识别经纬度坐标，coords={list(da.coords)}")

    return lat_name, lon_name

def standardize_2d_or_3d(da):
    """
    统一 DataArray 维度顺序：
    - 若有 time: (time, lat, lon)
    - 若无 time : (lat, lon)
    """
    lat_name, lon_name = detect_lat_lon(da)

    if "time" in da.dims:
        da = da.transpose("time", lat_name, lon_name)
    else:
        da = da.transpose(lat_name, lon_name)

    # 保证坐标递增
    if np.any(np.diff(da[lat_name].values) < 0):
        da = da.sortby(lat_name)
    if np.any(np.diff(da[lon_name].values) < 0):
        da = da.sortby(lon_name)

    return da

def squeeze_time_if_needed(da):
    if "time" in da.dims and da.sizes["time"] == 1:
        da = da.isel(time=0)
    return da

# =========================================================
# 3. 主处理
# =========================================================
failed_dates = []
saved_files = []

for year in range(start_year, end_year + 1):
    print("\n" + "=" * 100)
    print(f"🚀 开始处理 {year} 年")
    print("=" * 100)

    et_year_dir = os.path.join(et_root, str(year))
    pet_year_dir = os.path.join(pet_root, str(year))
    out_year_dir = os.path.join(output_root, str(year))

    os.makedirs(out_year_dir, exist_ok=True)

    if not os.path.exists(et_year_dir):
        print(f"❌ ET 目录不存在: {et_year_dir}")
        continue

    if not os.path.exists(pet_year_dir):
        print(f"❌ PET 目录不存在: {pet_year_dir}")
        continue

    et_files = sorted([
        f for f in os.listdir(et_year_dir)
        if f.lower().endswith(".nc") and f.startswith("ET_")
    ])

    if len(et_files) == 0:
        print(f"❌ {year} 年 ET 文件为空")
        continue

    print(f"📁 {year} 年 ET 文件数: {len(et_files)}")

    for idx, et_file in enumerate(et_files, start=1):
        date_str = et_file.replace("ET_", "").replace(".nc", "")   # 2001-01-01

        pet_file = os.path.join(
            pet_year_dir,
            f"ChinaMet_010deg_petPM_{date_str.replace('-', '_')}.nc"
        )
        et_path = os.path.join(et_year_dir, et_file)
        out_path = os.path.join(out_year_dir, f"Diff_{date_str}.nc")

        if not os.path.exists(pet_file):
            print(f"⚠️ PET 文件缺失，跳过: {pet_file}")
            failed_dates.append(date_str)
            continue

        try:
            ds_et = xr.open_dataset(et_path)
            ds_pet = xr.open_dataset(pet_file)

            et_var = detect_main_var(ds_et)
            pet_var = detect_main_var(ds_pet)

            da_et = ds_et[et_var].astype("float32")
            da_pet = ds_pet[pet_var].astype("float32")

            da_et = standardize_2d_or_3d(da_et)
            da_pet = standardize_2d_or_3d(da_pet)

            da_et = squeeze_time_if_needed(da_et)
            da_pet = squeeze_time_if_needed(da_pet)

            et_lat, et_lon = detect_lat_lon(da_et)
            pet_lat, pet_lon = detect_lat_lon(da_pet)

            # 强制 ET 使用 PET 的坐标名
            rename_map = {}
            if et_lat != pet_lat:
                rename_map[et_lat] = pet_lat
            if et_lon != pet_lon:
                rename_map[et_lon] = pet_lon
            if len(rename_map) > 0:
                da_et = da_et.rename(rename_map)

            # 对齐检查：维度长度
            if da_et.sizes[pet_lat] != da_pet.sizes[pet_lat] or da_et.sizes[pet_lon] != da_pet.sizes[pet_lon]:
                raise ValueError(
                    f"网格尺寸不一致: ET=({da_et.sizes[pet_lat]}, {da_et.sizes[pet_lon]}) "
                    f"PET=({da_pet.sizes[pet_lat]}, {da_pet.sizes[pet_lon]})"
                )

            # 对齐检查：坐标数组
            lat_equal = np.allclose(da_et[pet_lat].values, da_pet[pet_lat].values, equal_nan=True)
            lon_equal = np.allclose(da_et[pet_lon].values, da_pet[pet_lon].values, equal_nan=True)

            if not lat_equal or not lon_equal:
                raise ValueError(f"网格坐标不一致: lat_equal={lat_equal}, lon_equal={lon_equal}")

            # 差值
            da_diff = (da_et - da_pet).astype("float32")

            # 加回 time 维
            time_val = np.datetime64(date_str)
            da_diff = da_diff.expand_dims(time=[time_val])

            # 命名为 Diff
            da_diff.name = "Diff"
            ds_out = da_diff.to_dataset()

            # 保留属性
            ds_out["Diff"].attrs = {
                "long_name": "ET minus PET",
                "description": "Daily ET minus PET over China, aligned to PET grid",
                "units": da_pet.attrs.get("units", ""),
                "source_ET": et_path,
                "source_PET": pet_file,
                "dtype_saved": "float32"
            }

            encoding = {
                "Diff": {
                    "zlib": True,
                    "complevel": 4,
                    "dtype": "float32",
                    "_FillValue": np.float32(np.nan)
                }
            }
            for c in ds_out.coords:
                if c not in encoding:
                    encoding[c] = {}

            ds_out.to_netcdf(out_path, encoding=encoding)

            if year == start_year and idx <= 3:
                saved_files.append(out_path)
            elif random.random() < 0.003:
                saved_files.append(out_path)

            if idx % 30 == 0 or idx == 1 or idx == len(et_files):
                arr = ds_out["Diff"].isel(time=0).values
                valid = np.isfinite(arr)
                print(f"  ✅ {year} 已完成 {idx}/{len(et_files)} | {date_str} | 有效像元={valid.sum()}")

            ds_et.close()
            ds_pet.close()
            ds_out.close()

        except Exception as e:
            print(f"❌ {date_str} 处理失败: {e}")
            failed_dates.append(date_str)

# =========================================================
# 4. 随机绘图检查
# =========================================================
if PLOT_RANDOM_DAY and len(saved_files) > 0:
    print("\n" + "=" * 100)
    print("🖼️ 随机抽取一个 ET-PET 文件绘图检查")
    print("=" * 100)

    rand_file = random.choice(saved_files)
    print(f"随机抽中的文件: {rand_file}")

    ds_check = xr.open_dataset(rand_file)
    da_check = ds_check["Diff"]

    da_plot = da_check.isel(time=0) if "time" in da_check.dims else da_check
    lat_name, lon_name = detect_lat_lon(da_plot)

    arr = da_plot.values
    valid = np.isfinite(arr)

    date_str = str(np.datetime_as_string(ds_check["time"].values[0], unit="D")) if "time" in ds_check.coords else "NoTime"

    print(f"随机日期: {date_str}")
    print(f"有效像元数: {valid.sum()}")
    if valid.sum() > 0:
        print(f"最小值: {np.nanmin(arr):.4f}")
        print(f"最大值: {np.nanmax(arr):.4f}")
        print(f"均值  : {np.nanmean(arr):.4f}")

    plt.figure(figsize=(10, 8))
    im = plt.pcolormesh(
        da_plot[lon_name].values,
        da_plot[lat_name].values,
        arr,
        shading="auto",
        cmap="viridis"
    )
    plt.colorbar(im, label="Diff (ET - PET)")
    plt.title(f"Random Check: {os.path.basename(rand_file)} | {date_str}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    plt.show()

    ds_check.close()

# =========================================================
# 5. 汇总
# =========================================================
print("\n" + "=" * 100)
print("🎯 全部处理完成")
print("=" * 100)

if len(failed_dates) == 0:
    print("✅ 2001-2022 每日 ET-PET 文件全部处理成功")
else:
    print(f"⚠️ 共有 {len(failed_dates)} 个日期处理失败或缺文件")
    print(failed_dates[:50])
    if len(failed_dates) > 50:
        print("... 其余失败日期已省略")