import os
import glob
import random
import numpy as np
import xarray as xr
import rasterio

# =========================================================
# 1. 路径设置
# =========================================================
etpet_dir = r"E:\data\ET_minus_PET_Daily\2009"

spring_mask_path = r"E:\数据集\作物种植文件\0.1度重采样文件_修正版\Spring_Wheat_01deg_filtered.tif"
winter_mask_path = r"E:\数据集\作物种植文件\0.1度重采样文件_修正版\Winter_Wheat_01deg_filtered.tif"


# =========================================================
# 2. 工具函数：判断方向
# =========================================================
def get_axis_direction(arr, axis_name="lat"):
    arr = np.asarray(arr)
    if arr.size < 2:
        return "无法判断"

    diff = arr[1] - arr[0]
    if diff > 0:
        return f"{axis_name}递增↑"
    elif diff < 0:
        return f"{axis_name}递减↓"
    else:
        return "方向异常（坐标重复）"


# =========================================================
# 3. 工具函数：打印 NC 信息
# =========================================================
def inspect_nc_file(nc_path):
    print("\n" + "=" * 90)
    print(f"📂 随机抽取的 ET-PET 文件: {nc_path}")
    print("=" * 90)

    ds = xr.open_dataset(nc_path)

    print("\n📌 Dataset 概览:")
    print(ds)

    # 自动识别坐标名
    lat_name = None
    lon_name = None
    for c in ds.coords:
        cl = c.lower()
        if cl in ["lat", "latitude", "y"]:
            lat_name = c
        if cl in ["lon", "longitude", "x"]:
            lon_name = c

    if lat_name is None or lon_name is None:
        raise ValueError("❌ 未识别到 lat/lon 坐标，请检查文件坐标名。")

    # 自动识别变量名（优先 Diff）
    if "Diff" in ds.data_vars:
        var_name = "Diff"
    else:
        var_name = list(ds.data_vars)[0]

    lat = ds[lat_name].values
    lon = ds[lon_name].values
    data = ds[var_name]

    lat_res = float(np.abs(lat[1] - lat[0])) if len(lat) > 1 else np.nan
    lon_res = float(np.abs(lon[1] - lon[0])) if len(lon) > 1 else np.nan

    print("\n" + "-" * 90)
    print("📌 ET-PET 文件空间信息")
    print("-" * 90)
    print(f"变量名: {var_name}")
    print(f"shape: {data.shape}")
    print(f"dtype: {data.dtype}")
    print(f"lat长度: {len(lat)}")
    print(f"lon长度: {len(lon)}")
    print(f"lat范围: {lat.min()} ~ {lat.max()}")
    print(f"lon范围: {lon.min()} ~ {lon.max()}")
    print(f"lat分辨率: {lat_res}")
    print(f"lon分辨率: {lon_res}")
    print(f"纬度方向: {get_axis_direction(lat, 'lat')}")
    print(f"经度方向: {get_axis_direction(lon, 'lon')}")

    ds.close()

    return {
        "path": nc_path,
        "shape": data.shape,
        "lat_len": len(lat),
        "lon_len": len(lon),
        "lat_min": float(lat.min()),
        "lat_max": float(lat.max()),
        "lon_min": float(lon.min()),
        "lon_max": float(lon.max()),
        "lat_res": lat_res,
        "lon_res": lon_res,
        "lat_dir": get_axis_direction(lat, "lat"),
        "lon_dir": get_axis_direction(lon, "lon"),
    }


# =========================================================
# 4. 工具函数：打印 tif 掩膜信息
# =========================================================
def inspect_mask(mask_path, mask_name):
    print("\n" + "=" * 90)
    print(f"📂 {mask_name} 掩膜文件: {mask_path}")
    print("=" * 90)

    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        transform = src.transform
        bounds = src.bounds
        crs = src.crs
        height = src.height
        width = src.width
        res_x, res_y = src.res

        # 构造像元中心点坐标
        lon = np.array([transform.c + (i + 0.5) * transform.a for i in range(width)])
        lat = np.array([transform.f + (j + 0.5) * transform.e for j in range(height)])

        print("\n📌 掩膜空间信息")
        print("-" * 90)
        print(f"shape: {arr.shape}")
        print(f"dtype: {arr.dtype}")
        print(f"CRS: {crs}")
        print(f"bounds: {bounds}")
        print(f"transform: {transform}")
        print(f"分辨率: lon_res = {res_x}, lat_res = {abs(res_y)}")
        print(f"lon长度: {len(lon)}")
        print(f"lat长度: {len(lat)}")
        print(f"lon范围: {lon.min()} ~ {lon.max()}")
        print(f"lat范围: {lat.min()} ~ {lat.max()}")
        print(f"纬度方向: {get_axis_direction(lat, 'lat')}")
        print(f"经度方向: {get_axis_direction(lon, 'lon')}")
        print(f"有效像元数(>0): {np.sum(arr > 0)}")

    return {
        "path": mask_path,
        "shape": arr.shape,
        "lat_len": len(lat),
        "lon_len": len(lon),
        "lat_min": float(lat.min()),
        "lat_max": float(lat.max()),
        "lon_min": float(lon.min()),
        "lon_max": float(lon.max()),
        "lat_res": float(abs(res_y)),
        "lon_res": float(res_x),
        "lat_dir": get_axis_direction(lat, "lat"),
        "lon_dir": get_axis_direction(lon, "lon"),
        "crs": str(crs),
        "bounds": bounds,
    }


# =========================================================
# 5. 工具函数：对比 NC 与 掩膜
# =========================================================
def compare_info(nc_info, mask_info, mask_name):
    print("\n" + "=" * 90)
    print(f"🔍 ET-PET 与 {mask_name} 掩膜匹配性检查")
    print("=" * 90)

    shape_match = nc_info["shape"] == mask_info["shape"]
    lat_len_match = nc_info["lat_len"] == mask_info["lat_len"]
    lon_len_match = nc_info["lon_len"] == mask_info["lon_len"]

    lat_res_match = np.isclose(nc_info["lat_res"], mask_info["lat_res"], atol=1e-8)
    lon_res_match = np.isclose(nc_info["lon_res"], mask_info["lon_res"], atol=1e-8)

    lat_min_match = np.isclose(nc_info["lat_min"], mask_info["lat_min"], atol=1e-6)
    lat_max_match = np.isclose(nc_info["lat_max"], mask_info["lat_max"], atol=1e-6)
    lon_min_match = np.isclose(nc_info["lon_min"], mask_info["lon_min"], atol=1e-6)
    lon_max_match = np.isclose(nc_info["lon_max"], mask_info["lon_max"], atol=1e-6)

    lat_dir_match = nc_info["lat_dir"] == mask_info["lat_dir"]
    lon_dir_match = nc_info["lon_dir"] == mask_info["lon_dir"]

    print(f"shape是否一致:        {'✅' if shape_match else '❌'}   NC={nc_info['shape']} | MASK={mask_info['shape']}")
    print(f"lat长度是否一致:      {'✅' if lat_len_match else '❌'}   NC={nc_info['lat_len']} | MASK={mask_info['lat_len']}")
    print(f"lon长度是否一致:      {'✅' if lon_len_match else '❌'}   NC={nc_info['lon_len']} | MASK={mask_info['lon_len']}")

    print(f"lat分辨率是否一致:    {'✅' if lat_res_match else '❌'}   NC={nc_info['lat_res']} | MASK={mask_info['lat_res']}")
    print(f"lon分辨率是否一致:    {'✅' if lon_res_match else '❌'}   NC={nc_info['lon_res']} | MASK={mask_info['lon_res']}")

    print(f"lat最小值是否一致:    {'✅' if lat_min_match else '❌'}   NC={nc_info['lat_min']} | MASK={mask_info['lat_min']}")
    print(f"lat最大值是否一致:    {'✅' if lat_max_match else '❌'}   NC={nc_info['lat_max']} | MASK={mask_info['lat_max']}")
    print(f"lon最小值是否一致:    {'✅' if lon_min_match else '❌'}   NC={nc_info['lon_min']} | MASK={mask_info['lon_min']}")
    print(f"lon最大值是否一致:    {'✅' if lon_max_match else '❌'}   NC={nc_info['lon_max']} | MASK={mask_info['lon_max']}")

    print(f"纬度方向是否一致:      {'✅' if lat_dir_match else '❌'}   NC={nc_info['lat_dir']} | MASK={mask_info['lat_dir']}")
    print(f"经度方向是否一致:      {'✅' if lon_dir_match else '❌'}   NC={nc_info['lon_dir']} | MASK={mask_info['lon_dir']}")

    all_match = all([
        shape_match, lat_len_match, lon_len_match,
        lat_res_match, lon_res_match,
        lat_min_match, lat_max_match,
        lon_min_match, lon_max_match,
        lat_dir_match, lon_dir_match
    ])

    print("\n" + "-" * 90)
    if all_match:
        print(f"✅ 结论：ET-PET 文件与 {mask_name} 掩膜可直接匹配。")
    else:
        print(f"⚠️ 结论：ET-PET 文件与 {mask_name} 掩膜仍存在不一致，后续需先对齐/重采样。")
    print("-" * 90)


# =========================================================
# 6. 主程序
# =========================================================
def main():
    # 找到全部 2009 年 ET-PET 文件
    nc_files = sorted(glob.glob(os.path.join(etpet_dir, "Diff_2009-*.nc")))
    if len(nc_files) == 0:
        raise FileNotFoundError(f"❌ 在目录中未找到 ET-PET 文件: {etpet_dir}")

    random_nc = random.choice(nc_files)

    # 检查 ET-PET
    nc_info = inspect_nc_file(random_nc)

    # 检查春小麦掩膜
    spring_info = inspect_mask(spring_mask_path, "Spring_Wheat")

    # 检查冬小麦掩膜
    winter_info = inspect_mask(winter_mask_path, "Winter_Wheat")

    # 对比
    compare_info(nc_info, spring_info, "Spring_Wheat")
    compare_info(nc_info, winter_info, "Winter_Wheat")


if __name__ == "__main__":
    main()