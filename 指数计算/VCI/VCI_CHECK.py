import os
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import numpy as np

# ==============================
# 1. 路径
# ==============================
vci_dir = r"E:\data\VCI\Spring_Wheat_2001_2020"
shp_path = r"E:\数据集\中国_省\海南省_省.shp"
out_dir = r"E:\data\VCI\Spring_Wheat_2001_2020_noHainan"

os.makedirs(out_dir, exist_ok=True)

# ==============================
# 2. 读取海南 shp（只读一次）
# ==============================
print("🚀 读取海南省边界...")

gdf = gpd.read_file(shp_path)
gdf = gdf.to_crs("EPSG:4326")

# ==============================
# 3. 处理年份（2002–2020）
# ==============================
years = [str(y) for y in range(2001, 2021)]

for year in years:

    print(f"\n👉 处理 {year}")

    vci_path = os.path.join(vci_dir, f"VCI_{year}.nc")
    out_path = os.path.join(out_dir, f"VCI_{year}_noHainan.nc")

    # ==============================
    # 4. 读取 VCI
    # ==============================
    ds = xr.open_dataset(vci_path)
    vci = ds["VCI"]

    # ==============================
    # 5. 坐标标准化
    # ==============================
    if "lon" in vci.coords:
        vci = vci.rename({"lon": "x"})
    if "lat" in vci.coords:
        vci = vci.rename({"lat": "y"})

    # 防止翻转（关键）
    if vci.y[0] > vci.y[-1]:
        vci = vci.sortby("y")

    # 写入 CRS
    vci.rio.write_crs("EPSG:4326", inplace=True)

    # ==============================
    # 6. 生成海南掩膜
    # ==============================
    mask_hainan = vci.rio.clip(gdf.geometry, gdf.crs, drop=False)

    # ==============================
    # 7. 剔除海南区域
    # ==============================
    vci_no_hainan = vci.where(mask_hainan.isnull())

    # ==============================
    # 8. 压缩保存
    # ==============================
    vci_no_hainan = vci_no_hainan.round(2)
    vci_no_hainan.name = "VCI"

    encoding = {
        "VCI": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 5
        }
    }

    vci_no_hainan.to_netcdf(out_path, encoding=encoding)

    print(f"✅ 已完成 {year}")

print("\n🎉 2002–2020 海南剔除完成！")