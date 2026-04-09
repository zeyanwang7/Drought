import rasterio
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 路径
# ===============================
spring_path = r"E:\数据集\作物种植文件\0.1度重采样文件_修正版\Spring_Wheat_01deg_filtered.tif"
winter_path = r"E:\数据集\作物种植文件\0.1度重采样文件_修正版\Winter_Wheat_01deg_filtered.tif"
shp_path = r"E:\数据集\中国_省\中国_省2.shp"

# ===============================
# 读取省界
# ===============================
try:
    gdf = gpd.read_file(shp_path, encoding='utf-8')
except:
    gdf = gpd.read_file(shp_path, encoding='gbk')


# ===============================
# 绘图函数
# ===============================
def plot_crop(raster_path, title):

    print(f"\n🎨 绘制: {title}")

    with rasterio.open(raster_path) as src:
        data = src.read(1)
        bounds = src.bounds
        crs = src.crs

    # CRS对齐
    if gdf.crs != crs:
        gdf_plot = gdf.to_crs(crs)
    else:
        gdf_plot = gdf

    mask = np.where(data == 1, 1, np.nan)

    fig, ax = plt.subplots(figsize=(10, 8))

    # ===============================
    # 栅格（正确）
    # ===============================
    im = ax.imshow(
        mask,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin='lower'
    )

    # ===============================
    # 省界
    # ===============================
    gdf_plot.boundary.plot(ax=ax, linewidth=0.8, color='black')

    # ===============================
    # 🚀 关键修复（强制坐标方向一致）
    # ===============================
    ax.set_ylim(bounds.bottom, bounds.top)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.show()


# ===============================
# 执行
# ===============================
plot_crop(spring_path, "Spring Wheat (Filtered)")
plot_crop(winter_path, "Winter Wheat (Filtered)")