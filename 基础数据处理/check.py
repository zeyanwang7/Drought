import rioxarray
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os

# --- 配置路径 ---
# 替换为您导出的 NDWI 文件路径
tif_path = r"E:\数据集\作物物候\NDWI_China_010deg_2020_04_001.tif"
# 建议下载 1:1000万或更详细的中国省界 Shapefile，如果没有，代码将尝试下载在线底图
# 也可以直接使用 geopandas 自带的低精度地图或您本地的省界 shp
province_shp = None  # 如果有本地 shp 路径请填写，例如 r"D:\map\china_provinces.shp"


def check_and_plot(tif_file, shp_file=None):
    if not os.path.exists(tif_file):
        print(f"错误：找不到文件 {tif_file}")
        return

    # 1. 读取 TIF 数据
    rds = rioxarray.open_rasterio(tif_file)

    # 检查基本信息
    print("-" * 30)
    print(f"文件名: {os.path.basename(tif_file)}")
    print(f"坐标系 (CRS): {rds.rio.crs}")
    print(f"影像尺寸 (WxH): {rds.rio.width} x {rds.rio.height}")
    print(f"分辨率: {rds.rio.resolution()}")
    print(f"数值范围: [{rds.min().values:.3f}, {rds.max().values:.3f}]")
    print("-" * 30)

    # 2. 准备绘图
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 设置中国区域范围 [经度min, 经度max, 纬度min, 纬度max]
    ax.set_extent([73, 135, 18, 54], crs=ccrs.PlateCarree())

    # 3. 绘制 TIF 栅格
    # 如果 NDWI 的维度中有 'band'，我们选第一个波段
    if 'band' in rds.dims:
        data_to_plot = rds.isel(band=0)
    else:
        data_to_plot = rds

    # 使用 pcolormesh 绘图
    im = data_to_plot.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='YlGn',  # 植被指数常用黄-绿配色
        add_colorbar=True,
        cbar_kwargs={'label': 'NDWI Value', 'shrink': 0.6}
    )

    # 4. 叠加省界
    if shp_file and os.path.exists(shp_file):
        china_map = gpd.read_file(shp_file)
        china_map.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5, transform=ccrs.PlateCarree())
    else:
        # 如果没有本地 shp，使用 Cartopy 提供的国家/海岸线底图
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.6)
        print("提示：未加载本地省界 SHP，使用系统内置基础底图。")

    # 5. 修饰地图
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linestyle='--')
    plt.title(f"Check NDWI Grid: {os.path.basename(tif_file)}", fontsize=14)

    plt.show()


# --- 执行 ---
check_and_plot(tif_path, province_shp)