import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
import matplotlib.path as mpath
import numpy as np
import pandas as pd
import os

# --- 1. 路径设置 ---
spei_path = r'E:\data\meteorologicaldata\SPEI\SPEI_30d_8dStep_Maize.nc'
shp_path = r'E:\数据集\中国_省\中国_省2.shp'
output_dir = r'E:\计算结果'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_filename = os.path.join(output_dir, '玉米典型干旱演变图_修正裁剪版.png')

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 加载数据 ---
ds = xr.open_dataset(spei_path)
spei_period = ds['SPEI_30d'].sel(time=slice('2011-06-10', '2011-08-25'))
time_steps = spei_period.time.values

# --- 3. 准备 SHP 要素与裁剪路径 ---
reader = shpreader.Reader(shp_path)
provinces_geom = list(reader.geometries())
china_feature = ShapelyFeature(provinces_geom, ccrs.PlateCarree(),
                               edgecolor='black', facecolor='none')


# 构造一个合并后的 Path 对象
def get_china_mask_path(geoms):
    all_paths = []
    for geom in geoms:
        # 处理 MultiPolygon
        parts = geom.geoms if hasattr(geom, 'geoms') else [geom]
        for part in parts:
            # 获取外部轮廓
            coords = np.asarray(part.exterior.coords)
            all_paths.append(mpath.Path(coords))
    # 合并所有省份路径
    if not all_paths: return None
    combined_path = mpath.Path.make_compound_path(*all_paths)
    return combined_path


china_mask_path = get_china_mask_path(provinces_geom)

# --- 4. 设置绘图布局 ---
ncols = 4
nrows = int(np.ceil(len(time_steps) / ncols))
fig, axes = plt.subplots(
    nrows=nrows, ncols=ncols,
    figsize=(22, 5 * nrows),
    subplot_kw={'projection': ccrs.PlateCarree()}
)
axes = axes.flatten()

levels = [-3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3]
cmap = plt.get_cmap('RdYlBu')

# --- 5. 循环绘图 ---
print(f"正在进行地理边界裁剪绘图...")
for i, t in enumerate(time_steps):
    ax = axes[i]
    data = spei_period.sel(time=t)
    date_str = pd.to_datetime(t).strftime('%Y年%m月%d日')

    # A. 核心修改：设置子图的边界为中国地图形状
    # 这会强制该子图只在 Path 内部绘制内容
    if china_mask_path is not None:
        ax.set_boundary(china_mask_path, transform=ccrs.PlateCarree())

    # B. 绘制数据
    im = data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        levels=levels,
        add_colorbar=False,
        extend='both'
    )

    # C. 添加省份边界线 (置于顶层)
    ax.add_feature(china_feature, linewidth=0.6, zorder=10)

    # D. 设置范围 (需包含九段线范围)
    ax.set_extent([73, 135, 15, 54], crs=ccrs.PlateCarree())
    ax.set_title(f"{date_str}", fontsize=14)

# --- 6. 完善细节 ---
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.subplots_adjust(bottom=0.12, wspace=0.1, hspace=0.3)
cbar_ax = fig.add_axes([0.3, 0.06, 0.4, 0.012])
fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='SPEI')

plt.suptitle("玉米关键生育期 SPEI 时空演变 (2011年)", fontsize=26, y=0.98, fontweight='bold')

# --- 7. 保存 ---
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"✅ 绘图成功！已解决 ValueError。图片保存至: {output_filename}")
plt.show()