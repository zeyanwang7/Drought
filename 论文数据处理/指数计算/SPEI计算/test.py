import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# --- 1. 设置文件路径 ---
# 你可以把路径换成 SPEI 或 SVPD 的文件
file_path = r'E:\data\meteorologicaldata\IndexCalculationResults\SPEI\SPEI_30d_8dStep_Wheat_2001_2022.nc'
var_name = 'SPEI_30d'  # 对应文件中的变量名


def plot_check_map(time_index=100):
    """
    time_index: 整数，代表时间轴上的第几个步长（例如 100 代表 2003 年左右）
    """
    # --- 2. 加载数据 ---
    ds = xr.open_dataset(file_path)
    data = ds[var_name].isel(time=time_index)
    time_str = str(data.time.values)[:10]

    # --- 3. 创建画布 ---
    plt.figure(figsize=(12, 8), dpi=100)
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 添加地理底图
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', linewidth=0.5)

    # --- 4. 绘图 ---
    # cmap 选择 'RdYlBu'，干旱（负值）显红色，湿润显蓝色
    im = data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='RdYlBu',
        vmin=-2.5, vmax=2.5,  # 标准化指数通常在这个范围
        add_colorbar=True,
        cbar_kwargs={'label': f'{var_name} Value', 'shrink': 0.6}
    )

    # --- 5. 细节优化 ---
    ax.set_title(f"Spatial Check: {var_name} on {time_str}\n(Masked for Wheat Area)", fontsize=14)

    # 设置显示范围（中国区域）
    ax.set_extent([73, 135, 18, 54], crs=ccrs.PlateCarree())

    # 打印统计信息，检查是否存在非掩膜数据
    valid_count = np.sum(~np.isnan(data.values))
    print(f"当前时段: {time_str}")
    print(f"有效像素数（掩膜内）: {valid_count}")
    print(f"数值范围: {data.min().values:.2f} 到 {data.max().values:.2f}")

    plt.show()


if __name__ == "__main__":
    # 你可以尝试换不同的 time_index 看看不同年份
    plot_check_map(time_index=460)  # 460 大约是 2011 年左右