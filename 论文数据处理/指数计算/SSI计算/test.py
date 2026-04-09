import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# --- 1. 设置文件路径 ---
ssi_path = r'E:\data\meteorologicaldata\IndexCalculationResults\SSI\SSI_30d_8day_Wheat_Gringorten.nc'


def plot_ssi_check(time_index=None):
    # 加载数据
    if not os.path.exists(ssi_path):
        print("❌ 找不到 SSI 文件，请先确保计算已完成。")
        return

    ds = xr.open_dataset(ssi_path)
    ssi = ds['SSI']

    # 2. 选择时间点
    if time_index is None:
        # 默认选一个 2022 年夏季的时间点 (2022年长江流域高温干旱非常典型)
        # 或者随机选一个：time_index = np.random.randint(0, len(ssi.time))
        target_date = '2022-08-21'
        try:
            data_to_plot = ssi.sel(time=target_date, method='nearest')
        except:
            data_to_plot = ssi.isel(time=-10)  # 报错则选倒数第10期
    else:
        data_to_plot = ssi.isel(time=time_index)

    current_time = str(data_to_plot.time.values)[:10]
    print(f"📊 正在绘制时间点: {current_time}")

    # 3. 绘图设置
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 添加地理要素
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', linewidth=0.5)

    # 绘制 SSI
    # vmin/vmax 设置为 -3 到 3 是标准化指数的标准量程
    im = data_to_plot.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='RdYlBu',  # 红(干) - 黄 - 蓝(湿)
        vmin=-3, vmax=3,
        add_colorbar=True,
        cbar_kwargs={'label': 'SSI (Gringorten)', 'pad': 0.02, 'shrink': 0.6}
    )

    # 设置范围 (中国区域)
    ax.set_extent([73, 135, 18, 54], crs=ccrs.PlateCarree())

    plt.title(f"Rice SSI Spatial Distribution ({current_time})", fontsize=14)

    # 检查有效值百分比
    valid_count = np.sum(~np.isnan(data_to_plot.values))
    print(f"✅ 该期有效像元数: {valid_count}")

    plt.show()


if __name__ == "__main__":
    import os

    plot_ssi_check()