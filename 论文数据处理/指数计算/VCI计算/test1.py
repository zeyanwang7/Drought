import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# --- 1. 设置路径 ---
vci_path = r'E:\data\NDVI\VCI\VCI_8d_01deg_Rice.nc'


def plot_check_rice_vci():
    print("🚀 正在读取水稻 VCI 数据进行绘图检查...")

    # 读取数据
    ds = xr.open_dataset(vci_path)
    vci = ds['VCI']

    # --- 2. 选取典型时间点 ---
    # 选取 2022 年 8 月中旬（长江流域极端干旱期）
    target_date = '2022-08-15'
    try:
        sample_data = vci.sel(time=target_date, method='nearest')
    except:
        # 如果日期不匹配，随机取最后一年的 8 月份一期
        sample_data = vci.isel(time=-20)

    actual_date = np.datetime_as_string(sample_data.time.values, unit='D')
    print(f"✅ 绘图目标日期: {actual_date}")

    # --- 3. 绘图 ---
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig = plt.figure(figsize=(12, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # 添加地图背景
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    # 增加省界以方便精确定位水稻产区
    ax.add_feature(cfeature.STATES.with_scale('50m'), linewidth=0.3, edgecolor='gray')

    # 绘制 VCI 图像
    # VCI 标准色阶：红(旱) -> 黄 -> 绿(好)
    im = sample_data.plot(
        ax=ax,
        x='Lon', y='Lat',
        cmap='RdYlGn',  # 红黄绿配色
        vmin=0, vmax=100,
        add_colorbar=True,
        cbar_kwargs={'label': 'VCI (%)', 'shrink': 0.7, 'pad': 0.02}
    )

    # 设置中国范围
    ax.set_extent([73, 135, 18, 54], crs=ccrs.PlateCarree())

    plt.title(f"水稻 VCI 空间分布检查 ({actual_date})", fontsize=14)

    # --- 4. 统计诊断 ---
    print("-" * 30)
    print(f"数据诊断信息:")
    print(f"- 有效像素值范围: {np.nanmin(sample_data):.2f} 至 {np.nanmax(sample_data):.2f}")
    print(f"- 均值: {np.nanmean(sample_data):.2f}")

    plt.show()


if __name__ == "__main__":
    plot_check_rice_vci()