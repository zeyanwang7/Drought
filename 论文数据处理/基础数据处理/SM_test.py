import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import random

# 路径设置
base_path = r'E:\data\Soil_Humidity'
avg_path = r'E:\data\Soil_Humidity_Average_0_100'

# 随机选一个年份和一天
target_year = random.randint(2000, 2022)
target_day = 180  # 约6月中下旬

print(f"检查日期: {target_year} 年第 {target_day} 天")

# 1. 读取原始层级 (抽取 10cm, 50cm, 100cm 作为代表) 和 平均值层级
try:
    # 加载代表性原始层
    ds_10 = xr.open_dataset(os.path.join(base_path, '10cm', f'SMCI_9km_{target_year}_10cm.nc'))
    ds_50 = xr.open_dataset(os.path.join(base_path, '50cm', f'SMCI_9km_{target_year}_50cm.nc'))
    ds_100 = xr.open_dataset(os.path.join(base_path, '100cm', f'SMCI_9km_{target_year}_100cm.nc'))

    # 加载你计算的平均值层
    ds_avg = xr.open_dataset(os.path.join(avg_path, f'SMCI_9km_{target_year}_Avg_0_100cm.nc'))

    # 2. 绘图对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.3)


    # 定义绘图函数
    def plot_sm(ds, var_name, ax, title):
        data = ds[var_name].isel(time=target_day)
        im = data.plot(ax=ax, cmap='YlGnBu', add_colorbar=True,
                       cbar_kwargs={'label': 'Soil Moisture'})
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()


    # 绘制各图
    plot_sm(ds_10, 'SMCI', axes[0, 0], f"Original 10cm (Year:{target_year}, Day:{target_day})")
    plot_sm(ds_50, 'SMCI', axes[0, 1], f"Original 50cm (Year:{target_year}, Day:{target_day})")
    plot_sm(ds_100, 'SMCI', axes[1, 0], f"Original 100cm (Year:{target_year}, Day:{target_day})")
    plot_sm(ds_avg, 'SMCI_avg', axes[1, 1], f"Calculated 0-100cm Average")

    plt.suptitle(f"Verification of Depth Averaging (0-100cm)", fontsize=18, y=0.98)
    plt.show()

    # 3. 打印数值检查 (取中心点坐标)
    lat_idx, lon_idx = len(ds_10.lat) // 2, len(ds_10.lon) // 2
    val_10 = ds_10.SMCI.isel(time=target_day, lat=lat_idx, lon=lon_idx).values
    val_50 = ds_50.SMCI.isel(time=target_day, lat=lat_idx, lon=lon_idx).values
    val_100 = ds_100.SMCI.isel(time=target_day, lat=lat_idx, lon=lon_idx).values
    val_avg = ds_avg.SMCI_avg.isel(time=target_day, lat=lat_idx, lon=lon_idx).values

    print("\n----- 随机中心点数值校验 -----")
    print(f"10cm 原始值: {val_10:.4f}")
    print(f"50cm 原始值: {val_50:.4f}")
    print(f"100cm 原始值: {val_100:.4f}")
    print(f"0-100cm 计算平均值: {val_avg:.4f}")

    # 释放资源
    for d in [ds_10, ds_50, ds_100, ds_avg]: d.close()

except Exception as e:
    print(f"绘图检查失败: {e}")