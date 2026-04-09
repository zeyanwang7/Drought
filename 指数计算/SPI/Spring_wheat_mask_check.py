import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import numpy as np

# 1. 加载文件
prec_nc_path = r'E:\data\meteorologicaldata\precipitation\2000\ChinaMet_010deg_prec_2000_01_01.nc'
wheat_tif_path = r'E:\数据集\作物种植文件\0.1度重采样文件\Spring_Wheat_01deg.tif'

# 读取 NC
ds_prec = xr.open_dataset(prec_nc_path)
# 【核心步骤】检查纬度是否为升序（从南到北），如果是，则翻转为降序（从北到南）
# 这样才能与 TIF 的 Top-down 布局匹配
if ds_prec.lat.values[0] < ds_prec.lat.values[-1]:
    ds_prec = ds_prec.reindex(lat=ds_prec.lat[::-1])

# 读取重采样后的 TIF
da_wheat = rioxarray.open_rasterio(wheat_tif_path).squeeze()

# 2. 提取绘图数据
# 假设 NC 中的变量名是 'prec'
prec_data = ds_prec['prec'].isel(time=0) if 'time' in ds_prec.dims else ds_prec['prec']

# 3. 验证绘图：叠置分析
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# 左图：查看降水背景下的作物遮罩
prec_data.plot(ax=ax[0], cmap='Blues', add_colorbar=True, label='Precipitation')
# 叠加作物（将非种植区设为透明）
wheat_masked = da_wheat.where(da_wheat > 0)
wheat_masked.plot(ax=ax[0], cmap='Reds_r', add_colorbar=False, alpha=0.7)
ax[0].set_title("Overlay Check: Wheat(Red) on Prec(Blue)")

# 右图：直接对比矩阵形状和数值
# 获取 TIF 的四至范围，用于 imshow 准确定位
# 获取真实的地理范围 [左, 右, 下, 上]
left, bottom, right, top = da_wheat.rio.bounds()

# 修改右图绘图代码
ax[1].imshow(da_wheat.values, cmap='Greens', origin='lower')
ax[1].set_title("Corrected Matrix View (TIF)")
ax[1].set_xlabel("Longitude")
ax[1].set_ylabel("Latitude")

plt.tight_layout()
plt.show()

# 4. 打印数组形状确保一致
print(f"降水矩阵形状: {prec_data.shape}")
print(f"小麦矩阵形状: {da_wheat.shape}")