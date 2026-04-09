import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import os
import numpy as np

# 1. 定义文件路径
input_tif = r'E:\数据集\作物种植文件\作物原始种植分布文件\小麦\Winter_Wheat_Stable_2009_2015.tif'
reference_nc = r'E:\data\meteorologicaldata\precipitation\2000\ChinaMet_010deg_prec_2000_01_01.nc'
output_folder = r'E:\数据集\作物种植文件\0.1度重采样文件'
output_path = os.path.join(output_folder, 'Winter_Wheat_01deg_MaxResampled.tif')

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 2. 读取参考网格信息
ds_ref = xr.open_dataset(reference_nc)
if ds_ref.rio.crs is None:
    ds_ref.rio.write_crs("EPSG:4326", inplace=True)

# 3. 读取原始 TIF
src_da = rioxarray.open_rasterio(input_tif)

# 4. 使用 'max' 算法执行重采样
# 这种方式会确保在大像元范围内的 100 个小像元中，只要有一个最大值（通常是1），结果即为最大值
resampled_da = src_da.rio.reproject_match(
    ds_ref,
    resampling=Resampling.max  # 关键：改用最大值重采样
)

# 5. 处理无效值 (NoData)
# 重采样可能会引入一些极小值或背景值，强制转为二值 (0 或 1)
# 假设 1 代表种植，0 代表未种植
resampled_da.values = np.where(resampled_da.values >= 0.5, 1, 0).astype(np.uint8)

# 6. 保存并绘图
resampled_da.rio.to_raster(output_path)
print(f"最大值重采样完成，文件已保存: {output_path}")

plt.figure(figsize=(12, 7))
# 使用布尔显示，更清晰地观察种植区覆盖情况
resampled_da.sel(band=1).plot(cmap='Greens', add_colorbar=True)
plt.title("Spring Wheat Distribution (0.1° Max Resampling)")
plt.show()