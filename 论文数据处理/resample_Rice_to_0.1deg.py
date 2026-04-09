import xarray as xr
import rioxarray
from rasterio.enums import Resampling

# 1. 文件路径设置
rice_tif_path = r'E:\data\全国1Km作物种植数据\水稻\CHN_Rice_Union_2000_2019.tif'
precip_nc_path = r'E:\data\meteorologicaldata\Precip_8Day_Standard\ChinaMet_0.10deg_Precip_8Day_2019.nc'
output_path = r'E:\data\全国1Km作物种植数据\水稻\CHN_Rice_Resampled_01deg.tif'

# 2. 读取降水 NC 文件作为基准网格
# 注意：NC文件通常包含多个维度（时间、经度、纬度），我们只需要它的空间结构
ds_precip = xr.open_dataset(precip_nc_path)

# 如果 NC 文件没有定义投影，手动指定 WGS84 (EPSG:4326)
if ds_precip.rio.crs is None:
    ds_precip.rio.write_crs("EPSG:4326", inplace=True)

# 3. 读取水稻 TIF 文件
da_rice = rioxarray.open_rasterio(rice_tif_path, masked=True)

# 4. 执行对齐重采样 (Match Reprojection)
# 这里选用 Resampling.average，结果代表 0.1° 网格内的水稻种植比例（0-1之间）
# 如果你只需要 0/1 结果，请将 average 改为 mode
rice_resampled = da_rice.rio.reproject_match(
    ds_precip,
    resampling=Resampling.average
)

# 5. 结果处理：将无效值（Nodata）填充为 0 并保存
rice_resampled = rice_resampled.fillna(0)

# 6. 保存为 TIF 文件
rice_resampled.rio.to_raster(output_path)

print(f"重采样完成！")
print(f"输出文件路径: {output_path}")
print(f"输出分辨率: {rice_resampled.rio.transform()[0]} 度")
print(f"输出维度 (高度, 宽度): {rice_resampled.shape[-2:]}")