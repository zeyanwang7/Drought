import os
import numpy as np
import xarray as xr
from osgeo import gdal

# --- 参数设置 ---
input_tif = r'E:\data\全国1Km作物种植数据\玉米\Maize_Final_Presence5_2000_2019_Fixed.tif'
ref_nc = r'E:\data\meteorologicaldata\precipitation\2020\ChinaMet_010deg_prec_2020_01_01.nc'
output_resampled = r'E:\data\全国1Km作物种植数据\玉米\Maize_Distribution_010deg_Average.tif'


def resample_to_average_grid():
    # 启用 GDAL 异常抛出
    gdal.UseExceptions()

    # 1. 使用 xarray 获取 NC 文件的空间元数据
    ds = xr.open_dataset(ref_nc)

    lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
    lat_name = 'lat' if 'lat' in ds.coords else 'latitude'

    lon = ds.coords[lon_name].values
    lat = ds.coords[lat_name].values

    cols = len(lon)
    rows = len(lat)

    res_lon = abs(lon[1] - lon[0])
    res_lat = abs(lat[1] - lat[0])

    # 计算精确边界 (minX, minY, maxX, maxY)
    min_x = lon.min() - res_lon / 2
    max_x = lon.max() + res_lon / 2
    min_y = lat.min() - res_lat / 2
    max_y = lat.max() + res_lat / 2

    print(f"--- 空间元数据提取成功 ---")
    print(f"目标尺寸: {rows}行 x {cols}列")
    print(f"采样方式: 平均值采样 (反映种植面积比例)")

    # 2. 配置 WarpOptions
    # 注意：平均值采样会产生浮点数，所以输出类型改为 GDT_Float32
    warp_options = gdal.WarpOptions(
        format='GTiff',
        outputBounds=(min_x, min_y, max_x, max_y),
        width=cols,
        height=rows,
        dstSRS='EPSG:4326',
        resampleAlg=gdal.GRA_Average,  # 核心修改：平均值重采样
        outputType=gdal.GDT_Float32,  # 输出浮点数以保留比例信息
        srcNodata=0,
        dstNodata=0,
        creationOptions=['COMPRESS=LZW', 'TILED=YES']
    )

    # 3. 执行 Warp
    gdal.Warp(destNameOrDestDS=output_resampled,
              srcDSOrSrcDSTab=input_tif,
              options=warp_options)

    print(f"\n--- 处理完成 ---")
    print(f"结果已生成: {output_resampled}")
    print("提示：现在的像素值为 0 到 1 之间的比例，0 代表无种植，1 代表全覆盖。")
    ds.close()


if __name__ == "__main__":
    try:
        resample_to_average_grid()
    except Exception as e:
        print(f"\n！！！程序运行失败 ！！！")
        print(f"错误原因: {e}")