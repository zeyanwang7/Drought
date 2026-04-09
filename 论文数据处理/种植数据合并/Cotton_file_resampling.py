import os
from osgeo import gdal, osr
import xarray as xr

# 开启异常抛出
gdal.UseExceptions()

# --- 1. 路径设置 ---
union_tif_path = r'E:\data\新疆棉花作物种植地图\Cotton_2018_2021_Union.tif'
ref_nc_path = r'E:\data\meteorologicaldata\pet_8day_results\PET_8day_2000.nc'
output_resampled_path = r'E:\data\新疆棉花作物种植地图\Cotton_Union_01deg_Presence.tif'


def resample_presence_to_01deg():
    # --- 2. 使用 xarray 提取参考网格 (NC文件) 的地理信息 ---
    print(f"🚀 正在提取参考网格信息: {os.path.basename(ref_nc_path)}")
    try:
        ds = xr.open_dataset(ref_nc_path)

        # 自动识别经纬度变量名
        lon_name = 'lon' if 'lon' in ds.coords else 'longitude'
        lat_name = 'lat' if 'lat' in ds.coords else 'latitude'

        lon = ds.coords[lon_name].values
        lat = ds.coords[lat_name].values

        target_cols = len(lon)
        target_rows = len(lat)

        # 计算分辨率
        res_x = (lon[-1] - lon[0]) / (target_cols - 1)
        res_y = (lat[-1] - lat[0]) / (target_rows - 1)

        # 构建基准 GeoTransform (左上角坐标)
        # 注意：经纬度 NC 通常是中心对齐，需要减去半个像元得到边缘坐标
        ref_geotrans = [lon.min() - res_x / 2, res_x, 0, lat.max() + abs(res_y) / 2, 0, -abs(res_y)]

        # 构建 WGS84 坐标系
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ref_proj = srs.ExportToWkt()

        # 目标边界 [minX, minY, maxX, maxY]
        target_bounds = [
            ref_geotrans[0],
            ref_geotrans[3] + ref_geotrans[5] * target_rows,
            ref_geotrans[0] + ref_geotrans[1] * target_cols,
            ref_geotrans[3]
        ]

        ds.close()
        print(f"✅ 成功锁定参考网格: {target_cols}x{target_rows}")
    except Exception as e:
        print(f"💥 读取 NC 文件失败: {e}")
        return

    # --- 3. 使用 GDAL Warp 执行重采样 (最大值法) ---
    print(f"\n正在进行“有/无”判定重采样 (10m -> 0.1度)...")

    # 修复：gdal.GRIORA_Max 改为 gdal.GRA_Max
    # 逻辑：在 0.1 度网格内，只要包含 10m 的 1 值，结果即为 1
    warp_options = gdal.WarpOptions(
        format='GTiff',
        dstSRS=ref_proj,
        outputBounds=target_bounds,
        width=target_cols,
        height=target_rows,
        resampleAlg=gdal.GRA_Max,  # 关键修复点
        srcNodata=0,
        dstNodata=0,
        creationOptions=['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES'],
        multithread=True
    )

    try:
        gdal.Warp(output_resampled_path, union_tif_path, options=warp_options)
        print(f"\n✨ 处理成功完成！")
        print(f"结果保存至: {output_resampled_path}")
        print("💡 结果含义: 1 代表该 0.1 度格网内存在棉花种植区，0 代表无。")
    except Exception as e:
        print(f"💥 重采样过程中出错: {e}")


if __name__ == "__main__":
    resample_presence_to_01deg()