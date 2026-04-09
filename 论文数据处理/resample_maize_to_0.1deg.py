#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
玉米种植数据重采样工具
功能：将玉米种植数据重采样到0.1度分辨率，并与基准NC文件对齐
以2000年8天的0.1°降水NC文件为基准
"""

import os
import numpy as np
from osgeo import gdal, osr
import netCDF4 as nc
import sys
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 设置GDAL环境
gdal.UseExceptions()


class MaizeResampler:
    """玉米种植数据重采样器"""

    def __init__(self, maize_dir, precip_dir, output_dir=None):
        """
        初始化重采样器

        参数:
        maize_dir: 玉米数据目录
        precip_dir: 降水数据目录（包含基准NC文件）
        output_dir: 输出目录
        """
        self.maize_dir = maize_dir
        self.precip_dir = precip_dir

        if output_dir is None:
            # 默认输出到玉米目录下的resampled_0.1deg子目录
            output_dir = os.path.join(maize_dir, "resampled_0.1deg")
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 目标分辨率
        self.target_res = 0.1

        print("=" * 70)
        print("玉米种植数据重采样工具")
        print("=" * 70)
        print(f"玉米数据目录: {maize_dir}")
        print(f"降水数据目录: {precip_dir}")
        print(f"输出目录: {output_dir}")
        print(f"目标分辨率: {self.target_res} 度")
        print("=" * 70)

    def analyze_nc_grid(self, nc_file_path):
        """
        分析基准NC文件的网格信息

        参数:
        nc_file_path: NC文件路径

        返回:
        网格信息字典
        """
        print(f"\n{'=' * 50}")
        print(f"分析基准NC文件网格信息")
        print(f"{'=' * 50}")

        try:
            if not os.path.exists(nc_file_path):
                raise FileNotFoundError(f"找不到NC文件: {nc_file_path}")

            print(f"基准文件: {os.path.basename(nc_file_path)}")

            # 打开NC文件
            ds = nc.Dataset(nc_file_path, 'r')

            # 查找变量
            lat_var_name, lon_var_name = None, None

            # 常见的变量名
            potential_lat_names = ['lat', 'latitude', 'Lat', 'Latitude', 'LAT', 'LATITUDE', 'y']
            potential_lon_names = ['lon', 'longitude', 'Lon', 'Longitude', 'LON', 'LONGITUDE', 'x']

            for var_name in ds.variables:
                var = ds.variables[var_name]
                if len(var.shape) == 1:  # 坐标变量
                    if var_name.lower() in [n.lower() for n in potential_lat_names]:
                        lat_var_name = var_name
                    elif var_name.lower() in [n.lower() for n in potential_lon_names]:
                        lon_var_name = var_name

            # 如果没找到，尝试其他方法
            if not lat_var_name or not lon_var_name:
                for var_name in ds.variables:
                    if 'lat' in var_name.lower():
                        lat_var_name = var_name
                    elif 'lon' in var_name.lower():
                        lon_var_name = var_name

            if not lat_var_name or not lon_var_name:
                # 尝试从维度中查找
                for dim_name in ds.dimensions:
                    if 'lat' in dim_name.lower():
                        lat_var_name = dim_name
                    elif 'lon' in dim_name.lower():
                        lon_var_name = dim_name

            if not lat_var_name or not lon_var_name:
                raise ValueError("无法找到经纬度变量")

            # 读取经纬度数据
            lats = ds.variables[lat_var_name][:]
            lons = ds.variables[lon_var_name][:]

            # 检查纬度方向
            lat_increasing = lats[1] > lats[0] if len(lats) > 1 else True

            # 计算参数
            lat_res = abs(float(lats[1] - lats[0])) if len(lats) > 1 else self.target_res
            lon_res = abs(float(lons[1] - lons[0])) if len(lons) > 1 else self.target_res

            lat_min = float(lats.min())
            lat_max = float(lats.max())
            lon_min = float(lons.min())
            lon_max = float(lons.max())

            if not lat_increasing:
                lat_min, lat_max = lat_max, lat_min

            rows = len(lats)
            cols = len(lons)

            # 计算地理变换
            geo_transform = (lon_min, lon_res, 0, lat_max, 0, -lat_res)

            # 获取坐标系
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)  # WGS84
            proj = srs.ExportToWkt()

            ds.close()

            nc_info = {
                'file_path': nc_file_path,
                'lat_var': lat_var_name,
                'lon_var': lon_var_name,
                'lat_min': lat_min,
                'lat_max': lat_max,
                'lon_min': lon_min,
                'lon_max': lon_max,
                'lat_res': lat_res,
                'lon_res': lon_res,
                'rows': rows,
                'cols': cols,
                'geo_transform': geo_transform,
                'projection': proj
            }

            print(f"网格信息提取成功:")
            print(f"  纬度变量: {lat_var_name}")
            print(f"  经度变量: {lon_var_name}")
            print(f"  纬度范围: {lat_min:.4f} 到 {lat_max:.4f}")
            print(f"  经度范围: {lon_min:.4f} 到 {lon_max:.4f}")
            print(f"  分辨率: {lat_res:.4f} × {lon_res:.4f} 度")
            print(f"  网格尺寸: {cols}列 × {rows}行")
            print(f"  左上角: ({lon_min:.4f}, {lat_max:.4f})")

            return nc_info

        except Exception as e:
            print(f"❌ 分析NC文件失败: {str(e)}")
            raise

    def resample_maize_to_nc_grid(self, maize_tif_path, nc_info, output_path=None,
                                  resample_method='max'):
        """
        将玉米种植数据重采样到NC文件网格

        参数:
        maize_tif_path: 玉米TIFF文件路径
        nc_info: NC文件网格信息
        output_path: 输出文件路径
        resample_method: 重采样方法
            'max': 最大值法（如果0.1度网格中有玉米，则标记为1）
            'mode': 众数法
            'average': 平均值法

        返回:
        输出文件路径
        """
        print(f"\n{'=' * 50}")
        print(f"重采样玉米数据到NC网格")
        print(f"{'=' * 50}")

        try:
            # 检查输入文件
            if not os.path.exists(maize_tif_path):
                raise FileNotFoundError(f"找不到玉米文件: {maize_tif_path}")

            print(f"输入文件: {os.path.basename(maize_tif_path)}")

            # 设置输出路径
            if output_path is None:
                basename = os.path.basename(maize_tif_path)
                name_without_ext = os.path.splitext(basename)[0]
                output_path = os.path.join(
                    self.output_dir,
                    f"{name_without_ext}_0.1deg_aligned.tif"
                )

            # 读取玉米数据
            print("读取玉米分布数据...")
            ds_maize = gdal.Open(maize_tif_path, gdal.GA_ReadOnly)
            if ds_maize is None:
                raise RuntimeError(f"无法打开文件: {maize_tif_path}")

            # 获取输入数据信息
            geo_trans_maize = ds_maize.GetGeoTransform()
            proj_maize = ds_maize.GetProjection()
            cols_maize = ds_maize.RasterXSize
            rows_maize = ds_maize.RasterYSize
            maize_data = ds_maize.GetRasterBand(1).ReadAsArray()
            nodata_maize = ds_maize.GetRasterBand(1).GetNoDataValue() or 0

            print(f"玉米数据信息:")
            print(f"  原始尺寸: {cols_maize}列 × {rows_maize}行")
            print(f"  原始分辨率: {abs(geo_trans_maize[1]):.8f} × {abs(geo_trans_maize[5]):.8f}")
            print(f"  数值范围: [{maize_data.min()}, {maize_data.max()}]")

            # 统计玉米像素
            maize_pixels = np.count_nonzero(maize_data)
            total_pixels = cols_maize * rows_maize
            maize_ratio = maize_pixels / total_pixels * 100
            print(f"  玉米像素: {maize_pixels:,} / {total_pixels:,} ({maize_ratio:.2f}%)")

            # 获取NC网格信息
            target_geo_transform = nc_info['geo_transform']
            target_proj = nc_info['projection']
            target_cols = nc_info['cols']
            target_rows = nc_info['rows']
            target_lon_min = nc_info['lon_min']
            target_lon_max = nc_info['lon_max']
            target_lat_min = nc_info['lat_min']
            target_lat_max = nc_info['lat_max']

            print(f"\n目标网格信息:")
            print(f"  尺寸: {target_cols}列 × {target_rows}行")
            print(f"  分辨率: {nc_info['lon_res']:.4f} × {nc_info['lat_res']:.4f} 度")
            print(f"  空间范围: [{target_lon_min:.4f}, {target_lat_min:.4f}] 到 "
                  f"[{target_lon_max:.4f}, {target_lat_max:.4f}]")

            # 选择重采样算法
            gdal_methods = {
                'max': gdal.GRA_Max,  # 最大值法：如果0.1度网格中有玉米，则标记为1
                'mode': gdal.GRA_Mode,  # 众数法
                'average': gdal.GRA_Average,  # 平均值法
                'nearest': gdal.GRA_NearestNeighbour,  # 最近邻
                'min': gdal.GRA_Min,  # 最小值法
            }

            if resample_method not in gdal_methods:
                print(f"警告: 不支持的 resample_method '{resample_method}'，使用默认的 'max'")
                resample_method = 'max'

            resample_alg = gdal_methods[resample_method]
            print(f"重采样方法: {resample_method} ({resample_alg})")

            # 执行重采样对齐
            print("正在执行重采样对齐...")

            # 使用GDAL Warp进行重采样
            tmp_ds = gdal.Warp('', ds_maize, format='MEM',
                               outputBounds=[target_lon_min, target_lat_min,
                                             target_lon_max, target_lat_max],
                               width=target_cols,
                               height=target_rows,
                               dstSRS=target_proj,
                               resampleAlg=resample_alg,
                               srcNodata=nodata_maize,
                               dstNodata=nodata_maize,
                               warpMemoryLimit=1024)

            if tmp_ds is None:
                raise RuntimeError("GDAL Warp重采样失败")

            resampled_data = tmp_ds.ReadAsArray()

            # 后处理重采样数据
            print("正在后处理重采样数据...")

            # 对于分类数据，确保是0-1整数
            if resample_method in ['max', 'mode', 'nearest']:
                resampled_data = np.round(resampled_data).astype(np.uint8)

            # 确保值在0-1范围内
            resampled_data = np.clip(resampled_data, 0, 1)

            # 转换为整数类型
            resampled_data = resampled_data.astype(np.uint8)

            # 统计重采样结果
            maize_cells = np.count_nonzero(resampled_data)
            total_cells = target_cols * target_rows
            maize_ratio_resampled = maize_cells / total_cells * 100

            unique_values = np.unique(resampled_data)

            print(f"\n重采样结果统计:")
            print(f"  玉米网格数: {maize_cells:,} / {total_cells:,}")
            print(f"  玉米比例: {maize_ratio_resampled:.2f}%")
            print(f"  唯一值: {unique_values}")
            print(f"  数据范围: [{resampled_data.min()}, {resampled_data.max()}]")

            if maize_pixels > 0:
                change_ratio = (maize_cells - maize_pixels) / maize_pixels * 100
                print(f"  像素变化: {change_ratio:+.2f}%")

            # 保存重采样结果
            print(f"\n正在保存结果...")
            self._save_geotiff(
                output_path=output_path,
                data=resampled_data,
                geo_transform=target_geo_transform,
                projection=target_proj,
                nodata=nodata_maize,
                description=f"Maize distribution resampled to 0.1 degree aligned with NC grid"
            )

            # 清理资源
            ds_maize = None
            tmp_ds = None

            print(f"\n✅ 重采样完成!")
            print(f"   输入: {os.path.basename(maize_tif_path)}")
            print(f"   输出: {os.path.basename(output_path)}")

            return output_path

        except Exception as e:
            print(f"\n❌ 重采样失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _save_geotiff(self, output_path, data, geo_transform, projection,
                      nodata=0, description=""):
        """保存GeoTIFF文件"""

        driver = gdal.GetDriverByName("GTiff")

        rows, cols = data.shape
        creation_options = [
            'COMPRESS=LZW',
            'PREDICTOR=2',
            'TILED=YES',
            'BIGTIFF=IF_SAFER'
        ]

        out_ds = driver.Create(
            output_path,
            cols,
            rows,
            1,
            gdal.GDT_Byte,
            options=creation_options
        )

        out_ds.SetGeoTransform(geo_transform)
        out_ds.SetProjection(projection)

        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(data)
        out_band.SetNoDataValue(nodata)

        if description:
            out_band.SetDescription(description)

        # 计算统计信息
        out_band.FlushCache()
        out_band.ComputeStatistics(False)

        # 设置元数据
        metadata = {
            'TIFFTAG_DOCUMENTNAME': description,
            'TIFFTAG_DATETIME': datetime.now().strftime('%Y:%m:%d %H:%M:%S'),
            'Processing_Software': 'MaizeResampler',
            'NoData_Value': str(nodata),
            'Resolution': f"{geo_transform[1]:.4f} degree",
            'Data_Type': 'Binary (0: No maize, 1: Maize)'
        }
        out_ds.SetMetadata(metadata)

        out_ds.FlushCache()
        out_ds = None

        print(f"  已保存: {os.path.basename(output_path)}")
        print(f"  尺寸: {cols}列 × {rows}行")
        print(f"  玉米网格比例: {np.count_nonzero(data) / data.size * 100:.2f}%")

    def verify_alignment(self, maize_tif_path, nc_file_path):
        """
        验证玉米数据与NC文件的对齐情况

        参数:
        maize_tif_path: 重采样后的玉米TIFF文件
        nc_file_path: NC文件路径
        """
        print(f"\n{'=' * 50}")
        print(f"验证对齐情况")
        print(f"{'=' * 50}")

        try:
            # 打开玉米数据
            ds_maize = gdal.Open(maize_tif_path, gdal.GA_ReadOnly)
            if ds_maize is None:
                print(f"❌ 无法打开玉米文件: {maize_tif_path}")
                return False

            maize_geo = ds_maize.GetGeoTransform()
            maize_proj = ds_maize.GetProjection()
            maize_cols = ds_maize.RasterXSize
            maize_rows = ds_maize.RasterYSize

            # 获取NC网格信息
            nc_info = self.analyze_nc_grid(nc_file_path)

            # 比较参数
            alignment_ok = True
            issues = []

            # 比较地理变换
            if abs(maize_geo[0] - nc_info['geo_transform'][0]) > 0.0001:
                alignment_ok = False
                issues.append(f"经度起始位置不匹配: {maize_geo[0]:.6f} vs {nc_info['geo_transform'][0]:.6f}")

            if abs(maize_geo[3] - nc_info['geo_transform'][3]) > 0.0001:
                alignment_ok = False
                issues.append(f"纬度起始位置不匹配: {maize_geo[3]:.6f} vs {nc_info['geo_transform'][3]:.6f}")

            if abs(maize_geo[1] - nc_info['lon_res']) > 0.0001:
                alignment_ok = False
                issues.append(f"经度分辨率不匹配: {maize_geo[1]:.6f} vs {nc_info['lon_res']:.6f}")

            if abs(maize_geo[5] - (-nc_info['lat_res'])) > 0.0001:
                alignment_ok = False
                issues.append(f"纬度分辨率不匹配: {maize_geo[5]:.6f} vs {-nc_info['lat_res']:.6f}")

            # 比较网格尺寸
            if maize_cols != nc_info['cols']:
                alignment_ok = False
                issues.append(f"列数不匹配: {maize_cols} vs {nc_info['cols']}")

            if maize_rows != nc_info['rows']:
                alignment_ok = False
                issues.append(f"行数不匹配: {maize_rows} vs {nc_info['rows']}")

            ds_maize = None

            if alignment_ok:
                print(f"✅ 对齐验证通过!")
                print(f"  网格尺寸: {maize_cols}列 × {maize_rows}行")
                print(f"  分辨率: {maize_geo[1]:.6f} × {abs(maize_geo[5]):.6f} 度")
                print(f"  左上角: ({maize_geo[0]:.6f}, {maize_geo[3]:.6f})")
                return True
            else:
                print(f"❌ 对齐验证失败!")
                for issue in issues:
                    print(f"  {issue}")
                return False

        except Exception as e:
            print(f"❌ 验证失败: {str(e)}")
            return False


def find_maize_union_file(maize_dir):
    """查找玉米并集文件"""
    import glob

    # 可能的文件名模式
    patterns = [
        "CHN_Maize_Union_2000_2019.tif",
        "CHN_Maize_Union_*.tif",
        "CHN_Maize_*_Union_*.tif"
    ]

    for pattern in patterns:
        search_path = os.path.join(maize_dir, pattern)
        files = glob.glob(search_path)
        if files:
            return files[0]

    return None


def find_nc_file(precip_dir, year=2000):
    """查找NC文件"""
    import glob

    # 可能的文件名模式
    patterns = [
        f"ChinaMet_0.10deg_Precip_8Day_{year}.nc",
        "ChinaMet_0.10deg_Precip_8Day_*.nc",
        "*0.10deg*Precip*.nc"
    ]

    for pattern in patterns:
        search_path = os.path.join(precip_dir, pattern)
        files = glob.glob(search_path)
        if files:
            return files[0]

    return None


def main():
    """主函数"""
    print("玉米种植数据重采样工具")
    print("=" * 70)

    # 配置路径
    maize_dir = r"E:\data\全国1Km作物种植数据\玉米"
    precip_dir = r"E:\data\meteorologicaldata\Precip_8Day_Standard"

    # 检查路径是否存在
    if not os.path.exists(maize_dir):
        print(f"错误: 玉米路径不存在 - {maize_dir}")
        return 1

    if not os.path.exists(precip_dir):
        print(f"错误: 降水数据路径不存在 - {precip_dir}")
        return 1

    # 查找玉米并集文件
    print("查找玉米并集文件...")
    maize_union_file = find_maize_union_file(maize_dir)

    if not maize_union_file:
        print("错误: 未找到玉米并集文件")
        print("请确保已生成玉米并集文件 (CHN_Maize_Union_2000_2019.tif)")
        print("或修改代码中的文件名模式")
        return 1

    print(f"找到玉米并集文件: {os.path.basename(maize_union_file)}")

    # 查找基准NC文件
    print("\n查找基准NC文件...")
    nc_file = find_nc_file(precip_dir, 2000)

    if not nc_file:
        print("错误: 未找到基准NC文件")
        print("请检查降水数据目录中是否存在ChinaMet_0.10deg_Precip_8Day_2000.nc文件")
        return 1

    print(f"找到基准NC文件: {os.path.basename(nc_file)}")

    # 创建重采样器
    resampler = MaizeResampler(
        maize_dir=maize_dir,
        precip_dir=precip_dir
    )

    # 分析NC文件网格
    print("\n" + "=" * 70)
    nc_info = resampler.analyze_nc_grid(nc_file)

    # 执行重采样
    print("\n" + "=" * 70)
    print("开始重采样...")
    print("=" * 70)

    try:
        # 重采样玉米数据
        output_file = resampler.resample_maize_to_nc_grid(
            maize_tif_path=maize_union_file,
            nc_info=nc_info,
            output_path=None,  # 使用默认输出路径
            resample_method='max'  # 使用最大值法：如果0.1度网格中有玉米，则标记为1
        )

        # 验证对齐
        print("\n" + "=" * 70)
        print("验证对齐结果...")
        print("=" * 70)

        alignment_ok = resampler.verify_alignment(output_file, nc_file)

        if alignment_ok:
            print(f"\n✅ 重采样完成且对齐验证通过!")
        else:
            print(f"\n⚠️ 重采样完成但对齐验证有警告!")

        print(f"\n输出文件:")
        print(f"  {output_file}")

        # 显示文件信息
        ds = gdal.Open(output_file, gdal.GA_ReadOnly)
        if ds:
            data = ds.GetRasterBand(1).ReadAsArray()
            maize_cells = np.count_nonzero(data)
            total_cells = data.size
            maize_ratio = maize_cells / total_cells * 100

            print(f"\n重采样结果统计:")
            print(f"  网格尺寸: {data.shape[1]}列 × {data.shape[0]}行")
            print(f"  玉米网格数: {maize_cells:,}")
            print(f"  玉米比例: {maize_ratio:.2f}%")

            ds = None

        return 0

    except Exception as e:
        print(f"\n❌ 重采样失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # 运行主程序
    start_time = datetime.now()
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    exit_code = main()

    end_time = datetime.now()
    elapsed = end_time - start_time

    print(f"\n结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {elapsed.total_seconds():.1f} 秒")

    sys.exit(exit_code)