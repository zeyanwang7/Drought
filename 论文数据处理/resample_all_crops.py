#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小麦和水稻种植数据重采样工具
功能：将小麦和水稻种植数据重采样到0.1度分辨率，并与基准NC文件对齐
修复版本：处理统计计算错误
"""

import os
import numpy as np
from osgeo import gdal, osr
import netCDF4 as nc
import sys
from datetime import datetime
import glob
import warnings

warnings.filterwarnings('ignore')

# 设置GDAL环境
gdal.UseExceptions()


class CropResampler:
    """作物种植数据重采样器"""

    def __init__(self, output_dir=None):
        """
        初始化重采样器

        参数:
        output_dir: 输出目录
        """
        if output_dir is None:
            # 默认输出到当前目录的resampled_0.1deg子目录
            output_dir = os.path.join(os.getcwd(), "resampled_0.1deg")
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 目标分辨率
        self.target_res = 0.1

        print("=" * 70)
        print("作物种植数据重采样工具")
        print("=" * 70)
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

    def resample_crop_to_nc_grid(self, crop_tif_path, nc_info, output_path=None,
                                 resample_method='max', crop_type='crop'):
        """
        将作物种植数据重采样到NC文件网格

        参数:
        crop_tif_path: 作物TIFF文件路径
        nc_info: NC文件网格信息
        output_path: 输出文件路径
        resample_method: 重采样方法
            'max': 最大值法（如果0.1度网格中有作物，则标记为1）
        crop_type: 作物类型 ('wheat', 'rice', 'maize')

        返回:
        输出文件路径
        """
        print(f"\n{'=' * 50}")
        print(f"重采样{crop_type}数据到NC网格")
        print(f"{'=' * 50}")

        try:
            # 检查输入文件
            if not os.path.exists(crop_tif_path):
                raise FileNotFoundError(f"找不到{crop_type}文件: {crop_tif_path}")

            print(f"输入文件: {os.path.basename(crop_tif_path)}")

            # 设置输出路径
            if output_path is None:
                basename = os.path.basename(crop_tif_path)
                name_without_ext = os.path.splitext(basename)[0]
                output_path = os.path.join(
                    self.output_dir,
                    f"{name_without_ext}_0.1deg_aligned.tif"
                )

            # 读取作物数据
            print(f"读取{crop_type}分布数据...")
            ds_crop = gdal.Open(crop_tif_path, gdal.GA_ReadOnly)
            if ds_crop is None:
                raise RuntimeError(f"无法打开文件: {crop_tif_path}")

            # 获取输入数据信息
            geo_trans_crop = ds_crop.GetGeoTransform()
            proj_crop = ds_crop.GetProjection()
            cols_crop = ds_crop.RasterXSize
            rows_crop = ds_crop.RasterYSize
            crop_data = ds_crop.GetRasterBand(1).ReadAsArray()

            # 检查地理变换
            if geo_trans_crop is None or geo_trans_crop == (0, 1, 0, 0, 0, 1):
                print(f"⚠️ 文件缺少地理变换信息，使用默认值...")
                # 使用中国范围的默认地理变换
                geo_trans_crop = (70.0, 0.00833333, 0, 55.0, 0, -0.00833333)

            nodata_crop = ds_crop.GetRasterBand(1).GetNoDataValue()
            if nodata_crop is None:
                nodata_crop = 0

            print(f"{crop_type}数据信息:")
            print(f"  原始尺寸: {cols_crop}列 × {rows_crop}行")
            print(f"  原始分辨率: {abs(geo_trans_crop[1]):.8f} × {abs(geo_trans_crop[5]):.8f}")
            print(f"  数值范围: [{crop_data.min()}, {crop_data.max()}]")

            # 统计作物像素
            crop_pixels = np.count_nonzero(crop_data)
            total_pixels = cols_crop * rows_crop
            crop_ratio = crop_pixels / total_pixels * 100
            print(f"  {crop_type}像素: {crop_pixels:,} / {total_pixels:,} ({crop_ratio:.2f}%)")

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

            # 使用最大值法重采样
            print(f"重采样方法: max (如果0.1度网格中有作物，则标记为1)")
            resample_alg = gdal.GRA_Max

            # 执行重采样对齐
            print("正在执行重采样对齐...")

            # 检查坐标系是否可用
            if proj_crop is None or proj_crop == "":
                print(f"⚠️ 源文件无坐标系信息，使用WGS84...")
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(4326)
                proj_crop = srs.ExportToWkt()

            # 使用GDAL Warp进行重采样
            try:
                # 设置warp选项
                warp_options = gdal.WarpOptions(
                    format='MEM',
                    outputBounds=[target_lon_min, target_lat_min, target_lon_max, target_lat_max],
                    width=target_cols,
                    height=target_rows,
                    dstSRS=target_proj,
                    resampleAlg=resample_alg,
                    srcNodata=nodata_crop,
                    dstNodata=nodata_crop,
                    warpMemoryLimit=1024,
                    errorThreshold=0
                )

                # 执行warp
                tmp_ds = gdal.Warp('', ds_crop, options=warp_options)

            except Exception as warp_error:
                print(f"GDAL Warp失败: {warp_error}")
                print("尝试使用替代方法...")
                tmp_ds = self._simple_resample(
                    crop_data, geo_trans_crop, target_geo_transform,
                    target_cols, target_rows, nodata_crop
                )

            if tmp_ds is None:
                raise RuntimeError("重采样失败")

            resampled_data = tmp_ds.ReadAsArray()

            # 后处理重采样数据
            print("正在后处理重采样数据...")

            # 确保是0-1整数
            resampled_data = np.round(resampled_data).astype(np.float32)
            resampled_data = np.clip(resampled_data, 0, 1)

            # 统计重采样结果
            crop_cells = np.count_nonzero(resampled_data)
            total_cells = target_cols * target_rows
            crop_ratio_resampled = crop_cells / total_cells * 100 if total_cells > 0 else 0

            unique_values = np.unique(resampled_data)

            print(f"\n重采样结果统计:")
            print(f"  {crop_type}网格数: {crop_cells:,} / {total_cells:,}")
            print(f"  {crop_type}比例: {crop_ratio_resampled:.2f}%")
            print(f"  唯一值: {unique_values}")
            print(f"  数据范围: [{resampled_data.min()}, {resampled_data.max()}]")

            if crop_pixels > 0 and crop_cells > 0:
                change_ratio = (crop_cells - crop_pixels) / crop_pixels * 100
                print(f"  像素变化: {change_ratio:+.2f}%")

            # 保存重采样结果
            print(f"\n正在保存结果...")
            self._save_geotiff_safe(
                output_path=output_path,
                data=resampled_data,
                geo_transform=target_geo_transform,
                projection=target_proj,
                nodata=nodata_crop,
                description=f"{crop_type.capitalize()} distribution resampled to 0.1 degree aligned with NC grid"
            )

            # 清理资源
            ds_crop = None
            if tmp_ds:
                tmp_ds = None

            print(f"\n✅ {crop_type}重采样完成!")
            print(f"   输入: {os.path.basename(crop_tif_path)}")
            print(f"   输出: {os.path.basename(output_path)}")

            return output_path

        except Exception as e:
            print(f"\n❌ {crop_type}重采样失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _simple_resample(self, src_data, src_geo_transform, target_geo_transform,
                         target_cols, target_rows, nodata_value):
        """
        简单重采样方法

        参数:
        src_data: 源数据
        src_geo_transform: 源地理变换
        target_geo_transform: 目标地理变换
        target_cols: 目标列数
        target_rows: 目标行数
        nodata_value: 无数据值

        返回:
        重采样后的数据集
        """
        print("使用简单重采样方法...")

        src_rows, src_cols = src_data.shape

        # 获取源和目标地理参数
        src_x_min, src_x_res, _, src_y_max, _, src_y_res = src_geo_transform
        src_y_res = abs(src_y_res)  # 确保为正

        target_x_min, target_x_res, _, target_y_max, _, target_y_res = target_geo_transform
        target_y_res = abs(target_y_res)

        # 创建目标数组
        target_data = np.zeros((target_rows, target_cols), dtype=np.float32)

        # 计算每个目标网格对应的源数据范围
        for i in range(target_rows):
            for j in range(target_cols):
                # 计算目标网格的边界
                target_left = target_x_min + j * target_x_res
                target_right = target_left + target_x_res
                target_top = target_y_max - i * target_y_res
                target_bottom = target_top - target_y_res

                # 计算在源数据中的对应位置
                src_left_idx = int((target_left - src_x_min) / src_x_res)
                src_right_idx = int((target_right - src_x_min) / src_x_res)
                src_top_idx = int((src_y_max - target_top) / src_y_res)
                src_bottom_idx = int((src_y_max - target_bottom) / src_y_res)

                # 确保索引在范围内
                src_left_idx = max(0, min(src_cols - 1, src_left_idx))
                src_right_idx = max(0, min(src_cols, src_right_idx))
                src_top_idx = max(0, min(src_rows - 1, src_top_idx))
                src_bottom_idx = max(0, min(src_rows, src_bottom_idx))

                # 提取源数据窗口
                if src_left_idx < src_right_idx and src_top_idx < src_bottom_idx:
                    window = src_data[src_top_idx:src_bottom_idx, src_left_idx:src_right_idx]

                    # 使用最大值：如果窗口中有任何作物，则目标网格为1
                    if np.any(window > 0):
                        target_data[i, j] = 1
                    else:
                        target_data[i, j] = 0
                else:
                    target_data[i, j] = nodata_value

        # 创建内存数据集
        driver = gdal.GetDriverByName('MEM')
        target_ds = driver.Create('', target_cols, target_rows, 1, gdal.GDT_Float32)

        target_ds.SetGeoTransform(target_geo_transform)

        # 设置WGS84坐标系
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        target_ds.SetProjection(srs.ExportToWkt())

        target_band = target_ds.GetRasterBand(1)
        target_band.WriteArray(target_data)
        target_band.SetNoDataValue(nodata_value)

        return target_ds

    def _save_geotiff_safe(self, output_path, data, geo_transform, projection,
                           nodata=0, description=""):
        """安全保存GeoTIFF文件，避免统计计算错误"""

        driver = gdal.GetDriverByName("GTiff")

        rows, cols = data.shape

        # 检查是否有有效数据
        has_valid_data = np.any(data != nodata) if nodata is not None else np.any(data >= 0)

        if not has_valid_data:
            print(f"⚠️ 警告: 输出数据可能全部为无效值，继续保存...")

        # 创建选项
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

        if out_ds is None:
            raise RuntimeError("无法创建输出文件")

        out_ds.SetGeoTransform(geo_transform)
        out_ds.SetProjection(projection)

        out_band = out_ds.GetRasterBand(1)

        # 确保数据是uint8类型
        if data.dtype != np.uint8:
            data = data.astype(np.uint8)

        out_band.WriteArray(data)

        if nodata is not None:
            out_band.SetNoDataValue(float(nodata))

        if description:
            out_band.SetDescription(description)

        # 安全地计算统计信息
        try:
            if has_valid_data:
                out_band.FlushCache()
                # 只在实际有数据时计算统计
                if np.count_nonzero(data != nodata) > 0:
                    out_band.ComputeStatistics(False)
                else:
                    print(f"  跳过统计计算: 无有效像素")
            else:
                print(f"  跳过统计计算: 无有效数据")
        except Exception as stats_error:
            print(f"  统计计算失败: {stats_error}")
            # 手动设置统计信息
            valid_data = data[data != nodata] if nodata is not None else data
            if len(valid_data) > 0:
                out_band.SetStatistics(
                    float(valid_data.min()),
                    float(valid_data.max()),
                    float(valid_data.mean()),
                    float(valid_data.std())
                )

        # 设置元数据
        metadata = {
            'TIFFTAG_DOCUMENTNAME': description,
            'TIFFTAG_DATETIME': datetime.now().strftime('%Y:%m:%d %H:%M:%S'),
            'Processing_Software': 'CropResampler',
            'NoData_Value': str(nodata) if nodata is not None else 'None',
            'Resolution': f"{geo_transform[1]:.4f} degree",
            'Data_Type': 'Binary (0: No crop, 1: Crop)',
            'Rows': str(rows),
            'Columns': str(cols),
            'Valid_Pixels': str(np.count_nonzero(data != nodata)) if nodata is not None else str(np.count_nonzero(data))
        }
        out_ds.SetMetadata(metadata)

        out_ds.FlushCache()
        out_ds = None

        print(f"  已保存: {os.path.basename(output_path)}")
        print(f"  尺寸: {cols}列 × {rows}行")

        valid_pixels = np.count_nonzero(data != nodata) if nodata is not None else np.count_nonzero(data)
        total_pixels = data.size
        valid_ratio = valid_pixels / total_pixels * 100 if total_pixels > 0 else 0
        print(f"  有效像素比例: {valid_ratio:.2f}%")

        # 计算作物像素（值=1的像素）
        crop_pixels = np.count_nonzero(data == 1)
        crop_ratio = crop_pixels / total_pixels * 100 if total_pixels > 0 else 0
        print(f"  作物像素比例: {crop_ratio:.2f}%")


def find_crop_union_file(crop_dir, crop_name):
    """查找作物并集文件"""
    # 可能的文件名模式
    patterns = [
        f"CHN_{crop_name}_Union_2000_2019.tif",
        f"CHN_{crop_name}_Union_*.tif",
        f"CHN_{crop_name}_*_Union_*.tif"
    ]

    for pattern in patterns:
        search_path = os.path.join(crop_dir, pattern)
        files = glob.glob(search_path)
        if files:
            return files[0]

    return None


def find_nc_file(precip_dir, year=2000):
    """查找NC文件"""
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


def resample_all_crops():
    """重采样所有作物数据"""
    print("作物种植数据重采样工具")
    print("=" * 70)

    # 配置路径
    wheat_dir = r"E:\data\全国1Km作物种植数据\小麦"
    rice_dir = r"E:\data\全国1Km作物种植数据\水稻"
    maize_dir = r"E:\data\全国1Km作物种植数据\玉米"
    precip_dir = r"E:\data\meteorologicaldata\Precip_8Day_Standard"

    # 检查路径是否存在
    paths = {
        '小麦': wheat_dir,
        '水稻': rice_dir,
        '玉米': maize_dir,
        '降水数据': precip_dir
    }

    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"错误: {name}路径不存在 - {path}")
            return 1

    # 查找基准NC文件
    print("查找基准NC文件...")
    nc_file = find_nc_file(precip_dir, 2000)

    if not nc_file:
        print("错误: 未找到基准NC文件")
        print("请检查降水数据目录中是否存在ChinaMet_0.10deg_Precip_8Day_2000.nc文件")
        return 1

    print(f"找到基准NC文件: {os.path.basename(nc_file)}")

    # 创建重采样器
    resampler = CropResampler()

    # 分析NC文件网格
    print("\n" + "=" * 70)
    nc_info = resampler.analyze_nc_grid(nc_file)

    # 处理各种作物
    crop_configs = [
        {
            'name': '小麦',
            'dir': wheat_dir,
            'crop_name': 'Wheat',
            'union_name': 'Wheat'
        },
        {
            'name': '水稻',
            'dir': rice_dir,
            'crop_name': 'Rice',
            'union_name': 'Rice'
        },
        {
            'name': '玉米',
            'dir': maize_dir,
            'crop_name': 'Maize',
            'union_name': 'Maize'
        }
    ]

    results = {}

    for config in crop_configs:
        print(f"\n{'=' * 70}")
        print(f"处理{config['name']}数据")
        print(f"{'=' * 70}")

        # 查找作物并集文件
        print(f"查找{config['name']}并集文件...")
        crop_union_file = find_crop_union_file(config['dir'], config['union_name'])

        if not crop_union_file:
            print(f"警告: 未找到{config['name']}并集文件")
            print(f"请确保已生成{config['name']}并集文件 (CHN_{config['union_name']}_Union_2000_2019.tif)")
            continue

        print(f"找到{config['name']}并集文件: {os.path.basename(crop_union_file)}")

        try:
            # 重采样作物数据
            output_file = resampler.resample_crop_to_nc_grid(
                crop_tif_path=crop_union_file,
                nc_info=nc_info,
                output_path=None,
                resample_method='max',  # 使用最大值法
                crop_type=config['name'].lower()
            )

            # 记录结果
            results[config['name']] = {
                'input': crop_union_file,
                'output': output_file,
                'status': 'success'
            }

            # 显示输出文件信息
            try:
                ds = gdal.Open(output_file, gdal.GA_ReadOnly)
                if ds:
                    data = ds.GetRasterBand(1).ReadAsArray()
                    crop_cells = np.count_nonzero(data == 1)
                    total_cells = data.size
                    crop_ratio = crop_cells / total_cells * 100 if total_cells > 0 else 0

                    print(f"\n{config['name']}重采样结果:")
                    print(f"  网格尺寸: {data.shape[1]}列 × {data.shape[0]}行")
                    print(f"  {config['name']}网格数: {crop_cells:,}")
                    print(f"  {config['name']}比例: {crop_ratio:.2f}%")

                    # 检查数据一致性
                    unique_values = np.unique(data)
                    print(f"  数据值: {unique_values}")

                    ds = None
            except Exception as e:
                print(f"  读取输出文件失败: {e}")

        except Exception as e:
            print(f"\n❌ {config['name']}重采样失败: {str(e)}")
            results[config['name']] = {
                'error': str(e),
                'status': 'failed'
            }

    # 输出汇总报告
    print(f"\n{'=' * 70}")
    print(f"重采样完成汇总")
    print(f"{'=' * 70}")

    successful = []
    failed = []

    for crop_name, result in results.items():
        if result.get('status') == 'success':
            successful.append(f"{crop_name}: {os.path.basename(result['output'])}")
        else:
            failed.append(f"{crop_name}: {result.get('error', '未知错误')}")

    print(f"成功处理: {len(successful)} 种作物")
    for item in successful:
        print(f"  ✓ {item}")

    if failed:
        print(f"\n处理失败: {len(failed)} 种作物")
        for item in failed:
            print(f"  ✗ {item}")

    # 保存汇总报告
    report_file = os.path.join(resampler.output_dir, "resampling_report.txt")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("作物种植数据重采样报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"基准NC文件: {os.path.basename(nc_file)}\n")
        f.write(f"输出目录: {resampler.output_dir}\n")
        f.write("\n处理结果:\n")

        for crop_name, result in results.items():
            f.write(f"\n{crop_name}:\n")
            if result.get('status') == 'success':
                f.write(f"  状态: 成功\n")
                f.write(f"  输入文件: {os.path.basename(result['input'])}\n")
                f.write(f"  输出文件: {os.path.basename(result['output'])}\n")
            else:
                f.write(f"  状态: 失败\n")
                f.write(f"  错误: {result.get('error', '未知错误')}\n")

    print(f"\n详细报告: {report_file}")

    return 0


def main():
    """主函数"""
    start_time = datetime.now()
    print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        exit_code = resample_all_crops()

        end_time = datetime.now()
        elapsed = end_time - start_time

        print(f"\n结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"总耗时: {elapsed.total_seconds():.1f} 秒")

        return exit_code

    except KeyboardInterrupt:
        print(f"\n⚠️ 用户中断处理")
        return 1
    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())