#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
玉米种植分布数据合并工具
功能：将2000-2019年玉米种植分布数据合并为并集
"""

import os
import numpy as np
from osgeo import gdal
import sys
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# 设置GDAL环境
gdal.UseExceptions()


def merge_maize_union(maize_dir, start_year=2000, end_year=2019, base_year=2019):
    """
    合并玉米种植分布数据（2000-2019年并集）

    参数:
    maize_dir: 玉米数据文件夹路径
    start_year: 起始年份
    end_year: 结束年份
    base_year: 基准对齐年份

    返回:
    输出文件路径
    """
    try:
        print("=" * 70)
        print("玉米种植分布数据合并工具")
        print("=" * 70)
        print(f"数据目录: {maize_dir}")
        print(f"年份范围: {start_year} - {end_year}")
        print(f"基准年份: {base_year}")
        print("=" * 70)

        # 1. 设置基准文件路径
        base_file = os.path.join(maize_dir, f"CHN_Maize_{base_year}.tif")
        output_file = os.path.join(maize_dir, f"CHN_Maize_Union_{start_year}_{end_year}.tif")

        # 检查基准文件是否存在
        if not os.path.exists(base_file):
            # 尝试查找其他年份作为基准
            print(f"警告: 基准文件 {os.path.basename(base_file)} 不存在")

            # 查找可用的基准文件
            available_years = []
            for year in range(start_year, end_year + 1):
                test_file = os.path.join(maize_dir, f"CHN_Maize_{year}.tif")
                if os.path.exists(test_file):
                    available_years.append(year)

            if not available_years:
                raise FileNotFoundError(f"错误: 在 {maize_dir} 中未找到任何玉米数据文件")

            # 使用中间的年份作为基准
            base_year = available_years[len(available_years) // 2]
            base_file = os.path.join(maize_dir, f"CHN_Maize_{base_year}.tif")
            print(f"使用 {base_year} 年数据作为基准")

        if not os.path.exists(base_file):
            raise FileNotFoundError(f"错误: 找不到基准文件 {base_file}")

        # 2. 读取基准文件的参数
        print(f"读取基准文件: {os.path.basename(base_file)}")
        ds_base = gdal.Open(base_file, gdal.GA_ReadOnly)
        if ds_base is None:
            raise RuntimeError(f"无法打开基准文件: {base_file}")

        geo_trans = ds_base.GetGeoTransform()
        proj = ds_base.GetProjection()
        cols = ds_base.RasterXSize
        rows = ds_base.RasterYSize

        # 检查地理变换参数
        if geo_trans[1] == 0 or geo_trans[5] == 0:
            print("警告: 地理变换参数可能有问题，请检查基准文件")

        # 打印基准文件信息
        print(f"基准文件信息:")
        print(f"  尺寸: {cols}列 × {rows}行")
        print(f"  分辨率: {abs(geo_trans[1]):.8f} × {abs(geo_trans[5]):.8f}")
        print(f"  左上角: ({geo_trans[0]:.6f}, {geo_trans[3]:.6f})")
        print(f"  坐标系: {proj[:50]}...")

        # 计算输出范围
        min_x = geo_trans[0]
        max_y = geo_trans[3]
        max_x = min_x + geo_trans[1] * cols
        min_y = max_y + geo_trans[5] * rows

        print(f"空间范围:")
        print(f"  X: {min_x:.6f} 到 {max_x:.6f}")
        print(f"  Y: {min_y:.6f} 到 {max_y:.6f}")

        # 初始化并集矩阵和有效数据掩码
        union_mask = np.zeros((rows, cols), dtype=np.uint8)
        valid_data_mask = np.zeros((rows, cols), dtype=np.bool_)

        processed_years = []
        skipped_years = []

        print(f"\n开始处理 {end_year - start_year + 1} 个年份的数据...")

        # 3. 循环处理每一年
        for year in range(start_year, end_year + 1):
            file_name = f"CHN_Maize_{year}.tif"
            file_path = os.path.join(maize_dir, file_name)

            if not os.path.exists(file_path):
                print(f"  [{year:4d}] 跳过: 文件不存在 - {file_name}")
                skipped_years.append(year)
                continue

            print(f"  [{year:4d}] 处理: {file_name}", end="", flush=True)

            try:
                # 打开源文件
                ds_src = gdal.Open(file_path, gdal.GA_ReadOnly)
                if ds_src is None:
                    print(f" - 无法打开文件，跳过")
                    skipped_years.append(year)
                    continue

                # 使用gdal.Warp进行对齐
                warp_options = gdal.WarpOptions(
                    format='VRT',
                    outputBounds=[min_x, min_y, max_x, max_y],
                    width=cols,
                    height=rows,
                    dstSRS=proj,
                    srcNodata=0,
                    dstNodata=0,
                    resampleAlg=gdal.GRA_NearestNeighbour,
                    errorThreshold=0,
                    warpMemoryLimit=512
                )

                # 执行重采样对齐
                tmp_ds = gdal.Warp('', ds_src, options=warp_options)

                if tmp_ds is None:
                    print(f" - 对齐失败，跳过")
                    skipped_years.append(year)
                    ds_src = None
                    continue

                # 读取对齐后的数据
                data = tmp_ds.ReadAsArray()
                if data is None:
                    print(f" - 读取数据失败，跳过")
                    skipped_years.append(year)
                    ds_src = None
                    tmp_ds = None
                    continue

                # 转换数据类型
                data = data.astype(np.float32)

                # 统计源数据信息
                src_total = data.size

                # 创建有效数据掩码（值大于0且不是NaN）
                valid_mask = (data > 0) & (~np.isnan(data))

                # 更新全局有效掩码
                valid_data_mask = np.logical_or(valid_data_mask, valid_mask)

                # 将有效的玉米数据加入并集
                binary_data = np.where(data > 0, 1, 0).astype(np.uint8)
                union_mask = np.logical_or(union_mask, binary_data).astype(np.uint8)

                # 统计该年份的有效区域
                valid_pixels = np.count_nonzero(valid_mask)
                valid_ratio = valid_pixels / src_total * 100 if src_total > 0 else 0

                # 统计玉米像素
                maize_pixels = np.count_nonzero(binary_data)
                maize_ratio = maize_pixels / src_total * 100 if src_total > 0 else 0

                print(f" - 完成 (玉米像素: {maize_pixels:,} [{maize_ratio:.1f}%])")

                # 释放资源
                ds_src = None
                tmp_ds = None

                processed_years.append(year)

            except Exception as e:
                print(f" - 处理失败: {str(e)[:50]}...")
                skipped_years.append(year)
                continue

        # 检查是否处理了任何文件
        if len(processed_years) == 0:
            raise RuntimeError("错误: 没有成功处理任何年份的数据")

        print(f"\n数据对齐完成，开始清理无效区域...")

        # 4. 清理无效区域
        # 应用有效数据掩码：只保留在至少一个年份中有效的区域
        union_mask_cleaned = np.where(valid_data_mask, union_mask, 0).astype(np.uint8)

        # 5. 检测并修复边缘伪影
        print(f"检测边缘伪影...")

        # 计算每一行是否有有效数据
        row_has_data = np.any(valid_data_mask, axis=1)

        # 找出第一个和最后一个有数据的行
        valid_rows = np.where(row_has_data)[0]
        if len(valid_rows) > 0:
            first_valid_row = valid_rows[0]
            last_valid_row = valid_rows[-1]

            print(f"  有效数据行范围: 第 {first_valid_row} 行 到 第 {last_valid_row} 行")
            print(f"  总行数: {rows}")

            # 检查是否有顶部空白区域
            if first_valid_row > 0:
                top_blank = first_valid_row
                print(f"  检测到顶部空白区域: {top_blank} 行")

            # 检查是否有底部空白区域
            if last_valid_row < rows - 1:
                bottom_blank = rows - last_valid_row - 1
                print(f"  检测到底部空白区域: {bottom_blank} 行")

                # 清理底部空白区域
                if bottom_blank > 10:
                    union_mask_cleaned[last_valid_row + 1:, :] = 0
                    print(f"  已清理底部 {bottom_blank} 行的空白区域")

        # 6. 计算统计信息
        total_pixels = rows * cols
        maize_pixels = np.count_nonzero(union_mask_cleaned)
        maize_ratio = maize_pixels / total_pixels * 100 if total_pixels > 0 else 0

        # 7. 保存最终结果
        print(f"\n正在保存结果...")
        print(f"输出文件: {output_file}")

        driver = gdal.GetDriverByName("GTiff")
        if driver is None:
            raise RuntimeError("无法获取GeoTiff驱动")

        # 创建选项
        creation_options = [
            'COMPRESS=LZW',
            'PREDICTOR=2',
            'TILED=YES',
            'BLOCKXSIZE=256',
            'BLOCKYSIZE=256',
            'BIGTIFF=IF_SAFER'
        ]

        out_ds = driver.Create(
            output_file,
            cols,
            rows,
            1,
            gdal.GDT_Byte,
            options=creation_options
        )

        if out_ds is None:
            raise RuntimeError("无法创建输出文件")

        out_ds.SetGeoTransform(geo_trans)
        out_ds.SetProjection(proj)

        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(union_mask_cleaned)
        out_band.SetNoDataValue(0)

        # 设置波段描述
        out_band.SetDescription(f"Maize Union {start_year}-{end_year}")

        # 计算统计信息
        out_band.FlushCache()
        out_band.ComputeStatistics(False)

        # 设置元数据
        out_ds.SetMetadata({
            'TIFFTAG_DOCUMENTNAME': f'Maize Union Mask {start_year}-{end_year}',
            'TIFFTAG_IMAGEDESCRIPTION': 'Maize planting union mask',
            'Base_Year': str(base_year),
            'Start_Year': str(start_year),
            'End_Year': str(end_year),
            'Processed_Years': ','.join(map(str, processed_years)),
            'Skipped_Years': ','.join(map(str, skipped_years)) if skipped_years else 'None',
            'Total_Cells': str(total_pixels),
            'Maize_Cells': str(maize_pixels),
            'Maize_Ratio': f'{maize_ratio:.2f}%',
            'Processing_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Processing_Software': 'MaizeUnionMerger v1.0'
        })

        # 确保写入磁盘
        out_ds.FlushCache()
        out_ds = None

        # 8. 输出统计报告
        print("\n" + "=" * 70)
        print("处理完成！")
        print("=" * 70)
        print(f"文件统计:")
        print(f"  处理成功的年份: {len(processed_years)} 年")
        print(f"  跳过的年份: {len(skipped_years)} 年")

        if processed_years:
            print(f"  处理年份列表: {sorted(processed_years)}")

        if skipped_years:
            print(f"  跳过年份列表: {sorted(skipped_years)}")

        print(f"\n空间统计:")
        print(f"  总像素数: {total_pixels:,}")
        print(f"  玉米像素数: {maize_pixels:,}")
        print(f"  覆盖率: {maize_ratio:.2f}%")

        print(f"\n输出文件:")
        print(f"  {output_file}")
        print("=" * 70)

        # 9. 保存详细的文本报告
        report_file = output_file.replace('.tif', '_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("玉米种植分布合并报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"数据目录: {maize_dir}\n")
            f.write(f"年份范围: {start_year}-{end_year}\n")
            f.write(f"基准年份: {base_year}\n")
            f.write(f"输出文件: {output_file}\n")
            f.write("\n处理统计:\n")
            f.write(f"  成功处理年份: {len(processed_years)}\n")
            f.write(f"  处理年份列表: {sorted(processed_years)}\n")
            f.write(f"  跳过年份: {len(skipped_years)}\n")
            if skipped_years:
                f.write(f"  跳过年份列表: {sorted(skipped_years)}\n")

            f.write("\n空间统计:\n")
            f.write(f"  网格尺寸: {cols}列 × {rows}行\n")
            f.write(f"  总像素数: {total_pixels:,}\n")

            f.write("\n玉米种植统计:\n")
            f.write(f"  玉米像素数: {maize_pixels:,}\n")
            f.write(f"  覆盖率: {maize_ratio:.2f}%\n")

            f.write("\n文件信息:\n")
            f.write(f"  左上角: ({geo_trans[0]:.6f}, {geo_trans[3]:.6f})\n")
            f.write(f"  像素大小: {geo_trans[1]:.8f} × {abs(geo_trans[5]):.8f}\n")
            f.write(f"  坐标系: {proj[:100]}...\n")

        print(f"详细报告: {report_file}")

        return output_file

    except Exception as e:
        print(f"\n❌ 处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def check_maize_files(maize_dir, start_year=2000, end_year=2019):
    """
    检查玉米数据文件是否存在

    参数:
    maize_dir: 玉米数据目录
    start_year: 起始年份
    end_year: 结束年份

    返回:
    存在文件的年份列表
    """
    print("检查玉米数据文件...")
    print("-" * 40)

    existing_years = []
    missing_years = []

    for year in range(start_year, end_year + 1):
        file_name = f"CHN_Maize_{year}.tif"
        file_path = os.path.join(maize_dir, file_name)

        if os.path.exists(file_path):
            existing_years.append(year)

            # 获取文件大小
            file_size = os.path.getsize(file_path)
            size_mb = file_size / (1024 * 1024)

            # 获取更详细的信息
            try:
                ds = gdal.Open(file_path, gdal.GA_ReadOnly)
                if ds is not None:
                    cols = ds.RasterXSize
                    rows = ds.RasterYSize
                    data = ds.GetRasterBand(1).ReadAsArray()
                    maize_pixels = np.count_nonzero(data)
                    total_pixels = cols * rows
                    maize_ratio = maize_pixels / total_pixels * 100 if total_pixels > 0 else 0
                    ds = None
                    print(f"  ✓ {file_name} ({size_mb:.1f} MB, {cols}×{rows}, 玉米: {maize_ratio:.1f}%)")
                else:
                    print(f"  ✓ {file_name} ({size_mb:.1f} MB)")
            except:
                print(f"  ✓ {file_name} ({size_mb:.1f} MB)")
        else:
            missing_years.append(year)
            print(f"  ✗ {file_name} (缺失)")

    print("-" * 40)
    print(f"总计: {len(existing_years)} 个文件存在, {len(missing_years)} 个文件缺失")

    if missing_years:
        print(f"缺失年份: {missing_years}")

    return existing_years


def verify_output_file(output_file):
    """
    验证输出文件

    参数:
    output_file: 输出文件路径

    返回:
    验证信息字典
    """
    if not os.path.exists(output_file):
        return {"status": "error", "message": "输出文件不存在"}

    try:
        ds = gdal.Open(output_file, gdal.GA_ReadOnly)
        if ds is None:
            return {"status": "error", "message": "无法打开输出文件"}

        geo_trans = ds.GetGeoTransform()
        proj = ds.GetProjection()
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        data = ds.GetRasterBand(1).ReadAsArray()

        total_pixels = cols * rows
        maize_pixels = np.count_nonzero(data)
        maize_ratio = maize_pixels / total_pixels * 100 if total_pixels > 0 else 0

        # 检查数据值
        unique_values = np.unique(data)

        ds = None

        return {
            "status": "success",
            "file_size_mb": os.path.getsize(output_file) / (1024 * 1024),
            "cols": cols,
            "rows": rows,
            "total_pixels": total_pixels,
            "maize_pixels": maize_pixels,
            "maize_ratio": maize_ratio,
            "unique_values": unique_values.tolist(),
            "resolution_x": abs(geo_trans[1]),
            "resolution_y": abs(geo_trans[5])
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def main():
    """主函数"""
    print("玉米种植分布数据合并工具")
    print("=" * 70)

    # 配置路径
    maize_dir = r"E:\data\全国1Km作物种植数据\玉米"

    # 检查路径是否存在
    if not os.path.exists(maize_dir):
        print(f"错误: 路径不存在 - {maize_dir}")
        print("请检查路径是否正确，或修改代码中的路径。")
        return 1

    # 检查玉米数据文件
    existing_years = check_maize_files(maize_dir, 2000, 2019)

    if not existing_years:
        print("\n错误: 未找到任何玉米数据文件!")
        print("请检查:")
        print(f"1. 路径是否正确: {maize_dir}")
        print("2. 文件命名格式应为: CHN_Maize_YYYY.tif (如 CHN_Maize_2000.tif)")
        return 1

    print(f"\n找到 {len(existing_years)} 个可用的数据文件")

    # 显示文件统计
    if existing_years:
        print(f"可用年份: {sorted(existing_years)}")

        # 计算缺失年份
        all_years = list(range(2000, 2020))
        missing_years = [y for y in all_years if y not in existing_years]
        if missing_years:
            print(f"缺失年份: {missing_years}")
            print("注意: 缺失年份的数据将不会被包含在并集中")

    # 开始合并处理
    print("\n" + "=" * 70)
    print("开始合并处理...")
    print("=" * 70)

    try:
        output_file = merge_maize_union(
            maize_dir=maize_dir,
            start_year=2000,
            end_year=2019,
            base_year=2019
        )

        print(f"\n✅ 合并完成!")
        print(f"输出文件: {output_file}")

        # 验证输出文件
        print(f"\n验证输出文件...")
        verify_info = verify_output_file(output_file)

        if verify_info["status"] == "success":
            print(f"  验证通过!")
            print(f"  文件大小: {verify_info['file_size_mb']:.1f} MB")
            print(f"  网格尺寸: {verify_info['cols']}列 × {verify_info['rows']}行")
            print(f"  玉米像素: {verify_info['maize_pixels']:,}")
            print(f"  覆盖率: {verify_info['maize_ratio']:.2f}%")
            print(f"  像素值: {verify_info['unique_values']}")
            print(f"  分辨率: {verify_info['resolution_x']:.8f} × {verify_info['resolution_y']:.8f}")
        else:
            print(f"  验证失败: {verify_info['message']}")

        return 0

    except KeyboardInterrupt:
        print(f"\n⚠️ 用户中断处理")
        return 1
    except Exception as e:
        print(f"\n❌ 处理失败: {str(e)}")
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