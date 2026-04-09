import os
import numpy as np
from osgeo import gdal
#2000年到2019年小麦种植分布合并

def align_and_union_wheat_improved(folder_path, start_year=2000, end_year=2019, base_year=2019):
    """
    改进版：处理小麦种植数据对齐与并集运算，专门处理对齐时产生的无效边缘区域

    参数:
    folder_path: 数据文件夹路径
    start_year: 起始年份
    end_year: 结束年份
    base_year: 基准对齐年份

    返回:
    输出文件的路径
    """
    try:
        # 1. 设置基准文件路径
        base_file = os.path.join(folder_path, f"CHN_Wheat_{base_year}.tif")
        output_file = os.path.join(folder_path, f"CHN_Wheat_Union_{start_year}_{end_year}_cleaned.tif")

        print("=" * 60)
        print(f"小麦种植数据并集处理工具")
        print(f"基准年份: {base_year}")
        print(f"处理范围: {start_year} - {end_year}")
        print(f"数据路径: {folder_path}")
        print("=" * 60)

        # 检查基准文件
        if not os.path.exists(base_file):
            raise FileNotFoundError(f"错误：找不到基准文件 {base_file}")

        # 检查文件夹是否存在
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"错误：文件夹不存在 {folder_path}")

        # 2. 读取基准文件的参数
        ds_base = gdal.Open(base_file, gdal.GA_ReadOnly)
        if ds_base is None:
            raise RuntimeError(f"无法打开基准文件: {base_file}")

        geo_trans = ds_base.GetGeoTransform()
        proj = ds_base.GetProjection()
        cols = ds_base.RasterXSize
        rows = ds_base.RasterYSize

        # 验证地理变换参数
        if geo_trans[1] == 0 or geo_trans[5] == 0:
            print("警告：地理变换参数可能有问题，请检查基准文件")

        print(f"基准文件: {os.path.basename(base_file)}")
        print(f"输出尺寸: {cols} 列 × {rows} 行")
        print(
            f"地理范围: [{geo_trans[0]}, {geo_trans[3] + geo_trans[5] * rows}] × [{geo_trans[0] + geo_trans[1] * cols}, {geo_trans[3]}]")
        print("-" * 60)

        # 计算输出范围
        min_x = geo_trans[0]
        max_y = geo_trans[3]
        max_x = min_x + geo_trans[1] * cols
        min_y = max_y + geo_trans[5] * rows

        # 初始化并集矩阵和有效数据掩码
        union_mask = np.zeros((rows, cols), dtype=np.uint8)
        valid_data_mask = np.zeros((rows, cols), dtype=np.bool_)

        processed_years = []
        skipped_years = []
        all_year_valid_masks = []

        print(f"开始处理数据对齐和并集运算...")
        print(f"正在检查并处理 {end_year - start_year + 1} 个年份的数据")

        # 3. 循环处理每一年
        for year in range(start_year, end_year + 1):
            file_name = f"CHN_Wheat_{year}.tif"
            file_path = os.path.join(folder_path, file_name)

            if not os.path.exists(file_path):
                print(f"  [{year}] 跳过：文件不存在 - {file_name}")
                skipped_years.append(year)
                continue

            print(f"  [{year}] 正在处理: {file_name}", end="", flush=True)

            try:
                # 打开源文件
                ds_src = gdal.Open(file_path, gdal.GA_ReadOnly)
                if ds_src is None:
                    print(f" - 无法打开文件，跳过")
                    skipped_years.append(year)
                    continue

                # 获取源文件信息（用于调试）
                src_proj = ds_src.GetProjection()
                src_geo_trans = ds_src.GetGeoTransform()
                src_cols = ds_src.RasterXSize
                src_rows = ds_src.RasterYSize

                # 使用gdal.Warp进行对齐
                warp_options = gdal.WarpOptions(
                    format='VRT',
                    outputBounds=[min_x, min_y, max_x, max_y],
                    width=cols,
                    height=rows,
                    dstSRS=proj,
                    srcNodata=0,  # 源数据的无数据值
                    dstNodata=0,  # 目标数据的无数据值
                    resampleAlg=gdal.GRA_NearestNeighbour,
                    errorThreshold=0,
                    warpMemoryLimit=1024  # 内存限制(MB)
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
                src_nonzero = np.count_nonzero(data)
                src_total = data.size

                # 创建有效数据掩码（值大于0且不是NaN）
                valid_mask = (data > 0) & (~np.isnan(data))

                # 保存该年份的有效掩码
                all_year_valid_masks.append(valid_mask.copy())

                # 更新全局有效掩码（至少在一个年份中有效的区域）
                valid_data_mask = np.logical_or(valid_data_mask, valid_mask)

                # 将有效的小麦数据加入并集
                binary_data = np.where(data > 0, 1, 0).astype(np.uint8)
                union_mask = np.logical_or(union_mask, binary_data).astype(np.uint8)

                # 统计该年份的有效区域
                valid_pixels = np.count_nonzero(valid_mask)
                valid_ratio = valid_pixels / src_total * 100 if src_total > 0 else 0

                print(f" - 完成 (有效像素: {valid_pixels:,} [{valid_ratio:.1f}%])")

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
            raise RuntimeError("错误：没有成功处理任何年份的数据")

        print("-" * 60)
        print(f"数据对齐完成，开始清理无效区域...")

        # 4. 清理无效区域（处理黑色长条问题）
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

            # 检查是否有底部空白区域（黑色长条）
            if last_valid_row < rows - 1:
                bottom_blank = rows - last_valid_row - 1
                print(f"  检测到底部空白区域（黑色长条）: {bottom_blank} 行")

                # 清理底部空白区域
                if bottom_blank > 10:  # 如果空白超过10行，认为是伪影
                    union_mask_cleaned[last_valid_row + 1:, :] = 0
                    print(f"  已清理底部 {bottom_blank} 行的空白区域")

        # 6. 检测完全无效的列
        col_has_data = np.any(valid_data_mask, axis=0)
        valid_cols = np.where(col_has_data)[0]
        if len(valid_cols) > 0:
            first_valid_col = valid_cols[0]
            last_valid_col = valid_cols[-1]

            left_blank = first_valid_col
            right_blank = cols - last_valid_col - 1

            if left_blank > 0:
                print(f"  检测到左侧空白区域: {left_blank} 列")
            if right_blank > 0:
                print(f"  检测到右侧空白区域: {right_blank} 列")

        # 7. 计算清理前后的统计信息
        total_pixels = rows * cols
        wheat_pixels_before = np.count_nonzero(union_mask)
        wheat_pixels_after = np.count_nonzero(union_mask_cleaned)
        valid_pixels_total = np.count_nonzero(valid_data_mask)

        wheat_ratio_before = (wheat_pixels_before / total_pixels) * 100 if total_pixels > 0 else 0
        wheat_ratio_after = (wheat_pixels_after / total_pixels) * 100 if total_pixels > 0 else 0
        valid_ratio_total = (valid_pixels_total / total_pixels) * 100 if total_pixels > 0 else 0

        # 8. 保存最终结果
        print(f"\n正在保存清理后的结果...")
        print(f"输出文件: {output_file}")

        driver = gdal.GetDriverByName("GTiff")
        if driver is None:
            raise RuntimeError("无法获取GeoTiff驱动")

        # 创建选项：使用LZW压缩，分块存储，适合大文件
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
        out_band.SetDescription(f"Wheat Union {start_year}-{end_year} (Cleaned)")

        # 计算统计信息
        out_band.FlushCache()
        out_band.ComputeStatistics(False)

        # 设置元数据
        out_ds.SetMetadata({
            'TIFFTAG_DOCUMENTNAME': f'Wheat Union Mask {start_year}-{end_year}',
            'TIFFTAG_IMAGEDESCRIPTION': 'Cleaned wheat planting union mask with edge artifacts removed',
            'Base_Year': str(base_year),
            'Start_Year': str(start_year),
            'End_Year': str(end_year),
            'Processed_Years': ','.join(map(str, processed_years)),
            'Skipped_Years': ','.join(map(str, skipped_years)) if skipped_years else 'None',
            'Processing_Date': np.datetime64('today').astype(str)
        })

        # 确保写入磁盘
        out_ds.FlushCache()
        out_ds = None

        # 9. 输出统计报告
        print("=" * 60)
        print(f"处理完成！")
        print("=" * 60)
        print(f"文件统计:")
        print(f"  处理成功的年份: {len(processed_years)} 年")
        print(f"  跳过的年份: {len(skipped_years)} 年")
        if skipped_years:
            print(f"  跳过年份列表: {skipped_years}")

        print(f"\n空间统计:")
        print(f"  总像素数: {total_pixels:,}")
        print(f"  有效数据区域: {valid_pixels_total:,} 像素 [{valid_ratio_total:.2f}%]")

        print(f"\n小麦种植统计 (清理前):")
        print(f"  小麦像素数: {wheat_pixels_before:,}")
        print(f"  覆盖率: {wheat_ratio_before:.2f}%")

        print(f"\n小麦种植统计 (清理后):")
        print(f"  小麦像素数: {wheat_pixels_after:,}")
        print(f"  覆盖率: {wheat_ratio_after:.2f}%")

        print(f"\n清理效果:")
        if wheat_pixels_before > 0:
            change_pixels = wheat_pixels_after - wheat_pixels_before
            change_ratio = (change_pixels / wheat_pixels_before) * 100
            print(f"  像素变化: {change_pixels:+,} 像素 [{change_ratio:+.2f}%]")

        print(f"\n输出文件已保存至:")
        print(f"  {output_file}")
        print("=" * 60)

        return output_file

    except Exception as e:
        print(f"\n❌ 处理过程中发生错误: {str(e)}")
        print("=" * 60)
        raise


def check_single_file(file_path):
    """
    检查单个TIFF文件的统计信息，用于诊断问题

    参数:
    file_path: TIFF文件路径
    """
    try:
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return

        print(f"\n检查文件: {os.path.basename(file_path)}")
        print("-" * 40)

        ds = gdal.Open(file_path, gdal.GA_ReadOnly)
        if ds is None:
            print("无法打开文件")
            return

        # 获取基本信息
        geo_trans = ds.GetGeoTransform()
        proj = ds.GetProjection()
        cols = ds.RasterXSize
        rows = ds.RasterYSize

        print(f"尺寸: {cols} × {rows}")
        print(f"地理变换: {geo_trans}")

        # 读取数据
        data = ds.ReadAsArray()
        print(f"数据类型: {data.dtype}")
        print(f"数据形状: {data.shape}")

        # 统计信息
        unique_values = np.unique(data)
        print(f"唯一值: {unique_values}")

        total_pixels = data.size
        nonzero_pixels = np.count_nonzero(data)
        nonzero_ratio = (nonzero_pixels / total_pixels) * 100

        print(f"总像素: {total_pixels:,}")
        print(f"非零像素: {nonzero_pixels:,} [{nonzero_ratio:.2f}%]")

        # 检查每行/每列的情况
        zero_rows = np.all(data == 0, axis=1)
        zero_cols = np.all(data == 0, axis=0)

        zero_row_count = np.sum(zero_rows)
        zero_col_count = np.sum(zero_cols)

        print(f"全零行数: {zero_row_count} [{zero_row_count / rows * 100:.1f}%]")
        print(f"全零列数: {zero_col_count} [{zero_col_count / cols * 100:.1f}%]")

        # 找出连续的全零区域
        if zero_row_count > 0:
            zero_row_indices = np.where(zero_rows)[0]
            print(
                f"全零行索引范围: {zero_row_indices[0] if len(zero_row_indices) > 0 else 'N/A'} 到 {zero_row_indices[-1] if len(zero_row_indices) > 0 else 'N/A'}")

        ds = None

    except Exception as e:
        print(f"检查文件时出错: {str(e)}")


if __name__ == "__main__":
    # --- 请在此处修改你的文件夹路径 ---
    target_path = r"E:\data\全国1Km作物种植数据\小麦"

    # 可选：先检查单个文件的问题
    # check_single_file(os.path.join(target_path, "CHN_Wheat_2019.tif"))
    # check_single_file(os.path.join(target_path, "CHN_Wheat_2000.tif"))

    # 运行主函数
    try:
        result_file = align_and_union_wheat_improved(
            folder_path=target_path,
            start_year=2000,
            end_year=2019,
            base_year=2019
        )

        print(f"\n✅ 处理完成！输出文件: {result_file}")

        # 可选：验证输出文件
        # check_single_file(result_file)

    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")