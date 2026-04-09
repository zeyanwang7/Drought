import os
import numpy as np
from osgeo import gdal

# --- 参数设置 ---
input_dir = r'E:\data\全国1Km作物种植数据\小麦'
ref_year = 2019  # 基准年份
output_path = os.path.join(input_dir, 'Wheat_Final_Presence5_2000_2019_Fixed.tif')
start_year = 2000
end_year = 2019
threshold = 5


def process_and_fix_black_rectangle():
    # 1. 获取基准文件元数据
    ref_file = os.path.join(input_dir, f"CHN_Wheat_{ref_year}.tif")
    if not os.path.exists(ref_file):
        print(f"错误: 找不到基准文件 {ref_file}")
        return

    ref_ds = gdal.Open(ref_file)
    ref_proj = ref_ds.GetProjection()
    ref_geotrans = ref_ds.GetGeoTransform()
    cols = ref_ds.RasterXSize
    rows = ref_ds.RasterYSize

    # 初始化累加计数器 (16位整数以防溢出)
    count_array = np.zeros((rows, cols), dtype=np.int16)

    print(f"开始处理，基准尺寸: {rows}x{cols}")

    # 2. 遍历并对齐数据
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(input_dir, f"CHN_Wheat_{year}.tif")
        if not os.path.exists(file_path):
            continue

        # 使用 Warp 强制对齐到基准范围
        # srcNodata 和 dstNodata 都设为 0，确保背景一致
        tmp_ds = gdal.Warp('', file_path, format='VRT',
                           dstSRS=ref_proj,
                           outputBounds=(ref_geotrans[0],
                                         ref_geotrans[3] + ref_geotrans[5] * rows,
                                         ref_geotrans[0] + ref_geotrans[1] * cols,
                                         ref_geotrans[3]),
                           width=cols, height=rows,
                           srcNodata=0, dstNodata=0,
                           resampleAlg=gdal.GRA_NearestNeighbour)

        data = tmp_ds.ReadAsArray()
        # 核心：统计有种植的像元 (假设 1 为种植)
        count_array += (data == 1).astype(np.int16)
        print(f"已处理: {year}")
        tmp_ds = None

    # 3. 执行 10 年判定逻辑
    # 满足条件为 1，不满足为 0
    result_array = np.where(count_array >= threshold, 1, 0).astype(np.uint8)

    # 4. 写入输出文件（解决黑块的关键配置）
    driver = gdal.GetDriverByName('GTiff')
    # 使用 LZW 压缩并设置预处理，提升 QGIS 兼容性
    options = ['COMPRESS=LZW', 'PREDICTOR=2', 'TILED=YES']
    out_ds = driver.Create(output_path, cols, rows, 1, gdal.GDT_Byte, options=options)
    out_ds.SetProjection(ref_proj)
    out_ds.SetGeoTransform(ref_geotrans)

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(result_array)

    # 将 0 设置为 NoData，这样 QGIS 就会把 0 (无数据) 渲染为透明，而不是黑色
    out_band.SetNoDataValue(0)

    out_ds.FlushCache()
    out_ds = None
    print(f"\n处理完成！生成文件：{output_path}")


if __name__ == "__main__":
    process_and_fix_black_rectangle()