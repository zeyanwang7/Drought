import os
import numpy as np
from osgeo import gdal

# 开启异常抛出
gdal.UseExceptions()

# --- 1. 路径设置 ---
input_dir = r'E:\data\新疆棉花作物种植地图\Merged_Output'
output_path = r'E:\data\新疆棉花作物种植地图\Cotton_2018_2021_Union.tif'

years = [2018, 2019, 2020, 2021]
input_files = [os.path.join(input_dir, f'Cotton_{year}_TOA_10m_1_filter.tif') for year in years]

# --- 2. 预检查与参数获取 ---
for f in input_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"找不到文件: {f}，请确认年度合并已完成。")

ds_first = gdal.Open(input_files[0])
proj = ds_first.GetProjection()
geotrans = ds_first.GetGeoTransform()
cols = ds_first.RasterXSize
rows = ds_first.RasterYSize

# --- 3. 创建输出文件（二值图，1比特或字节存储） ---
driver = gdal.GetDriverByName('GTiff')
# 使用 GDT_Byte，开启压缩以节省空间
ds_out = driver.Create(output_path, cols, rows, 1, gdal.GDT_Byte,
                       options=['COMPRESS=LZW', 'BIGTIFF=YES', 'TILED=YES'])
ds_out.SetProjection(proj)
ds_out.SetGeoTransform(geotrans)
out_band = ds_out.GetRasterBand(1)

# --- 4. 分块逻辑或运算（Union） ---
block_size = 4096  # 10m数据建议块大一点，提高IO效率
print(f"🚀 开始计算 2018-2021 并集掩膜...")
print(f"图像尺寸: {cols} x {rows}")

for i in range(0, rows, block_size):
    row_count = min(block_size, rows - i)
    for j in range(0, cols, block_size):
        col_count = min(block_size, cols - j)

        # 初始化当前块的并集矩阵（全0）
        union_block = np.zeros((row_count, col_count), dtype=bool)

        for f in input_files:
            ds = gdal.Open(f)
            # 只读取当前块
            data = ds.GetRasterBand(1).ReadAsArray(j, i, col_count, row_count)
            # 逻辑或：只要有一年非0，该位置即为True
            union_block = np.logical_or(union_block, data > 0)
            ds = None

        # 将布尔矩阵转为 0/1 写入
        out_band.WriteArray(union_block.astype(np.uint8), j, i)

# 刷新缓存并关闭
out_band.FlushCache()
ds_out = None
print(f"\n✅ 并集合并完成！")
print(f"结果路径: {output_path}")