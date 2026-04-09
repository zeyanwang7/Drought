import os
from osgeo import gdal
import time

# 显式开启异常处理，消除警告
gdal.UseExceptions()

# ---------- 路径配置 ----------
# 原始数据根目录
base_dir = r'E:\数据集\种植作物数据集'
# 输出结果存放目录
output_root = r'E:\数据集\全国拼接结果_WGS84'

# 确保输出目录存在
if not os.path.exists(output_root):
    os.makedirs(output_root)

# 设置目标坐标系为 WGS84 (度)
dst_crs = 'EPSG:4326'


def batch_process_remainder():
    total_start_time = time.time()
    print("=" * 60)
    print(f"任务启动：准备处理 2002 年至 2024 年的数据 (已跳过 2001)")
    print("=" * 60)

    # 循环范围调整为 2002 到 2024 (range(2002, 2025))
    for year in range(2021, 2025):
        year_str = str(year)
        year_path = os.path.join(base_dir, year_str)
        year_start_time = time.time()

        # 1. 检查文件夹
        if not os.path.exists(year_path):
            print(f"跳过：未找到 {year_str} 年文件夹。")
            continue

        # 2. 搜集该年份下所有的 .tif 文件
        tif_files = [
            os.path.join(year_path, f)
            for f in os.listdir(year_path)
            if f.lower().endswith('.tif')
        ]

        if not tif_files:
            print(f"跳过：{year_str} 年文件夹中没有找到栅格文件。")
            continue

        print(f"正在拼接 {year_str} 年：发现 {len(tif_files)} 个分幅，请稍候...")

        # 3. 输出路径与临时路径
        output_filename = f"China_Maize_{year_str}_WGS84.tif"
        output_path = os.path.join(output_root, output_filename)
        vrt_path = os.path.join(output_root, f"temp_{year_str}.vrt")

        try:
            # 构建虚拟栅格 (srcNodata=0 处理重叠)
            vrt_options = gdal.BuildVRTOptions(srcNodata=0, VRTNodata=0)
            gdal.BuildVRT(vrt_path, tif_files, options=vrt_options)

            # 执行实质性的拼接与投影
            gdal.Warp(
                output_path,
                vrt_path,
                format='GTiff',
                dstSRS=dst_crs,
                resampleAlg='near',
                creationOptions=[
                    'COMPRESS=LZW',
                    'TILED=YES',
                    'BIGTIFF=YES'
                ],
                multithread=True
            )

            # --- 进度打印 ---
            year_end_time = time.time()
            duration = year_end_time - year_start_time
            print(f"✅ [进度提示] {year_str} 年全国数据拼接已完成！ (本年耗时: {duration:.2f} 秒)")
            print("-" * 60)

        except Exception as e:
            print(f"❌ [报错] 处理 {year_str} 年时发生异常: {e}")

        finally:
            # 清理临时文件
            if os.path.exists(vrt_path):
                os.remove(vrt_path)

    total_end_time = time.time()
    total_min = (total_end_time - total_start_time) / 60
    print("=" * 60)
    print(f"🎉 2002-2024 年任务全部处理成功！总耗时: {total_min:.2f} 分钟")
    print("=" * 60)


if __name__ == '__main__':
    batch_process_remainder()