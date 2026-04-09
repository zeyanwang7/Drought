import xarray as xr
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
#将每日潜在蒸散发数据重采样到8天尺度
# --- 基础路径设置 ---
# 假设你的数据根目录在 pet 文件夹
root_dir = r'E:\data\meteorologicaldata\pet'
# 设置 8 天结果的输出目录
output_root = r'E:\data\meteorologicaldata\pet_8day_results'

if not os.path.exists(output_root):
    os.makedirs(output_root)


def preprocess_nc(ds):
    """预处理函数：从文件名提取准确的时间坐标"""
    path = ds.encoding['source']
    fname = os.path.basename(path)
    # 适配格式: ChinaMet_010deg_petPM_yyyy_mm_dd.nc
    parts = fname.replace('.nc', '').split('_')
    date_str = f"{parts[-3]}-{parts[-2]}-{parts[-1]}"
    ds = ds.expand_dims(time=[pd.to_datetime(date_str)])
    return ds


def batch_process_pet(start_year, end_year):
    print(f"开始批量处理 {start_year} 到 {end_year} 年的数据...")

    for year in range(start_year, end_year + 1):
        year_dir = os.path.join(root_dir, str(year))

        # 检查该年份文件夹是否存在
        if not os.path.exists(year_dir):
            print(f"警告：找不到年份文件夹 {year_dir}，跳过。")
            continue

        print(f"\n--- 正在处理 {year} 年 ---")

        # 搜索该年份下的所有日尺度 NC 文件
        file_pattern = os.path.join(year_dir, f"ChinaMet_010deg_petPM_{year}_*.nc")
        file_list = sorted(glob.glob(file_pattern))

        if not file_list:
            print(f"在 {year} 文件夹下未找到匹配文件。")
            continue

        try:
            # 1. 读取并合并该年份的所有天数据
            # chunks 设为 {'time': 10} 可以平衡读取速度和内存占用
            ds_year = xr.open_mfdataset(file_list, combine='nested', concat_dim='time',
                                        preprocess=preprocess_nc, chunks={'time': 10})

            # 2. 8天重采样（累加总和）
            ds_8day = ds_year.resample(time='8D', label='left').sum()

            # 3. 保存该年份的结果
            output_filename = f"PET_8day_{year}.nc"
            output_path = os.path.join(output_root, output_filename)

            # 写入文件
            ds_8day.to_netcdf(output_path)
            print(f"成功保存: {output_path} (时间步数: {len(ds_8day.time)})")

            # 释放内存
            ds_year.close()
            ds_8day.close()

        except Exception as e:
            print(f"处理 {year} 年时发生错误: {e}")

    print("\n所有年份处理完成！")
    print(f"结果存放在: {output_root}")


if __name__ == "__main__":
    # 执行 2000 到 2019 年的处理
    batch_process_pet(2020, 2022)