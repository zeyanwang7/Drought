import os
import glob
import xarray as xr
import pandas as pd
import re

# --- 1. 路径与参数设置 ---
# 输入每日 D 值根目录
input_daily_dir = r'E:\data\meteorologicaldata\P_minus_PET'
# 输出 30天累积+8天采样的中间结果文件
output_final_path = r'E:\data\meteorologicaldata\D_30d_accumulated_8day_step.nc'

YEAR_START = 2000
YEAR_END = 2022


def preprocess_add_time(ds):
    """从文件名 ChinaMet_010deg_D_YYYY_MM_DD.nc 中提取日期"""
    filepath = ds.encoding['source']
    filename = os.path.basename(filepath)
    match = re.search(r'(\d{4}_\d{2}_\d{2})', filename)
    if match:
        date_str = match.group(1)
        date_obj = pd.to_datetime(date_str, format='%Y_%m_%d')
        ds = ds.expand_dims(time=[date_obj])
    return ds


def run_step1_resampling():
    print(f"🚀 开始执行接力式计算：{YEAR_START} - {YEAR_END}")
    print(f"模式：30天每日滑动累积 -> 8天步长重采样")
    print("-" * 60)

    yearly_results = []
    # 缓冲区：存储前一年最后 29 天的原始 D 值
    buffer_prev_days = None

    for year in range(YEAR_START, YEAR_END + 1):
        # 1. 获取当年所有日尺度文件
        year_pattern = os.path.join(input_daily_dir, str(year), f"ChinaMet_010deg_D_{year}_*.nc")
        files = glob.glob(year_pattern)
        files.sort()

        if not files:
            print(f"⚠️  {year} 年未找到文件，跳过。")
            continue

        # 2. 读取当年数据并直接载入内存 (单年份数据量约 500MB，内存完全可控)
        ds_year = xr.open_mfdataset(files, combine='nested', concat_dim='time',
                                    preprocess=preprocess_add_time).load()

        # 3. 拼接前一年缓冲区，确保 1 月份计算不中断
        if buffer_prev_days is not None:
            # 将前一年 12 月的尾巴和今年拼接
            ds_to_calc = xr.concat([buffer_prev_days, ds_year], dim='time')
        else:
            # 只有 2000 年没有缓冲区
            ds_to_calc = ds_year

        # 4. 执行 30 天滑动累积计算
        # 注意：这里的窗口是基于“每日”序列的，非常精准
        ds_accumulated = ds_to_calc['D'].rolling(time=30, center=False, min_periods=30).sum()

        # 5. 只提取属于当年的部分（切掉缓冲区的日期）
        ds_year_only = ds_accumulated.sel(time=str(year))

        # 6. 执行 8 天步长采样 (对应论文中的 Weekly 更新频率)
        ds_8d = ds_year_only.resample(time='8D').last()
        yearly_results.append(ds_8d)

        # 7. 更新缓冲区：提取今年最后 29 天的原始 D 值给明年使用
        buffer_prev_days = ds_year.isel(time=slice(-29, None))

        # 8. 打印实时进度
        print(f"✅ {year} 年处理完成 | 该年采样期数: {len(ds_8d.time)} | 进度: {year}/{YEAR_END}")

    # --- 最终合并与保存 ---
    print("-" * 60)
    print("📦 正在合并 23 年重采样序列...")

    full_ds = xr.concat(yearly_results, dim='time').to_dataset(name='D_30d_acc')

    # 增加压缩编码，节省硬盘
    encoding = {'D_30d_acc': {'zlib': True, 'complevel': 4}}

    if os.path.exists(output_final_path):
        os.remove(output_final_path)

    full_ds.to_netcdf(output_final_path, encoding=encoding)

    print("-" * 60)
    print(f"🎉 任务成功完成！")
    print(f"重采样文件路径: {output_final_path}")
    print(f"总计观测期数: {len(full_ds.time)} (2000-2022)")


if __name__ == "__main__":
    run_step1_resampling()