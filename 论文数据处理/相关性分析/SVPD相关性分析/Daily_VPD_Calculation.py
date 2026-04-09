import xarray as xr
import numpy as np
import os
from glob import glob

# --- 1. 基础路径配置 ---
base_rhu_dir = r'E:\data\meteorologicaldata\RelativeHumidity'
base_tmax_dir = r'E:\data\meteorologicaldata\Tmpmax'
base_output_dir = r'E:\data\meteorologicaldata\VPD'

years = range(2000, 2024)  # 包含 2000 到 2023

# --- 2. 开始循环处理每一年的数据 ---
for year in years:
    print(f"\n{'=' * 30}")
    print(f"🚀 正在处理年份: {year}")

    # 构建当前年份的具体路径
    curr_rhu_dir = os.path.join(base_rhu_dir, str(year))
    curr_tmax_dir = os.path.join(base_tmax_dir, str(year))
    curr_output_dir = os.path.join(base_output_dir, str(year))

    # 检查输入文件夹是否存在
    if not (os.path.exists(curr_rhu_dir) and os.path.exists(curr_tmax_dir)):
        print(f"⚠️ 路径缺失，跳过 {year} 年。")
        continue

    # 创建输出文件夹
    if not os.path.exists(curr_output_dir):
        os.makedirs(curr_output_dir)

    # 获取该年份下所有的每日 .nc 文件
    rhu_files = sorted(glob(os.path.join(curr_rhu_dir, f"ChinaMet_010deg_rhu_{year}_*.nc")))
    tmax_files = sorted(glob(os.path.join(curr_tmax_dir, f"ChinaMet_010deg_tmpmax_{year}_*.nc")))

    if len(rhu_files) == 0 or len(tmax_files) == 0:
        print(f"❌ {year} 年文件夹内未找到 .nc 文件，跳过。")
        continue

    # --- 3. 批量读取与拼接 ---
    try:
        # 使用 preprocess 确保时间维度对齐（如果文件里没写时间维，可以用这个函数添加）
        ds_rhu = xr.open_mfdataset(rhu_files, concat_dim='time', combine='nested')
        ds_tmax = xr.open_mfdataset(tmax_files, concat_dim='time', combine='nested')

        # 提取变量 (根据 ChinaMet 变量名)
        rh = ds_rhu['rhu']
        tmax = ds_tmax['tmpmax']

        # --- 4. 物理公式计算 ---
        # 1. 计算最高温下的饱和水汽压 (kPa)
        es_tmax = 0.6108 * np.exp((17.27 * tmax) / (tmax + 237.3))

        # 2. 计算实际水汽压 (ea)，利用日均相对湿度
        ea = es_tmax * (rh / 100.0)

        # 3. 计算每日最大 VPD (kPa)
        vpd_max = es_tmax - ea

        # 配置元数据
        vpd_max.name = 'vpd_max'
        vpd_max.attrs = {
            'units': 'kPa',
            'long_name': 'Daily Maximum Vapor Pressure Deficit',
            'method': 'Calculated from ChinaMet Tmax and RHmean using Tetens formula'
        }

        # --- 5. 保存结果 ---
        # 保存为年文件
        output_filename = f'ChinaMet_010deg_VPDmax_{year}_Annual.nc'
        output_path = os.path.join(curr_output_dir, output_filename)

        print(f"💾 正在写入: {output_filename}")
        vpd_max.to_netcdf(output_path)

        # 释放内存，关闭数据集
        ds_rhu.close()
        ds_tmax.close()
        print(f"✅ {year} 年计算完成！")

    except Exception as e:
        print(f"💥 {year} 年处理出错: {str(e)}")

print("\n" + "#" * 30)
print("🎉 所有年份处理任务结束！")