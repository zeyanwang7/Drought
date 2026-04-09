import xarray as xr
import os
import pandas as pd
from datetime import datetime

# 1. 设置路径
prec_base_path = r'E:\data\meteorologicaldata\precipitation'
pet_base_path = r'E:\data\meteorologicaldata\pet'
output_base_path = r'E:\data\meteorologicaldata\P_minus_PET'  # 结果保存路径

# 如果输出目录不存在，则创建
if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)

# 2. 设置需要计算的年份范围
years = range(2000, 2023)  # 示例：计算2000年到2022年

for year in years:
    print(f"正在处理 {year} 年数据...")
    year_str = str(year)

    # 定义该年份下结果存储的文件夹
    year_output_dir = os.path.join(output_base_path, year_str)
    if not os.path.exists(year_output_dir):
        os.makedirs(year_output_dir)

    # 构造该年份的数据路径
    prec_year_path = os.path.join(prec_base_path, year_str)
    pet_year_path = os.path.join(pet_base_path, year_str)

    # 检查路径是否存在
    if not os.path.exists(prec_year_path) or not os.path.exists(pet_year_path):
        print(f"跳过 {year} 年：未找到对应文件夹")
        continue

    # 获取该年份下所有的日期（生成 01_01 到 12_31 的列表）
    # 使用 pandas 生成日期范围以处理闰年问题
    date_range = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31')

    for date in date_range:
        date_str = date.strftime('%Y_%m_%d')

        # 构造具体的文件名
        prec_file = f"ChinaMet_010deg_prec_{date_str}.nc"
        pet_file = f"ChinaMet_010deg_petPM_{date_str}.nc"

        prec_full_path = os.path.join(prec_year_path, prec_file)
        pet_full_path = os.path.join(pet_year_path, pet_file)

        # 检查两个文件是否都存在
        if os.path.exists(prec_full_path) and os.path.exists(pet_full_path):
            try:
                # 3. 读取数据
                ds_prec = xr.open_dataset(prec_full_path)
                ds_pet = xr.open_dataset(pet_full_path)

                # 提取变量（请确保变量名正确，通常可通过 ds.data_vars 查看）
                # 这里假设变量名分别为 'prec' 和 'pet'，如果不同请修改
                var_prec = list(ds_prec.data_vars)[0]
                var_pet = list(ds_pet.data_vars)[0]

                # 4. 计算 P - PET
                diff = ds_prec[var_prec] - ds_pet[var_pet]

                # 转换为 Dataset 并重命名变量
                ds_diff = diff.to_dataset(name='D')

                # 5. 保存结果
                output_file = f"ChinaMet_010deg_D_{date_str}.nc"
                ds_diff.to_netcdf(os.path.join(year_output_dir, output_file))

                # 关闭文件释放资源
                ds_prec.close()
                ds_pet.close()

            except Exception as e:
                print(f"处理日期 {date_str} 时出错: {e}")
        else:
            print(f"缺失文件: {date_str}")

print("全部计算完成！")