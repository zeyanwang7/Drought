import xarray as xr
import os

# --- 路径设置 ---
prec_dir = r'E:\data\meteorologicaldata\Precip_8Day_Standard'
pet_dir = r'E:\data\meteorologicaldata\pet_8day_results'
output_dir = r'E:\data\meteorologicaldata\D_value_8day'

# 创建输出文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def calculate_d_value(start_year, end_year):
    print(f"开始计算 {start_year} - {end_year} 年的水分盈亏量 (P-PET)...")

    for year in range(start_year, end_year + 1):
        # 构建文件路径
        prec_file = os.path.join(prec_dir, f"ChinaMet_0.10deg_Precip_8Day_{year}.nc")
        pet_file = os.path.join(pet_dir, f"PET_8day_{year}.nc")

        # 检查文件是否存在
        if not (os.path.exists(prec_file) and os.path.exists(pet_file)):
            print(f"警告：{year} 年数据不完整，跳过。")
            continue

        try:
            # 1. 加载数据
            ds_p = xr.open_dataset(prec_file)
            ds_pet = xr.open_dataset(pet_file)

            # 2. 获取变量名
            var_p = list(ds_p.data_vars)[0]
            var_pet = list(ds_pet.data_vars)[0]

            # 3. 强制对齐坐标 (防止经纬度极微小的差异)
            # join='inner' 确保只计算空间重叠部分
            ds_p, ds_pet = xr.align(ds_p, ds_pet, join='inner')

            # 4. 计算 P-PET
            da_d = ds_p[var_p] - ds_pet[var_pet]

            # 5. 转换为 Dataset 并将变量名命名为 'P-PET'
            ds_d = da_d.to_dataset(name='P-PET')

            # 6. 保存结果
            output_path = os.path.join(output_dir, f"D_value_8day_{year}.nc")
            ds_d.to_netcdf(output_path)

            print(f"成功计算并保存 {year} 年 P-PET 值。")

            # 释放内存
            ds_p.close()
            ds_pet.close()

        except Exception as e:
            print(f"{year} 年计算出错: {e}")

    print(f"\n全部处理完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    calculate_d_value(2020, 2022)