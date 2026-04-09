import xarray as xr
import os
import numpy as np

# 1. 设置路径
base_path = r'E:\data\Soil_Humidity'
output_path = r'E:\data\Soil_Humidity_Average_0_100'

if not os.path.exists(output_path):
    os.makedirs(output_path)

# 2. 定义参数
depths = [f"{i}cm" for i in range(10, 110, 10)]
years = range(2000, 2023)
variable_name = 'SMCI'

for year in years:
    print(f"--- 正在处理 {year} 年 ---")
    year_layers = []

    try:
        # 逐层读取
        for depth in depths:
            file_name = f"SMCI_9km_{year}_{depth}.nc"
            file_path = os.path.join(base_path, depth, file_name)

            # 使用 chunks 避免大内存占用
            ds = xr.open_dataset(file_path, chunks={'time': 50})
            year_layers.append(ds[variable_name])

        # 3. 计算均值
        combined = xr.concat(year_layers, dim='d')
        avg_sm = combined.mean(dim='d')

        # 4. 封装并保留元数据
        output_ds = avg_sm.to_dataset(name='SMCI_avg')

        # 读取一个模板文件来复制原始属性
        template_ds = xr.open_dataset(os.path.join(base_path, '10cm', f"SMCI_9km_{year}_10cm.nc"))
        output_ds.attrs = template_ds.attrs
        output_ds.attrs['history'] = f"Calculated depth average (0-100cm) from 10 layers"
        template_ds.close()

        # 5. 写入文件 - 修正 encoding 参数
        save_file = os.path.join(output_path, f"SMCI_9km_{year}_Avg_0_100cm.nc")

        # zlib=True 是开启压缩，complevel=4 是压缩等级 (1-9)
        encoding = {
            'SMCI_avg': {
                'zlib': True,
                'complevel': 4,
                'shuffle': True
            }
        }

        output_ds.to_netcdf(save_file, encoding=encoding)

        # 释放资源
        for layer in year_layers:
            layer.close()
        print(f"成功保存: {save_file}")

    except FileNotFoundError:
        print(f"跳过：{year} 年存在缺失文件。")
    except Exception as e:
        print(f"处理 {year} 年时发生错误: {str(e)}")

print("\n所有年度数据处理完成！")