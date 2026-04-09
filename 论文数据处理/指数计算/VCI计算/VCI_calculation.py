import xarray as xr
import numpy as np
import os

# --- 1. 路径设置 ---
input_dir = r'E:\data\NDVI\NDVI_8D'
output_dir = r'E:\data\NDVI\VCI'
crops = ['Wheat', 'Maize', 'Rice']


def calculate_crop_vci_safe():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for crop in crops:
        input_path = os.path.join(input_dir, f'NDVI_8d_01deg_{crop}.nc')
        output_path = os.path.join(output_dir, f'VCI_8d_01deg_{crop}.nc')

        if not os.path.exists(input_path):
            continue

        print(f"🚀 正在计算 {crop} 的 VCI 指数 (安全模式)...")

        # 使用 chunks 开启延迟加载，防止一次性读入 3GB 数据
        ds = xr.open_dataset(input_path, chunks={'time': 100})
        ndvi = ds['NDVI']

        # 1. 计算极值库 (这些数据量很小，可以直接 compute)
        print("   - 正在计算历史极值库...")
        ndvi_max = ndvi.groupby('time.dayofyear').max().compute()
        ndvi_min = ndvi.groupby('time.dayofyear').min().compute()

        # 2. 核心修正：按 dayofyear 循环计算 VCI，避免内存广播爆炸
        print("   - 正在逐时段执行归一化...")
        vci_list = []

        # 遍历一年中的 46 个 8天周期
        for doy in ndvi_max.dayofyear.values:
            # 提取所有年份中属于这一 doy 的数据
            ndvi_subset = ndvi.sel(time=(ndvi['time.dayofyear'] == doy))
            max_subset = ndvi_max.sel(dayofyear=doy)
            min_subset = ndvi_min.sel(dayofyear=doy)

            # 执行计算
            vci_subset = (ndvi_subset - min_subset) / (max_subset - min_subset) * 100
            vci_list.append(vci_subset)

        # 3. 合并所有时段，并按原始时间排序
        print("   - 正在重组时间序列...")
        vci_final = xr.concat(vci_list, dim='time').sortby('time')

        # 4. 限制范围并保存
        vci_final = vci_final.clip(0, 100).rename('VCI')

        # 确保坐标里没有多余的 dayofyear
        if 'dayofyear' in vci_final.coords:
            vci_final = vci_final.drop_vars('dayofyear')

        print(f"   - 正在保存到磁盘...")
        encoding = {'VCI': {'zlib': True, 'complevel': 4}}
        vci_final.to_dataset().to_netcdf(output_path, encoding=encoding)

        print(f"✅ {crop} VCI 处理完成！")


if __name__ == "__main__":
    calculate_crop_vci_safe()