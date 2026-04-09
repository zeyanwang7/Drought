import os
import xarray as xr
import numpy as np
from scipy.stats import norm
from osgeo import gdal
import pandas as pd

# --- 1. 路径与参数设置 ---
# 输入 8 天尺度的 VPD 均值目录（之前脚本生成的年度文件）
input_dir = r'E:\data\meteorologicaldata\VPD_8Day_result'
# 掩膜路径（水稻种植区）
mask_path = r'E:\data\全国1Km作物种植数据\水稻\Rice_Distribution_010deg_Average.tif'
# 输出结果保存路径
output_folder = r'E:\data\meteorologicaldata\IndexCalculationResults\SVPD'
output_path = os.path.join(output_folder, 'SVPD_010deg_8day_Rice_2000_2022.nc')

#棉花作物
# mask_path = r'E:\data\新疆棉花作物种植地图\Cotton_Union_01deg_Presence.tif'
# output_folder = r'E:\data\meteorologicaldata\IndexCalculationResults\SVPD'
# output_path = os.path.join(output_folder, 'SVPD_010deg_8day_Cotton_2000_2022.nc')
def calculate_svpd_final():
    # 确保存储目录存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # --- 1. 加载并修正掩膜方向 ---
    print("正在加载并修正掩膜方向...")
    mask_ds = gdal.Open(mask_path)
    mask_array = mask_ds.ReadAsArray()
    # TIF 坐标系通常与 NC 的 Lat 方向相反，进行上下翻转
    mask_array_fixed = np.flipud(mask_array)
    # 定义种植区阈值
    crop_mask = mask_array_fixed > 0

    # --- 2. 加载 VPD 气象数据 (2000-2022) ---
    print("正在加载 2000-2022 年 VPD 8天尺度数据...")
    years = range(2000, 2023)  # 包含 2000 到 2022
    files = [os.path.join(input_dir, f"VPDmax_30d_mean_8day_{year}.nc") for year in years]

    # 过滤掉不存在的文件
    existing_files = [f for f in files if os.path.exists(f)]
    if len(existing_files) == 0:
        print("❌ 错误：未在输入目录找到任何 .nc 文件，请检查路径。")
        return

    # 使用 dask 加载数据以优化内存
    ds = xr.open_mfdataset(existing_files, combine='nested', concat_dim='time',
                           chunks={'time': -1, 'lat': 100, 'lon': 100})

    # --- 3. 定义 SVPD 计算函数 ---
    def compute_svpd_core(x, is_crop):
        """
        基于 Gringorten 公式进行非参数标准化转换
        """
        if not is_crop or np.all(np.isnan(x)):
            return np.full(x.shape, np.nan)

        # 提取有效值（排除由于空值导致的计算错误）
        valid_indices = np.where(~np.isnan(x))[0]
        valid_data = x[valid_indices]
        n = len(valid_data)

        # 样本数太少（少于5年）则不计算，返回空
        if n < 5:
            return np.full(x.shape, np.nan)

        # Gringorten 公式计算经验概率
        # VPD 越大表示越干旱，Rank 越高，计算出的 SVPD 为正且值越大
        ranks = pd.Series(valid_data).rank(method='min').values
        p = (ranks - 0.44) / (n + 0.12)
        p = np.clip(p, 0.0001, 0.9999)

        # 转换为标准正态分布分位数
        result = np.full(x.shape, np.nan)
        result[valid_indices] = norm.ppf(p)
        return result

    # --- 4. 执行标准化计算 (逐期校正) ---
    print("正在执行逐像素、逐阶段 SVPD 标准化计算...")
    # 将掩膜转为 DataArray 以便对齐坐标
    mask_da = xr.DataArray(crop_mask, coords=[ds.lat, ds.lon], dims=['lat', 'lon'])

    # 变量名（对应之前脚本生成的变量）
    var_name = 'VPDmax_30d_avg'

    # 按 time.dayofyear 分组，确保同阶段横向对比
    svpd_final = xr.apply_ufunc(
        compute_svpd_core,
        ds[var_name].groupby('time.dayofyear'),
        mask_da,
        input_core_dims=[['time'], []],
        output_core_dims=[['time']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[np.float32]
    )

    # --- 5. 存储结果 ---
    print("正在写入合并后的 NC 文件...")
    # 转换维度顺序以符合标准
    svpd_final = svpd_final.transpose('time', 'lat', 'lon')
    svpd_ds = svpd_final.to_dataset(name='SVPD')

    # 添加属性说明
    svpd_ds['SVPD'].attrs = {
        'units': 'dimensionless',
        'long_name': 'Standardized Vapor Pressure Deficit',
        'calculation_method': 'Gringorten non-parametric method',
        'period': '2000-2022'
    }

    # 采用 Zlib 压缩
    encoding = {'SVPD': {'zlib': True, 'complevel': 5}}
    svpd_ds.to_netcdf(output_path, encoding=encoding)

    print(f"\n✅ SVPD 计算完成！")
    print(f"输出路径: {output_path}")


if __name__ == "__main__":
    calculate_svpd_final()