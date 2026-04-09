import xarray as xr
import numpy as np

# 加载 D 值文件
ds_d = xr.open_dataset(r'E:\data\meteorologicaldata\D_value_8day\D_value_8day_2001.nc')
d_var = ds_d['P-PET']

# 1. 检查最大值和最小值
d_max = d_var.max().values
d_min = d_var.min().values
d_mean = d_var.mean().values

print(f"数据最大值: {d_max}")
print(f"数据最小值: {d_min}")
print(f"全局平均值: {d_mean}")

# 2. 如果全是0，找出非零点的坐标
if d_max == 0 and d_min == 0:
    print("警报：该文件中所有数值均为 0！请检查原始 P 和 PET 数据。")
else:
    # 找到第一个非零值的索引
    nonzero_idx = np.where(d_var.values != 0)
    print(f"发现非零值！第一个非零点在索引: {nonzero_idx[0][0], nonzero_idx[1][0], nonzero_idx[2][0]}")