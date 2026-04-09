import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 原始数据
q = [16.86, 82.72, 285.19, 307.59, 368, 285.19, 218.98, 152.78, 101.85, 82.72, 74.95, 60.89, 42.16]
t = [0, 1, 2, 3,4, 6, 8, 10, 12, 13, 16, 20, 24]

# 要插值的目标时间点
t_target = np.arange(1, 25)  # 目标时间点 t = [1,2,3,...,24]

# 使用线性插值（默认）进行插值
interpolator = interp1d(t, q, kind='linear', fill_value="extrapolate")

# 计算对应的q值
q_target = interpolator(t_target)

# 输出插值结果
print("插值后的q值：", q_target)


